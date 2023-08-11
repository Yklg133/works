# Imports
import os
from PIL import Image
import numpy as np
import shutil
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
# Add dependencies to sys path, to be able to import from dependencies folder
sys.path.append('dependencies')

# Dependencies folder contains helpder functions to simplify training and evaluation from TorchVision repo
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# Hyperparameters
test_set_length = 500         # Test set (number of images)
train_batch_size = 2          # Train batch size
test_batch_size = 1           # Test batch size
num_classes = 3               # Number of classes
# Number of classes - background + cat + dog
learning_rate = 0.005         # Learning rate
num_epochs = 10               # Number of epochs
output_dir = "saved_model"    # Output directory to save the model

class OxfordPetDataset(data.Dataset):

    """ The dataset contains images, masks and annotations
        The dataset also includes the breed of cats and dogs """

    def __init__(self, root, transforms = None):

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, "Xmls"))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        xml = ET.parse(os.path.join(self.root, "Xmls", self.xmls[idx]))
        
        # Extract class(label) name from xml file
        annotation = xml.getroot()
        class_name = annotation[5][0].text

        # Assign label id, 1 - cat and 2 - dog
        if class_name == 'cat':
            label_id = 1
        elif class_name == 'dog':
            label_id = 2

        # Read image
        img = Image.open(img_path).convert("RGB")

        # Read mask and convert to numpy array
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Instances are encoded as different colors
        obj_ids = np.unique(mask)

        # First id is background, so remove it
        obj_ids = obj_ids[1:]

        # Creating binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bounding box coordinates for each mask
        boxes = []
        pos = np.where(masks[1])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        # Convert bounding box coordinates into torch tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)

        # Convert label into torch tensor
        labels = torch.tensor((label_id,), dtype = torch.int64)

        # Convert boolean to 1's and 0's
        masks = masks.astype('uint8')

        # The array contains two masks, first one is the mask without boundary around the object
        # The second array is the boundary around the object
        # Add both the arrays
        masks = masks[0] + masks[1]

        # Converting all the 1's into 0's and 0's into 1's
        # This is done in order to swap the mask covering the object to the mask covering the area other than object
        indices_one = masks == 1 
        indices_zero = masks == 0
        masks[indices_zero] = 1
        masks[indices_one] = 0

        # Add batch_size (1) to the mask shape
        masks = np.reshape(masks, (1, masks.shape[0], masks.shape[1]))

        # Convert mask into torch tensor
        masks = torch.as_tensor(masks, dtype = torch.uint8)

        image_id = torch.tensor([idx])

        # Area of the bouding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.tensor((label_id,), dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(num_classes):

    # Load instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for mask classification
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transforms(train):

    transforms = []

    # Convert numpy image to PyTorch Tensor
    transforms.append(T.ToTensor())

    if train:
        # Data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

if __name__ == '__main__':
    
    # Spliting the data into train and validation based on availability of annotations
    os.mkdir("OxfordDataset")
    os.mkdir("OxfordDataset/Images")
    os.mkdir("OxfordDataset/Masks")
    os.mkdir("OxfordDataset/Xmls")
    os.mkdir("OxfordDataset/Images_val")
    os.mkdir("OxfordDataset/Masks_val")

    xml_list = list(sorted(os.listdir('annotations/xmls')))

    for i in xml_list:
        source = os.path.join("annotations/xmls/", i)
        dest = "OxfordDataset/Xmls/"
        shutil.move(source, dest)

    images_list = []
    masks_list = []

    for xml_string in xml_list:
        image_string = xml_string.replace(".xml",".jpg")
        mask_string = xml_string.replace(".xml",".png")
        images_list.append(image_string)
        masks_list.append(mask_string)

    for i in masks_list:
        source = os.path.join("annotations/trimaps/", i)
        dest = "OxfordDataset/Masks/"
        shutil.move(source, dest)

    print("Done creating train masks folder")

    for i in images_list:
        source = os.path.join("images", i)
        dest = "OxfordDataset/Images/"
        shutil.move(source, dest)

    print("Done creating train images folder")

    val_masks_list = list(sorted(os.listdir("annotations/trimaps/")))
    val_images_list = list(sorted(os.listdir("images")))

    for i in val_masks_list:
        source = os.path.join("annotations/trimaps/", i)
        dest = "OxfordDataset/Masks_val/"
        shutil.move(source, dest)

    print("Done creating validation masks folder")

    for i in val_images_list:
        source = os.path.join("images", i)
        dest = "OxfordDataset/Images_val/"
        shutil.move(source, dest)

    print("Done creating validation images folder")

    # Remove corrupted images
    os.remove("OxfordDataset/Images/Egyptian_Mau_129.jpg")
    os.remove("OxfordDataset/Masks/Egyptian_Mau_129.png")
    os.remove("OxfordDataset/Xmls/Egyptian_Mau_129.xml")

    os.remove("OxfordDataset/Images/Egyptian_Mau_162.jpg")
    os.remove("OxfordDataset/Masks/Egyptian_Mau_162.png")
    os.remove("OxfordDataset/Xmls/Egyptian_Mau_162.xml")

    os.remove("OxfordDataset/Images/Egyptian_Mau_165.jpg")
    os.remove("OxfordDataset/Masks/Egyptian_Mau_165.png")
    os.remove("OxfordDataset/Xmls/Egyptian_Mau_165.xml")

    os.remove("OxfordDataset/Images/Egyptian_Mau_196.jpg")
    os.remove("OxfordDataset/Masks/Egyptian_Mau_196.png")
    os.remove("OxfordDataset/Xmls/Egyptian_Mau_196.xml")

    os.remove("OxfordDataset/Images/leonberger_18.jpg")
    os.remove("OxfordDataset/Masks/leonberger_18.png")
    os.remove("OxfordDataset/Xmls/leonberger_18.xml")

    os.remove("OxfordDataset/Images/miniature_pinscher_14.jpg")
    os.remove("OxfordDataset/Masks/miniature_pinscher_14.png")
    os.remove("OxfordDataset/Xmls/miniature_pinscher_14.xml")

    os.remove("OxfordDataset/Images/saint_bernard_108.jpg")
    os.remove("OxfordDataset/Masks/saint_bernard_108.png")
    os.remove("OxfordDataset/Xmls/saint_bernard_108.xml")
    
    # Set up the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define train and test dataset
    dataset = OxfordPetDataset('OxfordDataset', get_transforms(train = True))

    dataset_test = OxfordPetDataset('OxfordDataset', get_transforms(train = False))

    # Split the dataset into train and test
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-test_set_length])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_set_length:])

    # Define train and test dataloaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = train_batch_size, 
                    shuffle = True, num_workers = 4, collate_fn = utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = test_batch_size, 
                    shuffle = False, num_workers = 4, collate_fn = utils.collate_fn)

    print(f"We have: {len(indices)} images in the dataset, {len(dataset)} are training images and {len(dataset_test)} are test images")


    # Get the model using helper function
    model = get_model(num_classes)
    model.to(device)

    # Construct the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 0.0005)

    # Learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)


    for epoch in range(num_epochs):

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluate on the test datase
        evaluate(model, data_loader_test, device = device)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save the model state  
    torch.save(model.state_dict(), output_dir + "/model")