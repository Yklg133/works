# Imports
import numpy as np
import cv2
import os

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

saved_model = "saved_model"    # Output directory of the saved the model
num_classes = 3                # Number of classes
threshold = 0.1             # Minimum threshold for pixel-wise mask segmentation

filename = "nighttime1.jpeg"      # Image filename
img_path = "CA4/Part7/" + filename

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


img = cv2.imread(img_path)

# Create a copy of the original image
img_cp = img
# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert numpy to torch tensor and reshape
img = torchvision.transforms.ToTensor()(img)

# Load the model
loaded_model = get_model(num_classes)
loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 'model'), map_location = 'cpu'))

loaded_model.eval()
with torch.no_grad():
    prediction = loaded_model([img])

# Get bounding box from prediction
box = prediction[0]['boxes'][0]
box = np.array(box).astype("int")

# Get label id from prediction
label_id = prediction[0]['labels']
label_id = np.array(label_id)
label_id = label_id[0]

# Get score from prediction
score = prediction[0]['scores']
score = np.array(score)
score = score[0] * 100
score = np.round(score, decimals = 2)

# Set color of the mask based on the label_id
# In this case, red if the prediction is cat and green if the prediction is dog
if label_id == 1:
    label_name = "cat"
    color = [0, 0, 255]

elif label_id == 2:
    label_name = "dog"
    color = [0, 255, 0]

# Get mask from prediction
mask = prediction[0]['masks'][0][0]
mask = np.array(mask)
mask = (mask > threshold)
visible_mask = (mask * 255).astype("uint8")

instance = cv2.bitwise_and(img_cp, img_cp, mask = visible_mask)

color = np.array(color, dtype="uint8")

# Create color mask
roi = img_cp[mask]
blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
img_cp[mask] = blended

cv2.rectangle(img_cp, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

text = label_name + " " + str(score) + "%"

(text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
cv2.rectangle(img_cp, (box[0], box[1] - text_height - 2), (box[0] + text_width, box[1]), (0, 0, 255), -1)

cv2.putText(img_cp, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.imshow("image", img_cp)
cv2.imwrite("s_" + filename, img_cp)
cv2.waitKey(0)
