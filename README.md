# Train cats vs dogs instance segmentation model
## 1. Introduction
Train a cats vs dogs instance segmentation model from scratch using Mask R-CNN in PyTorch

In this project we use the popular Oxford-IIIT Pet Dataset.

Train.ipynb is the google colaboratory version of this project 

## 2 Prerequisites
- Python 3
- Numpy
- PIL
- PyTorch 1.8.1
- Torchvision
- Opencv (cv2)

## 3. Installation

1. Clone the respository
```bash
git clone https://github.com/harshatejas/cats_vs_dogs_instance_segmentation.git
cd cats_vs_dogs_instance_segmentation/
```
2. Dowload the dataset

   Head over to https://www.robots.ox.ac.uk/~vgg/data/pets/ to download The Oxford-IIIT Pet Dataset
   
   This will download and extract the images and annotations to cats_vs_dogs_instance_segmentation directory
   
```bash
# Download the dataset
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```
```bash
# Extract the dataset
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

   3. The dataset is split into train and validation by train.py, it is done based on the availability of annotations (.xml)
```bash
OxfordDataset/
 Images/
   [xxx].jpg
   ...
 Images_val/
   [xxx].jpg
   ...
 Masks/
   [xxx].png
   ...
 Masks_val/
   [xxx].png
   ...  
 Xmls/
   [xxx].xml
   ...
``` 

## 4. Train
Modify Hyperparameters in train.py

```bash
train.py
```

## 5. Test
predict.py is designed to run predictions on the images in validation folder (Images_val)

Change the filename and saved_model in predict.py

```bash
predict.py
```

## 6. Predicted Images
Here are some sample output images predicted by saved_model/model

![beagle_1](https://user-images.githubusercontent.com/52169316/124116214-51591000-da8c-11eb-9276-d1db222174f7.jpg)

![Egyptian_Mau_45](https://user-images.githubusercontent.com/52169316/124116253-5f0e9580-da8c-11eb-937a-69e797f953f4.jpg)

![shiba_inu_65](https://user-images.githubusercontent.com/52169316/124116308-73529280-da8c-11eb-9472-25ef174f1366.jpg)

![Ragdoll_43](https://user-images.githubusercontent.com/52169316/124116371-849b9f00-da8c-11eb-8aeb-9b06a37cb8e9.jpg)

![havanese_1](https://user-images.githubusercontent.com/52169316/124116473-9b41f600-da8c-11eb-8ee6-95a255821a6e.jpg)

![Russian_Blue_201](https://user-images.githubusercontent.com/52169316/124116511-a39a3100-da8c-11eb-9a0a-42fc835d76ac.jpg)

![samoyed_59](https://user-images.githubusercontent.com/52169316/124116556-b44aa700-da8c-11eb-8758-d0bb979726eb.jpg)

![Russian_Blue_41](https://user-images.githubusercontent.com/52169316/124116599-c4fb1d00-da8c-11eb-8670-3953b295c934.jpg)

![shiba_inu_25](https://user-images.githubusercontent.com/52169316/124116635-cd535800-da8c-11eb-8775-1904724b9498.jpg)

![British_Shorthair_21](https://user-images.githubusercontent.com/52169316/124117517-de509900-da8d-11eb-8a3a-d63a7a5f7596.jpg)
