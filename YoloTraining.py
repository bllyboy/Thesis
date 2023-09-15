import cv2
import pytesseract
from pathlib import Path
from PIL import Image
from IPython.display import display
import os
import torch
import sys

sys.path.append('C:/Users/Adam/yolov5')  # add path to yolov5 directory
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import non_max_suppression, scale_boxes
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the command to train the YOLOv5 model
# Path to YOLOv5 repository
yolo_dir = r'C:\Users\Adam\yolov5'

# Paths to training and testing data
train_data = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\train'
test_data = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\test'

# Path to the .yaml file
yaml_path = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\data.yaml'

# Learning rate scheduling parameters (cosine annealing)
#lr0 = 0.01  # Initial learning rate
#lrf = 0.0001  # Final learning rate

# Command to train the model
os.system(f'python {os.path.join(yolo_dir, "train.py")} --img 640 --batch 8 --epochs 45 --data {yaml_path} --weights yolov5s.pt --cos-lr')

#End of YOLO Training Block