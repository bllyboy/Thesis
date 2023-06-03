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

# Define the path to the labelled images folder
# labelled_images_path = "IDTrainPics"

# Define the percentage of images to be used for testing
# test_percent = 20

# Create train and test folders
# os.makedirs("train", exist_ok=True)
# os.makedirs("test", exist_ok=True)

# Get a list of all the labelled images
# labelled_images_list = os.listdir(labelled_images_path)

# Shuffle the labelled images list randomly
# random.shuffle(labelled_images_list)

# Determine the number of images for testing based on the test_percent
# test_count = int(len(labelled_images_list) * test_percent / 100)

# Move the first test_count images to the test folder and the remaining images to the train folder
# for i, img_path in enumerate(labelled_images_list):
#    if i < test_count:
#        os.rename(os.path.join(labelled_images_path, img_path), os.path.join("test", img_path))
#    else:
#        os.rename(os.path.join(labelled_images_path, img_path), os.path.join("train", img_path))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define the command to train the YOLOv5 model
# Path to YOLOv5 repository
# yolo_dir = r'C:\Users\Adam\yolov5'

# Paths to training and testing data
# train_data = r'\\train'
# test_data = r'\\test'

# Path to the .yaml file
# yaml_path = os.path.join(os.getcwd(), 'data.yaml')

# Command to train the model
# os.system(f'python {os.path.join(yolo_dir, "train.py")} --img 640 --batch 16 --epochs 50 --data {yaml_path} --weights yolov5s.pt')

# train_command = "python C:/Users/Adam/yolov5/train.py --data C:/Users/Adam/Documents/Adam/SCHOOL/FinalYear/Thesis/data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 3 --noval"


# Execute the command to train the YOLOv5 model
# os.system(train_command)


# Import YOLOv5


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
# model = attempt_load('C:/Users/Adam/yolov5/runs/train/exp31/weights/best.pt').to(device)

# Load image
# img_path = 'C:/Users/Adam/Documents/Adam/SCHOOL/FinalYear/Thesis/IDTrainPics/IMG_0661.jpg'
# img0 = cv2.imread(img_path)  # BGR
# img = letterbox(img0, 640, stride=32)[0]
# img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# img = np.ascontiguousarray(img)

# Perform inference
# img = torch.from_numpy(img).to(device)
# img = img.float()  # uint8 to fp16/32
# img /= 255.0  # (0 - 255) to (0.0 - 1.0)
# if img.ndimension() == 3:
#    img = img.unsqueeze(0)

# pred = model(img)[0]  # Perform inference

# Apply NMS
# pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)

# Process detections
# for i, det in enumerate(pred):  # detections per image
#    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#    if len(det):
# Rescale boxes from img_size to img0 size
#        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

# Extract ID card
#        for *xyxy, conf, cls in reversed(det):
#            x1, y1, x2, y2 = xyxy
#            id_card = img0[int(y1):int(y2), int(x1):int(x2)]
#            cv2.imshow('ID Card', id_card)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()

# Apply OCR to the cropped ID card
#            gray_id_card = cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY)

#            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#            text = pytesseract.image_to_string(gray_id_card)
#            print(text)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = attempt_load('C:/Users/Adam/yolov5/runs/train/exp31/weights/best.pt').to(device)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define your expected values
expected_name = "TITOUAH ADAM"
expected_address = "47 Triq Santa Margerita Tas-Sliema"


# expected_dob = "DOB"

def verify_information(extracted_text):
    extracted_text = extracted_text.upper().replace("\n", " ")  # Convert to upper case and replace newline characters
    extracted_text = ' '.join(extracted_text.split())  # Replace multiple spaces with a single space

    expected_name_upper = expected_name.upper().replace(" ", "")  # Convert to upper case and remove spaces
    expected_address_upper = expected_address.upper().replace(" ", "")  # Convert to upper case and remove spaces

    # Remove spaces from the extracted text as well before checking
    name_in_text = expected_name_upper in extracted_text.replace(" ", "")
    address_in_text = expected_address_upper in extracted_text.replace(" ", "")

    print("Extracted text:", extracted_text)
    print("Is the expected name in the text?", name_in_text)
    print("Is the expected address in the text?", address_in_text)

    return name_in_text and address_in_text


def get_text_from_nru(text):
    parts = text.split("Nru", 1)  # split the text by "Nru", but only for the first occurrence
    if len(parts) > 1:
        return "Nru" + parts[1]  # return the part of the text starting from "Nru"
    else:
        return text  # if "Nru" is not in the text, return the text unchanged




def process_image(img_path):
    img0 = cv2.imread(img_path)  # BGR
    img = letterbox(img0, 640, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Perform inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # (0 - 255) to (0.0 - 1.0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]  # Perform inference

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Extract ID card
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = xyxy
                id_card = img0[int(y1):int(y2), int(x1):int(x2)]
                # Apply OCR to the cropped ID card
                gray_id_card = cv2.cvtColor(id_card, cv2.COLOR_BGR2GRAY)

                # Resizing the image
                gray_id_card = cv2.resize(gray_id_card, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

                # Binarization
                (thresh, binarized_image) = cv2.threshold(gray_id_card, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Median filtering for noise reduction
                processed_image = cv2.medianBlur(binarized_image, 3)

                # OCR with English language specified
                custom_config = r'--oem 3 --psm 3'
                text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')
                text = get_text_from_nru(text)


                # Verify the information
                if verify_information(text):
                    print("User is verified")
                else:
                    print("User could not be verified")
                return text

    return ""


# Define Levenshtein distance function
# def levenshtein(s1, s2):
#    return distance(s1, s2)

# Extract texts from all images in the directory
image_directory = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\IDTrainPics\IMG_0661.JPG'

output_file_path = 'ThesisOCRResults.txt'

# No need for a loop here, just process the image directly
extracted_text = process_image(image_directory)
filename = os.path.basename(image_directory)  # Get the filename from the image_path

# Write the results to your output file
with open(output_file_path, 'w') as file:
    file.write(f'Filename: {filename}\n')
    file.write(f'Extracted Text: {extracted_text}\n\n')

#with open(output_file_path, 'w') as file:
#    for filename in os.listdir(image_directory):
#        if filename.endswith(".JPG"):  # Add more conditions if there are other image formats
#            image_path = os.path.join(image_directory, filename)
#            extracted_text = process_image(image_path)
#            file.write(f'Filename: {filename}\n')
#            file.write(f'Extracted Text: {extracted_text}\n\n')

            # Assuming you have a ground truth for each image:
            # ground_truth = get_ground_truth(filename)
            # lev_distance = levenshtein(extracted_text, ground_truth)
            # print(f'Levenshtein Distance for {filename}: {lev_distance}')
