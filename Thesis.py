import cv2
import pytesseract
from pathlib import Path
from PIL import Image
from IPython.display import display
import os
import torch
import sys
import random

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


# Begin OCR

names = ['idcard', 'nmidcard']

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # generate distinct colors for each class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = attempt_load('C:/Users/Adam/yolov5/runs/train/exp44/weights/best.pt').to(device)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define your expected values
expected_name = "TITOUAH ADAM"
expected_address = "47 Triq Santa Margerita Tas-Sliema"


#expected_dob = "DOB"

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


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_text_from_nru(text):
    parts = text.split("Nru", 1)  # split the text by "Nru", but only for the first occurrence
    if len(parts) > 1:
        return "Nru" + parts[1]  # return the part of the text starting from "Nru"
    else:
        return text  # if "Nru" is not in the text, return the text unchanged




# [The imports and initial setup remain unchanged]

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
            print("Detection found!")
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            #debug
            for *xyxy, conf, cls in reversed(det):
                print(f"Detected class: {int(cls)}, Confidence: {conf}")

            # Extract ID card only for Maltese ID class (class 0)
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} ({conf:.2f})"
                plot_one_box(xyxy, img0, color=colors[int(cls)], label=label, line_thickness=3)
                if int(cls) == 0:  # Check if the detected class is Maltese ID
                    print("Maltese ID found!")
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
    cv2.imwrite(f"debug_{i}.jpg", img0)
    # Resize for visualization
    max_size = 800
    h, w = img0.shape[:2]
    if h > w:
        new_h, new_w = max_size, int(max_size * w / h)
    else:
        new_w, new_h = max_size, int(max_size * h / w)

    resized_img = cv2.resize(img0, (new_w, new_h))

    cv2.imshow('Detections', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ""

# [The rest of the code remains unchanged]


#End of OCR Block

# Uncomment following
# Extract texts from all images in the directory
image_directory = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\IDTrainPics\maltatest.JPG'

output_file_path = 'ThesisOCRResults.txt'

# No need for a loop here, just process the image directly
extracted_text = process_image(image_directory)
filename = os.path.basename(image_directory)  # Get the filename from the image_path

# Write the results to your output file
with open(output_file_path, 'w') as file:
    file.write(f'Filename: {filename}\n')
    file.write(f'Extracted Text: {extracted_text}\n\n')


