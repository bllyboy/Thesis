import cv2
import numpy as np
import os

def rotate_image(image, angle):
    # Get image height and width
    (h, w) = image.shape[:2]

    # Define the center of the image
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def blur_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def darken_image(image, factor=0.5):
    return np.clip(factor * image, 0, 255).astype(np.uint8)


def zoom_out(image, factor=0.8):
    """
    Zooms out the image by resizing it to the specified factor and then
    centering it on a black canvas of the original size.

    Parameters:
    - image: The input image.
    - factor: The zoom out factor. E.g., 0.8 means the image will be resized to 80% of its original dimensions.

    Returns:
    - The zoomed-out image.
    """

    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h))

    # Create a black canvas of the original size
    canvas = np.zeros_like(image)

    # Place the resized image at the center of the canvas
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas


# Load the image
img_path = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\IDTrainPics\italyexample.JPG'
image = cv2.imread(img_path)

# Get directory and base filename
img_dir = os.path.dirname(img_path)
base_filename = os.path.basename(img_path)
base_filename_no_ext = os.path.splitext(base_filename)[0]

# Rotate/tilt the image
angle = 30  # The rotation angle, modify this as needed
rotated_image = rotate_image(image, angle)
cv2.imwrite(os.path.join(img_dir, f'{base_filename_no_ext}_rotated.jpg'), rotated_image)

# Blur the image
kernel_size = 11  # The size of the blur kernel, modify this as needed
blurred_image = blur_image(image, kernel_size)
cv2.imwrite(os.path.join(img_dir, f'{base_filename_no_ext}_blurred.jpg'), blurred_image)

# Darken the image
darken_factor = 0.3  # The darken factor, between 0 (black) and 1 (no change), modify this as needed
darkened_image = darken_image(image, darken_factor)
cv2.imwrite(os.path.join(img_dir, f'{base_filename_no_ext}_darkened.jpg'), darkened_image)

# Zoom out the image
zoom_factor = 0.65  # The zoom out factor, between 0 (image disappears) and 1 (no change), modify this as needed
zoomed_out_image = zoom_out(image, zoom_factor)
cv2.imwrite(os.path.join(img_dir, f'{base_filename_no_ext}_zoomed_out.jpg'), zoomed_out_image)
