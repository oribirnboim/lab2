import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import os
from pathlib import Path

def load_image(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def threshold(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def morpho(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

def label(image):
    labels = measure.label(image, connectivity=2)
    label_image = np.zeros_like(image, dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:  # Background
            continue
        label_image[labels == label] = 255
    return labels, label_image

def color_label_image(labels):
    return label2rgb(labels, bg_label=0)

def extract_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def load_images_from_folder(folder_path, extension="jpg"):
    folder = Path(folder_path).resolve()
    image_paths = list(folder.glob(f'*.{extension}'))
    
    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
    return images

def calculate_variance(images):
    # Stack images along the third dimension to create a 3D array
    stacked_images = np.stack(images, axis=-1)
    # Calculate the variance along the third dimension (axis=-1)
    variance_image = np.var(stacked_images, axis=-1)
    return variance_image

def visualize_variance(variance_image):
    plt.figure(figsize=(10, 8))
    plt.title("Variance Map of Magnetic Domains")
    plt.imshow(variance_image, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Variance')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    folder_path = 'part_1_batch_3'
    images = load_images_from_folder(folder_path, extension="jpg")
    
    if len(images) == 0:
        print(f"No images found in folder: {folder_path}")
    else:
        variance_image = calculate_variance(images)
        visualize_variance(variance_image)