import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import os
from pathlib import Path
from scipy.spatial.distance import cdist
import json

def load_image(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def threshold(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def morpho(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

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
    print(variance_image)
    return variance_image

def visualize_variance(variance_image):
    plt.figure(figsize=(10, 8))
    plt.title("Variance Map of Magnetic Domains", fontsize=20)
    img = plt.imshow(variance_image, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar(img, label='Variance')
    cbar.ax.tick_params(labelsize=12)  # Increase colorbar text size
    cbar.set_label('Variance', fontsize=20)  # Increase colorbar label size
    plt.axis('off')
    plt.show()

def select_defects(image):
    plt.figure(figsize=(10, 8))
    plt.title("Select Defects and Close Window")
    plt.imshow(image, cmap='hot', interpolation='nearest')
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return points

def save_defects(defects, filepath='defects.json'):
    with open(filepath, 'w') as file:
        json.dump(defects, file)

def load_defects(filepath='defects.json'):
    if Path(filepath).exists():
        with open(filepath, 'r') as file:
            defects = json.load(file)
        return defects
    return None

def distance_from_defects(image, defects):
    height, width = image.shape
    distances = np.full((height, width), np.inf)
    for defect in defects:
        defect_y, defect_x = int(defect[1]), int(defect[0])
        y, x = np.indices((height, width))
        distances = np.minimum(distances, np.sqrt((x - defect_x)**2 + (y - defect_y)**2))
    return distances

def plot_variance_vs_distance(variance_image, distance_image):
    distances = distance_image.flatten()
    variances = variance_image.flatten()
    plt.figure(figsize=(10, 8))
    plt.scatter(distances, variances, alpha=0.5)
    plt.title("Variance vs. Distance from Defects", fontsize=20)
    plt.xlabel("Distance from Defects (pixels)", fontsize=15)
    plt.ylabel("Variance", fontsize=15)
    plt.show()


def crop_image(image, crop_percent=10):
    """Crop the perimeter of the image by a given percentage."""
    height, width = image.shape[:2]
    crop_height = int(height * crop_percent / 100)
    crop_width = int(width * crop_percent / 100)
    cropped_image = image[crop_height:height-crop_height, crop_width:width-crop_width]
    return cropped_image

def crop_folder(input_folder, output_folder, crop_percent=10, extension="jpg"):
    input_folder_path = Path(input_folder).resolve()
    output_folder_path = Path(output_folder).resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_folder_path.glob(f'*.{extension}'))
    
    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        cropped_image = crop_image(image, crop_percent)
        output_image_path = output_folder_path / f"{idx:04d}.{extension}"
        cv2.imwrite(str(output_image_path), cropped_image)
        print(f"Saved cropped image: {output_image_path}")



if __name__ == "__main__":
    manual_selection = True
    folder_path = 'p1b3'
    images = load_images_from_folder(folder_path, extension="jpg")


    if len(images) == 0:
        print(f"No images found in folder: {folder_path}")
    else:
        variance_image = calculate_variance(images)
        # visualize_variance(variance_image)
        
        # Load defects if they exist, otherwise select and save them
        defect_filename = 'defects_' + folder_path + '.json'
        defects = load_defects(defect_filename)
        if (defects is None) or manual_selection:
            defects = select_defects(variance_image)
            save_defects(defects, defect_filename)
        
        # Calculate distance from defects
        distance_image = distance_from_defects(variance_image, defects)
        
        # Plot variance vs. distance
        plot_variance_vs_distance(variance_image, distance_image)



    # folder_path = 'part_1_batch_3'
    # output_folder = 'p1b3'
    # crop_percent = 20
    # crop_folder(folder_path, output_folder, crop_percent)
