import cv2
import numpy as np
import matplotlib.pyplot as plt
from part1 import *

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def absdiff_folder(folderpath, respath):
    # Ensure the result directory exists
    if not os.path.exists(respath):
        os.makedirs(respath)
    
    # List all jpg files in the folderpath
    files = [f for f in os.listdir(folderpath) if f.endswith('.jpg')]
    
    # Sort files based on the counter in their names
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    # Loop through the sorted files and calculate absolute differences
    for i in range(len(files) - 1):
        # Read the consecutive images
        img1_path = os.path.join(folderpath, files[i])
        img2_path = os.path.join(folderpath, files[i + 1])
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        # Compute the absolute difference
        abs_diff = cv2.absdiff(img1, img2)
        
        # Save the result
        diff_filename = f'diff_{i+1}.jpg'
        diff_path = os.path.join(respath, diff_filename)
        cv2.imwrite(diff_path, abs_diff)


def process_difference_images(folderpath, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all jpg files in the folderpath
    files = [f for f in os.listdir(folderpath) if f.endswith('.jpg')]
    
    # Sort files based on the counter in their names
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    for i, file in enumerate(files):
        # Read the difference image
        img_path = os.path.join(folderpath, file)
        difference_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the processing steps
        blurred = cv2.GaussianBlur(difference_image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        labels, _ = label(morph)
        color_labels = label2rgb(labels, bg_label=0)
        
        # Save the color labeled image
        color_label_filename = f'color_label_{i+1}.jpg'
        color_label_path = os.path.join(output_folder, color_label_filename)
        plt.imsave(color_label_path, color_labels)


if __name__ == "__main__":
    # folderpath = 'part_2_1_up'
    # diff_path = 'diff_part_2_1_up'
    # absdiff_folder(folderpath, diff_path)

    diff_folderpath = 'diff_part_2_1_up'
    output_folderpath = 'label_part_2_1_up'
    process_difference_images(diff_folderpath, output_folderpath)