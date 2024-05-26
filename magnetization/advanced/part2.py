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


def custom_label(image):
    labels = measure.label(image, connectivity=2)
    label_image = np.zeros_like(image, dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:  # Background
            continue
        label_image[labels == label] = 255
    return labels, label_image

def plot_average_area_vs_voltage(folderpath, start_voltage, end_voltage):
    # List all jpg files in the folderpath
    files = [f for f in os.listdir(folderpath) if f.endswith('.jpg')]
    
    # Sort files based on the counter in their names
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    num_images = len(files)
    voltage_interval = (end_voltage - start_voltage) / (num_images - 1)
    voltages = [start_voltage + i * voltage_interval for i in range(num_images)]
    
    average_areas = []
    
    for file in files:
        # Read the difference image
        img_path = os.path.join(folderpath, file)
        difference_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the processing steps
        blurred = cv2.GaussianBlur(difference_image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        # Label the connected components using your custom label function
        labels, _ = custom_label(morph)
        
        # Ensure labels is an integer array
        labels = labels.astype(np.int32)
        
        # Calculate the area of each labeled region
        areas = [np.sum(labels == i) for i in range(1, labels.max() + 1)]
        average_area = np.mean(areas) if areas else 0
        average_areas.append(average_area)
    
    # Plot the average areas vs. voltage
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, average_areas, marker='o', linestyle='', color='b')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Average Area of Labeled Regions (pixels)')
    plt.title('Average Area of Labeled Regions vs. Voltage')
    plt.grid(True)
    plt.show()


def get_label_sizes(image_path):
    # Read the difference image
    difference_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply the processing steps
    blurred = cv2.GaussianBlur(difference_image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Label the connected components using your label function
    labels, _ = label(morph)
    
    # Ensure labels is an integer array
    labels = labels.astype(np.int32)
    
    # Calculate the area of each labeled region
    areas = [np.sum(labels == i) for i in range(1, labels.max() + 1)]
    
    return areas



if __name__ == "__main__":
    # folderpath = 'part_2_batch_2'
    # diff_path = 'diff_part_2_batch_2'
    # absdiff_folder(folderpath, diff_path)

    # diff_folderpath = 'diff_part_2_batch_2'
    # output_folderpath = 'label_part_2_batch_2'
    # process_difference_images(diff_folderpath, output_folderpath)

    diff_folderpath = 'diff_part_2_1_up'
    plot_average_area_vs_voltage(diff_folderpath, 0.02, 6.04)


    # diff_folderpath = 'diff_part_2_batch_2'
    # plot_average_area_vs_voltage(diff_folderpath, 0, 6)


    # diff_photo = 'diff_part_2_1_up/diff_1.jpg'
    # print(get_label_sizes(diff_photo))