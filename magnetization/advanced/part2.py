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


if __name__ == "__main__":
    # # Load the two subsequent photos
    # # image1 = load_image('part_2_1_up/20240521_095933_150.jpg')
    # # image2 = load_image('part_2_1_up/20240521_095928_149.jpg')
    # image1 = load_image('part_2_1_up/20240521_100043_164.jpg')
    # image2 = load_image('part_2_1_up/20240521_100038_163.jpg')

    # # Ensure both images have the same dimensions
    # min_height = min(image1.shape[0], image2.shape[0])
    # min_width = min(image1.shape[1], image2.shape[1])
    # image1 = image1[:min_height, :min_width]
    # image2 = image2[:min_height, :min_width]

    # # Subtract the first image from the second one
    # difference_image = cv2.absdiff(image2, image1)


    # blurred = blur(difference_image)
    # thresholded = threshold(blurred)
    # morph = morpho(thresholded)
    # labels, label_image = label(morph)
    # color_labels = color_label_image(labels)

    # # Display the results
    # fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    # ax[0].imshow(difference_image, cmap='gray')
    # ax[0].set_title('Image 1')
    # ax[0].axis('off')

    # ax[1].imshow(label_image, cmap='gray')
    # ax[1].set_title('Labeled Binary Image')
    # ax[1].axis('off')

    # ax[2].imshow(color_labels)
    # ax[2].set_title('Color Labeled Image')
    # ax[2].axis('off')

    # plt.show()
    pass