import cv2
import numpy as np
import matplotlib.pyplot as plt
from part1 import *

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":
    # Load the two subsequent photos
    image1 = load_image('part_2_1_up/20240521_095933_150.jpg')
    image2 = load_image('part_2_1_up/20240521_095928_149.jpg')

    # Ensure both images have the same dimensions
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    # Subtract the first image from the second one
    difference_image = cv2.absdiff(image2, image1)


    blurred = blur(difference_image)
    thresholded = threshold(blurred)
    morph = morpho(thresholded)
    labels, label_image = label(morph)
    color_labels = color_label_image(labels)

    # Display the results
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    ax[0].imshow(difference_image, cmap='gray')
    ax[0].set_title('Image 1')
    ax[0].axis('off')

    ax[1].imshow(label_image, cmap='gray')
    ax[1].set_title('Labeled Binary Image')
    ax[1].axis('off')

    ax[2].imshow(color_labels)
    ax[2].set_title('Color Labeled Image')
    ax[2].axis('off')

    plt.show()