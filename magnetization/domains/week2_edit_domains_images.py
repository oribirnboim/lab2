import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


def cut_images_in_directory(directory_path):
    # Get a list of all BMP files in the chosen directory
    bmp_files = [file for file in os.listdir(directory_path) if file.endswith(".bmp")]

    # Loop through each BMP file
    for bmp_file in bmp_files:
        # Construct the full path of the BMP file
        bmp_path = os.path.join(directory_path, bmp_file)

        # Open the BMP file using Pillow
        image = Image.open(bmp_path)

        # Get the width and height of the image
        width, height = image.size

        # Set the coordinates for the cropping region
        left = 50
        top = 50
        right = width - 50
        bottom = height - 50

        # Check if the cropping region is valid
        if left < right and top < bottom:
            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))

            # Save the cropped image, overwriting the original BMP file
            cropped_image.save(bmp_path)

            # Close the image files
            cropped_image.close()
            image.close()
        else:
            print(f"Invalid cropping coordinates for {bmp_file}. Skipping.")


def threshold_image2(image_path, target_color, channel_error):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (if it's in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate lower and upper bounds for each channel
    lower_bound = np.array([max(0, target - error) for target, error in zip(target_color, channel_error)])
    upper_bound = np.array([min(255, target + error) for target, error in zip(target_color, channel_error)])

    # Create a binary mask based on the threshold
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Invert the mask, so the target +- error becomes black and the rest becomes white
    result = cv2.bitwise_not(mask)

    return result

def threshold_images_in_directory(directory_path, n):
    # Get a list of all image files in the chosen directory
    image_files = sorted([file for file in os.listdir(directory_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    # Define target colors and channel errors for dark and bright images
    target_bright = [215, 133, 0]  # Example RGB values for bright images
    bright_error = [30, 37, 0]  # Example error for each channel for bright images
    target_dark = [215, 133, 0]  # Example RGB values for dark images
    dark_error = [30, 37, 0]  # Example error for each channel for dark images
# 232, 115 - 232, 130 - 210, 98 - 232, 170 - dark



    # Loop through each image file
    for i, image_file in enumerate(image_files):
        # Construct the full path of the image file
        image_path = os.path.join(directory_path, image_file)

        # Choose the appropriate target color and channel error
        target_color = target_dark if i <= int(n) else target_bright
        channel_error = dark_error if i <= int(n) else bright_error

        # Apply the thresholding using threshold_image2 function
        thresholded_image = threshold_image2(image_path, target_color, channel_error)

        # Save the thresholded image using imwrite, overwriting the original file
        cv2.imwrite(image_path, thresholded_image)



def gradient_images_in_directory(directory_path):
    # Get a list of all image files in the chosen directory
    image_files = [file for file in os.listdir(directory_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Loop through each image file
    for image_file in image_files:
        # Construct the full path of the image file
        image_path = os.path.join(directory_path, image_file)

        # Compute gradient
        gradient_image = compute_gradient(image_path)

        # Save the gradient image, overwriting the original file
        cv2.imwrite(image_path, gradient_image)


def calculate_average_pixel(image_path):
    image = Image.open(image_path)
    # Calculate the average RGB value for the entire image
    average_rgb = np.mean(image, axis=(0, 1)).astype(int)
    return average_rgb


def threshold_image(image_path, threshold_value):
    # Open the image file using Pillow
    original_image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = original_image.convert('L')

    # Apply thresholding
    thresholded_image = grayscale_image.point(lambda x: 0 if x < threshold_value else 255, '1')

    # Close the image files
    grayscale_image.close()
    original_image.close()

    return thresholded_image


def threshold_image2(image_path, target_color, channel_error):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (if it's in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate lower and upper bounds for each channel
    lower_bound = np.array([max(0, target - error) for target, error in zip(target_color, channel_error)])
    upper_bound = np.array([min(255, target + error) for target, error in zip(target_color, channel_error)])

    # Create a binary mask based on the threshold
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Invert the mask, so the target +- error becomes black and the rest becomes white
    result = cv2.bitwise_not(mask)

    return result


def compute_gradient(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply the Sobel operator to compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return gradient_magnitude_normalized


def show_images_side_by_side(original_image, thresholded_image):
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    thresholded_array = np.array(thresholded_image)

    # Plotting the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(thresholded_array, cmap='gray')
    axes[1].set_title('Thresholded Image')
    axes[1].axis('off')

    plt.show()


def choose_random_image(directory_path):
    # Get a list of all BMP files in the chosen directory
    bmp_files = [file for file in os.listdir(directory_path) if file.lower().endswith('.bmp')]

    # Check if there are BMP files in the directory
    if not bmp_files:
        print(f"No BMP files found in {directory_path}")
        return None

    # Choose a random BMP file
    random_bmp_file = random.choice(bmp_files)

    # Return the full path to the randomly chosen BMP file
    return os.path.join(directory_path, random_bmp_file)


def threshold_random_image(directory_name, threshold_value, it):
    target_bright = [230, 150, 0]  # Example RGB values
    bright_error = [10, 20, 0]  # Example error for each channel
    target_dark = [215, 105, 0]
    dark_error = [20, 20, 0]

    for i in range(it):
        image_path = choose_random_image(directory_name)
        original_image = Image.open(image_path)
        thresholded_image = threshold_image2(image_path, target_dark, dark_error)
        show_images_side_by_side(original_image, thresholded_image)


def count_black_pixels_in_directory(directory_path):
    # Get the total number of BMP files in the chosen directory
    num_files = len([file for file in os.listdir(directory_path) if file.lower().endswith('.bmp')])

    # Initialize a list to store black pixel count for each image
    black_pixel_counts = []

    # Loop through each BMP file from 0 to num_files-1
    for i in range(num_files):
        bmp_file = f"{i}.bmp"  # Assuming filenames are 0.bmp, 1.bmp, ..., n.bmp
        print(bmp_file)

        # Construct the full path of the BMP file
        bmp_path = os.path.join(directory_path, bmp_file)

        # Open the BMP file using Pillow
        image = Image.open(bmp_path)

        # Get pixel data
        pixel_data = list(image.getdata())

        # Count black pixels (pixels with value 0)
        black_pixel_count = pixel_data.count(0)

        # Append the black pixel count to the list
        black_pixel_counts.append(black_pixel_count)

        # Close the image file
        image.close()

    return black_pixel_counts


def rename_and_sort_files(directory_path):
    # Get a list of all BMP files in the chosen directory
    bmp_files = [file for file in os.listdir(directory_path) if file.lower().endswith('.bmp')]

    # Extract number3 from each filename and create a list of tuples (original name, number3)
    file_info = [(bmp_file, int(bmp_file.split('_')[-1].split('.')[0])) for bmp_file in bmp_files]

    # Sort the list of tuples based on number3
    file_info.sort(key=lambda x: x[1])

    # Rename the BMP files in sorted order starting from 0
    for i, (original_name, _) in enumerate(file_info):
        new_name = f"{i}.bmp"
        original_path = os.path.join(directory_path, original_name)
        new_path = os.path.join(directory_path, new_name)
        os.rename(original_path, new_path)
        print(f"Renamed {original_name} to {new_name}")


def organize_directory(directory_name):
    """
    1- rename and sort the directory
    2- cut images
    """
    rename_and_sort_files(directory_name)
    cut_images_in_directory(directory_name)


def analyse_pixels_directory(directory_name):
    """
    1- threshold images in directory
    2- calc black pixels
    """
    threshold_value = 20
    total_pixels = 668 * 924
    # threshold_images_in_directory(directory_name, threshold_value)
    pixels_list = count_black_pixels_in_directory(directory_name)
    pixels_list = [count / total_pixels for count in pixels_list]
    return pixels_list


def organize_all_directories(base_directory):
    # Get a list of all directories inside the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Iterate over each directory and call analyse_directory
    for subdir in subdirectories:
        full_path = os.path.join(base_directory, subdir)
        organize_directory(full_path)


if __name__ == '__main__':
    directory_name = r'domains/second_up4'
    threshold_value = 140
    it = 8
    start_bright=27
    organize_directory(directory_name)
    # threshold_random_image(directory_name, threshold_value, it)

    # Extract values from exel
    excel_file_path = r'domains/voltages.xlsx'
    df = pd.read_excel(excel_file_path)
    up5 = df['up5'].tolist()
    up5 = [value for value in up5 if not np.isnan(value)]

    up4 = df['up4'].tolist()
    up4 = [value for value in up4 if not np.isnan(value)]

    up3 = df['up3'].tolist()
    up3 = [value for value in up3 if not np.isnan(value)]

    up_outer = df['up_outer'].tolist()
    up_outer = [value for value in up_outer if not np.isnan(value)]

