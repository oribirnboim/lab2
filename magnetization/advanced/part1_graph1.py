import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from part1 import load_images_from_folder

def calculate_average_distance(variance_image, up, down):
    # Get the coordinates of pixels above the upper threshold and below the lower threshold
    above_threshold_indices = np.argwhere(variance_image > up)
    below_threshold_indices = np.argwhere(variance_image < down)

    # If there are no pixels below the threshold, return an error message
    if below_threshold_indices.size == 0:
        raise ValueError("No pixels found with variance below the 'down' threshold")

    # Calculate distances
    distances = []
    for idx in above_threshold_indices:
        # Calculate the distance from the current pixel to all pixels below the threshold
        dists = np.sqrt((below_threshold_indices[:, 0] - idx[0]) ** 2 + (below_threshold_indices[:, 1] - idx[1]) ** 2)
        # Find the minimum distance
        min_dist = np.min(dists)
        distances.append(min_dist)

    # Calculate the average distance
    avg_distance = np.mean(distances)

    return avg_distance, distances


def plot_average_distance(variance_image, up, down):
    avg_distance, distances = calculate_average_distance(variance_image, up, down)

    plt.figure(figsize=(10, 6))
    plt.plot(distances, label='Distances')
    plt.axhline(y=avg_distance, color='r', linestyle='--', label=f'Average Distance = {avg_distance:.2f}')
    plt.xlabel('Pixel Index')
    plt.ylabel('Distance to Closest Pixel Below Threshold')
    plt.title(f'Average Distance of Pixels with Variance > {up} to Closest Pixel with Variance < {down}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with dummy data
folder_path = 'part_1_batch_3'
images = load_images_from_folder(folder_path, extension="jpg")
# Stack images along the third dimension to create a 3D array
stacked_images = np.stack(images, axis=-1)
# Calculate the variance along the third dimension (axis=-1)
variance_image = np.var(stacked_images, axis=-1)

up = 100
down = 50
plot_average_distance(variance_image, up, down)
