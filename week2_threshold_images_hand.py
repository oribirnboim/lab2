import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def threshold_image(image, g_value, g_error):
    # Extract the green channel
    green_channel = image[:, :, 1]

    # Define the threshold range for the green channel
    lower_g_bound = max(0, g_value - g_error)
    upper_g_bound = min(255, g_value + g_error)

    # Define the threshold range for the other channels
    lower_bounds = [215 - 30, lower_g_bound, 0]
    upper_bounds = [215 + 30, upper_g_bound, 0]

    # Create binary masks for each channel
    masks = [np.logical_and(image[:, :, i] >= lower_bounds[i], image[:, :, i] <= upper_bounds[i]) for i in range(3)]

    # Combine the masks to get the final result
    result = np.logical_and.reduce(masks)

    # Create a binary mask based on the threshold
    result = np.zeros_like(green_channel, dtype=np.uint8)
    result[masks[1]] = 255

    return result

class ImageThresholdingApp:
    def __init__(self, root, image_files):
        self.root = root
        self.image_files = image_files
        self.current_index = 0

        # Load the first image
        self.image = cv2.imread(image_files[self.current_index])
        self.label = tk.Label(root)
        self.update_display()

        # Create a slider for adjusting the green channel threshold
        self.slider_label = tk.Label(root, text="Adjust Green Channel Threshold")
        self.slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, length=400, command=self.update_threshold)
        self.slider.set(127)  # Default value
        self.slider_label.pack()
        self.slider.pack()

        # Create a button to save the thresholded image and move to the next one
        self.save_button = tk.Button(root, text="Save Thresholded Image", command=self.save_and_next)
        self.save_button.pack()

        # Bind arrow keys to slider
        self.root.bind("<Left>", lambda event: self.adjust_slider(-1))
        self.root.bind("<Right>", lambda event: self.adjust_slider(1))

    def adjust_slider(self, direction):
        current_value = self.slider.get()
        new_value = max(0, min(255, current_value + direction))
        self.slider.set(new_value)
        self.update_threshold(new_value)

    def update_display(self, image=None):
        if image is None:
            image = self.image

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to PhotoImage format
        photo = ImageTk.PhotoImage(Image.fromarray(image_rgb))

        # Destroy existing label widget
        if hasattr(self, 'label'):
            self.label.destroy()

        # Create a new label to display the image
        self.label = tk.Label(self.root, image=photo)
        self.label.photo = photo
        self.label.pack()

        # Keep a reference to the PhotoImage to prevent it from being garbage-collected
        self.label.image = photo

    def update_threshold(self, g_value):
        g_value = int(g_value)

        # Create a copy of the original image for thresholding
        temp_image = self.image.copy()

        # Apply the thresholding using the current green channel value
        thresholded_image = threshold_image(temp_image, g_value, 20)

        # Update the display with the thresholded image
        self.update_display(thresholded_image)

        # Save the thresholded image, replacing the original image
        cv2.imwrite(self.image_files[self.current_index], thresholded_image)

    def save_and_next(self):
        # Move to the next image
        self.current_index += 1
        if self.current_index < len(self.image_files):
            # Load the next image
            self.image = cv2.imread(self.image_files[self.current_index])

            # Create a copy of the original image for thresholding
            temp_image = self.image.copy()

            # Apply the thresholding using the current green channel value
            thresholded_image = threshold_image(temp_image, self.slider.get(), 20)

            # Save the thresholded image, replacing the original image
            cv2.imwrite(self.image_files[self.current_index], thresholded_image)

            # Update the display with the new image
            self.update_display(thresholded_image)

            # Reset the slider to the default value
            self.slider.set(127)
        else:
            # If there are no more images, close the application
            self.root.quit()

def main():
    # Ask the user to select a directory
    directory_path = filedialog.askdirectory(title="Select Directory")
    if not directory_path:
        return

    # Get a list of all image files in the chosen directory, sorted by name
    image_files = sorted([file for file in os.listdir(directory_path) if file.lower().endswith('.bmp')],
                         key=lambda x: int(x.split('.')[0]))

    if not image_files:
        print("No image files found in the selected directory.")
        return

    # Create the Tkinter application
    root = tk.Tk()
    root.title("Image Thresholding App")

    # Create an instance of the ImageThresholdingApp
    app = ImageThresholdingApp(root, [os.path.join(directory_path, file) for file in image_files])

    # Set up the main loop for the Tkinter application
    root.mainloop()

if __name__ == "__main__":
    main()
