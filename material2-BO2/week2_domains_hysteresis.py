from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import week2_edit_domains_images as edit_images


def plot_hysteresis(v, pixels):
    colors = np.arange(len(v))

    # Plotting the lists
    plt.scatter(v, pixels, c=colors, marker='o', cmap='coolwarm')

    # Adding labels and title
    plt.xlabel("Voltage")
    plt.ylabel("Black pixels (normalized)")
    plt.title("Hyeteresis loop")

    colorbar = plt.colorbar()
    colorbar.set_label('Color by X-values')
    plt.grid
    # Show the plot
    plt.show()


if __name__ == '__main__':
    directory_name = r'C:\Users\TLPL-324\lab2\domains\try_domains'
    pixels_list = edit_images.analyse_pixels_directory(directory_name)


    # Extract values from exel
    excel_file_path = r'C:\Users\TLPL-324\lab2\domains\voltages.xlsx'
    df = pd.read_excel(excel_file_path)
    up5 = df['up5'].tolist()
    up5 = [value for value in up5 if not np.isnan(value)]

    up4 = df['up4'].tolist()
    up4 = [value for value in up4 if not np.isnan(value)]

    up3 = df['up3'].tolist()
    up3 = [value for value in up3 if not np.isnan(value)]

    up_outer = df['up_outer'].tolist()
    up_outer = [value for value in up_outer if not np.isnan(value)]

    plot_hysteresis(up5, pixels_list)



