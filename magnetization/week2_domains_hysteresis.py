import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import week2_edit_domains_images as edit_images
import matplotlib.lines as mlines


def plot_hysteresis(v, pixels):
    pixels_error = [0.012] * len(pixels)
    v_error = [0.001] * len(v)

    colors = np.arange(len(v))

    # Plotting the lists
    plt.scatter(v, pixels, c=colors, marker='o', cmap='coolwarm')
    plt.errorbar(v, pixels, xerr=v_error, yerr=pixels_error, fmt='none', ecolor='orange', capsize=5,
                 label='Error Bars')

    # Adding labels and title
    plt.xlabel("Voltage(V)", fontsize=12)
    plt.ylabel("Black pixels percentage(%)", fontsize=12)
    plt.title("Hysteresis loop for 4V cycle", fontsize=14)

    colorbar = plt.colorbar()
    colorbar.set_label('Color by order of measurement')
    plt.grid()
    # Set denser grid
    plt.xticks(np.arange(-6, 6, 0.5))
    plt.yticks(np.arange(0, 101, 10))

    # Show the plot
    plt.show()


def plot_single_hysteresis():
    directory_name1 = r'domains/up5'
    pixels_list1 = edit_images.analyse_pixels_directory(directory_name1)
    pixels_list1 = [x*100 + 100 for x in pixels_list1]
    # directory_name2 = r'domains/up4'
    # pixels_list2 = edit_images.analyse_pixels_directory(directory_name2)
    # pixels_list2 = [x*100 for x in pixels_list2]
    # directory_name3 = r'domains/up3'
    # pixels_list3 = edit_images.analyse_pixels_directory(directory_name3)
    # pixels_list3 = [x*100 - 100 for x in pixels_list3]

    # Extract values from exel
    excel_file_path = r'C:\Users\TLPL-324\lab2\domains\voltages.xlsx'
    df = pd.read_excel(excel_file_path)
    up5 = df['up5'].tolist()
    up5 = [value for value in up5 if not np.isnan(value)]

    # up4 = df['up4'].tolist()
    # up4 = [value for value in up4 if not np.isnan(value)]
    #
    # up3 = df['up3'].tolist()
    # up3 = [value for value in up3 if not np.isnan(value)]
    #
    # up_outer = df['up_outer'].tolist()
    # up_outer = [value for value in up_outer if not np.isnan(value)]

    plot_hysteresis(up5, pixels_list1)


def plot_hysteresis2(v, pixels, label):
    pixels_error = [0.012] * len(pixels)
    v_error = [0.001] * len(v)
    # cmap = 'viridis', 'plasma', 'inferno', 'magma'

    colors = np.arange(len(v))
    # Plotting the lists
    plt.scatter(v, pixels, marker='o', label=label, s=40, c=colors, cmap='viridis', )
    plt.errorbar(v, pixels, xerr=v_error, yerr=pixels_error, fmt='none', ecolor='orange', capsize=5)

    # Adding labels and title
    plt.xlabel("Voltage(V)", fontsize=12)
    plt.ylabel("Black pixels percentage", fontsize=12)
    plt.title("All Hysteresis Loops", fontsize=14)

    plt.grid()
    plt.xticks(np.arange(-6, 6, 0.5))
    plt.yticks(np.arange(-100, 201, 10))

    legend_dot = mlines.Line2D([], [], color='black', marker='', linestyle='None', markersize=0,
                               label="top- 5V cycle\nmiddle- 4V cycle\nbottom- 3V cycle")
    plt.legend(handles=[legend_dot])


def plot_all_hysteresis():
    directory_name1 = r'domains/up5'
    pixels_list1 = edit_images.analyse_pixels_directory(directory_name1)
    pixels_list1 = [x * 100 + 100 for x in pixels_list1]

    directory_name2 = r'domains/up4'
    pixels_list2 = edit_images.analyse_pixels_directory(directory_name2)
    pixels_list2 = [x * 100 for x in pixels_list2]

    directory_name3 = r'domains/up3'
    pixels_list3 = edit_images.analyse_pixels_directory(directory_name3)
    pixels_list3 = [x * 100 - 100 for x in pixels_list3]

    excel_file_path = r'C:\Users\TLPL-324\lab2\domains\voltages.xlsx'
    df = pd.read_excel(excel_file_path)

    up5 = df['up5'].tolist()
    up5 = [value for value in up5 if not np.isnan(value)]

    up4 = df['up4'].tolist()
    up4 = [value for value in up4 if not np.isnan(value)]

    up3 = df['up3'].tolist()
    up3 = [value for value in up3 if not np.isnan(value)]

    # up_outer = df['up_outer'].tolist() #outer is not used
    # up_outer = [value for value in up_outer if not np.isnan(value)]

    # Plotting on the same graph with colormap
    plt.figure(figsize=(10, 6))

    plot_hysteresis2(up5, pixels_list1, 'top- 5V cycle')
    plot_hysteresis2(up4, pixels_list2, 'middle- 4V cycle')
    plot_hysteresis2(up3, pixels_list3, 'bottom- 3V cycle')
    colorbar = plt.colorbar()
    colorbar.set_label('Color by order of measurement')
    plt.show()


if __name__ == '__main__':
    plot_all_hysteresis()
