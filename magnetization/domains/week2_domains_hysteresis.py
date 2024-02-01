import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import week2_edit_domains_images as edit_images
import matplotlib.lines as mlines
from matplotlib import cm


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
    colors = np.arange(len(v))
    # Plotting the lists
    scatter = plt.scatter(v, pixels, marker='o', label=label, s=40, c=colors, cmap='viridis', )
    error = plt.errorbar(v, pixels, xerr=v_error, yerr=pixels_error, fmt='none', ecolor='orange', capsize=5)

    # Adding labels and title
    plt.xlabel("H[a.u.]", fontsize=12)
    plt.ylabel("B[a.u.]", fontsize=12)
    # plt.title("All Hysteresis Loops", fontsize=14)

    plt.grid()
    plt.xticks(np.arange(-6, 6, 0.5))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(fontsize=11)
    return scatter, error


# def plot_all_hysteresis():
#     directory_name1 = r'domains/up5'
#     pixels_list1 = edit_images.analyse_pixels_directory(directory_name1)
#     pixels_list1 = [x * 100 for x in pixels_list1]
#
#     directory_name2 = r'domains/up4'
#     pixels_list2 = edit_images.analyse_pixels_directory(directory_name2)
#     pixels_list2 = [x * 100 for x in pixels_list2]
#
#     directory_name3 = r'domains/up3'
#     pixels_list3 = edit_images.analyse_pixels_directory(directory_name3)
#     pixels_list3 = [x * 100 for x in pixels_list3]
#
#     excel_file_path = r'C:\Users\TLPL-324\lab2\magnetization\domains\voltages.xlsx'
#     df = pd.read_excel(excel_file_path)
#
#     up5 = df['up5'].tolist()
#     up5 = [value for value in up5 if not np.isnan(value)]
#
#     up4 = df['up4'].tolist()
#     up4 = [value for value in up4 if not np.isnan(value)]
#
#     up3 = df['up3'].tolist()
#     up3 = [value for value in up3 if not np.isnan(value)]
#
#     # up_outer = df['up_outer'].tolist() #outer is not used
#     # up_outer = [value for value in up_outer if not np.isnan(value)]
#
#     # Plotting on the same graph with colormap
#     plt.figure(figsize=(10, 6))
#
#     plot_hysteresis2(up5, pixels_list1, 'top- 5V cycle')
#     plot_hysteresis2(up4, pixels_list2, 'middle- 4V cycle')
#     plot_hysteresis2(up3, pixels_list3, 'bottom- 3V cycle')
#     colorbar = plt.colorbar()
#     colorbar.set_label('Color by order of measurement')
#     plt.show()

def plot_all_hysteresis():
    directory_name1 = r'domains/up5'
    pixels_list1 = edit_images.analyse_pixels_directory(directory_name1)
    pixels_list1 = [x * 100 for x in pixels_list1]

    directory_name2 = r'domains/up4'
    pixels_list2 = edit_images.analyse_pixels_directory(directory_name2)
    pixels_list2 = [x * 100 for x in pixels_list2]

    directory_name3 = r'domains/up3'
    pixels_list3 = edit_images.analyse_pixels_directory(directory_name3)
    pixels_list3 = [x * 100 for x in pixels_list3]

    excel_file_path = r'C:\Users\TLPL-324\lab2\magnetization\domains\voltages.xlsx'
    df = pd.read_excel(excel_file_path)

    up5 = df['up5'].tolist()
    up5 = [value*10 for value in up5 if not np.isnan(value)]

    up4 = df['up4'].tolist()
    up4 = [value*10 for value in up4 if not np.isnan(value)]

    up3 = df['up3'].tolist()
    up3 = [value*10 for value in up3 if not np.isnan(value)]

    # Creating subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 17), sharex=False)

    # Plotting the first subplot (top)
    pixels_error1 = [0.012] * len(pixels_list1)
    v_error1 = [0.001] * len(up5)
    colors1 = np.arange(len(up5))
    scatter1 = axs[0].scatter(up5, pixels_list1, marker='o', label="5V cycle", s=65, c=colors1, cmap='viridis')
    error1 = axs[0].errorbar(up5, pixels_list1, xerr=v_error1, yerr=pixels_error1, fmt='none', ecolor='orange', capsize=5)
    axs[0].set_title("All Hysteresis Loops", fontsize=34)
    # axs[0].set_xlabel("H[a.u.]", fontsize=12)
    # axs[0].set_ylabel("B[a.u.]", fontsize=12)
    axs[0].grid()
    axs[0].set_xticks(np.arange(-60, 60, 5))
    axs[0].set_yticks(np.arange(0, 101, 10))
    legend_text1 = '5V cycle'
    legend_dot1 = mlines.Line2D([], [], color='black', marker='', linestyle='None', markersize=0,
                               label=legend_text1)
    axs[0].legend(handles=[legend_dot1], loc='upper left')
    axs[0].tick_params(axis='both', which='major', labelsize=14)  # Set the font size for major ticks


    # Plotting the second subplot (middle)
    pixels_error2 = [0.012] * len(pixels_list2)
    v_error2 = [0.001] * len(up4)
    colors2 = np.arange(len(up4))
    scatter2 = axs[1].scatter(up4, pixels_list2, marker='o', label="4V cycle", s=65, c=colors2, cmap='viridis')
    error2 = axs[1].errorbar(up4, pixels_list2, xerr=v_error2, yerr=pixels_error2, fmt='none', ecolor='orange', capsize=5)
    # axs[1].set_xlabel("H[a.u.]", fontsize=12)
    axs[1].set_ylabel("B[a.u.]", fontsize=24)
    axs[1].grid()
    axs[1].set_xticks(np.arange(-60, 60, 5))
    axs[1].set_yticks(np.arange(0, 101, 10))
    legend_text2 = '4V cycle'
    legend_dot2 = mlines.Line2D([], [], color='black', marker='', linestyle='None', markersize=0,
                               label=legend_text2)
    axs[1].legend(handles=[legend_dot2], loc='upper left')
    axs[1].tick_params(axis='both', which='major', labelsize=14)  # Set the font size for major ticks

    # Plotting the third subplot (bottom)
    pixels_error3 = [0.012] * len(pixels_list3)
    v_error3 = [0.001] * len(up3)
    colors3 = np.arange(len(up3))
    scatter3 = axs[2].scatter(up3, pixels_list3, marker='o', label="3V cycle", s=65, c=colors3, cmap='viridis')
    error3 = axs[2].errorbar(up3, pixels_list3, xerr=v_error3, yerr=pixels_error3, fmt='none', ecolor='orange', capsize=5)
    axs[2].set_xlabel("H[a.u.]", fontsize=24)
    # axs[2].set_ylabel("B[a.u.]", fontsize=12)
    axs[2].grid()
    axs[2].set_xticks(np.arange(-60, 60, 5))
    axs[2].set_yticks(np.arange(0, 101, 10))
    legend_text3 = '3V cycle'
    legend_dot3 = mlines.Line2D([], [], color='black', marker='', linestyle='None', markersize=0,
                               label=legend_text3)
    axs[2].legend(handles=[legend_dot3], loc='upper left')
    axs[2].tick_params(axis='both', which='major', labelsize=14)  # Set the font size for major ticks


    cax_position = axs[2].get_position().bounds
    cax_position = [cax_position[0] + 0.9, cax_position[1], 0.05, cax_position[3]+0.6]  # Adjust the first value for the right shift
    cbar = plt.colorbar(scatter3, ax=axs, orientation='vertical', pad=0.02, cax=plt.axes(cax_position))
    cbar.set_label('Color by order of measurement', fontsize=18, rotation=90, labelpad=15)


    # Adjusting layout
    plt.tight_layout()
    plt.savefig(r'C:\Users\TLPL-324\lab2\magnetization\domains\hysteresis domains.png', bbox_inches='tight')
    # Display the plot
    plt.show()


if __name__ == '__main__':
    plot_all_hysteresis()
    # directory_name1 = r'domains/up5'
    # pixels_list1 = edit_images.analyse_pixels_directory(directory_name1)
    # pixels_list1 = [x * 100 for x in pixels_list1]
    #
    #
    # excel_file_path = r'C:\Users\TLPL-324\lab2\magnetization\domains\voltages.xlsx'
    # df = pd.read_excel(excel_file_path)
    #
    # up5 = df['up5'].tolist()
    # up5 = [value for value in up5 if not np.isnan(value)]
    #
    # plot_hysteresis2(up5, pixels_list1, "5V")