import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data(directory):
    res = []
    file_names = []  # To store the corresponding file names
    # Specify the directory where your CSV files are located

    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Loop through each CSV file and plot the data
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(directory, csv_file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        # Plot the data
        res.append((np.array(df.iloc[:, 4]), np.array(df.iloc[:, 10])))
        file_names.append(csv_file)  # Store the file name

    return res, file_names


def fix_data_set(data_set):
    res = center_data(data_set)
    res = sort_by_x(res)
    res = flip_if_needed(res)
    return res



def center_data(data_set):
    x, y = data_set
    return (x - 0.5*(np.max(x)+np.min(x)), y - 0.5*(np.max(y)+np.min(y)))

def sort_by_x(data_set):
    x, y = data_set
    indices = np.argsort(x)
    return (x[indices], y[indices])


def flip_if_needed(data_set):
    x, y = data_set
    if np.average(y[:len(y)//4]) > 0:
        x = -x
        return (x[::-1], y[::-1])
    return (x, y)


def fix_data(data):
    res = []
    for i in range(len(data)):
        res.append(fix_data_set(data[i]))
    return res


if __name__ == "__main__":
    directory_name = 'material1-BO1'
    # directory_name = 'material2-BO2'
    # directory_name = r'material3-BO3\area'
    # directory_name = r'material1-BO1\area'
    data, file_names = get_data(directory_name)[::1]
    data = fix_data(data)

    bad_material_2 = [0, 2, 4, 10]
    bad_material_1 = [2, 3, 5, 9, 10]

    bad_material = bad_material_1

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for indices in good_material
    for i in bad_material:
        if 0 <= i < len(data):  # Check if the index is within the valid range
            data_set = data[i]
            x, y = data_set
            axs[0].scatter(x, y)

        print(f"Bad Material File: {file_names[i]}")


    axs[0].set_title('bad Material Indices')
    axs[0].grid()

    # Plot for other indices
    other_indices = [i for i in range(len(data)) if i not in bad_material]
    for i in other_indices:
        if 0 <= i < len(data):  # Check if the index is within the valid range
            data_set = data[i]
            x, y = data_set
            axs[1].scatter(x, y)

    axs[1].set_title('Other Indices')
    axs[1].grid()

    plt.show()

