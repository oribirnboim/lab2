import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(directory):
    data_and_filenames = []

    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Loop through each CSV file and plot the data
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(directory, csv_file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        # Plot the data
        data_and_filenames.append({
            'data': (np.array(df.iloc[:, 4]), np.array(df.iloc[:, 10])),
            'filename': os.path.splitext(csv_file)[0]  # Remove ".csv" extension
        })

    return data_and_filenames


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
    if np.average(y[:len(y)//2]) > 0:
        x = -x
        return (x[::-1], y[::-1])
    return (x, y)


def fix_data(data):
    res = []
    for i in range(len(data)):
        res.append(fix_data_set(data[i]))
    return res


if __name__ == "__main__":
    directory_name1 = r'material1-BO1\area'
    directory_name2 = r'material3-BO3\area'

    data_and_filenames1 = get_data(directory_name1)[::1]
    data_and_filenames1 = sorted(data_and_filenames1, key=lambda x: int(x['filename']))

    data1 = [item['data'] for item in data_and_filenames1]
    data1 = fix_data(data1)

    data_and_filenames2 = get_data(directory_name2)[::1]
    data_and_filenames2 = sorted(data_and_filenames2, key=lambda x: int(x['filename']))

    data2 = [item['data'] for item in data_and_filenames2]
    data2 = fix_data(data2)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for i, data_set in enumerate(data1):
        x, y = data_set
        filename = data_and_filenames1[i]['filename']
        axs[0].scatter(x, y, label=f'{filename} plates')

    axs[0].set_title('Material 1')
    axs[0].set_xlabel('B(V)')
    axs[0].set_ylabel('H(V)')
    axs[0].grid()
    axs[0].legend()

    for i, data_set in enumerate(data2):
        x, y = data_set
        filename = data_and_filenames2[i]['filename']
        axs[1].scatter(x, y, label=f'{filename} plates')

    axs[1].set_title('Material 2')
    axs[1].set_xlabel('B(V)')
    axs[1].set_ylabel('H(V)')
    axs[1].grid()
    axs[1].legend()

    plt.show()