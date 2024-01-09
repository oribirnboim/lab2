import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_data(directory):
    res = []
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
    return res


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
    directory_name = 'material1-BO1'
    # directory_name = 'mat2'
    data = get_data(directory_name)[::1]
    data = fix_data(data)
    for i in range(len(data)):
        data_set = data[i]
        plt.scatter(*data_set)
    plt.grid()
    plt.show()