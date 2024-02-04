import numpy as np
import matplotlib.pyplot as plt
from malus import process_data
import os
import pandas as pd


def get_data(directory_path):
    data_list = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory_path, filename)

            # Read Excel file skipping the first 6 rows
            df = pd.read_excel(file_path, skiprows=6)

            # Extract the first two columns as numpy arrays
            column1 = np.array(df.iloc[:, 0])
            column2 = np.array(df.iloc[:, 1])

            # Append the data as a tuple to the list
            data_list.append((column1, column2))

    return data_list


def fix_data(data):
    res = []
    for data_set in data:
        res.append(process_data(data_set, 10))
    return res


def plot_folder(folder_name):
    data = get_data(folder_name)
    print(data)
    data = fix_data(data)
    for data_set in data:
        x, y = data_set
        plt.plot(x, y)
    plt.grid()
    plt.show()



if __name__ == "__main__":
    plot_folder('quarter_waveplate')