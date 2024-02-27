import os
from scipy .optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def get_data(filename):
    try:
        # Assuming the Excel file is in the same directory as the script
        path = filename if filename.endswith('.xlsx') else filename + '.xlsx'

        # Reading only the first and second columns of the Excel file into a DataFrame
        df = pd.read_excel(path, usecols=[0, 1], skiprows=6)

        # Converting the DataFrame columns to NumPy arrays
        array1 = df.iloc[:, 0].to_numpy()
        array2 = df.iloc[:, 1].to_numpy()

        # Returning the NumPy arrays
        return array1, array2
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_data(data, step):
    x, y = data
    x = np.arange(0, len(x) * step, step)
    y = y * 10 ** 6
    return x, y


def malus2(file_name):
    data = get_data(file_name)
    x, y = process_data(data, 10)
    x_err = [2 for val in x]
    y_err = [0.05 for val in y]
    # plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='.', label='data')

    def model(theta, i, d):
        return i * np.cos((theta + d) * np.pi / 180) ** 2

    params, covariance = curve_fit(model, x, y, p0=[175, 0])
    variance = np.diag(covariance)
    return params[1], variance[1]


def orthogonal_deviation():
    file_numbers = []
    averages = []
    errors = []
    folder_path = r'C:\Users\TLPL-324\lab2\polarization\chilary_orthogonal'
    # Iterate over files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith("cm.xlsx"):
            file_number = int(filename[:-7])  # Extracting everything before "cm"
            file_numbers.append(file_number)
            file_path = os.path.join(folder_path, filename)

            # Read the file into a DataFrame
            df = pd.read_excel(file_path)

            # Extract the current column
            current_column = df.iloc[5:, 1]  # Considering B6 as starting point

            # Calculate average current and standard deviation
            avg_current = np.mean(current_column[1:])*10**6  # Skip the header row
            std_dev_current = np.std(current_column[1:])

            averages.append(avg_current)
            errors.append(std_dev_current)

    plt.errorbar(file_numbers, averages, xerr=0.5, yerr=errors, fmt='o', capsize=5)
    plt.xlabel('Solution length (m)')
    plt.ylabel('I(µA)')
    plt.title('Solution length Vs. I')
    plt.grid(True)
    plt.show()


def chialy_1():
    file_numbers = []
    d_values = []
    d_errors = []
    folder_path = r'C:\Users\TLPL-324\lab2\polarization\chilary1'
    # Iterate over files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith("cm.xlsx"):
            file_number = int(filename[:-7])  # Extracting everything before ".xlxc"
            file_numbers.append(file_number/100)
            file_path = os.path.join(folder_path, filename)

            # Call func2 from the malus module
            d, error = malus2(file_path)
            d_values.append(d)
            d_errors.append(error)

    plt.errorbar(file_numbers, d_values, xerr=0.5/100, yerr=d_errors, fmt='o', capsize=5)
    plt.xlabel('Solution length (m)')
    plt.ylabel('Phase(°)')
    plt.title('Solution length Vs. Phase')
    plt.grid(True)
    plt.show()

    return file_numbers, d_values, d_errors


if __name__ == '__main__':
    orthogonal_deviation()
    chialy_1()

