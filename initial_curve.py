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
    i = 0
    # Loop through each CSV file and plot the data
    for csv_file in csv_files:
        print(str(i) + ': ' + csv_file)
        i += 1
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


def initial_curve(data, x_width, y_width):
    x = []
    x_err = []
    y = []
    y_err = []
    for data_set in data:
        x1, y1 = get_endpoint(data_set, x_width, y_width)
        x.append(x1)
        y.append(y1)
        x_err.append(x_width)
        y_err.append(y_width)
    return x, y, x_err, y_err




def get_endpoint(data_set, x_width, y_width):
    x, y = data_set
    x_res = x[len(x)-1] - x_width
    y_res = np.max(y) - y_width
    return (x_res, y_res)


def plot_initial_curves():
    directory_name = 'mat1'
    error = (0.05, 0.05)
    data, names = get_data(directory_name)
    data = fix_data(data)
    x1, y1, xerr, yerr = [], [], [], []
    for data_set in data:
        x, y = data_set
        point = get_endpoint(data_set, *error)
        x1.append(point[0])
        xerr.append(error[0])
        y1.append(point[1])
        yerr.append(error[1])
    plt.errorbar(x=x1, y=y1, xerr=xerr, yerr=yerr, fmt='.')
    directory_name = 'mat2'
    data, names = get_data(directory_name)
    data = fix_data(data)
    x1, y1, xerr, yerr = [], [], [], []
    for data_set in data:
        x, y = data_set
        point = get_endpoint(data_set, *error)
        x1.append(point[0])
        xerr.append(error[0])
        y1.append(point[1])
        yerr.append(error[1])
    plt.errorbar(x=x1, y=y1, xerr=xerr, yerr=yerr, fmt='.')
    plt.ylim(0, 16)
    plt.xlim(0, 8)
    plt.title('initial magnetization curves')
    plt.legend(['material 1', 'material 2'])
    plt.xlabel('H[V]')
    plt.ylabel('B[V]')
    plt.grid()
    plt.show()


def plot_folder(name):
    data, names = get_data(name)
    for data_set in data:
        x, y = data_set
        plt.scatter(x, y)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_folder('mat2')
    plot_initial_curves()