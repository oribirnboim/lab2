import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_data(directory):
    res = []
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



if __name__ == "__main__":
    directory_names = ['material1-BO1', 'material2-BO2', 'material1-BO1/area']
    labels = ['material 1', 'material 2']
    data_sets = [[8, 7, 4, 2, 12, 6, 11, 0], [8, 5, 3, 1, 9, 6, 7]]
    uncertainties = [(0.2, 0.04), (0.2, 0.04), (0.2, 0.04)]
    for i in [2]:
        directory_name = directory_names[i]
        data = get_data(directory_name)
        # data = [data[i] for i in data_sets[i]]
        data = fix_data(data)
        x, y, x_err, y_err = initial_curve(data, *uncertainties[i])
        # plt.errorbar(x=x, y=y, xerr=x_err, yerr=y_err, fmt='o')
        for j in range(len(data)):
            # if j in data_sets[i]:
            data_set = data[j]
            plt.scatter(*data_set)
        plt.plot
    plt.title("initial curves")
    plt.title("hysteresis loops at different amplitudes")
    plt.title("subset that matches expectations")
    plt.title('varying number of plates')
    # plt.legend(labels)
    # plt.ylim(0, 0.9)
    # plt.xlim(0, 9)
    plt.xlabel("B[V]")
    plt.ylabel("H[V]")
    plt.grid()
    plt.show()