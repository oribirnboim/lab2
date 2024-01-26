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
    error = (0.1, 0.1)
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
    plt.title('Initial Magnetization Curves')
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


def plot_full_loops():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data, names = get_data('mat1')
    data = fix_data(data)
    for data_set in data:
        x, y = data_set
        ax1.scatter(x, y, s=1)
    ax1.grid()
    ax1.set_title('material 1')
    ax1.set_xlabel('H[V]')
    ax1.set_ylabel('B[V]')
    ax1.set_xlim(-7.5, 7.5)
    ax1.set_ylim(-15, 15)
    data, names = get_data('mat2')
    data = fix_data(data)
    for data_set in data:
        x, y = data_set
        ax2.scatter(x, y, s=1)
    ax2.grid()
    ax2.set_title('material 2')
    ax2.set_xlabel('H[V]')
    ax2.set_ylabel('B[V]')
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(-15, 15)
    fig.suptitle('B over H curves')
    plt.show()


def clean_file_names(names):
    return [int(name.split('.')[0]) for name in names]


def sample_multiple_slabs(data, field):
    y1, y_err= [], []
    yerr = 0.2
    for data_set in data:
        x, y = data_set
        index = np.searchsorted(x, field)
        y1.append(abs(y[index]))
        y_err.append(yerr)
    return y1, y_err



def plot_multiple_slabs():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    xlim = (-6, 6)
    ylim = (-25, 25)
    data, names = get_data('mat1/area_again')
    data = fix_data(data)
    for data_set in data:
        x, y = data_set
        ax1.scatter(x, y, s=1)
    ax1.grid()
    ax1.set_title('material 1')
    ax1.set_xlabel('H[V]')
    ax1.set_ylabel('B[V]')
    lgnd = ax1.legend(clean_file_names(names))
    for dot in lgnd.legendHandles:
        dot._sizes = [10]
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    data, names = get_data('mat2/area')
    data = fix_data(data)
    for data_set in data:
        x, y = data_set
        ax2.scatter(x, y, s=1)
    ax2.grid()
    ax2.set_title('material 2')
    ax2.set_xlabel('H[V]')
    ax2.set_ylabel('B[V]')
    lgnd = ax2.legend(clean_file_names(names))
    for dot in lgnd.legendHandles:
        dot._sizes = [10]
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    fig.suptitle('B Over H Curves for a Varying Amount of Slabs')
    plt.show()


def plot_flux_over_area():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data, names = get_data('mat1/area_again')
    data = fix_data(data)
    num_slabs = np.array(clean_file_names(names))
    flux, flux_err = sample_multiple_slabs(data, 0)
    ax1.errorbar(x=num_slabs, y=flux, yerr=flux_err, fmt='.')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 20)
    ax1.set_xlabel('number of slabs')
    ax1.set_ylabel('Retentivity [V]')
    ax1.set_title('material 1')
    ax1.grid()
    data, names = get_data('mat2/area')
    data = fix_data(data)
    num_slabs = np.array(clean_file_names(names))
    flux, flux_err = sample_multiple_slabs(data, 0)
    ax2.errorbar(x=num_slabs, y=flux, yerr=flux_err, fmt='.')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 20)
    ax2.set_xlabel('number of slabs')
    ax2.set_ylabel('Retentivity [V]')
    ax2.set_title('material 2')
    ax2.grid()
    plt.suptitle('B(H=0) for a varying number of slabs')
    plt.show()


def get_max_h(data):
    h_err = 0.1
    max_h, max_h_err = [], []
    for data_set in data:
        x, y = data_set
        biggets_h = np.max(x)
        max_h.append(biggets_h - h_err)
        max_h_err.append(h_err)
    return max_h, max_h_err


def plot_max_h():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data, names = get_data('mat1/area_again')
    data = fix_data(data)
    num_slabs = clean_file_names(names)
    max_h, max_h_err = get_max_h(data)
    ax1.errorbar(x=num_slabs, y=max_h, yerr=max_h_err, fmt='.')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.set_xlabel('number of slabs')
    ax1.set_ylabel('max(H) [V]')
    ax1.set_title('material 1')
    ax1.grid()

    data, names = get_data('mat1/area')
    data = fix_data(data)
    num_slabs = clean_file_names(names)
    max_h, max_h_err = get_max_h(data)
    ax2.errorbar(x=num_slabs, y=max_h, yerr=max_h_err, fmt='.')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 6)
    ax2.set_xlabel('number of slabs')
    ax2.set_ylabel('max(H) [V]')
    ax2.set_title('material 2')
    ax2.grid()
    fig.suptitle('amplitude of H depending on cross section')
    plt.show()

    

if __name__ == "__main__":
    # plot_initial_curves()
    # plot_full_loops()
    # plot_multiple_slabs()
    plot_flux_over_area()
    # plot_max_h()
    pass