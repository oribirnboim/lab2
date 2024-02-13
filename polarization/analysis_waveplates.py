import pandas as pd
import numpy as np
from numpy import sin, cos
import re
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from malus import process_data
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


step = 10

def amp2(state, A, phi, p1, p2):
    alpha, theta = state.T
    alpha, theta, phi, p1, p2 = np.radians(alpha), np.radians(theta), np.radians(phi), np.radians(p1), np.radians(p2)
    alpha, theta = alpha - p1, theta - p2
    first = sin(alpha)*sin(theta) + cos(alpha)*cos(theta)*cos(phi)
    second = cos(alpha)*cos(theta)*sin(phi)
    return A*(first**2 + second**2)


def get_data(directory_path):
    x_arrays = []
    y_arrays = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory_path, filename)
            x, y = get_single(file_path)
            x_arrays.append(x)
            y_arrays.append(y)

    x_res = np.concatenate(x_arrays)
    y_res = np.concatenate(y_arrays)
    return clean_nan(x_res, y_res)


def get_single(file_path):
    """
    Read an Excel file, skip the first 6 rows, and extract the first two columns as numpy arrays.
    Extract and return the numeric part from the file name.

    Parameters:
    - file_path: str
        The relative path to the Excel file.

    Returns:
    - Tuple of two numpy arrays and an integer
        Two arrays representing the first and second columns of the Excel file, and the extracted number.
    """
    # Extract numeric part from the file name using regular expression
    alpha = int(re.search(r'\d+', file_path).group())

    # Read the Excel file into a DataFrame, skipping the first 6 rows
    df = pd.read_excel(file_path, skiprows=6)

    # Extract the first two columns as numpy arrays
    t = df.iloc[:, 0].to_numpy()
    power = df.iloc[:, 1].to_numpy()
    raw_theta, power = process_data((t, power), step)
    theta = -raw_theta + alpha
    theta = theta % 360
    system_state = np.array([np.array([alpha, val]) for val in theta])

    return system_state, power


def clean_nan(x, y):
    valid_indices = np.isfinite(y)
    return x[valid_indices], y[valid_indices]


def plot_folder(folder):
    x, y = get_data(folder)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o')
    ax.set_xlabel('$\\alpha$ [$^\circ$]')
    ax.set_ylabel('$\\theta$ [$^\circ$]')
    ax.set_zlabel('laser strength [$\mu$A]')
    plt.show()


def d3_fit_quarter():
    x, y = get_data('quarter_waveplate')
    x_err = [2 for val in x]
    y_err = [2 for val in y]
    model = amp2
    print(x)
    params, covariance = curve_fit(model, x, y, p0=[175, 90, 0, 0])
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(y, model(x, *params))
    print(r2)
    fit_x = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    fit_y = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
    fit_x, fit_y = np.meshgrid(fit_x, fit_y)
    fit_z = amp2(np.vstack([fit_x.ravel(), fit_y.ravel()]).T, *params)
    fit_z = fit_z.reshape(fit_x.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the original data
    ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o', label='Original Data')

    # Plot the fitted surface as a continuous surface
    ax.plot_surface(fit_x, fit_y, fit_z, cmap='viridis', alpha=0.5, label='Fitted Surface')

    ax.set_xlabel('$\\alpha$ [$^\circ$]')
    ax.set_ylabel('$\\theta$ [$^\circ$]')
    ax.set_zlabel('I [$\mu$A]')
    ax.legend()
    plt.show()


def separate_fit_quarter():
    x, y = get_data('quarter_waveplate')
    model = amp2

    params, covariance = curve_fit(model, x, y, p0=[175, 90, 0, 0])
    variance = np.diag(covariance)
    r2 = r2_score(y, model(x, *params))

    # Get unique alpha values
    unique_alpha_values = np.unique(x[:, 0])

    # Use a colormap for the colors
    # colors = plt.cm.tab20(np.linspace(0, 1, len(unique_alpha_values)))
    colors = colors = ['b', 'g', 'r', 'c', 'b', 'g', 'r', 'c', 'b', 'g', 'r', 'c']

    # Separate data into two ranges: alpha < 100 and alpha >= 100
    alpha_divisions = [0, 41, 101, 360]

    # Create separate plots for each alpha range
    for i in range(len(alpha_divisions) - 1):
        alpha_range = np.logical_and(x[:, 0] >= alpha_divisions[i], x[:, 0] < alpha_divisions[i+1])

        x_alpha_range = x[alpha_range]
        y_alpha_range = y[alpha_range]

        # Create a new figure for each alpha range
        fig, ax = plt.subplots(figsize=(8, 6))

        for alpha, color in zip(unique_alpha_values, colors):
            mask = x_alpha_range[:, 0] == alpha
            x_alpha = x_alpha_range[mask]
            y_alpha = y_alpha_range[mask]

            # Check if the array is not empty before calculating the range
            if len(x_alpha[:, 1]) > 0:
                # Plot the data points
                x_temp, y_temp = x_alpha[:, 1], y_alpha
                x_err = [2 for val in x_temp]
                y_err = [2 for val in y_temp]
                ax.errorbar(x=x_temp, y=y_temp, xerr = x_err, yerr = y_err, c=color, fmt='.', label=f'$\\alpha={alpha}^\circ$')

                # Plot the fitted curve for the current alpha value
                fit_x_alpha = np.linspace(min(x_alpha[:, 1]), max(x_alpha[:, 1]), 100)
                fit_y_alpha = model(np.column_stack((np.full_like(fit_x_alpha, alpha), fit_x_alpha)), *params)
                ax.plot(fit_x_alpha, fit_y_alpha, color=color, linestyle='--')

        ax.set_xlabel('$\\theta$ [$^\circ$]')
        ax.set_ylabel('I [$\mu$A]')
        ax.grid()
        ax.set_ylim([0, 180])
        ax.legend()

    plt.show()


def fit_half():
    x, y = get_data('half_waveplate')
    print(x)
    x_err = [2 for val in x]
    y_err = [2 for val in y]
    model = amp2
    print(x)
    upper_bound = [300, 360, 360, 360]
    lower_bound = [0, 0, 0, 0]
    params, covariance = curve_fit(model, x, y, p0=[175, 180, 0, 0], bounds=(lower_bound, upper_bound))
    # params, covariance = curve_fit(model, x, y, p0=[175, 180, 0, 0])
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(y, model(x, *params))
    print(r2)
    fit_x = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    fit_y = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
    fit_x, fit_y = np.meshgrid(fit_x, fit_y)
    fit_z = amp2(np.vstack([fit_x.ravel(), fit_y.ravel()]).T, *params)
    fit_z = fit_z.reshape(fit_x.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the original data
    ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o', label='Original Data')

    # Plot the fitted surface as a continuous surface
    ax.plot_surface(fit_x, fit_y, fit_z, cmap='viridis', alpha=0.5, label='Fitted Surface')

    ax.set_xlabel('$\\alpha$ [$^\circ$]')
    ax.set_ylabel('$\\theta$ [$^\circ$]')
    ax.set_zlabel('laser strength [$\mu$A]')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # d3_fit_quarter()
    # separate_fit_quarter()
    # fit_quarter()
    # fit_half()
    # plot_folder('quarter_waveplate')
