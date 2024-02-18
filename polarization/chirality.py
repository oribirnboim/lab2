import matplotlib.pyplot as plt
import numpy as np
from numpy import cos
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import re
import pandas as pd
from malus import process_data
import os


step=10


def get_data(folder_name):
    data_list = []
    d = []
    for filename in os.listdir(folder_name):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder_name, filename)
            c1, c2, dist = get_single(file_path)
            data_list.append((c1, c2))
            d.append(dist)
    return data_list, np.array(d)


def get_single(file_path):
    df = pd.read_excel(file_path, skiprows=6)
    distance = extract_distance(file_path)
    first_column_array = df.iloc[:, 0].to_numpy()
    second_column_array = df.iloc[:, 1].to_numpy()
    theta, power = process_data((first_column_array, second_column_array), step)
    return theta, power, distance


def extract_distance(filename):
    match = re.match(r'.*\\(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def cos2(theta, I, phase):
    return I*cos((theta+phase)*np.pi/180)**2


def linear(x, a):
    return a*x


def plot_phase_red():
    data, distance = get_data('chilary1')
    phase = []
    phase_err = []
    for i in range(len(data)):
        data_set = data[i]
        x, y = data_set # commented code diplsys and prints everything this loop does
        # plt.plot(x, y)
        params, covariance = curve_fit(cos2, x, y, p0=[175, 0])
        variance = np.diag(covariance)
        # print(params, distance[i])
        # print(variance)
        phase.append(params[1])
        phase_err.append(variance[1])
        # r2 = r2_score(y, cos2(x, *params))
        # print(r2)
        # x_fit = np.linspace(np.min(x), np.max(x), 1000)
        # y_fit = cos2(x_fit, *params)
        # plt.plot(x_fit, y_fit, label='fit')
        # plt.xlabel('Degrees [$^\circ$]', fontsize=14)
        # plt.ylabel('I [$\mu$A]', fontsize=14)
        # plt.grid()
        # plt.title(distance[i])
        # plt.legend()
        # plt.show()
    phase, phase_err = np.array(phase), np.array(phase_err)
    phase -= phase[np.argmax(phase)]
    phase *= -1
    distance -= distance[np.argmin(distance)]
    distance_err = np.array([1 for _ in distance])
    plt.errorbar(x=distance, y=phase, xerr=distance_err, yerr=phase_err, fmt='.', label='extracted phase')
    params, covariance = curve_fit(linear, distance, phase)
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(phase, linear(distance, *params))
    print(r2)
    x_fit = np.linspace(np.min(distance), np.max(distance), 1000)
    y_fit = linear(x_fit, *params)
    plt.plot(x_fit, y_fit, label='fit')
    plt.legend()
    plt.grid()
    plt.xlabel('$d[cm]$')
    plt.ylabel('$\phi[^\circ]$')
    plt.show()


def plot_phase_green():
    data, distance = get_data('malus_green')
    phase = []
    phase_err = []
    for i in range(len(data)):
        data_set = data[i]
        x, y = data_set # commented code diplsys and prints everything this loop does
        if distance[i]==20:
            x, y = x[1:], y[1:]
        print(len(x), len(y), distance[i])
        plt.plot(x, y)
        params, covariance = curve_fit(cos2, x, y, p0=[175, 0])
        variance = np.diag(covariance)
        print(params, distance[i])
        print(variance)
        phase.append(params[1])
        # phase.append(x[np.argmax(y)])
        phase_err.append(variance[1])
        # r2 = r2_score(y, cos2(x, *params))
        # print(r2)
        x_fit = np.linspace(np.min(x), np.max(x), 1000)
        y_fit = cos2(x_fit, *params)
        plt.plot(x_fit, y_fit, label='fit')
        plt.xlabel('Degrees [$^\circ$]', fontsize=14)
        plt.ylabel('I [$\mu$A]', fontsize=14)
        plt.grid()
        plt.title(distance[i])
        plt.legend()
        plt.show()
    phase, phase_err = np.array(phase), np.array(phase_err)
    phase -= phase[np.argmax(phase)]
    phase *= -1
    distance -= distance[np.argmin(distance)]
    distance_err = np.array([1 for _ in distance])
    plt.errorbar(x=distance, y=phase, xerr=distance_err, yerr=phase_err, fmt='.', label='extracted phase')
    params, covariance = curve_fit(linear, distance, phase)
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(phase, linear(distance, *params))
    print(r2)
    x_fit = np.linspace(np.min(distance), np.max(distance), 1000)
    y_fit = linear(x_fit, *params)
    plt.plot(x_fit, y_fit, label='fit')
    plt.legend()
    plt.grid()
    plt.xlabel('$d[cm]$')
    plt.ylabel('$\phi[^\circ]$')
    plt.show()


if __name__ == "__main__":
    plot_phase_green()
    plot_phase_red()
