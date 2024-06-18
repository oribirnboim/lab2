import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def process_voltage(v):
    v = v[~np.isnan(v)]
    return np.average(v), np.sqrt(np.var(v))



def load_data(file_path, columns):
    df = pd.read_csv(file_path, usecols=columns)
    data = [df.iloc[:, i].to_numpy() for i in range(df.shape[1])]
    return data



def extract_angle(file_name):
    return int(file_name[6:-4])



def plot_calibration():
    file_path1 = 'angles/angles_0_2_5_7_10_12_15.csv'
    file_path2 = 'angles/angles_-1_-2_-3_-4_-5_-8_-10.csv'
    data = load_data(file_path1, [1+2*i for i in range(7)]) + load_data(file_path2, [1+2*i for i in range(7)])
    angle = np.array([0, 2, 5, 7, 10, 12, 15, -1, -2, -3, -4, -5, -8, -10])
    angle_err = np.array([0.25 for a in angle])
    angle = angle*np.pi/180
    angle_err = angle_err*np.pi/180
    v, v_err = [], []
    for i in range(len(data)):
        avg, err = process_voltage(data[i])
        v.append(avg), v_err.append(err)
    v = np.array(v)
    v_err = np.array(v_err)
    plt.errorbar(v, angle, xerr=v_err, yerr=angle_err, fmt='.', label = 'data')

    def model(x, a, b):
        return a*x + b
    
    popt, pcov = curve_fit(model, v, angle)
    x_fit = np.linspace(np.min(v), np.max(v), 1000)
    y_fit = np.array([model(x, *popt) for x in x_fit])
    plt.plot(x_fit, y_fit, label = 'fit')


    r2 = r2_score(angle, model(v, *popt))
    print('r^2 score = ', r2)

    print('parameters: ', *popt)

    plt.xlabel('voltage [V]')
    plt.ylabel('angle [rads]')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    plot_calibration()
