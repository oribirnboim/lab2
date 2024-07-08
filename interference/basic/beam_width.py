from single_slit import get_angle, truncate_at_nans, width_prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


calibrated_values = (-0.10387617716748218, -0.11527878759453152)


def load_data(file_path):
    df = pd.read_csv(file_path, usecols=['Voltage (V) Run #1', 'Relative Intensity Run #1'])
    v = df.iloc[:,1]
    intensity = df.iloc[:,0]
    return truncate_at_nans(v, intensity)


def width_prediction(w, lamda):
    return 2*lamda/w


def plot_single_slit(file_path):
    v, i = load_data(file_path)
    angle = get_angle(v, calibrated_values)
    tan_angle = np.tan(angle)
    x = tan_angle
    plt.plot(x, i)

    plt.xlabel(r'tan($\alpha$)')
    plt.ylabel('relative intensity')
    plt.grid()
    plt.show()


def plot_width():
    a = np.array([2, 4, 8])
    w = np.array([0.067, 0.032, 0.016])
    w_err = np.array([0.002, 0.002, 0.002])
    plt.errorbar(x=a, y=w, yerr=w_err, fmt='.', label='data')
    xlim = (1, 10)
    ylim = (0.0, 0.1)

    a_pre = np.linspace(*xlim, 1000)
    w_pre = width_prediction(a_pre, 632.8*np.power(10., -9))*np.power(10, 5)
    plt.plot(a_pre, w_pre, label = 'prediction')

    r2 = r2_score(y_true=w, y_pred=width_prediction(a, 632.8*np.power(10., -9))*np.power(10, 5))
    print(f'r^2 score is {r2}')

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.ylabel('main beam width [rads]')
    plt.xlabel(r'aperture width $[10^{-5} m]$')
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # print(width_prediction(2*np.power(10., -5), 632.8*np.power(10., -9)))
    # # plot_single_slit('1_slit_0.04a_aperture0.1_1.csv')
    # plot_single_slit('1_slit_0.08a_aperture0.1_1.csv')
    plot_width()