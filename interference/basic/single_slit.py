import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


calibrated_values = (-0.12245, 0.17400)



def get_angle(v, calibrated_values):
    a, b = calibrated_values
    return a*v + b


def truncate_at_nans(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length")
    
    nan_indices = np.where(np.isnan(arr1) | np.isnan(arr2))[0]
    if len(nan_indices) > 0:
        first_nan_index = nan_indices[0]
        return arr1[:first_nan_index], arr2[:first_nan_index]
    else:
        return arr1, arr2


def load_data(file_path):
    df = pd.read_csv(file_path, usecols=['Voltage (V) Run #1', 'Relative Intensity Run #1'])
    v = df.iloc[:,0]
    intensity = df.iloc[:,1]
    return truncate_at_nans(v, intensity)



def width_prediction(a, lamda):
    return 2*lamda/a


def plot_single_slit(file_path):
    v, i = load_data(file_path)
    angle = get_angle(v, calibrated_values)
    tan_angle = np.tan(angle)
    x = angle
    plt.plot(x, i, label='data')
    

    def model(x, amp, stretch):
        return amp*np.sinc(x/stretch/np.pi)**2
    
    popt, pcov = curve_fit(model, angle, i)
    perr = np.sqrt(np.diag(pcov))
    
    r2 = r2_score(i, model(x, *popt))
    print('r^2 score = ', r2)

    print('parameters:')
    print('amp = ', popt[0], 'with err = ', perr[0])
    print('stretch = ', popt[1], 'with err = ', perr[1])
    width = 2*np.pi*popt[1]
    width_err = 2*np.pi*perr[1]
    print('fit width = ', width, 'with err ', width_err)

    x_fit = np.linspace(np.min(x), np.max(x), 3000)
    y_fit = model(x_fit, *popt)
    plt.plot(x_fit, y_fit, label='fit')

    plt.xlabel(r'$\theta[rad]$')
    plt.ylabel('relative intensity')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    print(width_prediction(2*np.power(10., -5), 632.8*np.power(10., -9)))
    plot_single_slit('1_slit_0.02a__no_aperture_1.csv')
    # plot_single_slit('1_slit_0.02a__no_aperture_2.csv')
    # plot_single_slit('1_slit_0.02a_aperture0.1_1.csv')
    # plot_single_slit('1_slit_0.02a_aperture0.1_2.csv')