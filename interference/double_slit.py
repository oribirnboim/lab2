import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from single_slit import get_angle, truncate_at_nans
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


calibrated_values = (-0.10387617716748218, -0.11527878759453152)
dist = 0.5 # length of rotating arm



def load_data(file_path):
    df = pd.read_csv(file_path, usecols=[1, 2])
    v = df.iloc[:,1]
    intensity = df.iloc[:,0]
    return truncate_at_nans(v, intensity)


def plot_double(file_path):
    v, i = load_data(file_path)
    angle = get_angle(v, calibrated_values)
    tan_angle = np.tan(angle)
    # x = tan_angle
    x = angle
    plt.plot(x, i)

    def model(x, amp, width, k):
        x = x
        stretch = width/2
        return amp*(np.sinc(x/stretch)*np.cos(x*k))**2 + 0.027
    
    popt, pcov = curve_fit(model, angle, i, p0=[2.5, 0.03, 2*np.pi/0.01])
    perr = np.sqrt(np.diag(pcov))
    
    r2 = r2_score(i, model(x, *popt))
    print('r^2 score = ', r2)

    print('parameters:')
    print('amp = ', popt[0], 'with err = ', perr[0])
    print('width = ', popt[1], 'with err = ', perr[1])
    print('k = ', popt[2], 'with err = ', perr[2])

    x_fit = np.linspace(np.min(x), np.max(x), 3000)
    y_fit = model(x_fit, *popt)
    plt.plot(x_fit, y_fit)
    plt.xlim([-0.05, 0.05])
    # plt.xlabel(r'tan($\alpha$)')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('relative intensity')
    plt.grid()
    plt.show()


def prediction(w, lamda, d):
    width = 2*lamda/w
    k = np.pi*d/lamda
    return width, k



if __name__ == "__main__":
    file_path = '2_slits_0.04a_0.125d_aperture0.1_3.csv'
    print(*prediction(0.00004, 632.8*np.power(10., -9), 0.000125))
    plot_double(file_path)
    # file_path = '2_slits_0.04a_0.25d_1.csv'
    # print(*prediction(0.00004, 632.8*np.power(10., -9), 0.00025))
    # plot_double(file_path)