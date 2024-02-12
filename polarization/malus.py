from scipy .optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
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
    x = np.arange(0, len(x)*step, step)
    y = y * 10 ** 6
    return x, y



def malus2():
    data = get_data('malus2')
    x, y = process_data(data, 10)
    x_err = [2 for val in x]
    y_err = [0.05 for val in y]
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='.', label='data')
    def model(theta, i, d):
        return i*np.cos((theta+d)*np.pi/180)**2
    params, covariance = curve_fit(model, x, y, p0=[175, 0])
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(y, model(x, *params))
    print(r2)
    x_fit = np.linspace(np.min(x), np.max(x), 1000)
    y_fit = model(x_fit, *params)
    plt.plot(x_fit, y_fit, label='fit')
    plt.xlabel('Degrees [$^\circ$]', fontsize=14)
    plt.ylabel('I [$\mu$A]', fontsize=14)
    plt.grid()
    plt.legend()
    plt.title('Malus law for 2 polarisers', fontsize=18)
    plt.show()


def malus3(name, step):
    data = get_data(name)
    x, y = process_data(data, step)
    x_err = [2 for val in x]
    y_err = [0.05 for val in y]
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='.', label='data')
    def model(theta, i, d, error):
        angle = (theta+d)*np.pi/180
        return i*(np.cos(angle)*np.cos(np.pi/2 + error*np.pi/180 - angle))**2
    params, covariance = curve_fit(model, x, y, p0=[175, 0, 0])
    variance = np.diag(covariance)
    print(params)
    print(variance)
    r2 = r2_score(y, model(x, *params))
    print(r2)
    x_fit = np.linspace(np.min(x), np.max(x), 1000)
    y_fit = model(x_fit, *params)
    plt.plot(x_fit, y_fit, label='fit')
    plt.xlabel('Degrees [$^\circ$]', fontsize=14)
    plt.ylabel('I [$\mu$A]', fontsize=14)
    plt.grid()
    plt.legend()
    plt.title('Malus law for 3 polarisers', fontsize=18)
    plt.show()


if __name__ == "__main__":
    # malus2()
    # malus3('malus3_grey_1', 5)
    malus3('malus3_grey_2', 10)
    pass