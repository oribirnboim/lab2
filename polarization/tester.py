import numpy as np
from numpy import sin, cos


def model(theta, alpha):
    theta, alpha = to_rads(theta), to_rads(alpha)
    wave_x = lambda k: cos(alpha)*cos(k)
    initial_y = lambda k: sin(alpha)*cos(k)
    wave_y = lambda k: initial_y(k-np.pi/2)
    a = np.linspace(0, 2*np.pi)


def to_rads(a):
    return a*np.pi/180


def plot():
    theta = np.linspace(0, 180, 1000)
    alpha = 45



if __name__ == "__main__":
    plot()