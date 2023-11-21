import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import imutils
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from math import ceil


def alpha_curve(ncolors):
    x = np.array([0.5, 0.387, 0.24, 0.136, 0.04, 0.011])
    y = np.array([1.255, 1.25, 1.189, 1.124, 0.783, 0.402]) / 1.255

    # this is the function we want to fit to our data
    def func(x, a, b):
        'nonlinear function in a and b to fit to data'
        return a * x / (b + x)

    initial_guess = [1.2, 0.03]
    pars, pcov = curve_fit(func, x, y, p0=initial_guess)

    linspace = np.linspace(0, 1, ncolors)
    color_alpha = func(linspace, pars[0], pars[1])
    color_alpha /= np.amax(color_alpha)
    
    return color_alpha


def get_colormaps(max_z = 10):
    step = 0.1
    ncolors = int(max_z / step)
    low_cmap = int(ncolors-(ncolors/2))
    high_cmap = int(ncolors+(ncolors/2))

    color_alpha = alpha_curve(ncolors)
    color_alpha = [1 for x in range(0, ncolors)]
    color_alpha[0] = 0
    # for reds
    high_range = [round(x, 1) for x in np.arange(0, max_z, step)]
    reds = plt.get_cmap('Reds')(range(low_cmap, high_cmap))
    reds[:,-1] = color_alpha
    red_dict = {x: y for x,y in zip(high_range, reds)}

    # for blues
    low_range = [round(x, 1) for x in  np.arange(-max_z, 0, step)]
    blues = plt.get_cmap('Blues')(range(low_cmap, high_cmap))
    blues = np.flip(blues, axis=0)
    blues[:,-1] = np.flip(color_alpha)
    blue_dict = {x: y for x,y in zip(low_range, blues)}

    return red_dict, blue_dict