import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import imutils
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from math import ceil

def viz_before_after(before, title):
    og_img = np.reshape(before, (x, y, z))
    pred_img = model.predict(np.reshape(before, (1, x, y, z, 1)))
    pred_img = np.reshape(pred_img, (x, y, z))
    
    # Plot original and new image
    f, axarr = plt.subplots(3, 2, figsize=(20,30))
    cmap = plt.get_cmap('viridis')
    
    # Define plots
    f.suptitle(title, fontsize=16)
    
    # Middle slice
    z_plot = ceil(pred_img.shape[2]/2)
    axarr[0][0].imshow(np.reshape(og_img[:,:,z_plot], (x, y)), cmap=cmap)
    axarr[0][0].set_title('Original')
    axarr[0][1].imshow(pred_img[:,:,z_plot], cmap=cmap)
    axarr[0][1].set_title('Recreated')
    
    # Early slice
    z_plot = ceil(pred_img.shape[2]/4)
    axarr[1][0].imshow(np.reshape(og_img[:,:,z_plot], (x, y)), cmap=cmap)
    axarr[1][0].set_title('Original')
    axarr[1][1].imshow(pred_img[:,:,z_plot], cmap=cmap)
    axarr[1][1].set_title('Recreated')
    
    # Late slice
    z_plot = ceil(pred_img.shape[2]/4)
    z_plot *= 3
    axarr[2][0].imshow(np.reshape(og_img[:,:,z_plot], (x, y)), cmap=cmap)
    axarr[2][0].set_title('Original')
    axarr[2][1].imshow(pred_img[:,:,z_plot], cmap=cmap)
    axarr[2][1].set_title('Recreated')
    
    return og_img, pred_img

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

def brain_diff(a, b, threshold=3, self_stand=False, mean=0, std=0.1):
    ncolors = 256
    
    x, y = a.shape
    
    diff = a - b
    
    if self_stand:
        diff_scaled = (diff - np.mean(diff)) / np.std(diff)
    else:
        diff_scaled = (diff - mean) / std
    
    # Values that fall below the mean/threshold
    below_zero = diff_scaled.copy()
    below_zero[below_zero > -threshold] = 0

    # Values that fall above the mean/threshold
    above_zero = diff_scaled.copy()
    above_zero[above_zero < threshold] = 0
    
    # Log-ish Alpha scale
    color_alpha = alpha_curve(ncolors)
    
    # Above mean colormap
    color_array = plt.get_cmap('Reds')(range(ncolors))
    color_array[:,-1] = color_alpha
    map_object = LinearSegmentedColormap.from_list(name='above_mean',colors=color_array)
    plt.register_cmap(cmap=map_object)

    # Above mean colormap
    color_array = plt.get_cmap('Blues')(range(ncolors))
    color_array = np.flip(color_array, axis=0)
    color_array[:,-1] = np.flip(color_alpha) #np.linspace(1.0,0.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name='below_mean',colors=color_array)
    plt.register_cmap(cmap=map_object)
    
    # Define plots
    fig, ax = plt.subplots(1, figsize=(20, 30))
    
    # Plot middle slice
    index = 0
    ax.imshow(a, cmap='Greys')
    im = ax.imshow(above_zero, cmap='above_mean')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im = ax.imshow(below_zero, cmap='below_mean')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    

def build_overlay_image(a, b, threshold=2, mean=0, std =1):
    ncolors = 256

    x, y, z = a.shape
    z_plot = ceil(z / 2)

    diff = a - b

    diff_scaled = (diff - mean) / std 

    # Values that fall below the mean/threshold
    below_zero = diff_scaled.copy()
    below_zero[below_zero > -threshold] = 0
    
    # Values that fall above the mean/threshold
    above_zero = diff_scaled.copy()
    above_zero[above_zero < threshold] = 0
    
    return below_zero, above_zero

def viz_slices(a, b):
    ncolors = 256

    x, y, z = a.shape
    z_plot = ceil(z / 2)
    
    diff = a - b
    mean = np.mean(diff)
    std = np.std(diff)

    diff_norm = (diff - mean) / std

    # Log-ish Alpha scale
    color_alpha = alpha_curve(ncolors)
    color_ind = [0]
    color_ind.extend(range(40,ncolors))

    # Above mean colormap
    color_array = plt.get_cmap('Reds')(range(ncolors))
    color_array[:,-1] = color_alpha
    map_object = LinearSegmentedColormap.from_list(name='above_mean2',colors=color_array[color_ind])
    #     map_object.set_clim(0, 2.0)
    plt.register_cmap(cmap=map_object)

    # Above mean colormap
    color_array = plt.get_cmap('Blues')(range(ncolors))
    color_array = np.flip(color_array, axis=0)
    color_array[:,-1] = np.flip(color_alpha) #np.linspace(1.0,0.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name='below_mean2',colors=color_array[color_ind])
    plt.register_cmap(cmap=map_object)

    # Define plots
    fig, ax = plt.subplots(3, 3, figsize=(25, 30), sharex=True, sharey=True,)#, constrained_layout=True)
    vmin=0
    vmax=17
    vminm=-vmax
    vmaxm=vmin

    ranges = [(2,3), (3,4), (4, float('inf'))]
    for col, (min_thresh, max_thresh) in enumerate(ranges):
        above_zero_loc = (diff_norm >= min_thresh) & (diff_norm < max_thresh)
        below_zero_loc = (diff_norm <= -min_thresh) & (diff_norm > -max_thresh)

        above_zero = diff_norm.copy()
        above_zero[above_zero_loc == False] = 0

        below_zero = diff_norm.copy()
        below_zero[below_zero_loc == False] = 0

        # Plot middle slice
        index = 0
        z_plot = ceil(a.shape[2]/2)
        ax[index][col].imshow(a[:,:,z_plot], cmap='Greys')
        im = ax[index][col].imshow(above_zero[:,:,z_plot], cmap='above_mean2', vmin=vmin, vmax=vmax)
    #         fig.colorbar(im, ax=ax[col])
        im = ax[index][col].imshow(below_zero[:,:,z_plot], cmap='below_mean2', vmin=vminm, vmax=vmaxm)
    #         fig.colorbar(im, ax=ax[col])

        # Plot early slice
        z_plot = ceil(a.shape[2]/4)
        index = 1
        ax[index][col].imshow(a[:,:,z_plot], cmap='Greys')
        im = ax[index][col].imshow(above_zero[:,:,z_plot], cmap='above_mean2', vmin=vmin, vmax=vmax)
    #         fig.colorbar(im, ax=ax[col])
        im = ax[index][col].imshow(below_zero[:,:,z_plot], cmap='below_mean2', vmin=vminm, vmax=vmaxm)
    #         fig.colorbar(im, ax=ax[col])

        # Plot late slice
        z_plot = ceil(a.shape[2]/4)
        z_plot *= 3
        index = 2
        ax[index][col].imshow(a[:,:,z_plot], cmap='Greys')
        im1 = ax[index][col].imshow(above_zero[:,:,z_plot], cmap='above_mean2', vmin=vmin, vmax=vmax)
    #         fig.colorbar(im, ax=ax[col])
        im2 = ax[index][col].imshow(below_zero[:,:,z_plot], cmap='below_mean2', vmin=vminm, vmax=vmaxm)
    #         fig.colorbar(im, ax=ax[col])

        ax[0][col].set_title(f"{min_thresh} to {max_thresh}")

    fig.tight_layout(rect=[0, 0, 1, 0.5], w_pad=4, h_pad=4)
    fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.6, pad=0.01)
    fig.colorbar(im2, ax=ax.ravel().tolist(), shrink=0.6, pad=0.01)