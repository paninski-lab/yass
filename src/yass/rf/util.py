import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sklearn.decomposition as decomp
import networkx as nx
import scipy
import scipy.signal
import scipy.ndimage as img
import scipy.signal as sig

#there are two steps to getting the rf. The first step is to get a core set of significant pixels using a 2x2 
# spatial filter. The second step expands the rf around the coreset. This was originally because I was trying to get
#fatter rfs for the multi-rf splitting project

def get_seed(image, dim = 2):
    kernel = np.full((dim, dim), 1/(float(dim)**2))
    smoothed_image = sig.convolve2d(image, kernel, 'same')
    
    seed = np.vstack(np.where(np.abs((smoothed_image - np.mean(smoothed_image))) > 4*np.std(smoothed_image)))
    
    
    std_pixels = np.array(seed).T
    
    dists = scipy.spatial.distance.cdist(std_pixels, std_pixels)
    thresh = 1.5    

    upper=1
    lower=0
    bin_dists = np.where(dists<thresh, upper, lower)
    #print (bin_dists)

    #compute connected nodes and sum spikes over them
    G = nx.from_numpy_array(bin_dists)
    
    con =  list(nx.connected_components(G))
    
    
    cluster_list = [std_pixels[list(con[i])] for i in range(len(con))]

    return cluster_list

def expand_rf(image, cluster_list):
    std = np.std(image)
    std_pixels = np.vstack(np.where(np.abs(image - np.mean(image))>std*2))

    if(std_pixels.shape[0] ==0):
        np.asarray(std_pixels = [])
        return "poop", std_pixels
    
    std_pixels = np.array(std_pixels).T
    
    dists = scipy.spatial.distance.cdist(std_pixels, std_pixels)
    thresh = 1.5    

    upper=1
    lower=0
    bin_dists = np.where(dists<thresh, upper, lower)
    #print (bin_dists)

    #compute connected nodes and sum spikes over them
    G = nx.from_numpy_array(bin_dists)
    
    con =  list(nx.connected_components(G))
    test_elements = [element[0] for element in cluster_list]
    valid_list = []
    for element in con:
        pixels = std_pixels[list(element)]
        
        indicator = [np.any([np.all(pixel == test_element) for pixel in pixels]) for test_element in test_elements]
        
        if np.any(indicator) == True:
            valid_list.append(element)
    
    return [std_pixels[list(element)] for element in valid_list]

#wrapper for the 2 steps
def get_rf(image, dim):
    seed = get_seed(image)
    return expand_rf(image, seed)


def classifiy_contours(gaussian_sd,
                       green_val,
                       fr_rates,
                       sd_mean_noise_th,
                       sd_ratio_noise_th,
                       green_noise_th,
                       midget_on_th,
                       midget_off_th,
                       large_on_th,
                       large_off_th,
                       sbc_fr_th):
    
    n_units = gaussian_sd.shape[0]
    labels = np.ones(n_units, 'int32')*-1
    
    # label:
    # 0: on parasol
    # 1: off parasol
    # 2: on midget
    # 3: off midget
    # 4: on large
    # 5: off large
    # 6: sbc
    # 7: unknown
    cell_types = ['On-Parasol', 'Off-Parasol', 'On-Midget', 'Off-Midget',
                  'On-Large', 'Off-Large', 'SBC', 'Unknown']
    
    sd_mean = np.mean(gaussian_sd, 1)
    max_sd = np.max(gaussian_sd, 1)
    max_sd[max_sd==0] = 1
    sd_ratio = np.min(gaussian_sd, 1)/max_sd
    
    # first find out bad contours
    idx_noise = np.logical_or(
        sd_mean < sd_mean_noise_th,
        sd_ratio < sd_ratio_noise_th)
    idx_noise = np.logical_or(idx_noise, np.abs(green_val) < green_noise_th)
    idx_noise = np.logical_and(idx_noise, labels==-1)
    labels[idx_noise] = 7
    
    # classify on
    idx_on = green_val >= green_noise_th
    
    idx_on_midget = np.logical_and(idx_on, sd_mean < midget_on_th)
    idx_on_midget = np.logical_and(idx_on_midget, labels==-1)
    labels[idx_on_midget] = 2
    
    idx_on_para = np.logical_and(idx_on, sd_mean < large_on_th)
    idx_on_para = np.logical_and(idx_on_para, labels==-1)
    labels[idx_on_para] = 0
    
    idx_on_large = np.logical_and(idx_on, labels==-1)
    labels[idx_on_large] = 4
    
    # classify off
    idx_off = green_val <= -green_noise_th

    idx_off_midget = np.logical_and(idx_off, sd_mean < midget_off_th)
    idx_off_midget = np.logical_and(idx_off_midget, labels==-1)
    labels[idx_off_midget] = 3
    
    idx_off_large = np.logical_and(idx_off, sd_mean >= large_off_th)
    idx_off_large = np.logical_and(idx_off_large, labels==-1)
    labels[idx_off_large] = 5
    
    idx_off_para = np.logical_and(idx_off, labels==-1)
    idx_off_para = np.logical_and(idx_off_para, fr_rates>sbc_fr_th)
    labels[idx_off_para] = 1
    
    idx_sbc = labels==-1
    labels[idx_sbc] = 6

    return labels, cell_types


def get_circle_plotting_data(i_cell, Gaussian_params):
    # Adapted from Nora's matlab code, hasn't been tripled checked

    circle_samples = np.arange(0,2*np.pi+0.1,0.1)
    x_circle = np.cos(circle_samples)
    y_circle = np.sin(circle_samples)

    # Get Gaussian parameters
    angle = Gaussian_params[i_cell,5]
    sd = Gaussian_params[i_cell,3:5]
    x_shift = Gaussian_params[i_cell,1]
    y_shift = Gaussian_params[i_cell,2]

    R = np.asarray([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    L = np.asarray([[sd[0], 0],[0, sd[1]]])
    circ = np.concatenate([x_circle.reshape((-1,1)),y_circle.reshape((-1,1))],axis=1)

    X = np.dot(R,np.dot(L,np.transpose(circ)))
    X[0] = X[0]+x_shift
    X[1] = np.abs(X[1]+y_shift)
    plotting_data = X

    return plotting_data


def contour_plots(contours, labels, cell_type_name, stim_size):
    
    n_types = len(cell_type_name)
    
    n_col = 5
    n_row = int(np.ceil(n_types/n_col))
    for j in range(n_types):
        plt.subplot(n_row, n_col, j+1)
        idx = np.where(labels == j)[0]
        for unit in idx:
            plt.plot(contours[unit][:, 0], contours[unit][:, 1], 'k', alpha=0.5)
        plt.xlim([0, stim_size[1]])
        plt.ylim([0, stim_size[0]])
        plt.title(cell_type_name[j])


def fr_plots(f_rates, mean_sd, labels, cell_type_name):
    
    f_rates[f_rates == 0] = np.exp(-11)
    mean_sd[mean_sd == 0] = np.exp(-11)

    f_rates = np.log(f_rates)
    mean_sd = np.log(mean_sd)

    n_types = len(cell_type_name)
    max_sd = np.max(mean_sd)
    max_fr = np.max(f_rates)
    
    n_col = 2
    n_row = int(np.ceil(n_types/n_col))
    for j in range(n_types):
        plt.subplot(n_row, n_col, j+1)
        idx = np.where(labels == j)[0]
        plt.scatter(mean_sd[idx], f_rates[idx], color='k', alpha=0.5)
        plt.ylim([-2, max_fr])
        plt.xlim([-2, max_sd])

        if j % n_col == 0:
            plt.ylabel('log of firing rates (Hz)')
        if j >= ((n_row-1)*n_col):
            plt.xlabel('log of mean Gaussian SD')
        plt.title(cell_type_name[j])


def make_classification_plot(gaussian_sd,
                             green_val,
                             f_rates,
                             sd_mean_noise_th,
                             sd_ratio_noise_th,
                             green_noise_th,
                             midget_on_th,
                             midget_off_th,
                             large_on_th,
                             large_off_th,
                             sbc_fr_th,
                             contours,
                             stim_size):
    
    labels, cell_type_name = classifiy_contours(
        gaussian_sd,
        green_val,
        f_rates,
        sd_mean_noise_th,
        sd_ratio_noise_th,
        green_noise_th/100,
        midget_on_th,
        midget_off_th,
        large_on_th,
        large_off_th,
        sbc_fr_th)
    
    colors = ['blue','red','green','cyan',
              'magenta','brown','pink', 'black']
    
    # label:
    # 0: on parasol
    # 1: off parasol
    # 2: on midget
    # 3: off midget
    # 4: on large
    # 5: off large
    # 6: sbc
    # 8: unknown
    n_cell_types = len(cell_type_name)
    
    
    mean_sd = np.mean(gaussian_sd, 1)
    max_sd = np.max(gaussian_sd, 1)
    max_sd[max_sd==0] = 1
    sd_ratio = np.min(gaussian_sd, 1)/max_sd
    
    dot_size = 5    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    for j in range(n_cell_types):
        plt.scatter(mean_sd[labels==j], green_val[labels==j], s=dot_size, color=colors[j])
    plt.ylabel('Most extreme green value')
    plt.xlabel('Mean SD')
    plt.ylim([np.min(green_val), np.max(green_val)])
    plt.xlim([0, 6])

    plt.subplot(1,2,2)
    for j in range(n_cell_types):
        plt.scatter(mean_sd[labels==j], sd_ratio[labels==j], s=dot_size, color=colors[j])
    plt.ylabel('Min/Max ratio of Gaussian SD')
    plt.xlabel('Mean Gaussian SD')
    custom_lines = [Line2D([], [], color=colors[i], marker='o', linestyle='None') for i in range(len(cell_type_name)-1)]
    plt.legend(custom_lines, cell_type_name[:-1])

    #plt.ylim([-0.05, 0.05])
    #plt.xlim([0, 6])
    plt.show()
    
    plt.figure(figsize=(12, 12))
    contour_plots(contours, labels, cell_type_name, stim_size)
    plt.show()
    
    plt.figure(figsize=(12, 10))
    fr_plots(f_rates, mean_sd, labels, cell_type_name)
    plt.show()
    
    return labels
