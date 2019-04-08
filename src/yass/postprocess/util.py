import numpy as np
import os
import scipy

def get_weights(fname_templates, fname_spike_train, out_dir):
    '''
    compute weights, which is number of spikes in each unit
    '''

    # load templates and spike train
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)

    # number of units
    n_units = templates.shape[0]

    # compute weights
    weights = np.zeros(n_units, 'int32')
    unique_ids, unique_weights = np.unique(spike_train[:, 1],
                                           return_counts=True)

    weights[unique_ids] = unique_weights
    
    # save
    fname_weights = os.path.join(out_dir, 'weights.npy')
    np.save(fname_weights, weights)

    return fname_weights, n_units

def run_deconv(data, templates, up_factor):

    # get shape
    n_units, n_times, n_chans = templates.shape

    # norm of templates
    norm_temps = np.square(templates).sum(axis=(1,2))

    # calculate objective function in deconv
    temps = np.flip(templates, axis=1)
    obj = np.zeros((n_units, n_times))
    for j in range(n_units):
        for c in range(n_chans):
            obj[j] += np.convolve(temps[j,:,c], data[:,c], 'same')
    obj = 2*obj - norm_temps[:, np.newaxis]

    if np.max(obj) > 0:
        best_fit_unit = np.max(obj, axis=1).argmax()
        best_fit_time = obj[best_fit_unit].argmax()
        shift = best_fit_time - n_times//2
        shifted_temp = np.roll(templates[best_fit_unit], shift, axis=0)

        up_temp = scipy.signal.resample(
            x=shifted_temp,
            num=n_times * up_factor,
            axis=0)

        up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor + np.arange(up_factor))]
        up_shifted_temps = np.concatenate((up_shifted_temps, np.roll(up_shifted_temps, shift=1, axis=0)), 1)
        if shift > 0:
            up_shifted_temps[:shift+1] = 0
        elif shift < 0:
            up_shifted_temps[shift-1:] = 0
        elif shift == 0:
            up_shifted_temps[[0,-1]] = 0

        idx_best_fit = np.max(np.abs(data[:,None] - up_shifted_temps), (0,2)).argmin()
        residual = data - up_shifted_temps[:,idx_best_fit]
    else:
        residual = data
        best_fit_unit = None

    return residual, best_fit_unit

def partition_spike_time(save_dir,
                         fname_spike_index,
                         units_in):

    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data
    spike_index = np.load(fname_spike_index)

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    spike_index_list = [[] for ii in range(n_units)]
    for j in range(len(spike_index)):
        tt, ii = spike_index[j]
        spike_index_list[ii].append(tt)

    # save them
    fnames = []
    for unit in units_in:

        fname = os.path.join(save_dir, 'partition_{}.npy'.format(unit))
        np.save(fname,
                spike_index_list[unit])
        fnames.append(fname)
        
    return fnames
