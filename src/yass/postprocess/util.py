import numpy as np
import os
import scipy
import datetime as dt

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

def run_deconv(data, templates, up_factor, method='l1'):

    #start = dt.datetime.now().timestamp()

    # find overlapping units with data and run in on those units only
    data_ptp = data.ptp(0)
    vis_th = np.min((2, np.max(data_ptp)))
    vis_chans =  data_ptp > vis_th
    overlap_units = np.where(
        np.any((templates.ptp(1))[:, vis_chans] > vis_th, axis=1))[0]

    # if no overlap units, just skip
    if len(overlap_units) == 0:
        return data, None, -10000
    else:
        templates = templates[overlap_units]

    # get shape
    n_units, n_times, n_chans = templates.shape

    # norm of templates
    norm_temps = np.square(templates).sum(axis=(1,2))

    # calculate objective function in deconv
    #start2 = dt.datetime.now().timestamp()

    temps = np.flip(templates, axis=1)
    obj = np.zeros((n_units, n_times))
    for j in range(n_units):
        for c in range(n_chans):
            obj[j] += np.convolve(temps[j,:,c], data[:,c], 'same')
    obj = 2*obj - norm_temps[:, np.newaxis]
    #print ("  obj fun time: ",  dt.datetime.now().timestamp() - start2)

    max_obj = np.max(obj, axis=1)
    best_fit_unit = max_obj.argmax()    
    if np.max(max_obj) > 0:
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

        if method == 'l1':
            idx_best_fit = np.max(np.abs(data[:,None] - up_shifted_temps), (0,2)).argmin()
        elif method == 'l2':
            idx_best_fit = np.sum(np.square(data[:,None] - up_shifted_temps), (0,2)).argmin()
        residual = data - up_shifted_temps[:,idx_best_fit]

    else:
        residual = data

    #print ("  total time: ",  dt.datetime.now().timestamp() - start)
            # best fit unit relative to all units
    return residual, overlap_units[best_fit_unit], np.max(max_obj)

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
        if os.path.exists(fname)==False:
            np.save(fname,
                    spike_index_list[unit])
        fnames.append(fname)
        
    return fnames
