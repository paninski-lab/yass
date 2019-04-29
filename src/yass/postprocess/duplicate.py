import numpy as np
import os
import parmap
import scipy

def remove_duplicates(fname_templates, fname_weights, save_dir, units_in=None,
                      multi_processing=False, n_processors=1):

    # first find duplicates
    units, fnames_out = find_duplicates(
        fname_templates, save_dir, units_in, 
        multi_processing, n_processors)

    # load weights
    weights = np.load(fname_weights)

    # kill duplicates
    # for each pair, kill unit with lower weight
    units_kill = np.zeros(len(weights), 'bool')
    for ctr, unit in enumerate(units):

        # load duplicates
        duplicates = np.load(fnames_out[ctr])
        for dup_unit in duplicates:
            if weights[unit] > weights[dup_unit]:
                units_kill[dup_unit] = True
            else:
                units_kill[unit] = True

    units_keep = units_in[~units_kill[units_in]]

    return units_keep


def find_duplicates(fname_templates, save_dir, units_in=None,
                    multi_processing=False, n_processors=1):
        
    ''' Compute absolute max distance using denoised templates
        Distances are computed between absolute value templates, but
        errors are normalized
    '''

    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # units_in is all units if none
    if units_in is None:
        n_units = np.load(fname_templates).shape[0]
        units_in = np.arange(units_in)

    # compute overlapping units
    units_compare_all = compute_units_to_compare(fname_templates)
    np.save(os.path.join(save_dir, 'units_compare_all.npy'),
            units_compare_all)
    
    # collect inputs
    units_compare = []
    fnames_out = []
    for unit in units_in:
        # remove units that are not supposed to be compared
        nearby_units = units_compare_all[unit]
        if len(nearby_units) > 0:
            nearby_units = nearby_units[
                np.in1d(nearby_units, units_in)]
        units_compare.append(nearby_units)

        # save file name
        fname = os.path.join(save_dir, 'unit_{}.npy'.format(unit))
        fnames_out.append(fname)

    # find duplicates
    if multi_processing:
        parmap.starmap(abs_max_dist,
                       list(zip(units_in, units_compare, fnames_out)),
                       fname_templates,
                       processes=n_processors,
                       pm_pbar=True)
    else:
        for ctr in range(len(units_in)):
            abs_max_dist(units_in[ctr],
                         units_compare[ctr], 
                         fnames_out[ctr],
                         fname_templates)

    return units_in, fnames_out

def compute_units_to_compare(fname_templates):

    templates = np.load(fname_templates)

    n_units = templates.shape[0]

    # get ptps
    max_val = templates.max(1)
    min_val = templates.min(1)
    ptps = (max_val - min_val).max(1)
    ptps_higher = np.maximum(ptps[:, None], ptps[None])

    # threshold on ptp diff
    diff_threshold = 1.2
    diff_rel_threshold = 0.15
    # calculate visible channels
    units_to_compare = []
    for unit in range(n_units-1):

        # 
        max_val_diff = np.max(np.abs(max_val[(unit+1):]-max_val[[unit]]), axis=1)
        min_val_diff = np.max(np.abs(min_val[(unit+1):]-min_val[[unit]]), axis=1)
        abs_diff = np.maximum(max_val_diff, min_val_diff)
        abs_diff_rel = abs_diff/ptps_higher[unit, (unit+1):]

        units_to_compare_ = np.where(
            np.logical_or(
                abs_diff < diff_threshold,
                abs_diff_rel < diff_rel_threshold))[0] + unit + 1
        # nearby units
        units_to_compare.append(units_to_compare_)

    units_to_compare.append([])

    return units_to_compare

def abs_max_dist(unit, candidates, fname_out, fname_templates,
                 up_factor=8, max_diff_threshold=1.2,
                 max_diff_rel_threshold=0.15):

    if os.path.exists(fname_out):
        return

    if len(candidates) == 0:
        np.save(fname_out, candidates)
        return

    # load templates
    templates = np.load(fname_templates)

    n_units, n_times, n_channels = templates.shape

    # get unit of interest
    template_unit = templates[unit]
    # get candidate templates as data
    templates_compare = templates[candidates]
    
    # find vis chans
    vis_chan_unit = np.max(
        np.abs(template_unit), axis=0) > max_diff_threshold/2
    vis_chan_candidates = np.max(
        np.abs(templates_compare), axis=(0, 1)) > max_diff_threshold/2
    vis_chan = np.logical_or(vis_chan_unit, vis_chan_candidates)
    vis_chan = np.where(vis_chan)[0]
    n_vis_chan = len(vis_chan)

    # mask it to vis chans
    template_unit = template_unit[:, vis_chan]
    templates_compare = templates_compare[:, :, vis_chan]

    # add zeros in between
    data = np.concatenate(
        (templates_compare,
         np.zeros((len(candidates), n_times, n_vis_chan), 'float32')),
        axis=1)
    data = data.reshape(-1, n_vis_chan)
    data = np.concatenate(
        (np.zeros((n_times, n_vis_chan), 'float32'), data),
        axis=0)
    
    # turn off templates
    templates = None

    # get objective value
    temps = np.flip(template_unit, axis=0)
    obj = np.zeros(data.shape[0])
    for c in range(n_vis_chan):
        obj += np.convolve(temps[:, c],
                           data[:, c], 'same')

    max_times = scipy.signal.argrelmax(obj, order=n_times)[0]
    # upsample best fit template
    up_temp = scipy.signal.resample(
        x=template_unit,
        num=n_times * up_factor//2,
        axis=0)
    
    # shift and downsample
    up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor//2
                                + np.arange(up_factor//2))]
    up_shifted_temps = np.concatenate(
        (up_shifted_temps,
         np.roll(up_shifted_temps, shift=1, axis=0)),
        axis=1)

    up_data_concat = np.zeros((data.shape[0], up_shifted_temps.shape[1], n_vis_chan), 'float32')
    for tt in max_times:
        up_data_concat[tt-n_times//2:tt+n_times//2+1] = up_shifted_temps

    diff = np.abs(data[:, None] - up_data_concat)
    diff = diff[n_times:].reshape((len(candidates), -1, diff.shape[1], n_vis_chan))
    diff = diff[:, :n_times]

    max_diff = np.min(np.max(diff, axis=(1,3)), axis=1)

    # ptp higher is max of ptp of a unit and its comparing unit
    ptp_higher = templates_compare.ptp(1).max(1)
    ptp_unit = template_unit.ptp(0).max()
    ptp_higher[ptp_higher < ptp_unit] = ptp_unit
    # get max difference relative to higher ptp
    max_diff_rel = max_diff/ptp_higher

    # 
    duplicates = candidates[np.logical_or(
        max_diff < max_diff_threshold,
        max_diff_rel < max_diff_rel_threshold)]

    # 
    np.save(fname_out, duplicates)

def abs_max_dist_orig(unit, candidates, fname_out, fname_templates,
                 up_factor=8, max_diff_threshold=1.2,
                 max_diff_rel_threshold=0.15):

    if os.path.exists(fname_out):
        return

    if len(candidates) == 0:
        np.save(fname_out, candidates)

    # load templates
    templates = np.load(fname_templates)

    # get shape
    n_units, n_times, n_channels = templates.shape

    # get relevant templates
    template_unit = templates[unit]
    templates_compare = templates[candidates]

    # turn off templates
    templates = None

    # do deconvolution
    temps = np.flip(templates_compare, axis=1)
    max_diff = np.zeros(len(candidates))
    for j in range(len(candidates)):

        # get objective value
        obj = np.zeros(n_times)
        for c in range(n_channels):
            obj += np.convolve(temps[j, :, c],
                               template_unit[:, c], 'same')

        # best fit time
        best_fit_time = obj.argmax()
        shift = best_fit_time - n_times//2
        shifted_temp = np.roll(templates_compare[j], shift, axis=0)

        # do upsample fit

        # upsample best fit template
        up_temp = scipy.signal.resample(
            x=shifted_temp,
            num=n_times * up_factor,
            axis=0)

        # shift and downsample
        up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor
                                    + np.arange(up_factor))]
        up_shifted_temps = np.concatenate(
            (up_shifted_temps,
             np.roll(up_shifted_temps, shift=1, axis=0)),
            axis=1)
        # mute rolled parts
        if shift > 0:
            up_shifted_temps[:shift+1] = 0
        elif shift < 0:
            up_shifted_temps[shift-1:] = 0
        elif shift == 0:
            up_shifted_temps[[0,-1]] = 0

        # find best fine shifts and compuate residual
        idx_best_fit = np.max(np.abs(template_unit[:,None] - up_shifted_temps),
                              axis=(0,2)).argmin()
        residual = template_unit - up_shifted_temps[:,idx_best_fit]

        max_diff[j] = np.max(np.abs(residual))

    # ptp higher is max of ptp of a unit and its comparing unit
    ptp_higher = templates_compare.ptp(1).max(1)
    ptp_unit = template_unit.ptp(0).max()
    ptp_higher[ptp_higher < ptp_unit] = ptp_unit
    # get max difference relative to higher ptp
    max_diff_rel = max_diff/ptp_higher

    # 
    duplicates = candidates[np.logical_or(
        max_diff < max_diff_threshold,
        max_diff_rel < max_diff_rel_threshold)]

    # 
    np.save(fname_out, duplicates)
