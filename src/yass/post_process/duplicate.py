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
    overlaps = compute_overlapping_units(fname_templates)

    # collect inputs
    units_compare = []
    fnames_out = []
    for unit in units_in:
        # only compare units whose id is bigger than unit
        # to prevent duplicate computation
        nearby_units = overlaps[unit]
        nearby_units = nearby_units[nearby_units > unit]
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

def compute_overlapping_units(fname_templates):

    templates = np.load(fname_templates)

    # get number of units
    n_units = templates.shape[0]

    # get ptps
    ptps = templates.ptp(1)

    # max channels
    mcs = ptps.argmax(1)

    # calculate visible channels
    overlaps = []
    for unit in range(n_units):

        # threshold is 0.5 of max ptp
        ptp_ratio = ptps[:, mcs[unit]]/ptps[unit, mcs[unit]]
        overlaps.append(np.where(
            np.logical_and(ptp_ratio > 0.5, ptp_ratio < 2))[0])

    return overlaps

    
def abs_max_dist(unit, candidates, fname_out, fname_templates,
                 up_factor=8, max_diff_threshold=1.2,
                 max_diff_rel_threshold=0.15):

    if os.path.exists(fname_out):
        return

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

