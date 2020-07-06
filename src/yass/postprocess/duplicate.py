import numpy as np
import os
import parmap
import scipy

def remove_duplicates(fname_templates, fname_weights,
                      save_dir, CONFIG, units_in=None, units_to_process=None,
                      multi_processing=False, n_processors=1):

    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load weights
    weights = np.load(fname_weights)

    # units_in is all units if none
    if units_in is None:
        units_in = np.arange(len(weights))
    if units_to_process is None:
        units_to_process = np.copy(units_in)

    # this allows units not in units_to_prcoess not get killed
    units_to_not_process = np.arange(len(weights))
    units_to_not_process = units_to_not_process[
        ~np.in1d(units_to_not_process, units_to_process)]
    weights[units_to_not_process] = np.max(weights) + 10

    # compute overlapping units
    fname_units_to_compare = os.path.join(save_dir, 'units_to_compare.npy')
    if os.path.exists(fname_units_to_compare):
        units_to_compare = np.load(fname_units_to_compare)[()]
    else:
        units_to_compare = compute_units_to_compare(
            fname_templates, units_in, units_to_process, CONFIG)
        # save it
        np.save(fname_units_to_compare,
                units_to_compare)

    ## partition templates
    #save_dir_partition = os.path.join(save_dir, 'partition')
    #if not os.path.exists(save_dir_partition):
    #    os.makedirs(save_dir_partition)
    #fnames_in = partition_templates(fname_templates,
    #                                units_to_compare,
    #                                save_dir_partition)

    #find duplicates
    #save_dir_result = os.path.join(save_dir, 'result')
    #if not os.path.exists(save_dir_result):
    #    os.makedirs(save_dir_result)

    fname_duplicates = os.path.join(save_dir, 'duplicates.npy')
    if os.path.exists(fname_duplicates):
        duplicates = np.load(fname_duplicates)[()]
    else:
        up_factor = 5
        max_diff_threshold = CONFIG.clean_up.abs_max_diff
        max_diff_rel_threshold = CONFIG.clean_up.rel_max_diff

        # find duplicates
        if multi_processing:

            # divide keys
            units = list(units_to_compare.keys())
            n_units =  len(units)
            sub_units_to_compare = []
            for j in range(n_processors):
                sub_keys = units[slice(j, n_units, n_processors)]
                sub_units_to_compare.append({k: units_to_compare[k] for k in sub_keys})

            # run duplicate detector
            duplicates_list = parmap.map(run_duplicate_detector,
                                         sub_units_to_compare,
                                         fname_templates,
                                         up_factor,
                                         max_diff_threshold,
                                         max_diff_rel_threshold,
                                         processes=n_processors)

            duplicates = {}
            for sub_list in duplicates_list:
                for unit in sub_list:
                    duplicates[unit] = sub_list[unit]

        else:
            duplicates = run_duplicate_detector(
                units_to_compare, fname_templates,
                up_factor, max_diff_threshold,
                max_diff_rel_threshold)

        # save it
        np.save(fname_duplicates, duplicates)

    fname_units_killed = os.path.join(save_dir, 'units_killed.npy')
    if os.path.exists(fname_units_killed):
        units_killed = np.load(fname_units_killed)
    else:
        units_killed = kill_duplicates(duplicates, weights)
        np.save(fname_units_killed, units_killed)

    return np.setdiff1d(units_in, units_killed)


def compute_units_to_compare(fname_templates, units_in,
                             units_to_process, CONFIG):

    # threshold on ptp diff
    diff_threshold = CONFIG.clean_up.abs_max_diff
    diff_rel_threshold = CONFIG.clean_up.rel_max_diff

    # load templates
    templates = np.load(fname_templates)
    #templates = templates[units_in]
    #n_units = templates.shape[0]

    # get ptps
    max_val = templates.max(1)
    min_val = templates.min(1)
    ptps = (max_val - min_val).max(1)
    ptps_higher = np.maximum(ptps[:, None], ptps[None])

    units_to_compare = {}
    idx_process = np.in1d(units_in, units_to_process)
    units_in_process = units_in[idx_process]
    units_in_dont_process = units_in[~idx_process]
    for ii, j in enumerate(units_in_process):

        if ii < len(units_in_process) - 1:
            # add within units_in_:
            max_val_diff = np.max(np.abs(max_val[units_in_process[ii+1:]] - max_val[[j]]), axis=1)
            min_val_diff = np.max(np.abs(min_val[units_in_process[ii+1:]] - min_val[[j]]), axis=1)
            abs_diff = np.maximum(max_val_diff, min_val_diff)
            abs_diff_rel = abs_diff/ptps_higher[j, units_in_process[ii+1:]]
            units_to_compare_1 = units_in_process[ii+1:][np.logical_or(
                abs_diff < diff_threshold, abs_diff_rel < diff_rel_threshold)]
        else:
            units_to_compare_1 = np.array(0, 'int32')

        # 
        max_val_diff = np.max(np.abs(max_val[units_in_dont_process] - max_val[[j]]), axis=1)
        min_val_diff = np.max(np.abs(min_val[units_in_dont_process] - min_val[[j]]), axis=1)
        abs_diff = np.maximum(max_val_diff, min_val_diff)
        abs_diff_rel = abs_diff/ptps_higher[j, units_in_dont_process]
        units_to_compare_2 = units_in_dont_process[np.logical_or(
                abs_diff < diff_threshold, abs_diff_rel < diff_rel_threshold)]

        # nearby units
        units_to_compare[j] = np.hstack((units_to_compare_1, units_to_compare_2))

    return units_to_compare


def run_duplicate_detector(
    units_to_compare, fname_templates,
    up_factor=5, max_diff_threshold=1.2,
    max_diff_rel_threshold=0.12):

    templates = np.load(fname_templates)

    duplicates = {}
    for unit in units_to_compare:

        # skip if already run
        #fname_out = os.path.join(save_dir, 'unit_{}.npz'.format(unit))
        #if os.path.exists(fname_out):
        #    continue

        # candidates
        candidates = units_to_compare[unit]
        # skip if no candidates
        if len(candidates) == 0:
            continue

        duplicates_ = abs_max_dist(
            templates[unit],
            templates[candidates],
            up_factor,
            max_diff_threshold,
            max_diff_rel_threshold)
        duplicates_ = candidates[duplicates_]

        duplicates[unit] = duplicates_

        ## save duplicates
        #np.savez(fname_out,
        #         unit=unit,
        #         duplicates=duplicates)

    return duplicates


def abs_max_dist(template_unit, templates_candidates,
                 up_factor=5, max_diff_threshold=1.2,
                 max_diff_rel_threshold=0.12):
    
    # find shift
    mc = template_unit.ptp(0).argmax()
    min_unit = template_unit[:, mc].argmin()
    min_candidates = templates_candidates[:, :, mc].argmin(1)
    shifts = min_candidates - min_unit

    # find vis chans
    vis_chan_unit = np.max(
        np.abs(template_unit), axis=0) > max_diff_threshold/2
    vis_chan_candidates = np.max(
        np.abs(templates_candidates), axis=(0, 1)) > max_diff_threshold/2
    vis_chan = np.logical_or(vis_chan_unit, vis_chan_candidates)
    vis_chan = np.where(vis_chan)[0]
    n_vis_chan = len(vis_chan)

    # temporal size
    n_times = template_unit.shape[0]

    # mask it to vis chans
    template_unit = template_unit[:, vis_chan]
    templates_candidates = templates_candidates[:, :, vis_chan]

    # upsample best fit template
    up_temp = scipy.signal.resample(
        x=template_unit,
        num=n_times * up_factor,
        axis=0)

    # shift and downsample
    up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor)
                                + np.arange(up_factor)]
    shifted_temps = np.concatenate(
        (np.roll(up_shifted_temps[:,[0]], shift=-2, axis=0),
         np.roll(up_shifted_temps, shift=-1, axis=0),
         up_shifted_temps,
         np.roll(up_shifted_temps, shift=1, axis=0),
         np.roll(up_shifted_temps, shift=2, axis=0)), axis=1)

    n_candidates = templates_candidates.shape[0]
    max_diff = np.ones(n_candidates)*100
    for j in range(n_candidates):

        if np.abs(shifts[j]) > n_times//4:
            continue

        # shift template
        temp_candidate = np.roll(
            templates_candidates[j], shift=-shifts[j], axis=0)

        # cut edges
        cut = np.max((np.abs(shifts[j]), 2))
        temp_candidate = temp_candidate[cut:-cut]

        # find diff
        max_diff[j] = np.min(np.max(
            np.abs(temp_candidate[:, None] - shifted_temps[cut:-cut]), axis=(0,2)))

    # ptp higher is max of ptp of a unit and its comparing unit
    ptp_higher = templates_candidates.ptp(1).max(1)
    ptp_unit = template_unit.ptp(0).max()
    ptp_higher[ptp_higher < ptp_unit] = ptp_unit
    # get max difference relative to higher ptp
    max_diff_rel = max_diff/ptp_higher

    # find duplicates
    duplicates = np.where(np.logical_or(
    max_diff < max_diff_threshold,
    max_diff_rel < max_diff_rel_threshold))[0]

    return duplicates


def kill_duplicates(duplicates_dict, weights):

    # kill duplicates
    # for each pair, kill unit with lower weight
    units_kill = np.zeros(len(weights), 'bool')
    for unit in duplicates_dict:
        duplicates = duplicates_dict[unit]
        for dup_unit in duplicates:
            if weights[unit] > weights[dup_unit]:
                units_kill[dup_unit] = True
            else:
                units_kill[unit] = True

    return np.where(units_kill)[0]
