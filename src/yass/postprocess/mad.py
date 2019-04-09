import numpy as np
import os
import parmap
import scipy

from yass.postprocess.util import run_deconv, partition_spike_time

def remove_high_mad(fname_templates,
                    fname_spike_train,
                    fname_weights,
                    reader,
                    save_dir,
                    units_in=None,
                    multi_processing=False,
                    n_processors=1):
    
    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # units_in is all units if none
    if units_in is None:
        n_units = np.load(fname_templates).shape[0]
        units_in = np.arange(units_in)

    # partition spike tImes
    spt_save_dir = os.path.join(save_dir, 'spike_times')
    fnames_spike_times = partition_spike_time(
        spt_save_dir, fname_spike_train, units_in)

    # collect save file name
    fnames_out = []
    for unit in units_in:
        fname = os.path.join(save_dir, 'unit_{}.npz'.format(unit))
        fnames_out.append(fname)
        
    if multi_processing:
        parmap.starmap(find_high_mad_unit, 
                   list(zip(units_in, fnames_spike_times, fnames_out)),
                   fname_templates,
                   units_in,
                   reader,
                   processes=n_processors,
                   pm_pbar=True)                
    else:
        for ctr in range(len(units_in)):
            find_high_mad_unit(
                units_in[ctr],
                fname_spike_times[ctr],
                fnames_out[ctr],
                fname_templates,
                units_in,
                reader)

    # load weights
    weights = np.load(fname_weights)

    # logic:
    # 1. if tmp['kill'] is True, it is a collision
    # 2. if tmp['kill'] is Fale and no matched unit, then clean unit
    # 3. if tmp['kill'] is Fale and matched to a clean unit, then collision
    # 4. if tmp['kill'] is False and matched to non-clean, kill smaller of two
    collision_units = []
    clean_units = []
    matched_pairs = []
    for ctr, fname in enumerate(fnames_out):
        tmp = np.load(fname)
        # logic #1
        if tmp['kill']:
            collision_units.append(units_in[ctr])
        # logic #2
        elif tmp['unit_matched'] == None:
            clean_units.append(units_in[ctr])
        else:
            matched_pairs.append([units_in[ctr], tmp['unit_matched']])
    collision_units = np.array(collision_units)
    clean_units = np.array(clean_units)
    matched_pairs = np.array(matched_pairs)

    # logic #3
    if len(matched_pairs) > 0:
        matched_to_clean = np.in1d(matched_pairs[:,1], clean_units)
        collision_units = np.hstack((collision_units,
                                     matched_pairs[matched_to_clean, 0]))

        # logic #4
        collision_pairs = matched_pairs[~matched_to_clean]
        if len(collision_pairs) > 0:
            for j in range(collision_pairs.shape[0]):
                k, k_ = collision_pairs[j]
                if weights[k] > weights[k_]:
                    collision_units = np.append(collision_units, k_)
                else:
                    collision_units = np.append(collision_units, k_)

    idx_keep = units_in[~np.in1d(units_in, collision_units)]

    return idx_keep

def find_high_mad_unit(unit,
                       fname_spike_time,
                       fname_out,
                       fname_templates,
                       units_in,
                       reader,
                       mad_gap=0.8,
                       mad_gap_breach=3,
                       up_factor=8,
                       residual_max_norm=1.2,
                       jitter=1):

    if os.path.exists(fname_out):
        return

    # load templates
    templates = np.load(fname_templates)

    # load spike times
    spt = np.load(fname_spike_time)

    # 500 is enough for mad calculation
    max_spikes = 500
    if len(spt) > max_spikes:
        spt = np.random.choice(a=spt,
                               size=max_spikes,
                               replace=False)

    # get waveforms
    wf, _ = reader.read_waveforms(spt)
    _, n_times, n_channels = wf.shape

    # get mean waveform
    mean_wf = np.mean(wf, axis=0)

    # limit to visible channels
    ptps = mean_wf.ptp(0)
    visch = np.where(ptps > 2)[0]
    wf = wf[:, :, visch]

    # max channel within visible channels
    mc = ptps[visch].argmax()

    # upsample
    wf_up = scipy.signal.resample(wf, n_times*up_factor,
                                  axis=1)
    n_times_up = wf_up.shape[1]

    # align on max channel
    mean_up = np.mean(wf_up[:, :, mc], axis=0)
    t_start = n_times_up//4
    t_end = n_times_up - n_times_up//4

    shifts = np.arange(-up_factor//2, up_factor//2+1)
    fits = np.zeros((wf.shape[0], len(shifts)))
    for j, shift in enumerate(shifts):
        fits[:,j] = np.sum(wf_up[:,t_start+shift:t_end+shift,mc]*mean_up[t_start:t_end], 1)
    best_shifts = shifts[fits.argmax(1)]

    # shift
    wf_aligned = np.zeros(wf_up.shape, 'float32')
    for j in range(wf.shape[0]):
        wf_aligned[j] = np.roll(wf_up[j], -best_shifts[j], axis=0)

    wf_aligned = wf_aligned[:,t_start:t_end]
    wf_aligned = wf_aligned[:,np.arange(0, wf_aligned.shape[1], up_factor)]

    # mad value for aligned waveforms
    t_mad = np.median(
            np.abs(np.median(wf_aligned, axis=0)[None] - wf_aligned), axis=0)

    mad_loc = (t_mad > mad_gap).sum(0)

    # if every visible channels are mad channels, kill it
    if np.all(mad_loc > mad_gap_breach):
        kill = True
        unit_matched = None

    # if no mad chanels, keep it
    elif np.all(mad_loc <= mad_gap_breach):
        kill = False
        unit_matched = None

    # if not, but if there is a unit that matches on
    # non mad channels, hold off
    # if there is no matching unit, keep it
    else:
        kill = False
        
        # load templates
        templates = np.load(fname_templates)[units_in]
        
        # unit idx within units_in
        unit_idx = np.where(units_in == unit)[0][0]

        # get mad and non mad channels
        mad_channels = visch[mad_loc > mad_gap_breach]
        non_mad_channels = np.setdiff1d(np.arange(n_channels), mad_channels)

        # get idx for all units but unit being tested
        idx_no_target = np.arange(len(units_in))
        idx_no_target = np.delete(idx_no_target, unit_idx)
        
        # run deconv without channels with high collisions
        residual, unit_matched = run_deconv(
            templates[unit_idx][:, non_mad_channels],
            templates[idx_no_target][:, :, non_mad_channels],
            up_factor)
        
        if np.max(np.abs(residual)) > residual_max_norm:
            unit_matched = None
        else:
            unit_matched = units_in[idx_no_target[unit_matched]]

    # save result
    np.savez(fname_out,
             kill=kill,
             unit_matched=unit_matched)
