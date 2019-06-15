import numpy as np
import os
import parmap
import scipy

from yass.postprocess.util import run_deconv, partition_spike_time

def remove_high_mad(fname_templates,
                    fname_spike_train,
                    fname_weights,
                    reader,
                    neigh_channels,
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
                       list(zip(fnames_spike_times, fnames_out)),
                       reader,
                       neigh_channels,
                       processes=n_processors,
                       pm_pbar=True)                
    else:
        for ctr in range(len(units_in)):
            find_high_mad_unit(
                fnames_spike_times[ctr],
                fnames_out[ctr],
                reader,
                neigh_channels)

    collision_units = []
    for ctr, fname in enumerate(fnames_out):
        tmp = np.load(fname)
        if tmp['kill']:
            collision_units.append(units_in[ctr])

    idx_keep = units_in[~np.in1d(units_in, collision_units)]

    return idx_keep


def find_high_mad_unit(
    fname_spike_time,
    fname_out,
    reader,
    neigh_channels,
    min_var=2,
    breach=10,
    up_factor=2,
    min_ptp=2):

    # load spike times
    spt = np.load(fname_spike_time)

    # 1000 is enough for mad calculation
    max_spikes = 1000
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
    visch = np.where(ptps > min_ptp)[0]
    if len(visch) == 0:
        visch = [ptps.argmax()]
    wf = wf[:, :, visch]
    ptps = ptps[visch]
    neigh_channels = neigh_channels[visch][:, visch]

    # upsample
    wf_up = scipy.signal.resample(wf, n_times*up_factor,
                                  axis=1)

    # max channel within visible channels
    e_var, t_var = get_mad(wf_up, up_factor, ptps.argmax())
    violation_per_channel = np.sum(e_var > (t_var+min_var), axis=0)

    channels_checked = np.zeros(len(visch), 'bool')
    mc = ptps.argmax()
    channels_checked[neigh_channels[mc]] = True
    
    while ((violation_per_channel.sum() > breach) and 
           (not np.all(channels_checked)) and 
           (violation_per_channel[channels_checked].sum() <= breach)
          ):
        
        cands = np.where(~channels_checked)[0]
        channel = cands[ptps[cands].argmax()]
        e_var, t_var = get_mad(wf_up, up_factor, channel)

        violation_temp = np.sum(e_var > (t_var+min_var), axis=0)
        violation_per_channel = np.vstack((
            violation_per_channel,
            violation_temp)).min(0)
        
        channels_checked[neigh_channels[channel]] = True

    if violation_per_channel.sum() > breach:
        kill = True
    else:
        kill = False

    # save result
    np.savez(fname_out,
             kill=kill,
             violation_per_channel=violation_per_channel,
             visch=visch)

def get_mad(wf_up, up_factor, channel):
    
    best_shifts = find_shifts(wf_up[:, :, channel], up_factor)

    # shift
    wf_aligned_up = np.zeros(wf_up.shape, 'float32')
    for j in range(wf_up.shape[0]):
        wf_aligned_up[j] = np.roll(wf_up[j], -best_shifts[j], axis=0)

    # get t_var
    t_var = get_t_var(wf_aligned_up)

    wf_aligned = wf_aligned_up[:,up_factor//2:-up_factor//2]
    wf_aligned = wf_aligned[:,np.arange(0, wf_aligned.shape[1], up_factor)]
    t_var = t_var[up_factor//2:-up_factor//2]
    t_var = t_var[np.arange(0, len(t_var), up_factor)]

    # mad value for aligned waveforms
    e_var = np.square(np.median(np.abs(np.median(wf_aligned, axis=0)[None] - wf_aligned), axis=0)/0.67449)
    
    return e_var, t_var


def get_t_var(wf):

    # get theoretical variance from shift
    mean_wf = np.mean(wf, 0)
    left_bound = np.roll(mean_wf, 1, 0)
    right_bound = np.roll(mean_wf, -1, 0)
    t_var = var_mixture_uniform(left_bound, mean_wf, mean_wf, right_bound)

    return t_var

    

def find_shifts(wf_up, up_factor):
    
    n_times_up = wf_up.shape[1]

    # align on max channel
    mean_up = np.mean(wf_up, axis=0)
    t_start = n_times_up//4
    t_end = n_times_up - n_times_up//4

    shifts = np.arange(-up_factor//2, up_factor//2+1)
    fits = np.zeros((wf_up.shape[0], len(shifts)))
    for j, shift in enumerate(shifts):
        fits[:,j] = np.sum(wf_up[:,t_start+shift:t_end+shift]*mean_up[t_start:t_end], 1)
    best_shifts = shifts[fits.argmax(1)]
    
    return best_shifts
    
def var_mixture_uniform(a1, a2, b1, b2):
    p = 0.5
    m1_a, m2_a = moment_1_2_unif(a1,a2)
    m1_b, m2_b = moment_1_2_unif(b1,b2)
    var_b = m2_b - m1_b**2
    # term1 
    t1 = p*(m2_a+m2_b) - (p**2)*(m1_a**2 + m1_b**2) + var_b
    # term2
    t2 = - 2*m1_a*m1_b*(p**2)
    # term3
    t3 = - 2*p*var_b

    return t1+t2+t3


def moment_1_2_unif(a, b):
    return (0.5*(a+b), (a**2 + b**2 + a*b)/3)
