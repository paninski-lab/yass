import numpy as np
import os
import parmap
import scipy.signal

def duplicate_l2(fname_templates, fname_spike_train, neigh_channels,
                 save_dir, n_spikes_big=300, min_ptp=2, units_in=None):

    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)

    n_units = templates.shape[0]
    n_spikes = np.zeros(n_units, 'int32')
    aa, bb = np.unique(spike_train[:,1], return_counts=True)
    n_spikes[aa] = bb

    if units_in is None:
        units_in = np.arange(n_units)

    templates = templates[units_in]
    n_spikes = n_spikes[units_in]

    idx_sort = np.argsort(n_spikes)
    templates = templates[idx_sort]
    n_spikes = n_spikes[idx_sort]

    # info necessary
    mcs = templates.ptp(1).argmax(1)
    norms = np.sum(np.square(templates), axis=1)
    vis_chans = templates.ptp(1) > min_ptp

    # idx of low fr units
    idx_small = np.where(n_spikes < n_spikes_big)[0]

    pairs = []
    for k in idx_small:
        idx_big_k = np.where(neigh_channels[mcs[k]][mcs[k+1:]])[0] + k+1
        if len(idx_big_k) > 0:
            #idx_big_k = idx_big[np.any(vis_chans[idx_big][:,vis_chans[k]], 1)]
            vis_chans_k = np.logical_or(vis_chans[k], vis_chans[idx_big_k])

            objs = np.zeros(len(idx_big_k))
            for ii, k2 in enumerate(idx_big_k):
                vis_chans_ = vis_chans_k[ii]
                objs[ii] = (2*np.sum(
                    templates[k,:,vis_chans_]*templates[k2,:,vis_chans_]) - 
                            1.2*np.sum(norms[[k, k2]][:, vis_chans_], 1).max())
            if np.max(objs) > 0:
                pairs.append([k, idx_big_k[np.argmax(objs)]])
            
    pairs = np.array(pairs)
    
    units_killed = units_in[idx_sort[pairs[:, 0]]]
    units_kill = units_in[idx_sort[pairs[:, 1]]]
    
    fname_units_killed = os.path.join(save_dir, 'units_killed.npy')
    fname_units_kill = os.path.join(save_dir, 'units_kill.npy')
    np.save(fname_units_killed, units_killed)
    np.save(fname_units_kill, units_kill)
    
    idx_keep = units_in[~np.in1d(units_in, units_killed)]

    return idx_keep
