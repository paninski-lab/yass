import os
import numpy as np
import networkx as nx

from yass.template import shift_chans, align_get_shifts_with_ref

def partition_input(save_dir,
                    fname_templates,
                    fname_spike_train,
                    fname_up=None):

    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)
    if fname_up is not None:
        up_data = np.load(fname_up)
        spike_train_up = up_data['spike_train_up']
        templates_up = up_data['templates_up']

    # re-organize spike times and templates id
    n_units = templates.shape[0]

    spike_train_list = [[] for ii in range(n_units)]
    for j in range(spike_train.shape[0]):
        tt, ii = spike_train[j]
        spike_train_list[ii].append(tt)

    if fname_up is not None:
        up_id_list = [[] for ii in range(n_units)]
        for j in range(spike_train.shape[0]):
            ii = spike_train[j, 1]
            up_id = spike_train_up[j, 1]
            up_id_list[ii].append(up_id)

    # partition upsampled templates also
    # and save them
    fnames = []
    for unit in range(n_units):

        fname = os.path.join(save_dir, 'partition_{}.npz'.format(unit))

        if fname_up is not None:
            unique_up_ids = np.unique(up_id_list[unit])
            up_templates = templates_up[unique_up_ids]
            new_id_map = {iid: ctr for ctr, iid in enumerate(unique_up_ids)}
            up_id2 = [new_id_map[iid] for iid in up_id_list[unit]]

            np.savez(fname,
                     template = templates[unit],
                     spike_times = spike_train_list[unit],
                     up_ids = up_id2,
                     up_templates = up_templates)
        else:
            np.savez(fname,
                     template = templates[unit],
                     spike_times = spike_train_list[unit])
       
        fnames.append(fname)
        
    return fnames

def merge_units(fname_templates, fname_spike_train, merge_pairs):
    
    # load templates
    templates = np.load(fname_templates)
    n_units, n_times, n_channels = templates.shape
    
    # load spike train
    spike_train = np.load(fname_spike_train)
    
    # make connected components
    merge_matrix = np.zeros((n_units, n_units),'int32')
    for pair in merge_pairs:
        merge_matrix[pair[0], pair[1]] = 1
        merge_matrix[pair[1], pair[0]] = 1
    G = nx.from_numpy_matrix(merge_matrix)
    merge_array=[]
    for cc in nx.connected_components(G):
        merge_array.append(list(cc))

    # get weights for merge
    weights = np.zeros(n_units)
    unique_ids, n_spikes = np.unique(spike_train[:,1], return_counts=True)
    weights[unique_ids] = n_spikes
    
    spike_train_new = np.copy(spike_train)
    templates_new = np.zeros((len(merge_array), n_times, n_channels))

    for new_id, units in enumerate(merge_array):
        
        # update templates
        if len(units) > 1:
            # align first
            aligned_temps, shifts = align_templates(templates[units])
            temp_ = np.average(aligned_temps, weights=weights[units], axis=2)
            templates_new[new_id] = temp_
            
        elif len(units) == 1:
            templates_new[new_id] = templates[units[0]]

        # update spike train id
        # also update time and upsampled templates based on the shift
        for ii, unit in enumerate(units):
            
            idx = spike_train[:,1] == unit

            # updated spike train id
            spike_train_new[idx,1] = new_id

            if len(units) > 1:
                # shift spike train time
                spt_old = spike_train[idx,0]
                shift_int = int(np.round(shifts[ii]))
                spike_train_new[idx,0] = spt_old - shift_int                

    return spike_train_new, templates_new, merge_array

def align_templates(templates):

    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    temps = templates[:, :, max_chan]

    best_shifts = align_get_shifts_with_ref(
                    temps, ref_template)

    aligned_templates = shift_chans(templates, best_shifts)
    
    return aligned_templates.transpose(1,2,0), best_shifts
