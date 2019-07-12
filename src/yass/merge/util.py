import os
import numpy as np
import networkx as nx

from yass.template import shift_chans, align_get_shifts_with_ref

def partition_input(save_dir,
                    fname_templates,
                    fname_spike_train,
                    fname_templates_up=None,
                    fname_spike_train_up=None):

    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data - only if incomplete files    
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)

    n_units = templates.shape[0]
            
    # partition upsampled templates also
    # and save them
    fnames = []
    read_flag = True
    for unit in range(n_units):
        fname = os.path.join(save_dir, 'partition_{}.npz'.format(unit))
        fnames.append(fname)

        # if previously computed
        if not os.path.exists(fname):
            # if at least 1 file is missing, then:
            #   Load upsampled dataset and compute partition
            if read_flag == True:
                # set flag to false so won't read this data again
                read_flag=False

                spike_train_list = [[] for ii in range(n_units)]
                for j in range(spike_train.shape[0]):
                    tt, ii = spike_train[j]
                    spike_train_list[ii].append(tt)

                if fname_templates_up is not None:
                    spike_train_up = np.load(fname_spike_train_up)
                    templates_up = np.load(fname_templates_up)

                    up_id_list = [[] for ii in range(n_units)]
                    for j in range(spike_train.shape[0]):
                        ii = spike_train[j, 1]
                        up_id = spike_train_up[j, 1]
                        up_id_list[ii].append(up_id)

            # proceed to compute partitioning
            if fname_templates_up is not None:
                unique_up_ids = np.unique(up_id_list[unit])
                if unique_up_ids.shape[0]==0:
                    np.savez(fname,
                         spike_times = [],
                         up_ids = [],
                         up_templates = [])
                else:
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
    templates_new = np.zeros((len(merge_array), n_times, n_channels),
                             'float32')

    for new_id, units in enumerate(merge_array):
        if len(units) > 1:

            # update templates
            temp = templates[units]
            # save only a unit with the highest weight
            id_keep = weights[units].argmax()
            templates_new[new_id] = templates[units[id_keep]]

            # update spike train
            # determine shifts
            mc = temp[id_keep].ptp(0).argmax()
            shifts = temp[:, :, mc].argmin(1) - temp[id_keep][:, mc].argmin()
            # update id and shift
            for ii, unit in enumerate(units):

                idx = spike_train[:, 1] == unit

                # updated id
                spike_train_new[idx, 1] = new_id
                # shift time
                spt_old = spike_train[idx, 0]
                spike_train_new[idx, 0] = spt_old + shifts[ii]

        elif len(units) == 1:
            templates_new[new_id] = templates[units[0]]

            idx = spike_train[:, 1] == unit[0]
            spike_train[idx, 1] = new_id

    return spike_train_new, templates_new, merge_array
