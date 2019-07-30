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

def merge_units(fname_templates, fname_spike_train,
                merge_pairs):
    
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
    
    spike_train_new = np.zeros((0, 2), 'int32')
    templates_new = np.zeros((len(merge_array), n_times, n_channels),
                             'float32')

    refactory_period = np.min(np.diff(np.sort(spike_train[:,0])))

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
            
            # we want to concatenate all spike times
            # but don't want them to violate refactroy period
            # do to so, keep all spike times of the unit with
            # largest weight.
            # for the rest, iteratively update from higher weights
            # to lower weights and only add spikes that don't
            # violate refractory period.
            spt_temp = np.zeros(0, 'int32')
            idx_sort_units = np.argsort(weights[units])[::-1]
            for ii in idx_sort_units:

                unit = units[ii]
                spt_old = spike_train[spike_train[:, 1] == unit, 0]
                spt_old = spt_old + shifts[ii]

                if len(spt_temp) == 0:
                    spt_temp = np.hstack((spt_temp, spt_old))
                    spt_temp = np.sort(spt_temp)
                else:
                    n_spt_temp = len(spt_temp)
                    n_spt_old = len(spt_old)

                    # concatenate old spikes to spt_temp
                    # also keep the labels of which ones are from spt_old
                    spt_temp = np.hstack((spt_temp, spt_old))
                    spt_temp_label = np.hstack((np.repeat(0, n_spt_temp),
                                                np.repeat(1, n_spt_old)))

                    # sort by time
                    idx_sort_temp = np.argsort(spt_temp)
                    spt_temp = spt_temp[idx_sort_temp]
                    spt_temp_label = spt_temp_label[idx_sort_temp]

                    # idx that violates refractory period
                    idx_violations = np.where(
                        np.diff(spt_temp) < refactory_period)[0]
                    spt_temp_label_violations = spt_temp_label[idx_violations]

                    # kill if idx violates and its label is 1 (meaning it is from spt_old)
                    idx_kill = idx_violations[spt_temp_label_violations == 1]
                    # if its label is 0 then idx +1 must have label 1
                    idx_kill2 = idx_violations[spt_temp_label_violations == 0] + 1
                    if np.any(spt_temp_label[idx_kill2] == 0):
                        raise ValueError('bug in the code!')

                    idx_kill = np.hstack((idx_kill, idx_kill2))
                    spt_temp = np.delete(spt_temp, idx_kill)

        elif len(units) == 1:
            templates_new[new_id] = templates[units[0]]

            spt_temp = spike_train[spike_train[:, 1] == units[0], 0]

        spike_train_temp = np.vstack(
            (spt_temp, np.repeat(new_id, len(spt_temp)).astype('int32'))).T
        spike_train_new = np.vstack((spike_train_new, spike_train_temp))

    # sort them by spike times
    idx_sort = np.argsort(spike_train_new[:, 0])
    spike_train_new = spike_train_new[idx_sort]

    return spike_train_new, templates_new, merge_array
