import numpy as np
import os
import parmap
import networkx as nx

from yass import read_config
from yass.geometry import n_steps_neigh_channels

def run_deduplication(batch_files_dir, output_directory):

    CONFIG = read_config()

    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
    w = 5

    batch_ids = list(np.arange(len(os.listdir(batch_files_dir))))
    if CONFIG.resources.multi_processing:
    #if False:
        parmap.map(run_deduplication_batch_simple,
                   batch_ids,
                   batch_files_dir,
                   output_directory,
                   neighbors,
                   w,
                   processes=CONFIG.resources.n_processors,
                   pm_pbar=True)
    else:
        for batch_id in batch_ids:
            run_deduplication_batch_simple(
                batch_id,
                batch_files_dir,
                output_directory,
                neighbors,
                w)

def run_deduplication_batch_simple(batch_id, batch_files_dir,
                output_directory, neighbors, w):
    # save name
    fname_save = os.path.join(
        output_directory,
        "dedup_"+str(batch_id).zfill(5)+'.npy')
    if os.path.exists(fname_save):
        return

    # load input data
    fname = os.path.join(
        batch_files_dir,
        "detect_"+str(batch_id).zfill(5)+'.npz')
    data = np.load(fname,allow_pickle=True)
    spike_index = data['spike_index']
    energy = data['energy']

    idx_survived = []
    for ctr in range(len(spike_index)):
        idx_survived.append(
            deduplicate_simple(spike_index[ctr],
                        energy[ctr],
                        neighbors, w))

    # save output
    np.save(fname_save, idx_survived)


def deduplicate_simple(spike_index, energy, neighbors, w=5):
    
    # sort by time
    idx_sort = np.argsort(spike_index[:,0])
    spike_index = spike_index[idx_sort]
    energy = energy[idx_sort]

    # number of data points
    n_data = spike_index.shape[0]

    # separate time and channel info
    TT = spike_index[:, 0]
    CC = spike_index[:, 1]

    # min and amx time
    T_max, T_min = np.max(TT), np.min(TT)
    # make time relative
    T_max -= T_min
    TT -= T_min
    T_min = 0

    # Index counting
    # index, index_counter[t], is the largest index with time <= t
    index_counter = np.zeros(T_max + w + 1, 'int32')
    t_now = T_min
    for j in range(n_data):
        if TT[j] > t_now:
            index_counter[t_now:TT[j]] = j - 1
            t_now = TT[j]
    index_counter[T_max:] = n_data - 1

    idx_survive = np.zeros(n_data, 'bool')
    for j in range(n_data):
        max_index = index_counter[TT[j] + w]
        min_index = index_counter[TT[j] - w - 1] + 1
        # max channels of temporally nearby spikes
        idx_temp_neighs = np.arange(min_index, max_index+1)
        cc_temporal_neighs = CC[idx_temp_neighs]

        # find spatially close spikes also
        idx_neighs = idx_temp_neighs[
            neighbors[CC[j]][cc_temporal_neighs]]

        if (len(idx_neighs)==0) or (energy[j] >= np.max(energy[idx_neighs]) - 1e-5):
            idx_survive[j] = True

    return idx_sort[idx_survive]


def run_deduplication_batch(batch_id, batch_files_dir,
                output_directory, neighbors):
    # save name
    fname_save = os.path.join(
        output_directory,
        "dedup_"+str(batch_id).zfill(5)+'.npy')
    if os.path.exists(fname_save):
        return

    # load input data
    fname = os.path.join(
        batch_files_dir,
        "detect_"+str(batch_id).zfill(5)+'.npz')
    data = np.load(fname,allow_pickle=True)
    spike_index = data['spike_index']
    energy = data['energy']

    idx_survived = []
    for ctr in range(len(spike_index)):
        idx_survived.append(
            deduplicate(spike_index[ctr],
                        energy[ctr],
                        neighbors))

    # save output
    np.save(fname_save, idx_survived)


def deduplicate(spike_index, energy, neighbors):
    
    # default window for deduplication in timesteps
    # Cat: TODO: read from CONFIG file
    w=5
    
    # sort by time
    idx_sort = np.argsort(spike_index[:,0])
    spike_index = spike_index[idx_sort]
    energy = energy[idx_sort]

    # number of data points
    n_data = spike_index.shape[0]

    # separate time and channel info
    TT = spike_index[:, 0]
    CC = spike_index[:, 1]

    # min and amx time
    T_max, T_min = np.max(TT), np.min(TT)
    # make time relative
    T_max -= T_min
    TT -= T_min
    T_min = 0

    # Index counting
    # index, index_counter[t], is the largest element with time <= t
    index_counter = np.zeros(T_max + w + 1, 'int32')
    t_now = T_min
    for j in range(n_data):
        if TT[j] > t_now:
            index_counter[t_now:TT[j]] = j - 1
            t_now = TT[j]
    index_counter[T_max:] = n_data - 1

    edges = []
    for j in range(n_data):
        max_index = index_counter[TT[j]+w]

        # max channels of temporally nearby spikes
        idx_temp_neighs = np.arange(j+1,max_index+1)
        cc_temporal_neighs = CC[idx_temp_neighs]

        # find spatially close spikes also
        idx_neighs = idx_temp_neighs[
            neighbors[CC[j]][cc_temporal_neighs]]

        edges += [[j, i] for i in idx_neighs]
    edges = tuple(edges)

    # Using cc, build connected components from the graph
    idx_survive = np.zeros(n_data, 'bool')
    G = nx.Graph()
    G.add_edges_from(edges)
    for cc in nx.connected_components(G):
        idx = list(cc)
        idx_survive[idx[np.argmax(energy[idx])]] = 1

    return idx_sort[idx_survive]
