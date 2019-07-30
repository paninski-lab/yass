import numpy as np
import logging
import os
from tqdm import tqdm
import parmap
import torch
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
#from numba import jit

from yass.util import absolute_path_to_asset
from yass.empty import empty
from yass.template import align_get_shifts_with_ref, shift_chans


def make_CONFIG2(CONFIG):
    ''' Makes a copy of several attributes of original config parameters
        to be sent into parmap function; original CONFIG can't be pickled;
    '''
    
    # make a copy of the original CONFIG object;
    # multiprocessing doesn't like the methods in original CONFIG        
    CONFIG2 = empty()
    CONFIG2.recordings=empty()
    CONFIG2.resources=empty()
    CONFIG2.deconvolution=empty()
    CONFIG2.data=empty()
    CONFIG2.cluster=empty()
    CONFIG2.cluster.prior=empty()
    CONFIG2.neuralnetwork = empty()

    CONFIG2.recordings.sampling_rate = CONFIG.recordings.sampling_rate
    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    
    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing
    CONFIG2.resources.n_sec_chunk = CONFIG.resources.n_sec_chunk

    CONFIG2.data.root_folder = CONFIG.data.root_folder
    CONFIG2.data.geometry = CONFIG.data.geometry
    CONFIG2.geom = CONFIG.geom
    
    CONFIG2.cluster.prior.a = CONFIG.cluster.prior.a
    CONFIG2.cluster.prior.beta = CONFIG.cluster.prior.beta
    CONFIG2.cluster.prior.lambda0 = CONFIG.cluster.prior.lambda0
    CONFIG2.cluster.prior.nu = CONFIG.cluster.prior.nu
    CONFIG2.cluster.prior.V = CONFIG.cluster.prior.V

    CONFIG2.neigh_channels = CONFIG.neigh_channels
    CONFIG2.cluster.max_n_spikes = CONFIG.cluster.max_n_spikes
    CONFIG2.cluster.knn_triage = CONFIG.cluster.knn_triage
    CONFIG2.cluster.min_fr = CONFIG.cluster.min_fr

    CONFIG2.spike_size = CONFIG.spike_size
    CONFIG2.spike_size_nn = CONFIG.spike_size_nn

    CONFIG2.neuralnetwork.apply_nn = CONFIG.neuralnetwork.apply_nn

    return CONFIG2

#@jit
def split_spikes(spike_index_list, spike_index, idx_keep):
    for j in idx_keep:
        tt, ii = spike_index[j]
        spike_index_list[ii].append(tt)
    return spike_index_list


# def split_spikes2(spike_index_list, spike_index, idx_keep, n_units):
    
    # for ii in range(n_units):
        # idx = np.where(spike_index[:,1]==ii)[0]
        # spike_index_list[ii] = spike_index[idx,0]
    # return spike_index_list


def split_parallel(units, spike_index):
    
    spike_index_list = []
    for ii in units:    
        idx = np.where(spike_index[:,1]==ii)[0]
        spike_index_list.append(spike_index[idx,0])
        
    return spike_index_list
   

def split_spikes_GPU(spike_index, idx_keep, n_units):
    
    spike_index_local = spike_index[idx_keep]
    spike_index_local = torch.from_numpy(spike_index_local).cuda()
    spike_index_list = []

    with tqdm(total=n_units) as pbar:
        for unit in range(n_units):
            idx = torch.where(spike_index_local[:,1]==unit, 
                              spike_index_local[:,1]*0+1, 
                              spike_index_local[:,1]*0)
            idx = torch.nonzero(idx)[:,0]
            spike_index_list.append(spike_index_local[idx,0].cpu().data.numpy())
            
            pbar.update()

    return spike_index_list


def split_spikes_parallel(spike_index_list, spike_index, idx_keep, 
                          n_units, CONFIG):
    
    np.save('/home/cat/spike_index.npy', spike_index[idx_keep])
    spike_index_local = spike_index[idx_keep]
    units = np.array_split(np.arange(n_units), CONFIG.resources.n_processors)
        
    print ("start parallel...# of chunks: ", len(units))
    res = parmap.map(split_parallel, units, spike_index_local, 
                      pm_processes=CONFIG.resources.n_processors)
    print ("end parallel...")
    print ("len res: ", len(res))

    spike_index_list = [[]]*n_units
    for i in range(len(res)):
        for ctr, unit in enumerate(units[i]):
            print ("saving unit: ", unit)
            spike_index_list[unit]==res[i][ctr]

    return spike_index_list


def partition_input(save_dir, max_time,
                    fname_spike_index,
                    CONFIG,
                    fname_templates_up=None,
                    fname_spike_train_up=None):

    print ("  partitioning input data (todo: skip if already computed)")
    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data
    spike_index = np.load(fname_spike_index, allow_pickle=True)
    # consider only spikes times less than max_time
    idx_keep = np.where(spike_index[:,0] < max_time)[0]

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    #spike_index_list = [[] for ii in range(n_units)]

    # Cat: TODO: have GPU-use flag in CONFIG file
    if CONFIG.resources.n_sec_chunk_gpu>0:
        spike_index_list = split_spikes_GPU(spike_index, idx_keep, n_units)
    else:
        spike_index_list = split_spikes_parallel(spike_index_list, spike_index, 
                                             idx_keep, n_units, CONFIG)

    # if there are upsampled data as input,
    # load and partition them also
    if fname_templates_up is not None:
        spike_index_up = np.load(fname_spike_train_up, allow_pickle=True)
        templates_up = np.load(fname_templates_up, allow_pickle=True)
        up_id_list = [[] for ii in range(n_units)]
        for j in idx_keep:
            ii = spike_index[j, 1]
            up_id = spike_index_up[j, 1]
            up_id_list[ii].append(up_id)

    fnames = []
    units = []
    for unit in range(n_units):

        # it needs at least 5 spikes to cluster
        if len(spike_index_list[unit]) < 5:
            continue

        fname = os.path.join(save_dir, 'partition_{}.npz'.format(unit))
        fnames.append(fname)
        units.append(unit)

        if os.path.exists(fname):
            continue

        if fname_templates_up is not None:
            unique_up_ids = np.unique(up_id_list[unit])
            if unique_up_ids.shape[0]==0:
                np.savez(fname,
                     spike_times = [],
                     up_ids = [],
                     up_templates = [])
                fnames.append(fname)

            else:
                up_templates = templates_up[unique_up_ids]
                new_id_map = {iid: ctr for ctr, iid in enumerate(unique_up_ids)}
                up_id2 = [new_id_map[iid] for iid in up_id_list[unit]]

                np.savez(fname,
                         spike_times = spike_index_list[unit],
                         up_ids = up_id2,
                         up_templates = up_templates)
        else:
            np.savez(fname,
                     spike_times = spike_index_list[unit])
        
    return units, fnames

def gather_clustering_result(result_dir, out_dir):

    '''load clustering results
    '''

    logger = logging.getLogger(__name__)
    
    logger.info("gathering clustering results")

    # make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # convert clusters to templates
    templates = []
    spike_indexes = []
    
    filenames = sorted(os.listdir(result_dir))
    for fname in filenames:
        data = np.load(os.path.join(result_dir, fname), allow_pickle=True)
        temp_temp = data['templates']
        if (temp_temp.shape[0]) != 0:
            templates.append(temp_temp)
            temp = data['spiketime']
            for s in range(len(temp)):
                spike_indexes.append(temp[s])

    spike_indexes = np.array(spike_indexes)    
    templates = np.vstack(templates)

    logger.info("units loaded: {}".format(len(spike_indexes)))

    fname_templates = os.path.join(out_dir, 'templates.npy')
    np.save(fname_templates, templates)

    # rearange spike indees from id 0..N
    logger.info("reindexing spikes")
    spike_train = np.zeros((0,2), 'int32')
    for k in range(spike_indexes.shape[0]):    
        temp = np.zeros((spike_indexes[k].shape[0],2), 'int32')
        temp[:,0] = spike_indexes[k]
        temp[:,1] = k
        spike_train = np.vstack((spike_train, temp))

    fname_spike_train = os.path.join(out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)

    return fname_templates, fname_spike_train


def load_align_waveforms(save_dir, fnames_input_data,
               reader_raw, reader_resid, raw_data, CONFIG):
    '''load and align waveforms first to run nn denoise
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_units = len(fnames_input_data)
    units = [unit for unit in range(n_units)]
    fnames_out = [os.path.join(save_dir, 'partition_{}.npz'.format(unit))
                  for unit in range(n_units)]

    if CONFIG.resources.multi_processing:
        parmap.map(load_align_waveforms_parallel,
                       list(zip(fnames_out, fnames_input_data, units)),
                       reader_raw,
                       reader_resid,
                       raw_data,
                       CONFIG,
                       processes=CONFIG.resources.n_processors,
                       pm_pbar=True)

    else:
        with tqdm(total=n_units) as pbar:
            for unit in units:
                load_align_waveforms_parallel(
                    [fnames_out[unit], fnames_input_data[unit], unit],      
                    reader_raw, reader_resid, raw_data, CONFIG)
                pbar.update()

    return fnames_out


#def load_align_waveforms_parallel(
#    fname_out, fname_input_data, unit,
#    reader_raw, reader_resid, raw_data, CONFIG):
def load_align_waveforms_parallel(data_in, 
                                  reader_raw, 
                                  reader_resid, 
                                  raw_data, 
                                  CONFIG):

    fname_out = data_in[0]
    fname_input_data = data_in[1]
    unit = data_in[2]
        
    input_data = np.load(fname_input_data, allow_pickle=True)
    if os.path.exists(fname_out):
        return

    # load spike times
    spike_times = input_data['spike_times']

    # calculate min_spikes
    n_spikes_in = len(spike_times)
    rec_len = np.max(spike_times) - np.min(spike_times)
    min_fr = CONFIG.cluster.min_fr
    n_sec_data = rec_len/float(CONFIG.recordings.sampling_rate)
    min_spikes = int(n_sec_data*min_fr)
    # if it will be subsampled, min spikes should decrease also
    min_spikes = int(min_spikes*np.min((
        1, CONFIG.cluster.max_n_spikes/
        float(n_spikes_in))))
    # min_spikes needs to be at least 5 to cluster
    min_spikes = max(min_spikes, 5)

    # subsample spikes
    (spike_times,
     idx_sampled,
     idx_inbounds) = subsample_spikes(
        spike_times,
        CONFIG.cluster.max_n_spikes,
        CONFIG.spike_size,
        reader_raw.rec_len)

    # first read waveforms in a bigger size
    # then align and cut down edges
    if CONFIG.neuralnetwork.apply_nn:
        spike_size_out = CONFIG.spike_size_nn
    else:
        spike_size_out = CONFIG.spike_size

    spike_size_read = (spike_size_out-1)*2 + 1

    if not raw_data:
        up_templates = input_data['up_templates']
        size_diff = (up_templates.shape[1] - spike_size_out)//2
        if size_diff > 0:
            up_templates = up_templates[:, size_diff:-size_diff]
        upsampled_ids = input_data['up_ids']
        upsampled_ids = upsampled_ids[
            idx_sampled][idx_inbounds].astype('int32')    

    if raw_data:
        channel = unit
    else:
        channel = np.argmax(up_templates.ptp(1).max(0))

    # load waveforms
    neighbor_chans = np.where(CONFIG.neigh_channels[channel])[0]
    if raw_data:
        wf, skipped_idx = reader_raw.read_waveforms(
            spike_times, spike_size_read, neighbor_chans)
        spike_times = np.delete(spike_times, skipped_idx)
    else:
        wf, skipped_idx = reader_resid.read_clean_waveforms(
            spike_times, upsampled_ids, up_templates,
            spike_size_read, neighbor_chans)
        spike_times = np.delete(spike_times, skipped_idx)
        upsampled_ids = np.delete(upsampled_ids, skipped_idx)

    # clip waveforms; seems necessary for neuropixel probe due to artifacts
    wf = wf.clip(min=-1000, max=1000)

    # align
    mc = np.where(neighbor_chans==channel)[0][0]
    shifts = align_get_shifts_with_ref(
        wf[:, :, mc])
    wf = shift_chans(wf, shifts)
    wf = wf[:, (spike_size_out//2):-(spike_size_out//2)]

    if raw_data:
        np.savez(fname_out,
                 spike_times=spike_times,
                 wf=wf,
                 shifts=shifts,
                 channel=channel,
                 min_spikes=min_spikes
                )
    else:
        np.savez(fname_out,
                 spike_times=spike_times,
                 wf=wf,
                 shifts=shifts,
                 upsampled_ids=upsampled_ids,
                 up_templates=up_templates,
                 channel=channel,
                 min_spikes=min_spikes
                )


def subsample_spikes(spike_times, max_spikes, spike_size, rec_len):
        
    # limit number of spikes
    if len(spike_times)>max_spikes:
        idx_sampled = np.random.choice(
            a=np.arange(len(spike_times)),
            size=max_spikes,
            replace=False)
        spike_times = spike_times[idx_sampled]
    else:
        idx_sampled = np.arange(len(spike_times))

    # limit indexes away from edge of recording
    idx_inbounds = np.where(np.logical_and(
                    spike_times>=spike_size//2,
                    spike_times<(rec_len - spike_size)))[0]
    spike_times = spike_times[
        idx_inbounds].astype('int32')

    return spike_times, idx_sampled, idx_inbounds

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def nn_denoise_wf(fnames_input_data, denoiser, devices):

    fnames_input_split = split(fnames_input_data, len(devices))
    processes = []
    for ii, device in enumerate(devices):
        p = mp.Process(target=nn_denoise_wf_parallel,
                       args=(fnames_input_split[ii],
                             denoiser, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def nn_denoise_wf_parallel(fnames, denoiser, device):

    denoiser = denoiser.to(device)

    for fname in fnames:
        temp = np.load(fname, allow_pickle=True)

        if 'denoised_wf' in temp.files:
            continue

        wf = temp['wf']
        n_data, n_times, n_chans = wf.shape
        if wf.shape[0]>0:
            wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
            wf_torch = torch.FloatTensor(wf_reshaped).to(device)
            denoised_wf = denoiser(wf_torch)[0]
            denoised_wf = denoised_wf.reshape(
                n_data, n_chans, n_times)
            denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

            # reshape it
            #window = np.arange(15, 40)
            #denoised_wf = denoised_wf[:, window]
            denoised_wf = denoised_wf.reshape(n_data, -1)
        else:
            denoised_wf = np.zeros((wf.shape[0],
                                    wf.shape[1]*wf.shape[2]),'float32')

        temp = dict(temp)
        temp['denoised_wf'] = denoised_wf
        np.savez(fname, **temp)

def denoise_wf(fnames_input_data):

    with tqdm(total=len(fnames_input_data)) as pbar:
        for fname in fnames_input_data:
            temp = np.load(fname, allow_pickle=True)

            if 'denoised_wf' in temp.files:
                continue

            wf = temp['wf']
            n_data, n_times, n_chans = wf.shape
            wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)

            # restrict to high energy locations
            energy = np.max(np.median(np.square(wf), 0), 1)
            idx = np.where(energy > 0.5)[0]
            if len(idx) == 0:
                idx = [energy.argmax()]
            wf_reshaped = wf_reshaped[:, idx]

            if len(idx) > 5:
                # denoise using pca
                pca = PCA(n_components=5)
                score = pca.fit_transform(wf_reshaped)
                denoised_wf = pca.inverse_transform(score)
            else:
                denoised_wf = wf_reshaped

            # reshape it
            #window = np.arange(15, 40)
            #denoised_wf = denoised_wf[:, window]
            denoised_wf = denoised_wf.reshape(
                n_data, n_chans, len(idx)).transpose(0, 2, 1)
            denoised_wf = denoised_wf.reshape(n_data, -1)

            temp = dict(temp)
            temp['denoised_wf'] = denoised_wf
            np.savez(fname, **temp)
            pbar.update()
