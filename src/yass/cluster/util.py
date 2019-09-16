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

def make_spike_index_from_spike_train(fname_spike_train, fname_templates, savedir):
    
    spike_train = np.load(fname_spike_train)
    templates = np.load(fname_templates)
    
    mcs = templates.ptp(1).argmax(1)
    spike_index = np.copy(spike_train)
    spike_index[:,1] = mcs[spike_train[:,1]]
    labels = np.copy(spike_train[:,1])
    
    fname_spike_index = os.path.join(savedir, 'spike_index.npy')
    fname_labels = os.path.join(savedir, 'labels.npy')

    np.save(fname_spike_index, spike_index)
    np.save(fname_labels, labels)

    return fname_spike_index, fname_labels


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
   

def split_spikes_GPU(spike_index, n_units):

    # Cat: TODO: have GPU-use flag in CONFIG file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spike_index = torch.from_numpy(spike_index).to(device)
    spike_index_list = []

    with tqdm(total=n_units) as pbar:
        for unit in range(n_units):
            print ("unit: ", unit)
            idx = torch.where(spike_index[:,1]==unit,
                              spike_index[:,1]*0+1,
                              spike_index[:,1]*0)
            idx = torch.nonzero(idx)[:,0]
            spike_index_list.append(spike_index[idx,0].cpu().data.numpy())
            
            pbar.update()

    return spike_index_list


def split_spikes_parallel(spike_index_list, spike_index, idx_keep, 
                          n_units, CONFIG):
    
    #np.save('/home/cat/spike_index.npy', spike_index[idx_keep])
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


def partition_input(save_dir,
                    fname_spike_index,
                    CONFIG,
                    fname_templates_up=None,
                    fname_spike_train_up=None):

    print ("  partitioning input data (todo: skip if already computed)")
    # make directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data
    spike_index = np.load(fname_spike_index, allow_pickle=True)

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    #spike_index_list = [[] for ii in range(n_units)]

    spike_index_list = split_parallel(np.arange(n_units), spike_index)
    #spike_index_list = split_spikes_GPU(spike_index[idx_keep], n_units)
    #spike_index_list = split_spikes_parallel(spike_index_list, spike_index,
    #                                     idx_keep, n_units, CONFIG)

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


def load_align_waveforms(save_dir, raw_data, fname_splits,
                         fname_spike_index, fname_labels_input,
                         fname_templates_input, reader_raw, reader_resid,
                         CONFIG):
    '''load and align waveforms first to run nn denoise
    '''

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fname_splits is None:
        split_labels = np.load(fname_spike_index)[:, 1]
    else:
        split_labels = np.load(fname_splits)
    all_split_labels = np.unique(split_labels)

    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        split_labels_in = []
        for j in range(n_processors):
            split_labels_in.append(
                all_split_labels[slice(j, len(all_split_labels), n_processors)])
            
        fnames_out_parallel = parmap.map(
            load_align_waveforms_parallel,
            split_labels_in,
            save_dir,
            raw_data,
            fname_splits,
            fname_spike_index,
            fname_labels_input,
            fname_templates_input,
            reader_raw,
            reader_resid,
            CONFIG,
            processes=n_processors)
        fnames_out = []
        [fnames_out.extend(el) for el in fnames_out_parallel]
        
        units_out = []
        [units_out.extend(el) for el in split_labels_in]

    else:
        units_out = all_split_labels
        fnames_out = load_align_waveforms_parallel(
            all_split_labels,
            save_dir,
            raw_data,
            fname_splits,
            fname_spike_index,
            fname_labels_input,
            fname_templates_input,
            reader_raw,
            reader_resid,
            CONFIG)

    return units_out, fnames_out


def load_align_waveforms_parallel(labels_in,
                                  save_dir,
                                  raw_data,
                                  fname_splits,
                                  fname_spike_index,
                                  fname_labels_input,
                                  fname_templates,
                                  reader_raw,
                                  reader_resid,
                                  CONFIG):
    
    spike_index = np.load(fname_spike_index)
    if fname_splits is None:
        split_labels = spike_index[:, 1]
    else:
        split_labels = np.load(fname_splits)
    
    # minimum number of spikes per cluster
    rec_len_sec = np.ptp(spike_index[:,0])
    min_spikes = int(rec_len_sec*CONFIG.cluster.min_fr/CONFIG.recordings.sampling_rate)

    # first read waveforms in a bigger size
    # then align and cut down edges
    if CONFIG.neuralnetwork.apply_nn:
        spike_size_out = CONFIG.spike_size_nn
    else:
        spike_size_out = CONFIG.spike_size
    spike_size_buffer = 3
    spike_size_read = spike_size_out + 2*spike_size_buffer
    
    # load data for making clean wfs
    if not raw_data:
        labels_input = np.load(fname_labels_input)
        templates = np.load(fname_templates)

        n_times_templates = templates.shape[1]
        if n_times_templates > spike_size_out:
            n_times_diff = (n_times_templates - spike_size_out)//2
            templates = templates[:, n_times_diff:-n_times_diff]

    # get waveforms and align
    fname_outs = []
    for id_ in labels_in:
        fname_out = os.path.join(save_dir, 'partition_{}.npz'.format(id_))
        fname_outs.append(fname_out)

        if os.path.exists(fname_out):
            continue

        idx_ = np.where(split_labels == id_)[0]
        
        # spike times
        spike_times = spike_index[idx_, 0]

        # if it will be subsampled, min spikes should decrease also
        subsample_ratio = np.min(
            (1, CONFIG.cluster.max_n_spikes/float(len(spike_times))))
        min_spikes = int(min_spikes*subsample_ratio)
        # min_spikes needs to be at least 20 to cluster
        min_spikes = np.max((min_spikes, 20))

        # subsample spikes
        (spike_times,
         idx_sampled) = subsample_spikes(
            spike_times,
            CONFIG.cluster.max_n_spikes)
        
        # max channel and neighbor channels
        channel = int(spike_index[idx_, 1][0])
        neighbor_chans = np.where(CONFIG.neigh_channels[channel])[0]

        if raw_data:
            wf, skipped_idx = reader_raw.read_waveforms(
                spike_times, spike_size_read, neighbor_chans)
            spike_times = np.delete(spike_times, skipped_idx)

        else:

            # get upsampled ids
            template_ids_ = labels_input[idx_][idx_sampled]
            unique_template_ids = np.unique(template_ids_)

            # ids relabelled
            templates_in = templates[unique_template_ids]
            template_ids_in = np.zeros_like(template_ids_)
            for ii, k in enumerate(unique_template_ids):
                template_ids_in[template_ids_==k] = ii
 
            # get clean waveforms
            wf, skipped_idx = reader_resid.read_clean_waveforms(
                spike_times, template_ids_in, templates_in,
                spike_size_read, neighbor_chans)
            spike_times = np.delete(spike_times, skipped_idx)
            template_ids_in = np.delete(template_ids_in, skipped_idx)

        # align
        if wf.shape[0] > 0:
            mc = np.where(neighbor_chans==channel)[0][0]
            shifts = align_get_shifts_with_ref(
                wf[:, :, mc], nshifts=3)
            wf = shift_chans(wf, shifts)
            wf = wf[:, spike_size_buffer:-spike_size_buffer]
        else:
            shifts = None

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
                     upsampled_ids=template_ids_in,
                     up_templates=templates_in,
                     channel=channel,
                     min_spikes=min_spikes
                    )

    return fname_outs


def subsample_spikes(spike_times, max_spikes):
        
    # limit number of spikes
    if len(spike_times)>max_spikes:
        idx_sampled = np.random.choice(
            a=np.arange(len(spike_times)),
            size=max_spikes,
            replace=False)
        spike_times = spike_times[idx_sampled]
    else:
        idx_sampled = np.arange(len(spike_times))

    return spike_times, idx_sampled

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
