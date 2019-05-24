import numpy as np
import logging
import os
from tqdm import tqdm
import parmap

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
    CONFIG2.cluster.min_spikes = CONFIG.cluster.min_spikes

    CONFIG2.recordings.spike_size_ms = CONFIG.recordings.spike_size_ms
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
    
    CONFIG2.spike_size = CONFIG.spike_size

    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing

    return CONFIG2

def partition_input(save_dir, max_time,
                    fname_spike_index,
                    fname_templates_up=None,
                    fname_spike_train_up=None):

    print ("  partitioning input (TODO: Parallize for larger datasets)")
    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data
    spike_index = np.load(fname_spike_index)

    # consider only spikes times less than max_time
    idx_keep = np.where(spike_index[:,0] < max_time)[0]

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    spike_index_list = [[] for ii in range(n_units)]
    for j in idx_keep:
        tt, ii = spike_index[j]
        spike_index_list[ii].append(tt)

    # if there are upsampled data as input,
    # load and partition them also
    if fname_templates_up is not None:
        spike_index_up = np.load(fname_spike_train_up)
        templates_up = np.load(fname_templates_up)
        up_id_list = [[] for ii in range(n_units)]
        for j in idx_keep:
            ii = spike_index[j, 1]
            up_id = spike_index_up[j, 1]
            up_id_list[ii].append(up_id)

    fnames = []
    for unit in range(n_units):

        fname = os.path.join(save_dir, 'partition_{}.npz'.format(unit))
        fnames.append(fname)

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
        
    return np.arange(n_units), fnames

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
        data = np.load(os.path.join(result_dir, fname))
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
        parmap.starmap(load_align_waveforms_parallel,
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
                    fnames_out[unit], fnames_input_data[unit], unit,      
                    reader_raw, reader_resid, raw_data, CONFIG)
                pbar.update()

    return fnames_out


def load_align_waveforms_parallel(
    fname_out, fname_input_data, unit,
    reader_raw, reader_resid, raw_data, CONFIG):

    ref_template = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'ref_template.npy')))

    input_data = np.load(fname_input_data)
    if os.path.exists(fname_out):
        return

    # load spike times
    spike_times = input_data['spike_times']
    (spike_times,
     idx_sampled,
     idx_inbounds) = subsample_spikes(
        spike_times,
        CONFIG.cluster.max_n_spikes,
        CONFIG.spike_size,
        reader_raw.rec_len)

    if not raw_data:
        up_templates = input_data['up_templates']
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
            spike_times, CONFIG.spike_size, neighbor_chans)
        spike_times = np.delete(spike_times, skipped_idx)
    else:
        wf, skipped_idx = reader_resid.read_clean_waveforms(
            spike_times, upsampled_ids, up_templates,
            CONFIG.spike_size, neighbor_chans)
        spike_times = np.delete(spike_times, skipped_idx)
        upsampled_ids = np.delete(upsampled_ids, skipped_idx)

    # clip waveforms; seems necessary for neuropixel probe due to artifacts
    wf = wf.clip(min=-1000, max=1000)

    # align
    mc = np.where(neighbor_chans==channel)[0][0]
    shifts = align_get_shifts_with_ref(
        wf[:, :, mc],
        ref_template)
    wf = shift_chans(wf, shifts)
    if raw_data:
        np.savez(fname_out,
                 spike_times=spike_times,
                 wf=wf,
                 shifts=shifts,
                 channel=channel
                )
    else:
        np.savez(fname_out,
                 spike_times=spike_times,
                 wf=wf,
                 shifts=shifts,
                 upsampled_ids=upsampled_ids,
                 up_templates=up_templates,
                 channel=channel
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
    