import os
import logging
import numpy as np
import parmap

from yass.deconvolute.util import (svd_shifted_templates,
                                   small_shift_templates,
                                   make_spt_list_parallel, clean_up,
                                   calculate_temp_temp_parallel)
from yass.deconvolute.deconvolve import (deconvolve_new_allcores_updated,
                                         deconvolve_match_pursuit)
                                         
from yass.deconvolute.match_pursuit import (MatchPursuit2)
from yass.deconvolute.match_pursuit_analyze import (MatchPursuitAnalyze)
from yass import read_config

import multiprocessing as mp

def run2(spike_index_all,
        templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'.format(templates.shape,
                                                 spike_index_all.shape))

    # ******************************************************************
    # ****************** DECONVOLUTION START ***************************
    # ******************************************************************

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates, (2, 1, 0))

    sampling_rate = CONFIG.recordings.sampling_rate
    n_shifts = CONFIG.deconvolution.upsample_factor
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd
    n_features = CONFIG.deconvolution.n_features
    max_spikes = CONFIG.deconvolution.max_spikes
    n_processors = CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk

    TMP_FOLDER = CONFIG.data.root_folder
    
    # Cat: TODO: read from CONFIG file
    buffer_size = 200

    # Grab length of .dat file to compute chunk indexes below
    standardized_filename = os.path.join(CONFIG.data.root_folder, 
                                    output_directory, recordings_filename)
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # Generate indexes in chunks (units of seconds); usually 10-60sec
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    # Make the 4 parameter list to be sent to deconvolution algorithm
    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list)) #[:2]
    proc_indexes = np.arange(len(idx_list))
    
    print("# of chunks for deconvolution: ", len(idx_list), " verbose mode: ",
          CONFIG.deconvolution.verbose)
            
    # need to transpose axes for match_pursuit 
    templates = np.swapaxes(templates,0,1)
    templates = np.swapaxes(templates,1,2)
    
    # make deconv directory
    deconv_dir = os.path.join(CONFIG.data.root_folder,
                              'tmp/deconv')
    if not os.path.isdir(deconv_dir):
        os.makedirs(deconv_dir)

    # ****************************************************************
    # ****************** LOOP OVER CHUNKS OF DATA ********************
    # ****************************************************************

    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    chunk_length = 3
    chunk_ctr = 0
    # svd approximate reconstruction rank
    approx_rank = 3 
    for c in range(0, len(idx_list), chunk_length):
        
        # select segments to be processed in current chunk
        idx_list_local = idx_list[c:c+chunk_length]
        print ("Local idx list: \n", idx_list_local)
        
        # make deconv chunk directory
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))
        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
        
        # Computing SVD for each template.
        print ("computing svd for chunk: ", chunk_ctr)
        if (os.path.exists(deconv_chunk_dir+"/svd.npz"))==False:
            temporal, singular, spatial = np.linalg.svd(
                np.transpose(np.flipud(templates), (2, 0, 1)))
       
            np.savez(deconv_chunk_dir+"/svd.npz", temporal=temporal, 
                    singular=singular, spatial=spatial)
        else:
            temp_data = np.load(deconv_chunk_dir+"/svd.npz")
            temporal = temp_data['temporal']        
            singular = temp_data['singular']        
            spatial = temp_data['spatial']        
            
        # compute vis chans for each template:
        vis_chan = visible_chans(templates)
            
        # Compute pairwise convolution of filters
        print ("pairwise conv filter chunk: ", chunk_ctr)
        n_time, n_chan, n_unit = templates.shape
        pairwise_conv = pairwise_filter_conv_local(deconv_chunk_dir, 
                                    n_time, n_unit, temporal, 
                                    singular, spatial, approx_rank,
                                    vis_chan, templates)
        
        # initialize match pursuit 
        print ("Initialize match pursuit object chunk: ", chunk_ctr)
        #mp_object = MatchPursuit2(templates, deconv_dir, vis_chan, 
        #                          threshold=2, obj_energy=False)
        mp_object = MatchPursuit3(templates, deconv_dir, vis_chan, 
                                  threshold=2, obj_energy=False)
                   
                #(self, data, temps, threshold=10, conv_approx_rank=3,
                 #implicit_subtraction=True, upsample=1, obj_energy=False,
                 #vis_su=2., keep_iterations=False, sparse_subtraction=False,
                 #broadcast_subtraction=False):
       
        # make arg list first
        max_iter = 20
        args_in = []
        for k in range(len(idx_list_local)):
            args_in.append([[idx_list_local[k], k],
                    chunk_ctr,
                    max_iter, 
                    buffer_size, 
                    standardized_filename,
                    n_channels])
        
        # assign pre-computed data to object attributes;
        # Cat: TODO: are these copied in memory by yass or do they
        #            simply become shared variables; 
        #            If they are copied try using code below for this:
        #spatial_mp = mp.Array('float32', spatial)
        #temporal_mp = mp.Array('float32', temporal)
        #singular_mp = mp.Array('float32', singular)
        #pairwise_conv_mp = mp.Array('float32', pairwise_conv)
        mp_object.spatial = spatial
        mp_object.temporal = temporal
        mp_object.singular = singular
        mp_object.pairwise_conv = pairwise_conv
        
                
        #if CONFIG.resources.multi_processing:
        if True:
            p = mp.Pool(CONFIG.resources.n_processors)
            res = p.map_async(mp_object.run, args_in).get(988895)
            p.close()
        else:
            spike_train = []
            for k in range(len(idx_list_local)):
                print ("Processing segment: "+str(k)+"/"+str(len(idx_list_local)))
                spike_train_temp = mp_object.run(
                    [idx_list_local[k], k],
                    chunk_ctr,
                    max_iter, 
                    buffer_size, 
                    standardized_filename,
                    n_channels,
                    spatial,
                    temporal,
                    singular,
                    pairwise_conv)            
        
        # run clean_spike functino
        n_features = 0
        temp_obj = MatchPursuitAnalyze(data, self.dec_spike_train, 
                                    self.temps, self.n_channels, n_features)
        cleaned_wfs = []
        units = np.arange(self.n_unit)
        for unit in units:
            temp = temp_obj.get_unit_spikes(unit)
            print (temp.shape)
            cleaned_wfs.append(temp)

        np.savez(self.chunk_dir+"/cleaned_wfs.npz", 
                            cleaned_wfs = cleaned_wfs,
                            dec_spike_train = self.dec_spike_train,
                            dist_metric = self.dist_metric)
        
        # re-run MFM on chunks of data
        
        
        
        
        # update templates for next iteration
        
        
        
        chunk_ctr+=1
                    
    final_spike_train = np.zeros((0,2),'int32')
    for k in range(len(spike_train)):
        final_spike_train= np.vstack((final_spike_train,spike_train[k]))

    # Cat: TODO: reorder spike train by time
    print ("deconv spike train: ", final_spike_train.shape)


    logger.info('spike_train.shape: {}'.format(final_spike_train.shape))

    return final_spike_train #, np.transpose(templates)


def visible_chans(temps):
    a = temps.ptp(0) #np.max(temps, axis=0) - np.min(temps, 0)
    vis_chan = a > 1

    return vis_chan
        
def pairwise_filter_conv_local(deconv_chunk_dir, n_time, n_unit, temporal, 
                         singular, spatial, approx_rank, vis_chan, temps):
    
    #print (deconv_chunk_dir+"/parwise_conv.npy")
    if os.path.exists(deconv_chunk_dir+"/pairwise_conv.npy")==False:
        print ("IN LOOP")
        conv_res_len = n_time * 2 - 1
        pairwise_conv = np.zeros([n_unit, n_unit, conv_res_len])
        for unit1 in range(n_unit):
            u, s, vh = temporal[unit1], singular[unit1], spatial[unit1]
            vis_chan_idx = vis_chan[:, unit1]
            for unit2 in range(n_unit):
                for i in range(approx_rank):
                    pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                        s[i] * u[:, i].flatten(), 'full')

        np.save(deconv_chunk_dir+"/pairwise_conv.npy", pairwise_conv)
    else:
        pairwise_conv = np.load(deconv_chunk_dir+"/pairwise_conv.npy")
        
    return pairwise_conv
    
        
def run2_match_old(spike_index_all,
        templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'.format(templates.shape,
                                                 spike_index_all.shape))

    # ******************************************************************
    # ****************** DECONVOLUTION START ***************************
    # ******************************************************************

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates, (2, 1, 0))

    sampling_rate = CONFIG.recordings.sampling_rate
    n_shifts = CONFIG.deconvolution.upsample_factor
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd
    n_features = CONFIG.deconvolution.n_features
    max_spikes = CONFIG.deconvolution.max_spikes
    n_processors = CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk

    TMP_FOLDER = CONFIG.data.root_folder
    
    
    # compute padding length for deconvolution
    buffer_size = 2 * n_temporal_big + n_explore

    # Grab length of .dat file to compute chunk indexes below
    filename_bin = os.path.join(CONFIG.data.root_folder, output_directory,
                                recordings_filename)
    fp = np.memmap(filename_bin, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # Generate indexes in chunks (units of seconds)
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    # Make the 4 parameter list to be sent to deconvolution algorithm
    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list)) #[:2]
    proc_indexes = np.arange(len(idx_list))
    
    print("# of chunks for deconvolution: ", len(idx_list), " verbose mode: ",
          CONFIG.deconvolution.verbose)
            
    # need to transpose axes for match_pursuit 
    templates = np.swapaxes(templates,0,1)
    templates = np.swapaxes(templates,1,2)
    
    # Deconv using parmap module
    if CONFIG.resources.multi_processing:
        spike_train = parmap.map(
            deconvolve_match_pursuit,
            list(zip(idx_list, proc_indexes)),
            templates,
            output_directory,
            TMP_FOLDER,
            filename_bin,
            buffer_size,
            n_channels,
            threshold_d,
            verbose=CONFIG.deconvolution.verbose,
            processes=n_processors,
            pm_pbar=True)
    else:
        spike_train = []
        for k in range(len(idx_list)):
            print ("Processing chunk: "+str(k)+"/"+str(len(idx_list)))
            spike_train.append(
                deconvolve_match_pursuit(
                [idx_list[k], k],
                templates,
                output_directory,
                TMP_FOLDER,
                filename_bin,
                buffer_size,
                n_channels,
                threshold_d,
                verbose=CONFIG.deconvolution.verbose))      

    # run match pursuit 
    print (len(spike_train))
    print (spike_train[0].shape)
    
    final_spike_train = np.zeros((0,2),'int32')
    for k in range(len(spike_train)):
        final_spike_train= np.vstack((final_spike_train,spike_train[k]))

    # Gather spikes
    #spike_train = np.vstack(spike_train)

    # sort spikes by time and remove templates with spikes < max_spikes
    #spike_train, templates = clean_up(spike_train, templates, max_spikes)

    # Optional save spike_train as txt file for human readability
    # filename_spike_train = os.path.join(CONFIG.data.root_folder,
    #                            output_directory, 'spike_train.txt')
    # np.savetxt(filename_spike_train ,spike_train, fmt='%d',)

    logger.info('spike_train.shape: {}'.format(final_spike_train.shape))

    return final_spike_train #, np.transpose(templates)

# **********************************************************************
# *************************** OLD DECONVOLUTION ************************
# **********************************************************************

def run(spike_index_all,
        templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'.format(templates.shape,
                                                 spike_index_all.shape))

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates, (2, 1, 0))

    sampling_rate = CONFIG.recordings.sampling_rate
    n_shifts = CONFIG.deconvolution.upsample_factor
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd
    n_features = CONFIG.deconvolution.n_features
    max_spikes = CONFIG.deconvolution.max_spikes
    n_processors = CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk

    TMP_FOLDER = CONFIG.data.root_folder

    # get spt_list
    print('making spt list')
    path_to_spt_list = os.path.join(TMP_FOLDER, output_directory,
                                    'spt_list.npy')
    if os.path.exists(path_to_spt_list):
        spt_list = np.load(path_to_spt_list)
    else:
        spike_index_chunks = np.array_split(spike_index_all, n_processors)
        spt_list = parmap.map(
            make_spt_list_parallel,
            spike_index_chunks,
            n_channels,
            processes=n_processors,
            pm_pbar=True)

        # Recombine the spikes across channels from different procs
        spt_list_total = []
        for k in range(n_channels):
            spt_list_total.append([])
            for p in range(n_processors):
                spt_list_total[k].append(spt_list[p][k])

            spt_list_total[k] = np.hstack(spt_list_total[k])
        np.save(path_to_spt_list, spt_list_total)

    # upsample template
    print('computing shifted templates')
    path_to_shifted_templates = os.path.join(TMP_FOLDER, output_directory,
                                             'shifted_templates.npy')
    if os.path.exists(path_to_shifted_templates):
        shifted_templates = np.load(path_to_shifted_templates)
    else:
        shifted_templates = small_shift_templates(templates, n_shifts)
        np.save(path_to_shifted_templates, shifted_templates)

    # svd templates
    print('computing svd templates')
    path_to_temporal_features = os.path.join(TMP_FOLDER, output_directory,
                                             'temporal_features.npy')
    path_to_spatial_features = os.path.join(TMP_FOLDER, output_directory,
                                            'spatial_features.npy')
    if os.path.exists(path_to_temporal_features):
        temporal_features = np.load(path_to_temporal_features)
        spatial_features = np.load(path_to_spatial_features)
    else:
        temporal_features, spatial_features = svd_shifted_templates(
            shifted_templates, n_features)
        np.save(path_to_spatial_features, spatial_features)
        np.save(path_to_temporal_features, temporal_features)

    # calculate convolution of pairwise templates
    print("computing temp_temp")
    path_to_temp_temp = os.path.join(TMP_FOLDER, output_directory,
                                     'temp_temp.npy')
    if os.path.exists(path_to_temp_temp):
        temp_temp = np.load(path_to_temp_temp)
    else:
        indexes = np.arange(temporal_features.shape[0])
        template_list = np.array_split(indexes, n_processors)

        print("...todo: single-core option")
        temp_temp_array = parmap.map(
            calculate_temp_temp_parallel,
            template_list,
            temporal_features,
            spatial_features,
            processes=n_processors,
            pm_pbar=True)

        temp_temp = np.concatenate((temp_temp_array), axis=1) * 2
        np.save(path_to_temp_temp, temp_temp)

    # ******************************************************************
    # ****************** DECONVOLUTION START ***************************
    # ******************************************************************

    # compute padding length for deconvolution
    buffer_size = 2 * n_temporal_big + n_explore

    # Grab length of .dat file to compute chunk indexes below
    filename_bin = os.path.join(CONFIG.data.root_folder, output_directory,
                                recordings_filename)
    fp = np.memmap(filename_bin, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # Generate indexes in chunks (units of seconds)
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    # Make the 4 parameter list to be sent to deconvolution algorithm
    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list))
    proc_indexes = np.arange(len(idx_list))
    #print (idx_list)
    #print (proc_indexes)
    print("# of chunks for deconvolution: ", len(idx_list), " verbose mode: ",
          CONFIG.deconvolution.verbose)

    # Deconv using parmap module
    if CONFIG.resources.multi_processing:
        spike_train = parmap.map(
            deconvolve_new_allcores_updated,
            list(zip(idx_list, proc_indexes)),
            output_directory,
            TMP_FOLDER,
            filename_bin,
            path_to_spt_list,
            path_to_temp_temp,
            path_to_shifted_templates,
            buffer_size,
            n_channels,
            temporal_features,
            spatial_features,
            n_explore,
            threshold_d,
            verbose=CONFIG.deconvolution.verbose,
            processes=n_processors,
            pm_pbar=True)
    else:
        spike_train = []
        for k in range(len(idx_list)):
            spike_train.append(
                deconvolve_new_allcores_updated(
                    [idx_list[k], k],
                    output_directory,
                    TMP_FOLDER,
                    filename_bin,
                    path_to_spt_list,
                    path_to_temp_temp,
                    path_to_shifted_templates,
                    buffer_size,
                    n_channels,
                    temporal_features,
                    spatial_features,
                    n_explore,
                    threshold_d,
                    verbose=CONFIG.deconvolution.verbose))

    # Gather spikes
    spike_train = np.vstack(spike_train)

    # sort spikes by time and remove templates with spikes < max_spikes
    #spike_train, templates = clean_up(spike_train, templates, max_spikes)

    # Optional save spike_train as txt file for human readability
    # filename_spike_train = os.path.join(CONFIG.data.root_folder,
    #                            output_directory, 'spike_train.txt')
    # np.savetxt(filename_spike_train ,spike_train, fmt='%d',)

    logger.info('spike_train.shape: {}'.format(spike_train.shape))

    return spike_train, np.transpose(templates)
    
