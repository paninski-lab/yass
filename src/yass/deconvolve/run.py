import os
import logging
import numpy as np
import parmap
import datetime as dt
from tqdm import tqdm
import torch

from yass import read_config
from yass.reader import READER
from yass.deconvolve.match_pursuit import MatchPursuit_objectiveUpsample
from yass.deconvolve.match_pursuit_gpu import deconvGPU

#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(fname_templates_in,
        output_directory,
        recordings_filename,
        recording_dtype,
        threshold=None,
        run_chunk_sec='full',
        save_up_data=True):
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
        standardized.bin

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

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_templates = os.path.join(
        output_directory, 'templates.npy')
    fname_spike_train = os.path.join(
        output_directory, 'spike_train.npy')
    fname_templates_up = os.path.join(
        output_directory, 'templates_up.npy')
    fname_spike_train_up = os.path.join(
        output_directory, 'spike_train_up.npy')
    fname_shifts = os.path.join(
        output_directory, 'shifts.npy')
                               
    print ("Processing templates: ", fname_templates_in)

    # Cat: TODO: use Peter's conditional (below) instead of single file check
    # if (os.path.exists(fname_templates) and
        # os.path.exists(fname_spike_train) and
        # os.path.exists(fname_templates_up) and
        # os.path.exists(fname_spike_train_up)):
        # return (fname_templates, fname_spike_train,
                # fname_templates_up, fname_spike_train_up)

    if os.path.exists(fname_spike_train):
        return (fname_templates, fname_spike_train,
                fname_templates_up, fname_spike_train_up,
                fname_shifts)
    # parameters
    # TODO: read from CONFIG
    if threshold is None:
        threshold = CONFIG.deconvolution.threshold
    elif threshold == 'max':
        min_norm_2 = np.square(
            np.load(fname_templates_in)).sum((1,2)).min()
        threshold = min_norm_2*0.8

    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    if CONFIG.deconvolution.deconv_gpu:
        n_sec_chunk = CONFIG.resources.n_sec_chunk_gpu
    else:
        n_sec_chunk = CONFIG.resources.n_sec_chunk

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    n_sec_chunk,
                    chunk_sec=chunk_sec)

    # deconv using GPU
    if CONFIG.deconvolution.deconv_gpu:
        deconv_ONgpu(fname_templates_in,
                     output_directory,
                     reader,
                     threshold,
                     fname_spike_train,
                     fname_spike_train_up,
                     fname_templates,
                     fname_templates_up,
                     fname_shifts,
                     CONFIG,
                     run_chunk_sec)
                                                    
    # deconv using CPU
    else:
        deconv_ONcpu(fname_templates_in,
                     output_directory,
                     reader,
                     threshold,
                     save_up_data,
                     fname_spike_train,
                     fname_spike_train_up,
                     fname_templates,
                     fname_templates_up,
                     CONFIG)

    return (fname_templates, fname_spike_train,
            fname_templates_up, fname_spike_train_up,
            fname_shifts)


def deconv_ONgpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 fname_spike_train,
                 fname_spike_train_up,
                 fname_templates,
                 fname_templates_up,
                 fname_shifts,
                 CONFIG,
                 run_chunk_sec):

    # *********** MAKE DECONV OBJECT ************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)

    #print (kfadfa)
    # Cat: TODO: gpu deconv requires own chunk_len variable
    n_sec=CONFIG.resources.n_sec_chunk_gpu
    #root_dir = '/media/cat/1TB/liam/49channels/data1_allset'
    root_dir = CONFIG.data.root_folder

    # Cat: TODO: read from CONFIG
    d_gpu.max_iter=1000
    d_gpu.deconv_thresh=threshold

    # Cat: TODO: make sure svd recomputed for higher rank etc.
    d_gpu.svd_flag = True

    # Cat: TODO read from CONFIG file 
    d_gpu.RANK = 10
    d_gpu.vis_chan_thresh = 1.0

    # debug/printout parameters
    # Cat: TODO: read all from CONFIG
    d_gpu.save_objective = False
    d_gpu.verbose = False
    d_gpu.print_iteration_counter = 50

    # add reader
    d_gpu.reader = reader

    # *********** INIT DECONV ****************
    begin=dt.datetime.now().timestamp()
    d_gpu.initialize()
    setup_time = np.round((dt.datetime.now().timestamp()-begin),4)
    print ("-------------------------------------------")
    print ("Total init time ", setup_time, 'sec')
    print ("-------------------------------------------")
    print ("")

    # ************ RUN DECONV ***************
    print ("Subtraction step...")
    begin=dt.datetime.now().timestamp()
    if True:
        chunks = []
        for k in range(0, CONFIG.rec_len//CONFIG.recordings.sampling_rate, 
                        CONFIG.resources.n_sec_chunk_gpu):
            chunks.append([k,k+n_sec])
    # run data on small chunk only
    else:
        chunks = [run_chunk_sec]

    # Cat: TODO : last chunk of data may be skipped if this doesn't work right.
    print ("  (TODO: Make sure last bit is added if rec_len not multiple of n_sec_gpu_chnk)")

    # loop over chunks and run sutraction step
    for chunk_id in tqdm(range(reader.n_batches)):
    #for chunk_id in range(reader.n_batches):
        d_gpu.run(chunk_id)
        #print ("\n-------------------\n")

    subtract_time = np.round((dt.datetime.now().timestamp()-begin),4)

    print ("-------------------------------------------")
    total_length_sec = int((d_gpu.reader.end - d_gpu.reader.start)/d_gpu.reader.sampling_rate)
    print ("Total Deconv Speed ", np.round(total_length_sec/(setup_time+subtract_time),2), " x Realtime")


    # ************** SAVE SPIKES & SHIFTS **********************
    print ("  gathering spike trains and shifts from deconv; no. of iterations: ", 
             len(d_gpu.spike_list))
    batch_size = d_gpu.reader.batch_size
    buffer_size = d_gpu.reader.buffer
    temporal_size = (CONFIG.recordings.sampling_rate/1000*
                        CONFIG.recordings.spike_size_ms)
    spike_train = [np.zeros((0,2),'int32')]
    shifts = []
    for k in tqdm(range(len(d_gpu.spike_list))):
        spike_times = d_gpu.spike_list[k][1].cpu().data.numpy()
        idx_keep = np.logical_and(spike_times >= buffer_size + temporal_size,
                                  spike_times < batch_size - buffer_size)
        idx_keep = np.where(idx_keep)[0]
        temp=np.zeros((len(idx_keep),2), 'int32')
        temp[:,0]=spike_times[idx_keep]+d_gpu.spike_list[k][0]
        temp[:,1]=d_gpu.spike_list[k][2].cpu().data.numpy()[idx_keep]

        spike_train.extend(temp)
        shifts.append(d_gpu.shift_list[k].cpu().data.numpy()[idx_keep])
            
    spike_train = np.vstack(spike_train)

    # add half the spike time back in to get to centre of spike
    spike_train[:,0] = spike_train[:,0]-temporal_size//2

    # save spike train
    print ("  saving spike_train: ", spike_train.shape)
    fname_spike_train = os.path.join(d_gpu.out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)
    np.save(fname_spike_train_up, spike_train)

    # save shifts
    fname_shifts = os.path.join(d_gpu.out_dir, 'shifts.npy')
    np.save(fname_shifts,np.hstack(shifts))

    # save templates and upsampled templates
    templates_in_original = np.load(fname_templates_in)
    np.save(fname_templates, templates_in_original)
    np.save(fname_templates_up, templates_in_original)


def deconv_ONcpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 save_up_data,
                 fname_spike_train,
                 fname_spike_train_up,
                 fname_templates,
                 fname_templates_up,
                 CONFIG):

    logger = logging.getLogger(__name__)

    conv_approx_rank = 5
    upsample_max_val = 8
    max_iter = 1000

    mp_object = MatchPursuit_objectiveUpsample(
        fname_templates=fname_templates_in,
        save_dir=output_directory,
        reader=reader,
        max_iter=max_iter,
        upsample=upsample_max_val,
        threshold=threshold,
        conv_approx_rank=conv_approx_rank,
        n_processors=CONFIG.resources.n_processors,
        multi_processing=CONFIG.resources.multi_processing)

    logger.info('Number of Units IN: {}'.format(mp_object.temps.shape[2]))

    # directory to save results for each segment
    seg_dir = os.path.join(output_directory, 'seg')
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # skip files/batches already completed; this allows more even distribution
    # across cores in case of restart
    # Cat: TODO: if cpu is still being used by endusers, may wish to implement
    #       dynamic file assignment here to deal with slow cores etc.
    fnames_out = []
    batch_ids = []
    for batch_id in range(reader.n_batches):
        fname_temp = os.path.join(seg_dir,
                          "seg_{}_deconv.npz".format(
                              str(batch_id).zfill(6)))
        if os.path.exists(fname_temp):
            continue
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)
    logger.info("running deconvolution on {} batches of {} seconds".format(
        len(batch_ids), CONFIG.resources.n_sec_chunk))

    if len(batch_ids)>0: 
        if CONFIG.resources.multi_processing:
            logger.info("running deconvolution with {} processors".format(
                CONFIG.resources.n_processors))
            batches_in = np.array_split(batch_ids, CONFIG.resources.n_processors)
            fnames_in = np.array_split(fnames_out, CONFIG.resources.n_processors)
            parmap.starmap(mp_object.run,
                           list(zip(batches_in, fnames_in)),
                           processes=CONFIG.resources.n_processors,
                           pm_pbar=True)
        else:
            logger.info("running deconvolution")
            for ctr in range(len(batch_ids)):
                mp_object.run([batch_ids[ctr]], [fnames_out[ctr]])

    # collect result
    res = []
    logger.info("gathering deconvolution results")
    for batch_id in range(reader.n_batches):
        fname_out = os.path.join(seg_dir,
                  "seg_{}_deconv.npz".format(
                      str(batch_id).zfill(6)))
        res.append(np.load(fname_out)['spike_train'])
    res = np.vstack(res)

    logger.info('Number of Spikes deconvolved: {}'.format(res.shape[0]))

    # save templates and upsampled templates
    np.save(fname_templates, np.load(fname_templates_in))
    #np.save(fname_templates,
    #        mp_object.temps.transpose(2,0,1))

    # since deconv spike time is not centered, get shift for centering
    shift = CONFIG.spike_size // 2

    # get spike train and save
    spike_train = np.copy(res)
    # map back to original id
    spike_train[:, 1] = np.int32(spike_train[:, 1]/mp_object.upsample_max_val)
    spike_train[:, 0] += shift
    # save
    np.save(fname_spike_train, spike_train)

    if save_up_data:
        # get upsampled templates and mapping for computing residual
        (templates_up,
         deconv_id_sparse_temp_map) = mp_object.get_sparse_upsampled_templates()

        np.save(fname_templates_up,
                templates_up.transpose(2,0,1))

        # get upsampled spike train
        spike_train_up = np.copy(res)
        spike_train_up[:, 1] = deconv_id_sparse_temp_map[
                    spike_train_up[:, 1]]
        spike_train_up[:, 0] += shift
        np.save(fname_spike_train_up, spike_train_up)

    # Compute soft assignments
    #soft_assignments, assignment_map = get_soft_assignments(
    #        templates=templates.transpose([2, 0, 1]),
    #        templates_upsampled=templates.transpose([2, 0, 1]),
    #        spike_train=spike_train,
    #        spike_train_upsampled=spike_train,
    #        filename_residual=deconv_obj.residual_fname,
    #        n_similar_units=2)

    #np.save(deconv_obj.root_dir + '/soft_assignment.npy', soft_assignments)
    #np.save(deconv_obj.root_dir + '/soft_assignment_map.npy', assignment_map)
