import os
import logging
import numpy as np
import parmap

from yass import read_config
from yass.reader import READER
from yass.deconvolve.match_pursuit import MatchPursuit_objectiveUpsample
#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(fname_templates_in,
        output_directory,
        recordings_filename,
        recording_dtype,
        run_chunk='full',
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

    if (os.path.exists(fname_templates) and
        os.path.exists(fname_spike_train) and
        os.path.exists(fname_templates_up) and
        os.path.exists(fname_spike_train_up)):
        return (fname_templates, fname_spike_train,
                fname_templates_up, fname_spike_train_up)
    
    # parameters
    # TODO: read from CONFIG
    threshold = CONFIG.deconvolution.threshold
    deconv_gpu = CONFIG.deconvolution.deconv_gpu
    conv_approx_rank = 10
    upsample_max_val = 32
    max_iter = 1000

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    CONFIG.resources.n_sec_chunk)

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


    # chunks to run deconv on
    if run_chunk == 'full':
        batches = np.arange(reader.n_batches)
    else:
        start_sec, end_sec = run_chunk

        idx_less = np.where(reader.idx_list[:, 0] <= start_sec)[0]
        start_batch = np.argmin(reader.idx_list[idx_less, 0])

        idx_bigger = np.where(reader.idx_list[:, 0] >= end_sec)[0]
        end_batch = np.argmin(reader.idx_list[idx_bigger, 0])

        batches = np.arange(start_batch, end_batch+1)

    # skip files/batches already completed; this allows more even distribution
    # across cores in case of restart
    # Cat: TODO: if cpu is still being used by endusers, may wish to implement
    #       dynamic file assignment here to deal with slow cores etc.
    fnames_out = []
    batch_ids = []
    for batch_id in batches:
        fname_temp = os.path.join(seg_dir,
                          "seg_{}_deconv.npz".format(
                              str(batch_id).zfill(6)))
        if os.path.exists(fname_temp):
            continue
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)

    if len(batch_ids)>0: 
        logger.info("computing deconvolution over data")
        if CONFIG.resources.multi_processing:
            batches_in = np.array_split(batch_ids, CONFIG.resources.n_processors)
            fnames_in = np.array_split(fnames_out, CONFIG.resources.n_processors)
            parmap.starmap(mp_object.run,
                           list(zip(batches_in, fnames_in)),
                           processes=CONFIG.resources.n_processors,
                           pm_pbar=True)
        else:
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
    np.save(fname_templates,
            mp_object.temps.transpose(2,0,1))

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

    return (fname_templates, fname_spike_train,
            fname_templates_up, fname_spike_train_up)