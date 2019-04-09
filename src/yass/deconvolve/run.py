import os
import logging
import numpy as np
import parmap

from yass import read_config
from yass.reader import READER
from yass.deconvolve.match_pursuit import MatchPursuit_objectiveUpsample
#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(fname_templates,
        output_directory,
        recordings_filename,
        recording_dtype):
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

    fname_up = os.path.join(output_directory,
                            'deconv_up_result.npz')
    fname_spike_train = os.path.join(output_directory,
                                     'spike_train.npy')

    if os.path.exists(fname_up):
        return fname_spike_train, fname_up
    
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
        fname_templates=fname_templates,
        save_dir=output_directory,
        reader=reader,
        max_iter=max_iter,
        upsample=upsample_max_val,
        threshold=threshold,
        conv_approx_rank=conv_approx_rank,
        n_processors=CONFIG.resources.n_processors,
        multi_processing=CONFIG.resources.multi_processing)

    # collect save file name
    
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
        
    # run deconv for each batch
    # if CONFIG.resources.multi_processing:
        # parmap.starmap(mp_object.run_units,
                       # list(zip(batch_ids, fnames_out)),
                       # processes=CONFIG.resources.n_processors,
                       # pm_pbar=True)

    if len(batch_ids)>0: 
        logger.info("computing deconvolution over data")
        if CONFIG.resources.multi_processing:
            batches_in = np.array_split(batch_ids, CONFIG.resources.n_processors)
            fnames_in = np.array_split(fnames_out, CONFIG.resources.n_processors)
            parmap.starmap(mp_object.run_cores,
                           list(zip(batches_in, fnames_in)),
                           processes=CONFIG.resources.n_processors,
                           pm_pbar=True)
        else:
            for ctr in range(len(batch_ids)):
                mp_object.run(batch_ids[ctr], fnames_out[ctr])

    # collect result
    res = []
    logger.info("gathering deconvolution results")
    for batch_id in range(reader.n_batches):
        fname_out = os.path.join(seg_dir,
                  "seg_{}_deconv.npz".format(
                      str(batch_id).zfill(6)))
                      
        res.append(np.load(fname_out)['spike_train'])
    res = np.vstack(res)
    
    # get upsampled templates and mapping for computing residual
    (templates_up,
     deconv_id_sparse_temp_map) = mp_object.get_sparse_upsampled_templates()

    # since deconv spike time is not centered, get shift for centering
    spike_size = np.load(fname_templates).shape[1]
    shift = spike_size // 2

    # get spike train and save
    spike_train = np.copy(res)
    # map back to original id
    spike_train[:, 1] = np.int32(spike_train[:, 1]/mp_object.upsample_max_val)
    spike_train[:, 0] += shift
    # save
    np.save(fname_spike_train, spike_train)

    # get upsampled data
    spike_train_up = np.copy(res)
    spike_train_up[:, 1] = deconv_id_sparse_temp_map[
                spike_train_up[:, 1]]
    spike_train_up[:, 0] += shift

    # save
    np.savez(fname_up,
             spike_train_up=spike_train_up,
             templates_up=templates_up.transpose(2,0,1),
             spike_train=spike_train,
             templates=mp_object.temps.transpose(2,0,1))

    np.save(os.path.join(os.path.split(fname_up)[0],'spike_train_up.npy'),
            spike_train_up)


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

    return fname_spike_train, fname_up
