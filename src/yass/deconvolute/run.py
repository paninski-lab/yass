import os.path
import logging

import numpy as np
import datetime

from yass.deconvolute.deconvolve import deconvolve
from yass.neuralnetwork import fix_indexes
from yass import read_config
from yass.batch import RecordingsReader

def run(spike_train_clear, templates, spike_index_collision,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------
    spike_train_clear: numpy.ndarray (n_clear_spikes, 2)
        A 2D array for clear spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        A 2D array for collided spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

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

    .. literalinclude:: ../examples/deconvolute.py
    """
    
    logger = logging.getLogger(__name__)
    start_time = datetime.datetime.now()
    
    # read config file
    CONFIG = read_config()

    # read recording
    recording_path = os.path.join(CONFIG.data.root_folder,
                                  output_directory,
                                  recordings_filename)
    recordings = RecordingsReader(recording_path)
    
    # make another memory mapping file with write-access
    writable_recording_path = os.path.join(CONFIG.data.root_folder,
                                           output_directory,
                                           'deconvolved_recording.bin')
    writable_recordings = np.memmap(writable_recording_path, 
                                    dtype='float32', mode='w+',
                                    shape=recordings.shape)
    writable_recordings = np.copy(recordings.data)
    
    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_collision.shape: {}'
                  .format(templates.shape, spike_index_collision.shape))
    
    
    # run deconvolution algorithm
    n_rf = int(CONFIG.deconvolution.n_rf*CONFIG.recordings.sampling_rate/10000)
    spike_train_deconv = deconvolve(writable_recordings, templates, 
                                    spike_index_collision,
                                    CONFIG.deconvolution.n_explore, 
                                    n_rf, 
                                    CONFIG.deconvolution.upsample_factor, 
                                    CONFIG.deconvolution.threshold_a, 
                                    CONFIG.deconvolution.threshold_dd)

    logger.debug('spike_train_deconv.shape: {}'
                 .format(spike_train_deconv.shape))

    # merge spikes in one array
    spike_train = np.concatenate((spike_train_deconv, spike_train_clear))
    spike_train = spike_train[np.argsort(spike_train[:, 0])]

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    idx_keep = np.zeros(spike_train.shape[0], 'bool')

    # TODO: check if we can remove this
    for k in range(templates.shape[2]):
        idx_c = np.where(spike_train[:, 1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True],
                                       np.diff(spike_train[idx_c, 0])
                                       > 1))]] = 1

    logger.debug('deduplicated spike_train_deconv.shape: {}'
                 .format(spike_train.shape))

    spike_train = spike_train[idx_keep]

    currentTime = datetime.datetime.now()
    logger.info("Deconvolution done in {0} seconds.".format(
            (currentTime - start_time).seconds))
    
    return spike_train
