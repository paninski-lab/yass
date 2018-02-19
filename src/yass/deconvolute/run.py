import os.path
import logging

import numpy as np
import datetime

from yass.deconvolute.deconvolve import deconvolve
from yass import read_config
from yass.batch import RecordingsReader


def run(spike_index, templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index: numpy.ndarray (n_data, 2)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels

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
                  'spike_index.shape: {}'
                  .format(templates.shape, spike_index.shape))

    # run deconvolution algorithm
    n_rf = int(CONFIG.deconvolution.n_rf*CONFIG.recordings.sampling_rate/1000)
    spike_train = deconvolve(writable_recordings, templates,
                             spike_index,
                             CONFIG.spikeSize,
                             CONFIG.deconvolution.n_explore,
                             n_rf,
                             CONFIG.deconvolution.upsample_factor,
                             CONFIG.deconvolution.threshold_a,
                             CONFIG.deconvolution.threshold_dd)

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    # sort spikes by time
    spike_train = spike_train[np.argsort(spike_train[:, 0])]

    currentTime = datetime.datetime.now()
    logger.info("Deconvolution done in {0} seconds.".format(
            (currentTime - start_time).seconds))

    return spike_train
