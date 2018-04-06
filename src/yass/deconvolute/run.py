import os.path
import logging

import numpy as np

from yass.deconvolute.deconvolve import deconvolve, fix_indexes
from yass import read_config
from yass.batch import BatchProcessor
from yass.util import file_loader


def run(spike_index, templates, output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index: numpy.ndarray (n_data, 2), str or pathlib.Path
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels. Or path to
        npy file

    templates: numpy.ndarray (n_channels, waveform_size, n_templates), str
    or pathlib.Path
        A 3D array with the templates. Or path to npy file

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

    spike_index = file_loader(spike_index)
    templates = file_loader(templates)

    logger = logging.getLogger(__name__)

    # read config file
    CONFIG = read_config()

    # read recording
    recording_path = os.path.join(CONFIG.data.root_folder,
                                  output_directory,
                                  recordings_filename)
    bp = BatchProcessor(recording_path,
                        buffer_size=templates.shape[1])

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'
                  .format(templates.shape, spike_index.shape))

    # run deconvolution algorithm
    n_rf = int(CONFIG.deconvolution.n_rf*CONFIG.recordings.sampling_rate/1000)

    # run nn preprocess batch-wsie
    mc = bp.multi_channel_apply
    res = mc(
        deconvolve,
        mode='memory',
        cleanup_function=fix_indexes,
        pass_batch_info=True,
        templates=templates,
        spike_index=spike_index,
        spike_size=CONFIG.spike_size,
        n_explore=CONFIG.deconvolution.n_explore,
        n_rf=n_rf,
        upsample_factor=CONFIG.deconvolution.upsample_factor,
        threshold_a=CONFIG.deconvolution.threshold_a,
        threshold_dd=CONFIG.deconvolution.threshold_dd)

    spike_train = np.concatenate([element for element in res], axis=0)

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    # sort spikes by time
    spike_train = spike_train[np.argsort(spike_train[:, 0])]

    return spike_train
