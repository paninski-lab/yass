import os.path
import logging

import numpy as np

from yass.deconvolute.legacy.deconvolve import deconvolve, fix_indexes
from yass import read_config
from yass.batch import BatchProcessor


def legacy(spike_index, templates, recordings_filename='standarized.bin'):
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
    logger = logging.getLogger(__name__)

    # read config file
    CONFIG = read_config()

    # NOTE: this is not the right way to set defaults, the function should
    # list all parameters needed and provide defaults for them, since the
    # current code looks for parameters inside CONFIG, we are doing it like
    # this, we will remove this deconv method so de are not refactoring
    # this, new deconv methods should list all parameters in the function
    # signature
    defaults = {
        'n_rf': 1.5,
        'threshold_a': 0.3,
        'threshold_dd': 0,
        'n_explore': 2,
        'upsample_factor': 5
    }

    CONFIG._set_param('deconvolution', defaults)

    # read recording
    recording_path = os.path.join(CONFIG.path_to_output_directory,
                                  'preprocess',
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
