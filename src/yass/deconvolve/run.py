import os
import logging
import numpy as np

from yass import read_config
from yass.deconvolve.deconvolve import Deconv

def run(spike_train_cluster,
        templates,
        output_directory='tmp/',
        recordings_filename='standardized.bin'):
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

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_cluster.shape: {}'.format(templates.shape,
                                                 spike_train_cluster.shape))

    Deconv(spike_train_cluster,
            templates,
            output_directory,
            recordings_filename,
            CONFIG)

    # Note: new self.templates and self.spike_train is computed above
    # no need to return them to deconv
    spike_train = np.load(os.path.join(CONFIG.path_to_output_directory,
                    'spike_train_post_deconv_post_merge.npy'))

    templates = np.load(os.path.join(CONFIG.path_to_output_directory,
                    'templates_post_deconv_post_merge.npy'))

    return spike_train, templates
