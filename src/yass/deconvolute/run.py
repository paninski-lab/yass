import os.path
import logging

from yass import read_config
from yass.util import file_loader, file_saver
from yass.deconvolute.legacy import legacy


def run(spike_index, templates, recordings_filename='standarized.bin',
        function=legacy):
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

    spike_train = function(spike_index, templates, recordings_filename)

    # save spike train
    path_to_spike_train = os.path.join(CONFIG.path_to_output_directory,
                                       'spike_train.npy')
    logger.info('Spike train saved in %s', path_to_spike_train)
    file_saver(spike_train, path_to_spike_train)

    return spike_train
