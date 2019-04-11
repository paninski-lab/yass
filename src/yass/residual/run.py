import os
import logging
import numpy as np

from yass import read_config
from yass.reader import READER
from yass.residual.residual import RESIDUAL

def run(fname_up,
        output_directory,
        recordings_filename,
        recording_dtype,
        dtype_out='float32'):
    """Compute residual

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

    # Note: this uses spike times occuring at beginning of spike
    fname_out = os.path.join(output_directory, 'residual.bin')
    if os.path.exists(fname_out):
        return fname_out, dtype_out

    # get data reader
    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    CONFIG.resources.n_sec_chunk)

    # get residual object
    residual_object = RESIDUAL(fname_up,
                               reader,
                               fname_out,
                               dtype_out)

    # partition spike times
    fname_partitioned = os.path.join(
        output_directory, 'spike_times_partitioned.npy')
    residual_object.partition_spike_time(fname_partitioned)

    # compute residual
    seg_dir = os.path.join(output_directory, 'segs')
    residual_object.compute_residual(seg_dir,
                                     CONFIG.resources.multi_processing,
                                     CONFIG.resources.n_processors)

    # concatenate all segments
    residual_object.save_residual()

    return fname_out, dtype_out
