"""
Preprocess pipeline
"""
import logging
import os.path

from yass import read_config
from yass.geometry import make_channel_index
from yass.preprocess.filter import butterworth
from yass.preprocess.standarize import standarize
from yass.preprocess import whiten


def run(output_directory='tmp/'):
    """Execute preprocess pipeline

    Returns
    -------
    WIP

    Parameters
    ----------
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

    * ``config.yaml`` - Copy of the configuration file
    * ``metadata.yaml`` - Experiment metadata
    * ``filtered.bin`` - Filtered recordings
    * ``filtered.yaml`` - Filtered recordings metadata
    * ``standarized.bin`` - Standarized recordings
    * ``standarized.yaml`` - Standarized recordings metadata
    * ``whitening_filter.npy`` - Whitening filter

    Examples
    --------

    .. literalinclude:: ../examples/preprocess.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    OUTPUT_DTYPE = CONFIG.preprocess.dtype

    logger.info('Output dtype for transformed data will be {}'
                .format(OUTPUT_DTYPE))

    TMP = os.path.join(CONFIG.data.root_folder, output_directory)

    if not os.path.exists(TMP):
        logger.info('Creating temporary folder: {}'.format(TMP))
        os.makedirs(TMP)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(TMP))

    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    params = dict(dtype=CONFIG.recordings.dtype,
                  n_channels=CONFIG.recordings.n_channels,
                  data_format=CONFIG.recordings.format)

    if CONFIG.preprocess.filter:
        path, params = butterworth(path, params['dtype'], params['n_channels'],
                                   params['data_format'],
                                   CONFIG.filter.low_pass_freq,
                                   CONFIG.filter.high_factor,
                                   CONFIG.filter.order,
                                   CONFIG.recordings.sampling_rate,
                                   CONFIG.resources.max_memory,
                                   os.path.join(TMP, 'filtered.bin'),
                                   OUTPUT_DTYPE)

    # standarize
    (standarized_path,
        standarized_params) = standarize(path,
                                         params['dtype'],
                                         params['n_channels'],
                                         params['data_format'],
                                         CONFIG.recordings.sampling_rate,
                                         CONFIG.resources.max_memory,
                                         os.path.join(TMP, 'standarized.bin'),
                                         OUTPUT_DTYPE)

    # Whiten
    whiten_filter = whiten.matrix(standarized_path,
                                  standarized_params['dtype'],
                                  standarized_params['n_channels'],
                                  standarized_params['data_format'],
                                  CONFIG.neighChannels,
                                  CONFIG.geom,
                                  CONFIG.spikeSize,
                                  CONFIG.resources.max_memory,
                                  TMP)

    channel_index = make_channel_index(CONFIG.neighChannels,
                                       CONFIG.geom)

    return standarized_path, standarized_params, channel_index, whiten_filter
