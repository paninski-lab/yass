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


def run(output_directory='tmp/', if_file_exists='skip'):
    """Preprocess pipeline: filtering, standarization and whitening filter

    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standarized data are
    processed in chunks and written to disk.

    Parameters
    ----------
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for every
      generated file (filtered, standarized and whitening filter). If
      'overwrite' it replaces the files if any exist, if 'abort' it raises
      a ValueError exception if any file exists, if 'skip' if skips the
      operation if any file exists

    Returns
    -------
    standarized_path: str

    standarized_params: str

    channel_index: numpy.ndarray

    whiten_filter: numpy.ndarray

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

    * ``filtered.bin`` - Filtered recordings
    * ``filtered.yaml`` - Filtered recordings metadata
    * ``standarized.bin`` - Standarized recordings
    * ``standarized.yaml`` - Standarized recordings metadata
    * ``whitening.npy`` - Whitening filter

    Examples
    --------

    .. literalinclude:: ../examples/preprocess.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    OUTPUT_DTYPE = CONFIG.preprocess.dtype
    TMP = os.path.join(CONFIG.data.root_folder, output_directory)

    logger.info('Output dtype for transformed data will be {}'
                .format(OUTPUT_DTYPE))

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

    # optionally filter the data
    if CONFIG.preprocess.filter:
        path, params = butterworth(path,
                                   params['dtype'],
                                   params['n_channels'],
                                   params['data_format'],
                                   CONFIG.filter.low_pass_freq,
                                   CONFIG.filter.high_factor,
                                   CONFIG.filter.order,
                                   CONFIG.recordings.sampling_rate,
                                   CONFIG.resources.max_memory,
                                   TMP,
                                   OUTPUT_DTYPE,
                                   if_file_exists=if_file_exists)

    # standarize
    (standarized_path,
        standarized_params) = standarize(path,
                                         params['dtype'],
                                         params['n_channels'],
                                         params['data_format'],
                                         CONFIG.recordings.sampling_rate,
                                         CONFIG.resources.max_memory,
                                         TMP,
                                         OUTPUT_DTYPE,
                                         if_file_exists=if_file_exists)

    # Whiten
    whiten_filter = whiten.matrix(standarized_path,
                                  standarized_params['dtype'],
                                  standarized_params['n_channels'],
                                  standarized_params['data_format'],
                                  CONFIG.neighChannels,
                                  CONFIG.geom,
                                  CONFIG.spikeSize,
                                  CONFIG.resources.max_memory,
                                  TMP,
                                  if_file_exists=if_file_exists)

    channel_index = make_channel_index(CONFIG.neighChannels,
                                       CONFIG.geom)

    return (str(standarized_path), standarized_params, channel_index,
            whiten_filter)
