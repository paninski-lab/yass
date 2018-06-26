"""
Preprocess pipeline
"""
import logging
import os.path
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

from yass import read_config
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
        Location to store results, relative to CONFIG.data.root_folder,
        defaults to tmp/. See list of files in Notes section below.

    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. Control de behavior for every
        generated file. If 'overwrite' it replaces the files if any exist,
        if 'abort' it raises a ValueError exception if any file exists,
        if 'skip' it skips the operation (and loads the files) if any of them
        exist

    Returns
    -------
    standarized_path: str
        Path to standarized data binary file

    standarized_params: str
        Path to standarized data parameters

    channel_index: numpy.ndarray
        Channel indexes

    whiten_filter: numpy.ndarray
        Whiten matrix

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

    * ``preprocess/filtered.bin`` - Filtered recordings
    * ``preprocess/filtered.yaml`` - Filtered recordings metadata
    * ``preprocess/standarized.bin`` - Standarized recordings
    * ``preprocess/standarized.yaml`` - Standarized recordings metadata
    * ``preprocess/whitening.npy`` - Whitening filter

    Everything is run on CPU.

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/preprocess.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    OUTPUT_DTYPE = CONFIG.preprocess.dtype
    PROCESSES = CONFIG.resources.processes

    logger.info('Output dtype for transformed data will be {}'
                .format(OUTPUT_DTYPE))

    TMP = Path(CONFIG.data.root_folder, output_directory,
               'preprocess/')
    TMP.mkdir(parents=True, exist_ok=True)
    TMP = str(TMP)

    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    params = dict(dtype=CONFIG.recordings.dtype,
                  n_channels=CONFIG.recordings.n_channels,
                  data_order=CONFIG.recordings.order)

    # filter and standarize
    if CONFIG.preprocess.apply_filter:
        filter_params = CONFIG.preprocess.filter

        (standarized_path,
         standarized_params) = butterworth(path,
                                           params['dtype'],
                                           params['n_channels'],
                                           params['data_order'],
                                           filter_params.low_pass_freq,
                                           filter_params.high_factor,
                                           filter_params.order,
                                           CONFIG.recordings.sampling_rate,
                                           CONFIG.resources.max_memory,
                                           TMP,
                                           OUTPUT_DTYPE,
                                           standarize=True,
                                           output_filename='standarized.bin',
                                           if_file_exists=if_file_exists,
                                           processes=PROCESSES)
    # just standarize
    else:
        (standarized_path,
         standarized_params) = standarize(path,
                                          params['dtype'],
                                          params['n_channels'],
                                          params['data_order'],
                                          CONFIG.recordings.sampling_rate,
                                          CONFIG.resources.max_memory,
                                          TMP,
                                          OUTPUT_DTYPE,
                                          output_filename='standarized.bin',
                                          if_file_exists=if_file_exists,
                                          processes=PROCESSES)

    # TODO: remove whiten_filter out of output argument
    whiten_filter = whiten.matrix(standarized_path,
                                  standarized_params['dtype'],
                                  standarized_params['n_channels'],
                                  standarized_params['data_order'],
                                  CONFIG.channel_index,
                                  CONFIG.spike_size,
                                  CONFIG.resources.max_memory,
                                  TMP,
                                  output_filename='whitening.npy',
                                  if_file_exists=if_file_exists)

    return str(standarized_path), standarized_params, whiten_filter
