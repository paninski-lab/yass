"""
Preprocess pipeline
"""
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

from yass import read_config
from yass.preprocess import batch


def run(if_file_exists='skip', function=batch.run, **function_kwargs):
    """Preprocess pipeline: filtering, standarization and whitening filter

    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standarized data are
    processed in chunks and written to disk.

    Parameters
    ----------
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
    CONFIG = read_config()

    TMP = Path(CONFIG.path_to_output_directory, 'preprocess')
    TMP.mkdir(parents=True, exist_ok=True)
    TMP = str(TMP)

    (standarized_path,
     standarized_params,
     whiten_filter) = function(CONFIG,
                               if_file_exists=if_file_exists,
                               **function_kwargs)

    return standarized_path, standarized_params, whiten_filter
