import os
import logging
import datetime

from yass import read_config
from yass.templates.util import get_and_merge_templates as gam_templates
from yass.util import check_for_files, LoadFile, file_loader


@check_for_files(filenames=[LoadFile('templates.npy')],
                 mode='values', relative_to='output_directory',
                 auto_save=True, prepend_root_folder=True)
def run(spike_train, output_directory='tmp/',
        recordings_filename='standarized.bin',
        if_file_exists='skip', save_results=False):
    """Compute templates


    Parameters
    ----------
    spike_train: numpy.ndarray, str or pathlib.Path
        Spike train from cluster step or path to npy file

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to generate the templates, defaults to
        standarized.bin

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for the
      templates.npy. file If 'overwrite' it replaces the files if exists,
      if 'abort' it raises a ValueError exception if exists,
      if 'skip' it skips the operation if the file exists (and returns the
      stored file)

    save_results: bool, optional
        Whether to templates to disk
        (in CONFIG.data.root_folder/relative_to/templates.npy),
        defaults to False


    Returns
    -------
    templates: npy.ndarray
        Ttemplates

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/templates.py
    """
    spike_train = file_loader(spike_train)

    CONFIG = read_config()

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    _b = datetime.datetime.now()

    logger.info("Getting Templates...")

    path_to_recordings = os.path.join(CONFIG.data.root_folder,
                                      output_directory,
                                      recordings_filename)
    merge_threshold = CONFIG.templates.merge_threshold

    spike_train, templates = gam_templates(
        spike_train, path_to_recordings, CONFIG.spike_size,
        CONFIG.templates_max_shift, merge_threshold, CONFIG.neigh_channels)

    Time['e'] += (datetime.datetime.now() - _b).total_seconds()

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Templates done in {0} seconds.".format(
        (currentTime - startTime).seconds))

    return templates
