from os.path import join
import logging
import datetime
from yass.util import file_loader, check_for_files, LoadFile
from yass.cluster.legacy import location


@check_for_files(filenames=[LoadFile(join('cluster',
                                          'spike_train_cluster.npy')),
                            LoadFile(join('cluster', 'tmp_loc.npy')),
                            LoadFile(join('cluster', 'vbPar.pickle'))],
                 mode='values', relative_to=None,
                 auto_save=True)
def run(spike_index, if_file_exists='skip', save_results=False,
        function=location):
    """Clustering step

    Parameters
    ----------
    scores: numpy.ndarray (n_spikes, n_features, n_channels), str or Path
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels. Or path to a npy file

    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    output_directory: str, optional
        Location to store/look for the generate spike train, relative to
        CONFIG.data.root_folder

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for the
      spike_train_cluster.npy. file If 'overwrite' it replaces the files if
      exists, if 'abort' it raises a ValueError exception if exists,
      if 'skip' it skips the operation if the file exists (and returns the
      stored file)

    save_results: bool, optional
        Whether to save spike train to disk
        (in CONFIG.data.root_folder/relative_to/spike_train_cluster.npy),
        defaults to False

    Returns
    -------
    spike_train: (TODO add documentation)

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/cluster.py

    """
    spike_index = file_loader(spike_index)
    logger = logging.getLogger(__name__)
    start = datetime.datetime.now()

    spike_train, tmp_loc, vbParam = function(spike_index)

    elapsed = (datetime.datetime.now() - start).seconds
    logger.info("Clustering done in {0} seconds.".format(elapsed))

    return spike_train, tmp_loc, vbParam
