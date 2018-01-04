import logging
import os.path

from ..batch import BatchPipeline
from ..batch import PipedTransformation as Transform
from ..preprocess.filter import butterworth
from ..preprocess.standarize import standarize


# FIXME: remove old factory code from here, i dont think we really need this
# the preprocessor runs this same code...
# TODO: documentation
# TODO: comment code, it's not clear what it does
def process_data(CONFIG):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    logger = logging.getLogger(__name__)

    # check if filtered data already exists
    path_to_filtered = os.path.join(CONFIG.data.root_folder,
                                    'tmp/filtered.bin')

    if os.path.exists(path_to_filtered):
        logger.info('Filtered data already exists in: {}'
                    .format(path_to_filtered))
    else:
        logger.info('Filtered data does not exists...')

        tmp = os.path.join(CONFIG.data.root_folder, 'tmp')

        if not os.path.exists(tmp):
            logger.info('Creating temporary folder: {}'.format(tmp))
            os.makedirs(tmp)
        else:
            logger.info('Temporary folder {} already exists, output will be '
                        'stored there'.format(tmp))

        path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
        dtype = CONFIG.recordings.dtype

        # initialize pipeline object, one batch per channel
        pipeline = BatchPipeline(path, dtype, CONFIG.recordings.n_channels,
                                 CONFIG.recordings.format,
                                 CONFIG.resources.max_memory, tmp)

        # add filter transformation if necessary
        sampling_freq = CONFIG.recordings.sampling_rate

        if CONFIG.preprocess.filter:
            filter_op = Transform(butterworth,
                                  'filtered.bin',
                                  mode='single_channel_one_batch',
                                  keep=True,
                                  low_freq=CONFIG.filter.low_pass_freq,
                                  high_factor=CONFIG.filter.high_factor,
                                  order=CONFIG.filter.order,
                                  sampling_freq=sampling_freq)
            pipeline.add([filter_op])

        # standarize
        standarize_op = Transform(standarize, 'standarized.bin',
                                  mode='single_channel_one_batch',
                                  keep=True,
                                  sampling_freq=sampling_freq)

        pipeline.add([standarize_op])

        pipeline.run()
