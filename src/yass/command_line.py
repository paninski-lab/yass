import os

import click
import numpy as np
import logging

from . import set_config
from . import preprocess
from . import process
from . import deconvolute
from . import read_config


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--output_file', type=click.Path(), default='spike_train.csv',
              help='Path to output file, defaults to spike_train.csv')
def sort(config, output_file):
    """
    Sort recordings using a configuration file located in CONFIG
    """
    return _run_pipeline(config, output_file)


def _run_pipeline(config, output_file):
    """
    Run the entire pipeline given a path to a config file
    and output path
    """

    # configure logging module to get useful information
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # cfg = yass.Config.from_yaml(config)

    # pp = Preprocessor(cfg)
    # score, spike_index_clear, spike_index_collision = pp.process()

    # mp = Mainprocessor(cfg, score, spike_index_clear, spike_index_collision)
    # spikeTrain_clear, spike_index_collision = mp.mainProcess()

    # dc = Deconvolution(cfg, np.transpose(
    #     mp.templates, [1, 0, 2]), spike_index_collision)
    # spikeTrain_col = dc.fullMPMU()

    # spikeTrain = np.concatenate((spikeTrain_col, spikeTrain_clear))
    # idx_sort = np.argsort(spikeTrain[:, 0])
    # spikeTrain = spikeTrain[idx_sort]

    # idx_keep = np.zeros(spikeTrain.shape[0], 'bool')
    # for k in range(mp.templates.shape[2]):
    #     idx_c = np.where(spikeTrain[:, 1] == k)[0]
    #     idx_keep[idx_c[np.concatenate(
    #         ([True], np.diff(spikeTrain[idx_c, 0]) > 1))]] = 1
    # spikeTrain = spikeTrain[idx_keep]

    # path_to_file = os.path.join(cfg.data.root_folder, output_file)

    # np.savetxt(path_to_file, spikeTrain, fmt='%i, %i')
    # print('Done, spike train saved in: {}'.format(path_to_file))

    # set yass configuration parameters
    set_config(config)
    CONFIG = read_config()

    # run preprocessor
    score, spike_index_clear, spike_index_collision = preprocess.run()

    # run processor
    (spike_train_clear, templates,
     spike_index_collision) = process.run(score, spike_index_clear,
                                          spike_index_collision)

    # run deconvolution
    spike_train = deconvolute.run(spike_train_clear, templates,
                                  spike_index_collision)

    path_to_file = os.path.join(CONFIG.data.root_folder, output_file)
    np.savetxt(path_to_file, spike_train, fmt='%i, %i')
    logger.info('Done, spike train saved in: {}'.format(path_to_file))


@cli.command()
@click.argument('spike_train', type=click.Path(exists=True))
def train(config, output_file):
    """Train neural networks using a SPIKE_TRAIN csv file whose first column
    is the spike time and second column is the spike ID
    """
    pass
