import os
import os.path as path
import logging
import shutil
from functools import partial

import click
import numpy as np

from . import set_config
from . import preprocess
from . import process
from . import deconvolute
from . import read_config
from . import geometry
from .export import generate
from .util import load_yaml
from .neuralnetwork import train_neural_networks
from .config import Config


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False,
                                          resolve_path=True))
def sort(config, output_file):
    """
    Sort recordings using a configuration file located in CONFIG
    """
    return _run_pipeline(config, output_file='spike_train.npy')


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
    ROOT_FOLDER = CONFIG.data.root_folder
    TMP_FOLDER = path.join(ROOT_FOLDER, 'tmp/')

    # run preprocessor
    score, spike_index_clear, spike_index_collision = preprocess.run()

    # run processor
    (spike_train_clear, templates,
     spike_index_collision) = process.run(score, spike_index_clear,
                                          spike_index_collision)

    # run deconvolution
    spike_train = deconvolute.run(spike_train_clear, templates,
                                  spike_index_collision)

    # save templates
    path_to_templates = os.path.join(TMP_FOLDER, 'templates.npy')
    logging.info('Saving templates in {}'.format(path_to_templates))
    np.save(path_to_templates, templates)

    path_to_spike_train = os.path.join(TMP_FOLDER, output_file)
    np.save(path_to_spike_train, spike_train)
    logger.info('Spike train saved in: {}'.format(path_to_spike_train))


@cli.command()
@click.argument('spike_train', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_train', type=click.Path(exists=True, dir_okay=False))
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
def train(spike_train, config_train, config):
    """Train neural networks using a SPIKE_TRAIN csv or npy file whose
    first column is the spike time and second column is the spike ID,
    a CONFIG_TRAIN yaml file with the training parameters and a CONFIG
    yaml file with the data parameters
    """

    # configure logging module to get useful information
    logging.basicConfig(level=logging.INFO)

    loadtxt = partial(np.loadtxt, dtype='int32', delimiter=',')
    fn = loadtxt if spike_train.endswith('.csv') else np.load

    spike_train = fn(spike_train)

    CONFIG_TRAIN = load_yaml(config_train)
    CONFIG = Config.from_yaml(config)

    train_neural_networks(CONFIG, CONFIG_TRAIN, spike_train)


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
@click.option('--output_dir', type=click.Path(file_okay=False),
              help=('Path to output directory, defaults to '
                    'CONFIG.data.root_folder/phy/'))
def export(config, output_dir):
    """Generates phy input files, 'yass sort' must be run first
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    CONFIG = load_yaml(config)
    ROOT_FOLDER = CONFIG['data']['root_folder']
    N_CHANNELS = CONFIG['recordings']['n_channels']

    TMP_FOLDER = path.join(ROOT_FOLDER, 'tmp/')

    # verify that the tmp/ folder exists, otherwise abort
    if not os.path.exists(TMP_FOLDER):
        click.echo("{} directory does not exist, this means you "
                   "haven't run 'yass sort', run it before running "
                   "'yass export' again...".format(TMP_FOLDER))
        raise click.Abort()

    if output_dir is None:
        PHY_FOLDER = path.join(ROOT_FOLDER, 'phy/')
    else:
        PHY_FOLDER = output_dir

    if not os.path.exists(PHY_FOLDER):
        logger.info('Creating directory: {}'.format(PHY_FOLDER))
        os.makedirs(PHY_FOLDER)

    # convert data to wide format

    # generate params.py
    params = generate.params(config)
    path_to_params = path.join(PHY_FOLDER, 'params.py')

    with open(path_to_params, 'w') as f:
        f.write(params)

    logger.info('Saved {}...'.format(path_to_params))

    # channel_positions.npy
    logger.info('Generating channel_positions.npy')
    path_to_geom = path.join(ROOT_FOLDER, CONFIG['data']['geometry'])
    geom = geometry.parse(path_to_geom, N_CHANNELS)
    path_to_channel_positions = path.join(PHY_FOLDER, 'channel_positions.npy')
    np.save(path_to_channel_positions, geom)
    logger.info('Saved {}...'.format(path_to_channel_positions))

    # channel_map.npy
    channel_map = generate.channel_map(N_CHANNELS)
    path_to_channel_map = path.join(PHY_FOLDER, 'channel_map.npy')
    np.save(path_to_channel_map, channel_map)
    logger.info('Saved {}...'.format(path_to_channel_map))

    # move tmp/score.npy to phy/pc_features.npy
    path_to_score = path.join(TMP_FOLDER, 'score.npy')
    path_to_pc_features = path.join(PHY_FOLDER, 'pc_features.npy')
    shutil.copy2(path_to_score, path_to_pc_features)
    logger.info('Copied {} to {}...'.format(path_to_score,
                                            path_to_pc_features))

    # load spike train
    path_to_spike_train = path.join(TMP_FOLDER, 'spike_train.npy')
    logger.info('Loading spike train from {}...'.format(path_to_spike_train))
    spike_train = np.load(path_to_spike_train)
    N_SPIKES, _ = spike_train.shape

    # load templates
    logging.info('Loading previously saved templates...')
    path_to_templates = path.join(TMP_FOLDER, 'templates.npy')
    templates = np.load(path_to_templates)
    _, _, N_TEMPLATES = templates.shape

    # pc_features_ind.npy
    path_to_pc_features_ind = path.join(PHY_FOLDER, 'pc_feature_ind.npy')
    ch_neighbors = geom.find_channel_neighbors
    neigh_channels = ch_neighbors(geom, CONFIG.recordings.spatial_radius)

    pc_feature_ind = generate.pc_feature_ind(N_SPIKES, N_TEMPLATES, N_CHANNELS,
                                             geom, neigh_channels, spike_train,
                                             templates)
    np.save(path_to_pc_features_ind,  pc_feature_ind)

    # similar_templates.npy
    path_to_templates = path.join(TMP_FOLDER, 'templates.npy')
    path_to_similar_templates = path.join(PHY_FOLDER, 'similar_templates.npy')
    templates = np.load(path_to_templates)
    similar_templates = generate.similar_templates(templates)
    np.save(path_to_similar_templates,  similar_templates)
    logger.info('Saved {}...'.format(path_to_similar_templates))

    # spike_templates.npy and spike_times.npy
    path_to_spike_templates = path.join(PHY_FOLDER, 'spike_templates.npy')
    np.save(path_to_spike_templates,  spike_train[:, 1])
    logger.info('Saved {}...'.format(path_to_spike_templates))

    path_to_spike_times = path.join(PHY_FOLDER, 'spike_times.npy')
    np.save(path_to_spike_times, spike_train[:, 0])
    logger.info('Saved {}...'.format(path_to_spike_times))

    # template_features.npy

    # template_feature_ind.npy
    path_to_template_feature_ind = path.join(PHY_FOLDER,
                                             'template_feature_ind.npy')
    template_feature_ind = generate.template_feature_ind(N_TEMPLATES)
    np.save(path_to_template_feature_ind, template_feature_ind)
    logger.info('Saved {}...'.format(path_to_template_feature_ind))

    # templates.npy
    path_to_phy_templates = path.join(PHY_FOLDER, 'templates.npy')
    np.save(path_to_phy_templates, np.transpose(templates, [2, 1, 0]))
    logging.info('Saved phy-compatible templates in {}'
                 .format(path_to_phy_templates))

    # templates_ind.npy
    templates_ind = generate.templates_ind(N_TEMPLATES, N_CHANNELS)
    path_to_templates_ind = path.join(PHY_FOLDER, 'templates_ind.npy')
    np.save(path_to_templates_ind, templates_ind)
    logger.info('Saved {}...'.format(path_to_templates_ind))

    # whitening_mat.npy and whitening_mat_inv.npy
    logging.info('Generating whitening_mat.npy and whitening_mat_inv.npy...')
    whitening_mat, whitening_mat_inv = generate.whitening_matrices(N_CHANNELS)
    path_to_whitening_mat = path.join(PHY_FOLDER, 'whitening_mat.npy')
    path_to_whitening_mat_inv = path.join(PHY_FOLDER, 'whitening_mat_inv.npy')
    np.save(path_to_whitening_mat, whitening_mat)
    np.save(path_to_whitening_mat_inv, whitening_mat_inv)
    logger.info('Saved {}...'.format(path_to_whitening_mat))
    logger.info('Saved {}...'.format(path_to_whitening_mat_inv))

    logging.info('Done.')
