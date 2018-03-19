"""
Command line utility:

Spike sorting `yass sort`
Neural network training `yass train`
Phy integration `yass export`
"""

import os
import os.path as path
import logging
import logging.config
import shutil

import click
import numpy as np


from yass import pipeline
from yass import geometry
from yass.export import generate
from yass.util import load_yaml
from yass.augment import train_neural_networks
from yass.config import Config


@click.group()
@click.version_option()
def cli():
    """Command line group
    """
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False,
                                          resolve_path=True))
@click.option('-l', '--logger_level',
              help='Python logger level, defaults to INFO',
              default='INFO')
@click.option('-c', '--clean',
              help='Delete CONFIG.data.root_folder/output_dir/ before running',
              is_flag=True, default=False)
@click.option('-o', '--output_dir',
              help='Output directory (relative to CONFIG.data.root_folder '
              'to store the output data, defaults to tmp/',
              default='tmp/')
@click.option('-cm', '--complete',
              help='Generates extra files (needed to generate phy files)',
              is_flag=True, default=False)
def sort(config, logger_level, clean, output_dir, complete):
    """
    Sort recordings using a configuration file located in CONFIG
    """
    return _run_pipeline(config, output_file='spike_train.npy',
                         logger_level=logger_level, clean=clean,
                         output_dir=output_dir, complete=complete)


def _run_pipeline(config, output_file, logger_level='INFO', clean=True,
                  output_dir='tmp/', complete=False):
    """
    Run the entire pipeline given a path to a config file
    and output path
    """
    pipeline.run(config, logger_level, clean, output_dir, complete)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.argument('config_train', type=click.Path(exists=True, dir_okay=False))
@click.option('-l', '--logger_level',
              help='Python logger level, defaults to INFO',
              default='INFO')
def train(directory, config_train, logger_level):
    """
    Train neural networks, DIRECTORY must be a folder containing the
    output of `yass sort`, CONFIG_TRAIN must be the location of a file with
    the training parameters
    """
    logging.basicConfig(level=getattr(logging, logger_level))
    logger = logging.getLogger(__name__)

    path_to_spike_train = path.join(directory, 'spike_train.npy')
    spike_train = np.load(path_to_spike_train)

    logger.info('Loaded spike train with: {:,} spikes and {:,} different IDs'
                .format(len(spike_train),
                        len(np.unique(spike_train[:, 1]))))

    path_to_config = path.join(directory, 'config.yaml')
    CONFIG = Config.from_yaml(path_to_config)

    CONFIG_TRAIN = load_yaml(config_train)

    train_neural_networks(CONFIG, CONFIG_TRAIN, spike_train,
                          data_folder=directory)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--output_dir', type=click.Path(file_okay=False),
              help=('Path to output directory, defaults to '
                    'directory/phy/'))
def export(directory, output_dir):
    """
    Generates phy input files, 'yass sort' (with the `--complete` option)
    must be run first to generate all the necessary files
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    TMP_FOLDER = directory
    PATH_TO_CONFIG = path.join(TMP_FOLDER, 'config.yaml')
    CONFIG = load_yaml(PATH_TO_CONFIG)
    ROOT_FOLDER = CONFIG['data']['root_folder']
    N_CHANNELS = CONFIG['recordings']['n_channels']

    # verify that the tmp/ folder exists, otherwise abort
    if not os.path.exists(TMP_FOLDER):
        click.echo("{} directory does not exist, this means you "
                   "haven't run 'yass sort', run it before running "
                   "'yass export' again...".format(TMP_FOLDER))
        raise click.Abort()

    if output_dir is None:
        PHY_FOLDER = path.join(TMP_FOLDER, 'phy/')
    else:
        PHY_FOLDER = output_dir

    if not os.path.exists(PHY_FOLDER):
        logger.info('Creating directory: {}'.format(PHY_FOLDER))
        os.makedirs(PHY_FOLDER)

    # TODO: convert data to wide format

    # generate params.py
    params = generate.params(PATH_TO_CONFIG)
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

    # load spike train
    path_to_spike_train = path.join(TMP_FOLDER, 'spike_train.npy')
    logger.info('Loading spike train from {}...'.format(path_to_spike_train))
    spike_train = np.load(path_to_spike_train)
    N_SPIKES, _ = spike_train.shape
    logger.info('Spike train contains {:,} spikes'.format(N_SPIKES))

    # load templates
    logging.info('Loading previously saved templates...')
    path_to_templates = path.join(TMP_FOLDER, 'templates.npy')
    templates = np.load(path_to_templates)
    _, _, N_TEMPLATES = templates.shape

    # pc_features_ind.npy
    path_to_pc_features_ind = path.join(PHY_FOLDER, 'pc_feature_ind.npy')
    ch_neighbors = geometry.find_channel_neighbors
    neigh_channels = ch_neighbors(geom, CONFIG['recordings']['spatial_radius'])

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

    # template_feature_ind.npy
    path_to_template_feature_ind = path.join(PHY_FOLDER,
                                             'template_feature_ind.npy')
    template_feature_ind = generate.template_feature_ind(N_TEMPLATES,
                                                         similar_templates)
    np.save(path_to_template_feature_ind, template_feature_ind)
    logger.info('Saved {}...'.format(path_to_template_feature_ind))

    # template_features.npy
    templates_score = np.load(path.join(TMP_FOLDER, 'templates_score.npy'))
    templates_main_channel = np.load(path.join(TMP_FOLDER,
                                     'templates_main_channel.npy'))
    waveforms_score = np.load(path.join(TMP_FOLDER, 'waveforms_score.npy'))

    path_to_template_features = path.join(PHY_FOLDER, 'template_features.npy')
    template_features = generate.template_features(N_SPIKES, N_CHANNELS,
                                                   N_TEMPLATES, spike_train,
                                                   templates_main_channel,
                                                   neigh_channels, geom,
                                                   templates_score,
                                                   template_feature_ind,
                                                   waveforms_score)
    np.save(path_to_template_features, template_features)
    logger.info('Saved {}...'.format(path_to_template_features))

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
    path_to_whitening = path.join(TMP_FOLDER, 'whitening.npy')
    path_to_whitening_mat = path.join(PHY_FOLDER, 'whitening_mat.npy')
    shutil.copy2(path_to_whitening, )
    logging.info('Saving copy of whitening: {} in {}'
                 .format(path_to_whitening, path_to_whitening_mat))

    path_to_whitening_mat_inv = path.join(PHY_FOLDER, 'whitening_mat_inv.npy')
    whitening = np.load(path_to_whitening)
    np.save(path_to_whitening_mat_inv, np.linalg.inv(whitening))
    logger.info('Saving inverse of whitening matrix in {}...'
                .format(path_to_whitening_mat_inv))

    logging.info('Done.')
