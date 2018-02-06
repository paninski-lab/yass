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

from . import set_config
from . import preprocess
from . import process
from . import deconvolute
from . import read_config
from . import geometry
from .export import generate
from .util import load_yaml, save_metadata, load_logging_config_file
from .neuralnetwork import train_neural_networks
from .config import Config
from .explore import RecordingExplorer
from .preprocess import dimensionality_reduction as dim_red


@click.group()
def cli():
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
    # load yass configuration parameters
    set_config(config)
    CONFIG = read_config()
    ROOT_FOLDER = CONFIG.data.root_folder
    TMP_FOLDER = path.join(ROOT_FOLDER, output_dir)

    # remove tmp folder if needed
    if os.path.exists(TMP_FOLDER) and clean:
        shutil.rmtree(TMP_FOLDER)

    # create TMP_FOLDER if needed
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    # load logging config file
    logging_config = load_logging_config_file()
    logging_config['handlers']['file']['filename'] = path.join(TMP_FOLDER,
                                                               'yass.log')
    logging_config['root']['level'] = logger_level

    # configure logging
    logging.config.dictConfig(logging_config)

    # instantiate logger
    logger = logging.getLogger(__name__)

    # run preprocessor
    (score, spike_index_clear,
     spike_index_collision) = preprocess.run(output_directory=output_dir)

    # run processor
    (spike_train_clear, templates,
     spike_index_collision) = process.run(score, spike_index_clear,
                                          spike_index_collision,
                                          output_directory=output_dir)

    # run deconvolution
    spike_train = deconvolute.run(spike_train_clear, templates,
                                  spike_index_collision,
                                  output_directory=output_dir)

    # save metadata in tmp
    path_to_metadata = path.join(TMP_FOLDER, 'metadata.yaml')
    logging.info('Saving metadata in {}'.format(path_to_metadata))
    save_metadata(path_to_metadata)

    # save config.yaml copy in tmp/
    path_to_config_copy = path.join(TMP_FOLDER, 'config.yaml')
    shutil.copy2(config, path_to_config_copy)
    logging.info('Saving copy of config: {} in {}'.format(config,
                                                          path_to_config_copy))

    # save templates
    path_to_templates = path.join(TMP_FOLDER, 'templates.npy')
    logging.info('Saving templates in {}'.format(path_to_templates))
    np.save(path_to_templates, templates)

    path_to_spike_train = path.join(TMP_FOLDER, output_file)
    np.save(path_to_spike_train, spike_train)
    logger.info('Spike train saved in: {}'.format(path_to_spike_train))

    # this part loads waveforms for all spikes in the spike train and scores
    # them, this data is needed to later generate phy files
    if complete:
        STANDARIZED_PATH = path.join(TMP_FOLDER, 'standarized.bin')
        PARAMS = load_yaml(path.join(TMP_FOLDER, 'standarized.yaml'))

        # load waveforms for all spikes in the spike train
        logger.info('Loading waveforms from all spikes in the spike train...')
        explorer = RecordingExplorer(STANDARIZED_PATH,
                                     spike_size=CONFIG.spikeSize,
                                     dtype=PARAMS['dtype'],
                                     n_channels=PARAMS['n_channels'],
                                     data_format=PARAMS['data_format'])
        waveforms = explorer.read_waveforms(spike_train[:, 0])

        path_to_waveforms = path.join(TMP_FOLDER, 'spike_train_waveforms.npy')
        np.save(path_to_waveforms, waveforms)
        logger.info('Saved all waveforms from the spike train in {}...'
                    .format(path_to_waveforms))

        # score all waveforms
        logger.info('Scoring waveforms from all spikes in the spike train...')
        path_to_rotation = path.join(TMP_FOLDER, 'rotation.npy')
        rotation = np.load(path_to_rotation)

        main_channels = explorer.main_channel_for_waveforms(waveforms)
        path_to_main_channels = path.join(TMP_FOLDER,
                                          'waveforms_main_channel.npy')
        np.save(path_to_main_channels, main_channels)
        logger.info('Saved all waveforms main channels in {}...'
                    .format(path_to_waveforms))

        waveforms_score = dim_red.score(waveforms, rotation, main_channels,
                                        CONFIG.neighChannels, CONFIG.geom)
        path_to_waveforms_score = path.join(TMP_FOLDER, 'waveforms_score.npy')
        np.save(path_to_waveforms_score, waveforms_score)
        logger.info('Saved all scores in {}...'.format(path_to_waveforms))

        # score templates
        # TODO: templates should be returned in the right shape to avoid .T
        templates_ = templates.T
        main_channels_tmpls = explorer.main_channel_for_waveforms(templates_)
        path_to_templates_main_c = path.join(TMP_FOLDER,
                                             'templates_main_channel.npy')
        np.save(path_to_templates_main_c, main_channels_tmpls)
        logger.info('Saved all templates main channels in {}...'
                    .format(path_to_templates_main_c))

        templates_score = dim_red.score(templates_, rotation,
                                        main_channels_tmpls,
                                        CONFIG.neighChannels, CONFIG.geom)
        path_to_templates_score = path.join(TMP_FOLDER, 'templates_score.npy')
        np.save(path_to_templates_score, templates_score)
        logger.info('Saved all templates scores in {}...'
                    .format(path_to_waveforms))


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
