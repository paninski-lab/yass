import logging


from yass.augment import (make_training_data, save_detect_network_params,
                          save_triage_network_params, save_ae_network_params)
from yass.neuralnetwork import train_detector, train_ae, train_triage
from yass.util import change_extension


def train_neural_networks(CONFIG, CONFIG_TRAIN, spike_train, data_folder):
    """Train all neural networks

    Parameters
    ----------
    """
    logger = logging.getLogger(__name__)

    chosen_templates = CONFIG_TRAIN['templates']['ids']
    min_amp = CONFIG_TRAIN['templates']['minimum_amplitude']
    nspikes = CONFIG_TRAIN['training']['n_spikes']
    n_filters_detect = CONFIG_TRAIN['network_detector']['n_filters']
    n_iter = CONFIG_TRAIN['training']['n_iterations']
    n_batch = CONFIG_TRAIN['training']['n_batch']
    l2_reg_scale = CONFIG_TRAIN['training']['l2_regularization_scale']
    train_step_size = CONFIG_TRAIN['training']['step_size']
    detectnet_name = CONFIG_TRAIN['network_detector']['name']+'.ckpt'
    n_filters_triage = CONFIG_TRAIN['network_triage']['n_filters']
    triagenet_name = CONFIG_TRAIN['network_triage']['name']+'.ckpt'
    n_features = CONFIG_TRAIN['network_autoencoder']['n_features']
    ae_name = CONFIG_TRAIN['network_autoencoder']['name']+'.ckpt'

    # generate training data
    logger.info('Generating training data...')
    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amp, nspikes,
                                      data_folder=data_folder)

    # train detector
    logger.info('Training detector network...')
    train_detector(x_detect, y_detect, n_filters_detect, n_iter, n_batch,
                   l2_reg_scale, train_step_size, detectnet_name)

    # save detector model parameters
    logger.info('Saving detector network parameters...')
    save_detect_network_params(filters=n_filters_detect,
                               size=x_detect.shape[1],
                               n_neighbors=x_detect.shape[2],
                               output_path=change_extension(detectnet_name,
                                                            'yaml'))

    # train triage
    logger.info('Training triage network...')
    train_triage(x_triage, y_triage, n_filters_triage, n_iter, n_batch,
                 l2_reg_scale, train_step_size, triagenet_name)

    # save triage model parameters
    logger.info('Saving triage network parameters...')
    save_triage_network_params(filters=n_filters_triage,
                               size=x_detect.shape[1],
                               n_neighbors=x_detect.shape[2],
                               output_path=change_extension(triagenet_name,
                                                            'yaml'))

    # train autoencoder
    logger.info('Training autoencoder network...')
    train_ae(x_ae, y_ae, n_features, n_iter, n_batch, train_step_size, ae_name)

    # save autoencoder model parameters
    logger.info('Saving autoencoder network parameters...')
    save_ae_network_params(n_input=x_ae.shape[1],
                           n_features=n_features,
                           output_path=change_extension(ae_name, 'yaml'))
