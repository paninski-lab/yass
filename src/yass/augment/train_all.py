import logging


from yass.augment.make import make_training_data
from yass.neuralnetwork import NeuralNetDetector, NeuralNetTriage, AutoEncoder


def train_neural_networks(CONFIG, CONFIG_TRAIN, spike_train, data_folder):
    """
    Train neural network

    Parameters
    ----------
    CONFIG
        YASS configuration file
    CONFIG_TRAIN
        YASS Neural Network configuration file
    spike_train: numpy.ndarray
        Spike train, first column is spike index and second is main channel
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
    detectnet_name = './'+CONFIG_TRAIN['network_detector']['name']+'.ckpt'
    n_filters_triage = CONFIG_TRAIN['network_triage']['n_filters']
    triagenet_name = './'+CONFIG_TRAIN['network_triage']['name']+'.ckpt'
    n_features = CONFIG_TRAIN['network_autoencoder']['n_features']
    ae_name = './'+CONFIG_TRAIN['network_autoencoder']['name']+'.ckpt'

    # generate training data for detection, triage and autoencoder
    logger.info('Generating training data...')
    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amp, nspikes,
                                      data_folder=data_folder)

    # train detector
    NeuralNetDetector.train(x_detect, y_detect, n_filters_detect, n_iter,
                            n_batch, l2_reg_scale, train_step_size,
                            detectnet_name)

    # train triage
    NeuralNetTriage.train(x_triage, y_triage, n_filters_triage, n_iter,
                          n_batch, l2_reg_scale, train_step_size,
                          triagenet_name)

    # train autoencoder
    AutoEncoder.train(x_ae, y_ae, n_features, n_iter, n_batch, train_step_size,
                      ae_name)
