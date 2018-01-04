import yass
from yass import read_config
from yass.augment import (make_training_data, save_detect_network_params,
                          save_triage_network_params, save_ae_network_params)
from yass.neuralnetwork import train_detector, train_ae, train_triage


def train_neural_networks(CONFIG, spike_train, chosen_templates, min_amp,
                          nspikes, n_filters_detect, n_iter, n_batch,
                          l2_reg_scale, train_step_size, detectnet_name,
                          n_filters_triage, triagenet_name, n_features,
                          ae_name):
    """Train all neural networks

    Parameters
    ----------
    """

    # generate training data
    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amp, nspikes)

    # train detector
    train_detector(x_detect, y_detect, n_filters_detect, n_iter, n_batch,
                   l2_reg_scale, train_step_size, detectnet_name)

    # save detector model parameters
    save_detect_network_params(filters=n_filters_detect,
                               size=x_detect.shape[1],
                               n_neighbors=x_detect.shape[2],
                               output_path=detectnet_name.replace('ckpt',
                                                                  'yaml'))

    # train triage
    train_triage(x_triage, y_triage, n_filters_triage, n_iter, n_batch,
                 l2_reg_scale, train_step_size, triagenet_name)

    # save triage model parameters
    save_triage_network_params(filters=n_filters_triage,
                               size=x_detect.shape[1],
                               n_neighbors=x_detect.shape[2],
                               output_path=triagenet_name.replace('ckpt',
                                                                  'yaml'))

    # train autoencoder
    train_ae(x_ae, y_ae, n_features, n_iter, n_batch, train_step_size, ae_name)

    # save autoencoder model parameters
    save_ae_network_params(n_input=x_ae.shape[1],
                           n_features=n_features,
                           output_path=ae_name.replace('ckpt', 'yaml'))
