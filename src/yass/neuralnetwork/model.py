import warnings
import logging

from yass.util import dict2yaml, get_version
from sklearn import metrics
import numpy as np
from keras.models import load_model


class Model:
    """tensorflow based Neural Network model inherit from this class
    """

    def _validate_dimensions(self, waveform_length, n_neighbors=None):
        if self.waveform_length != waveform_length:
            raise ValueError('waveform length from network ({}) does not '
                             'match input data ({})'
                             .format(self.waveform_length,
                                     waveform_length))

        if n_neighbors is not None:
            if self.n_neighbors != n_neighbors:
                raise ValueError('number of n_neighbors from network ({}) '
                                 'does not match input data ({})'
                                 .format(self.n_neighbors,
                                         n_neighbors))

    def _save_params(self, path, params):
        metadata = dict(yass_version=get_version())
        params['metadata'] = metadata

        dict2yaml(output_path=path, **params)

    def _evaluate(self):
        y_pred = self.predict(self.x_test)

        m = dict()

        _ = metrics.confusion_matrix(self.y_test, y_pred)
        cm = _ / _.sum(axis=1)
        m['tn'], m['fp'], m['fn'], m['tp'] = cm.flatten()

        m['acc'] = metrics.accuracy_score(self.y_test, y_pred)
        m['prec'] = metrics.precision_score(self.y_test, y_pred)
        m['rec'] = metrics.recall_score(self.y_test, y_pred)

        logger = logging.getLogger(__name__)

        logger.info('Test set metrics:\n\tTN: {tn}\n\tFP: {fp}\n\tFN: {fn}\n\t'
                    'TP: {tp}\n\tAccuracy: {acc}\n\tPrecision: {prec}\n\t'
                    'Recall: {rec}'
                    .format(**m))

        return m

    def _save_test_set(self):
        no_ext = self.path_to_model.replace('.ckpt', '')
        path_to_x_test = '{}-x-test.npy'.format(no_ext)
        path_to_y_test = '{}-y-test.npy'.format(no_ext)

        logger = logging.getLogger(__name__)
        logger.info('Saving x_test at %s', path_to_x_test)
        logger.info('Saving y_test at %s', path_to_y_test)

        np.save(path_to_x_test, self.x_test)
        np.save(path_to_y_test, self.y_test)

    def _load_test_set(self):
        no_ext = self.path_to_model.replace('.ckpt', '')
        path_to_x_test = '{}-x-test.npy'.format(no_ext)
        path_to_y_test = '{}-y-test.npy'.format(no_ext)

        logger = logging.getLogger(__name__)
        logger.info('Loading x_test at %s', path_to_x_test)
        logger.info('Loading y_test at %s', path_to_y_test)

        self.x_test = np.load(path_to_x_test)
        self.y_test = np.load(path_to_y_test)


class KerasModel:
    """Wrapper for keras model
    """

    def __init__(self, path_to_model, allow_longer_waveform_length=False,
                 allow_more_channels=False):
        self._model = load_model(path_to_model)

        self.path_to_model = path_to_model
        self.allow_longer_waveform_length = allow_longer_waveform_length
        self.allow_more_channels = allow_more_channels

        _, self.waveform_length, self.n_neighbors, _ = self._model.input_shape

    def predict_with_threshold(self, x, threshold, **kwargs):
        """Make predictions with threshold
        """
        _, waveform_length_in, n_neighbors_in = x.shape

        if not self.allow_longer_waveform_length:
            self._validate_waveform_length(waveform_length_in)
        else:
            if waveform_length_in > self.waveform_length:
                warnings.warn("Input waveform length ({}) is larger than "
                              "Network's ({}) loaded from: ({}), using only "
                              "({}) middle observations for predition"
                              .format(waveform_length_in,
                                      self.waveform_length,
                                      self.path_to_model,
                                      self.waveform_length))

                mid = int((waveform_length_in - 1)/2)
                R = int((self.waveform_length - 1)/2)
                x = x[:, mid - R: mid + R + 1, :]

            elif waveform_length_in < self.waveform_length:
                raise ValueError("Input waveform length ({}) is shorter than "
                                 "Network's ({}), cannot make predictions"
                                 .format(waveform_length_in,
                                         self.waveform_length))

        if not self.allow_more_channels:
            self._validate_n_neighbors(n_neighbors_in)
        else:
            if n_neighbors_in > self.n_neighbors:
                warnings.warn("Input number of neighbors ({}) is larger than "
                              "Network's ({}) loaded from: ({}), using only "
                              "first ({}) neighbors for predition"
                              .format(n_neighbors_in,
                                      self.n_neighbors,
                                      self.path_to_model,
                                      self.n_neighbors))

                x = x[:, :, :self.n_neighbors]

            elif n_neighbors_in < self.n_neighbors:
                raise ValueError("Input number of neighbors({}) is shorter "
                                 "than Network's ({}), cannot make predictions"
                                 .format(n_neighbors_in,
                                         self.n_neighbors))

        x = x[:, :, :, np.newaxis]

        return np.squeeze(self._model.predict_proba(x, **kwargs) > threshold)

    def _validate_waveform_length(self, waveform_length):
        if self.waveform_length != waveform_length:
            raise ValueError('waveform length from network ({}) does not '
                             'match input data ({})'
                             .format(self.waveform_length,
                                     waveform_length))

    def _validate_n_neighbors(self, n_neighbors):
        if self.n_neighbors != n_neighbors:
            raise ValueError('number of n_neighbors from network ({}) '
                             'does not match input data ({})'
                             .format(self.n_neighbors,
                                     n_neighbors))
