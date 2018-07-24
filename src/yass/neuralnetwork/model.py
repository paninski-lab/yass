import logging

from yass.util import dict2yaml, get_version
from sklearn import metrics


class Model:
    """Neural Network model inherit from this class
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
        metadata = dict(yass_version=get_version)
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
