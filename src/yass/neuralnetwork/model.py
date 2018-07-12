from yass.util import dict2yaml, get_version


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
