class Model:
    """Neural Network model inherit from this class
    """
    def _validate_dimensions(self, waveform_length, n_neighbors):
        if self.waveform_length != waveform_length:
            raise ValueError('waveform length from network ({}) does not '
                             'match input data ({})'
                             .format(self.waveform_length,
                                     waveform_length))

        if self.n_neighbors != n_neighbors:
            raise ValueError('number of n_neighbors from network ({}) does '
                             'not match input data ({})'
                             .format(self.n_neighbors,
                                     n_neighbors))
