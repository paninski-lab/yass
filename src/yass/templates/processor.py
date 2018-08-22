import os

import numpy as np

from yass.templates.util import get_templates, main_channels
from yass.templates.util import align as _align
from yass.geometry import order_channels_by_distance


# TODO: remove config
class TemplatesProcessor:
    """Provides functions for manipulating templates
    """

    def __init__(self, templates):
        self._update_templates(templates)

    @classmethod
    def from_spike_train(cls, CONFIG, half_waveform_length, spike_train,
                         path_to_data):

        # make sure standarized data already exists
        if not os.path.exists(path_to_data):
            raise ValueError('Standarized data does not exist in: {}, this is '
                             'needed to generate training data, run the '
                             'preprocesor first to generate it'
                             .format(path_to_data))

        n_spikes, _ = spike_train.shape

        # add weight of one to every spike
        weighted_spike_train = np.hstack((spike_train,
                                          np.ones((n_spikes, 1), 'int32')))

        # get templates
        templates, _ = get_templates(weighted_spike_train,
                                     path_to_data,
                                     CONFIG.resources.max_memory,
                                     half_waveform_length)

        templates = np.transpose(templates, (2, 1, 0))

        return cls(templates)

    def _update_templates(self, templates):
        self.templates = templates
        self.amplitudes = np.max(np.abs(templates), axis=(1, 2))
        self.main_channels = main_channels(templates)

    def _check_half_waveform_length(self, half_waveform_length):
        _, current_waveform_length, _ = self.templates.shape

        if half_waveform_length > current_waveform_length:
            raise ValueError('New half_waveform_length ({}) must be smaller'
                             'than current half_waveform_length ({})'
                             .format(half_waveform_length,
                                     current_waveform_length))

    def choose_with_indexes(self, indexes, inplace=False):
        """
        Keep only selected templates and from those, only the ones above
        certain value

        Returns
        -------
        """
        try:
            chosen_templates = self.templates[indexes]
        except IndexError:
            raise IndexError('Error getting chosen_templates, make sure '
                             'the ids exist')

        if inplace:
            self._update_templates(chosen_templates)
        else:
            return TemplatesProcessor(chosen_templates)

    def choose_with_minimum_amplitude(self, minimum_amplitude, inplace=False):
        chosen_templates = self.templates[self.amplitudes > minimum_amplitude]

        if inplace:
            self._update_templates(chosen_templates)
        else:
            return TemplatesProcessor(chosen_templates)

    def crop_temporally(self, half_waveform_length, inplace=False):

        self._check_half_waveform_length(half_waveform_length)

        _, current_waveform_length, _ = self.templates.shape
        mid_point = int(current_waveform_length/2)
        MID_POINT_IDX = slice(mid_point - half_waveform_length,
                              mid_point + half_waveform_length + 1)

        new_templates = self.templates[:, MID_POINT_IDX, :]

        if inplace:
            self._update_templates(new_templates)
        else:
            return TemplatesProcessor(new_templates)

    def align(self, half_waveform_length, inplace=False):
        # deactivated, need to fix align function
        pass
        # self._check_half_waveform_length(half_waveform_length)
        # new_templates = _align(self.templates, half_waveform_length)

        # if inplace:
        #   self._update_templates(new_templates)
        # else:
        #   return TemplatesProcessor(new_templates)

    def crop_spatially(self, neighbors, geometry, inplace=False):
        n_templates, waveform_length, _ = self.templates.shape

        # spatially crop (only keep neighbors)
        n_neigh_to_keep = np.max(np.sum(neighbors, 0))
        new_templates = np.zeros((n_templates, waveform_length,
                                  n_neigh_to_keep))

        for k in range(n_templates):

            # get neighbors for the main channel in the kth template
            ch_idx = np.where(neighbors[self.main_channels[k]])[0]

            # order channels
            ch_idx, _ = order_channels_by_distance(self.main_channels[k],
                                                   ch_idx, geometry)

            # new kth template is the old kth template by keeping only
            # ordered neighboring channels
            new_templates[k, :,
                          :ch_idx.shape[0]] = self.templates[k][:, ch_idx]

        if inplace:
            self._update_templates(new_templates)
        else:
            return TemplatesProcessor(new_templates)

    @property
    def values(self):
        return self.templates
