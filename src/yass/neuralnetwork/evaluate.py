"""
Functions for evaluating models
"""
import pandas as pd
import numpy as np

from yass.augment.make import testing_data


class TestSet:
    """Test set object
    """

    def __init__(self, amplitude_units_per_bin, *args, **kwargs):
        # make test dataset
        self.data, self.amplitudes, self.slices = testing_data(*args, **kwargs)

        # convert to data frame
        self.df = to_data_frame(self.data, self.amplitudes, self.slices,
                                amplitude_units_per_bin)

    @property
    def kinds(self):
        return self.slices.keys()

    def get_kind(self, kind):
        return self.data[self.slices[kind]]

    def compute_per_group(function):
        pass


def to_data_frame(array, amplitudes, slices, amplitude_units_per_bin=10):
    wfs = [a for a in array]

    if amplitude_units_per_bin:
        amplitudes = discretize(amplitudes, amplitude_units_per_bin)

    kinds = [[kind] * (slice_.stop - slice_.start) for
             kind, slice_ in slices.items()]
    kinds = [item for sublist in kinds for item in sublist]

    df = pd.DataFrame(data={'waveform': wfs,
                            'amplitude': amplitudes,
                            'kind': kinds})
    return df


def discretize(amplitudes, amplitude_units_per_bin):
    range_ = int(np.max(amplitudes) - np.min(amplitudes))
    bins = int(range_ / amplitude_units_per_bin)

    discretized = pd.qcut(amplitudes, bins)

    return [interval.right for interval in discretized]
