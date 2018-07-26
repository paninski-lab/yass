"""
Functions for evaluating models
"""
import pandas as pd
import numpy as np

from yass.augment.make import spikes
from yass.util import ensure_iterator


class TestSet:
    """Test set object
    """

    def __init__(self, amplitude_units_per_bin, *args, **kwargs):
        # make test dataset
        (self.data_clean, self.data_noisy,
         self.amplitudes, self.slices) = spikes(*args, **kwargs)

        # convert to data frame
        self.df_noisy = to_data_frame(self.data_noisy, self.amplitudes,
                                      self.slices, amplitude_units_per_bin)

        self.df_clean = to_data_frame(self.data_clean, self.amplitudes,
                                      self.slices, amplitude_units_per_bin)

    @property
    def kinds(self):
        return self.slices.keys()

    @property
    def stats(self):
        return {k: s.stop - s.start for k, s in self.slices.items()}

    @property
    def counts(self):
        """Return counts per kinds
        """
        pass

    @ensure_iterator('kind')
    def get_kind_clean(self, kind):
        return np.concatenate([self.data_clean[self.slices[k]] for k in kind])

    @ensure_iterator('kind')
    def get_kind_noisy(self, kind):
        return np.concatenate([self.data_noisy[self.slices[k]] for k in kind])

    def compute_per_group(function):
        pass


def to_data_frame(array, amplitudes, slices, amplitude_units_per_bin=10):
    wfs = [a for a in array]

    kinds = [[kind] * (slice_.stop - slice_.start) for
             kind, slice_ in slices.items()]
    kinds = [item for sublist in kinds for item in sublist]

    data = {'waveform': wfs, 'amplitude': amplitudes, 'kind': kinds}

    if amplitude_units_per_bin:
        amplitude_groups = discretize(amplitudes, amplitude_units_per_bin)
        data['amplitude_group'] = amplitude_groups

    df = pd.DataFrame(data=data)
    return df


def discretize(amplitudes, amplitude_units_per_bin):
    range_ = int(np.max(amplitudes) - np.min(amplitudes))
    bins = int(range_ / amplitude_units_per_bin)

    discretized = pd.qcut(amplitudes, bins, duplicates='drop')

    return [interval.right for interval in discretized]


def split(x, y, train_proportion=0.7):
    n = x.shape[0]

    train_idx = np.random.choice(n, size=int(n * train_proportion),
                                 replace=False)

    test_idx = np.ones(n, dtype=bool)
    test_idx[~train_idx] = False

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]
