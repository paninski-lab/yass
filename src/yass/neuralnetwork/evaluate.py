"""
Functions for evaluating models
"""
import pandas as pd
import numpy as np

from yass.augment.make import spikes
from yass.util import ensure_iterator
from yass.templates.util import amplitudes as compute_amplitudes


class Dataset:
    """
    Class for manipulating spikes, it also provides a function
    to make a dataset with simulated data
    """

    def __init__(self, data_clean, data_noisy, slices,
                 amplitude_units_per_bin,
                 spatial_sig=None, temporal_sig=None):
        # make test dataset
        self.data_clean = data_clean
        self.data_noisy = data_noisy
        self.amplitudes = compute_amplitudes(data_clean)
        self.slices = slices
        self.amplitude_units_per_bin = amplitude_units_per_bin
        self.spatial_sig = spatial_sig
        self.temporal_sig = temporal_sig

        # convert to data frame
        self.df_noisy = to_data_frame(self.data_noisy, self.amplitudes,
                                      self.slices, amplitude_units_per_bin)

        self.df_clean = to_data_frame(self.data_clean, self.amplitudes,
                                      self.slices, amplitude_units_per_bin)

    @classmethod
    def make(cls, amplitude_units_per_bin, *args, **kwargs):
        # make test dataset
        (data_clean, data_noisy,
         amplitudes, slices,
         spatial_sig,
         temporal_sig) = spikes(*args, **kwargs)

        return cls(data_clean, data_noisy, slices, amplitude_units_per_bin,
                   spatial_sig, temporal_sig)

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

    def _make_from_kind(self, kind, amplitude_units_per_bin):
        slice_ = self.slices[kind]

        data_clean = self.data_clean[slice_]
        data_noisy = self.data_noisy[slice_]
        slices = {kind: slice_}

        # TODO: create another constructor that only takes the indices
        # to create the new Dataset, to avoid redoing computations
        # and to keep new columnds added to the dfs
        return Dataset(data_clean, data_noisy, slices,
                       amplitude_units_per_bin if amplitude_units_per_bin
                       is not None else self.amplitude_units_per_bin)

    @ensure_iterator('kind')
    def get_kind(self, kind, amplitude_units_per_bin=None):
        if len(kind) == 1:
            return self._make_from_kind(kind[0], amplitude_units_per_bin)
        else:
            return [self._make_from_kind(k) for k in kind]


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


def _process_group(group_id, group, predict_function, metric_fn):
    """Process a group in a grouped dataframe
    """
    # TODO: remove try-except when old module is removed

    # get waveforms
    wfs = np.stack(group.waveform.values, axis=0)

    # make predictions
    try:
        # old module
        preds = predict_function(wfs)
    except Exception:
        # keras
        preds = np.squeeze(predict_function(wfs[:, :, :, np.newaxis]))

    # compute metric
    metric = metric_fn(preds, group)

    return group_id, metric


def compute_per_group(df, column, predict_function, metric_fn):
    """Compute a metric over groups in a dataframe
    """
    # group and compute proportion of correct predictions
    vals = [_process_group(group_id, group, predict_function, metric_fn)
            for group_id, group
            in df.groupby(column)]

    group_ids, metric = list(zip(*vals))

    return group_ids, metric
