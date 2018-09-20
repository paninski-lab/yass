"""
Functions for evaluating models
"""
import pandas as pd
import numpy as np

from yass.augment.make import spikes
from yass.util import ensure_iterator
from yass.templates.util import amplitudes as compute_amplitudes
from yass.templates.util import ptps as compute_ptps


class Dataset:
    """
    Class for manipulating spikes, it also provides a function
    to make a dataset with simulated data

    Parameters
    ----------
    data: numpy.ndarray (n_observations, waveform_length, n_channels)
        The data
    """

    def __init__(self, data, slices, units_per_bin,
                 data_clean=None, spatial_sig=None, temporal_sig=None):
        # datasets
        self.data = data
        self.data_clean = data_clean

        # These are computed on demand
        self._amplitudes = None
        self._ptps = None
        self._df = None
        self._df_clean = None

        # other properties
        self.slices = slices
        self.units_per_bin = units_per_bin
        self.spatial_sig = spatial_sig
        self.temporal_sig = temporal_sig

    @property
    def amplitudes(self):
        if self._amplitudes is None:
            self._amplitudes = compute_amplitudes(self.data)

        return self._amplitudes

    @property
    def ptps(self):
        if self._ptps is None:
            self._ptps = compute_ptps(self.data)

        return self._ptps

    @property
    def df(self):
        if self._df is None:
            self._df = to_data_frame(self.data, self.amplitudes,
                                     self.ptps, self.slices,
                                     self.units_per_bin)
        return self._df

    @property
    def df_clean(self):
        if self.data_clean is None:
            raise ValueError('Cannot access to clean data frame if no clean '
                             'data was provided in the constructor')

        if self._df_clean is None:
            self._df_clean = to_data_frame(self.data_clean,
                                           self.amplitudes,
                                           self.ptps, self.slices,
                                           self.units_per_bin)
        return self._df_clean

    @classmethod
    def make(cls, units_per_bin, include_clean_data=False, *args, **kwargs):
        # make test dataset
        (data_clean, data_noisy,
         amplitudes, slices,
         spatial_sig,
         temporal_sig) = spikes(*args, **kwargs)

        return cls(data_noisy, slices, units_per_bin,
                   data_clean if include_clean_data else None,
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

    def _make_from_kind(self, kind, units_per_bin):
        slice_ = self.slices[kind]

        data = self.data[slice_]
        data_clean = (None if self.data_clean is None
                      else self.data_clean[slice_])
        slices = {kind: slice_}

        # TODO: create another constructor that only takes the indices
        # to create the new Dataset, to avoid redoing computations
        # and to keep new columnds added to the dfs
        return Dataset(data, slices,
                       units_per_bin if units_per_bin
                       is not None else self.units_per_bin,
                       data_clean)

    @ensure_iterator('kind')
    def get_kind(self, kind, units_per_bin=None):
        if len(kind) == 1:
            return self._make_from_kind(kind[0], units_per_bin)
        else:
            return [self._make_from_kind(k, units_per_bin) for k in kind]


def to_data_frame(array, amplitudes, ptps, slices,
                  units_per_bin=10):
    wfs = [a for a in array]

    kinds = [[kind] * (slice_.stop - slice_.start) for
             kind, slice_ in slices.items()]
    kinds = [item for sublist in kinds for item in sublist]

    data = {'waveform': wfs, 'amplitude': amplitudes, 'kind': kinds,
            'ptp': ptps}

    if units_per_bin is not None:
        amplitude_groups = discretize(amplitudes, units_per_bin)
        ptp_groups = discretize(ptps, units_per_bin)
        data['amplitude_group'] = amplitude_groups
        data['ptp_group'] = ptp_groups

    df = pd.DataFrame(data=data)
    return df


def discretize(amplitudes, units_per_bin):
    range_ = float(np.max(amplitudes) - np.min(amplitudes))
    bins = int(range_ / units_per_bin)

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
