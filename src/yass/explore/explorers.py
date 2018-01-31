import collections
from math import sqrt, ceil, floor
from functools import partial
import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from yass import geometry as geom

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Circle
except ImportError:
    _matplotlib = False
    _matplotlib_message = ('Matplotlib is required to run this function, '
                           'install it and try again')
else:
    _matplotlib = True
    _matplotlib_message = None

from ..util import ensure_iterator, sample, requires
from ..batch import RecordingsReader
from .table import Table


# TODO: use functions in util module
def _is_iter(obj):
    return isinstance(obj, collections.Iterable)


# TODO: use functions in util module
def _grid_size(group_ids, max_cols=None):
    total = len(group_ids)
    sq = sqrt(total)
    cols = int(floor(sq))
    rows = int(ceil(sq))
    rows = rows + 1 if rows * cols < len(group_ids) else rows

    if max_cols and cols > max_cols:
        rows = ceil(total/max_cols)
        cols = max_cols

    return rows, cols


# TODO: use functions in util module
def _make_grid_plot(fn, group_ids, ax, sharex, sharey, max_cols=None):
    rows, cols = _grid_size(group_ids, max_cols)

    f, axs = ax.subplots(rows, cols, sharex=sharex, sharey=sharey)

    axs = axs if _is_iter(axs) else [axs]

    if cols > 1:
        axs = [item for sublist in axs for item in sublist]

    for g, ax in zip(group_ids, axs):
        fn(group_id=g, ax=ax)


# TODO: refactor this, templates should be computed from raw data instead
# of being passed as parameter, or maybe allow both ways?
class SpikeTrainExplorer(object):
    """Explore spike trains and templates

    Parameters
    ----------
    spike_train: str or np.ndarray
        Path to npy spike train file or spike train np.ndarray. The first
        column of the file should be the spike index and the second the spike
        ID
    recording_explorer: RecordingExplorer
        Recording explorer instance
    templates: np.ndarray, optional
        Templates, if None, templates are computed using the recording explorer
    projection_matrix: np.ndarray, optional
        Projection Matrix, if None, methods that return scores will not work
    """

    def __init__(self, spike_train, recording_explorer,
                 templates=None, projection_matrix=None):

        self.logger = logging.getLogger(__name__)

        if isinstance(spike_train, str):
            name, extension = spike_train.split('.')

            if extension == 'csv':
                self.spike_train = np.loadtxt(spike_train,
                                              dtype='int32', delimiter=',')
            elif extension == 'npy':
                self.spike_train = np.load(spike_train)
            else:
                raise ValueError('Unsupported extension: {}'.format(extension))
        else:
            self.spike_train = spike_train

        self.all_ids = np.unique(self.spike_train[:, 1])
        self.recording_explorer = recording_explorer
        self.projection_matrix = projection_matrix

        if templates is not None:
            self.templates = templates
        else:
            self.templates, self.weights = self._compute_templates()

        self._spike_groups = {id_: self.spike_train[self.spike_train[:, 1] ==
                              id_, 0] for id_ in self.all_ids}

        if projection_matrix is not None:
            ft_space = self._reduce_dimension
            self.templates_feature_space = ft_space(self.templates)
        else:
            self.templates_feature_space = None

    def _reduce_dimension(self, data, flatten=False):
        """Reduce dimensionality
        """
        R, n_features = self.projection_matrix.shape
        nchannel, R, n_data = data.shape

        reduced = np.transpose(np.reshape(np.matmul(np.reshape(np.transpose(
            data, [0, 2, 1]), (-1, R)), [self.projection_matrix]),
            (nchannel, n_data, n_features)), (0, 2, 1))

        if flatten:
            reduced = np.reshape(reduced, [reduced.shape[0], -1])

        return reduced

    def _compute_templates(self):
        """Compute templates from spike train

        Returns
        -------
        numpy.ndarray
            [n_channels, n_features, n_templates] array with the templates
        numpy.ndarray
            1D array with the number of spikes used to form the ith template
        """
        # get waveforms
        waveforms = [self.waveforms_for_group(id_) for id_ in self.all_ids]

        self.logger.debug('Loaded {:,} waveforms'.format(len(waveforms)))

        # compute templates for every group
        templates = [np.mean(w, axis=0) for w in waveforms]

        # get weights (number of waveforms used to compute each template)
        weights = np.array([len(w) for w in waveforms])

        # stack
        templates = np.stack(templates, axis=2)

        self.logger.debug('Finished computing templates, shape is: {}'
                          .format(templates.shape))

        # return transposed templates (makes plotting easier) and weights
        return templates.transpose(1, 0, 2), weights

    @property
    def spike_groups(self):
        """Grouped spike times by ID

        Returns
        -------
        dictionary
            Dictionary where each key is the group ID and the value is a 1D
            numpy.ndarray with the spike times for that group
        """
        return self._spike_groups

    @ensure_iterator('group_ids')
    def scores_for_groups(self, group_ids, channels, flatten=True):
        """Get scores for one or more groups

        Parameters
        ----------
        group_id: int or list
            The id for one or more group
        channels: list
            Which channels to return
        flatten: bool, optional
            Flatten scores along channels, defaults to True
        """
        waveforms = [self.waveforms_for_group(g, channels) for g in group_ids]
        lengths = np.hstack([np.ones(w.shape[0])*g for g, w in zip(group_ids,
                                                                   waveforms)])

        return self._reduce_dimension(np.vstack(waveforms),
                                      flatten=flatten), lengths

    def times_for_group(self, group_id):
        """Get the spiking times for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        matches_group = self.spike_train[:, 1] == group_id
        return self.spike_train[matches_group][:, 0]

    def main_channel_for_group(self, group_id):
        """Get the main channel for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        template = self.templates[:, :, group_id]
        main = np.argmax(np.max(template, axis=1))
        return main

    def neighbor_channels_for_group(self, group_id):
        """Get the neighbor channels for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        main = self.main_channel_for_group(group_id)
        neigh_matrix = self.recording_explorer.neigh_matrix
        return np.where(neigh_matrix[main])[0]

    def template_for_group(self, group_id):
        """Get the template for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        return self.templates[:, :, group_id]

    def waveforms_for_group(self, group_id, channels='all'):
        """Get the waveforms for a group in selected channels

        Parameters
        ----------
        group_id: int
            The id for the group
        """

        if isinstance(channels, str) and channels == 'all':
            channels = range(self.recording_explorer.n_channels)

        # get all spike times that form this group
        times = self.times_for_group(group_id)

        # get waveforms in selected channels
        read_wf = self.recording_explorer.read_waveform
        return np.stack([read_wf(t, channels) for t in times])

    def close_templates(self, group_id, k):
        """return K similar templates
        """
        difference = np.sum(np.square(self.templates -
                                      self.templates[:, :, [group_id]]),
                            axis=(0, 1))
        close_to_far_idx = np.argsort(difference)
        return close_to_far_idx[:k+1]

    def _plot_template(self, group_id, ax=None):
        """Plot a single template
        """
        ax = ax if ax else plt.gca()
        template = self.template_for_group(group_id)
        ax.plot(template.T)
        ax.set_title('Template {}'.format(group_id))
        plt.tight_layout()

    @requires(_matplotlib, _matplotlib_message)
    def plot_templates(self, group_ids, ax=None, sharex=True, sharey=False):
        """Plot templates

        Parameters
        ----------
        group_ids: int or list or str
            Groups to plot, it can be either a single group, a list of groups
            or 'all'
        """
        ax = ax if ax else plt

        if isinstance(group_ids, str) and group_ids == 'all':
            group_ids = self.all_ids

        group_ids = group_ids if _is_iter(group_ids) else [group_ids]
        _make_grid_plot(self._plot_template, group_ids, ax, sharex, sharey)

    @requires(_matplotlib, _matplotlib_message)
    def plot_pca(self, group_ids, channels, sample=None, ax=None):
        """
        Reduce dimensionality using PCA and plot data
        """
        ax = ax if ax else plt

        scores, labels = self.scores_for_groups(group_ids, channels)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scores)

        for color in np.unique(group_ids).astype('int'):
            x = reduced[labels == color, 0]
            y = reduced[labels == color, 1]

            if sample:
                x = np.random.choice(x, size=int(sample*len(x)), replace=False)
                y = np.random.choice(x, size=int(sample*len(y)), replace=False)

            plt.scatter(x, y, label='Group {}'.format(color, alpha=0.7))

        ax.legend()

    @requires(_matplotlib, _matplotlib_message)
    def plot_lda(self, group_ids, channels, sample=None, ax=None):
        """
        Reduce dimensionality using LDA and plot data
        """
        ax = plt if ax is None else ax

        scores, labels = self.scores_for_groups(group_ids, channels)

        lda = LDA(n_components=2)
        reduced = lda.fit_transform(scores, labels)

        for color in np.unique(group_ids).astype('int'):
            x = reduced[labels == color, 0]
            y = reduced[labels == color, 1]

            if sample:
                x = np.random.choice(x, size=int(sample*len(x)), replace=False)
                y = np.random.choice(x, size=int(sample*len(y)), replace=False)

            ax.scatter(x, y, label='Group {}'.format(color), alpha=0.7)

        ax.legend()

    @requires(_matplotlib, _matplotlib_message)
    def plot_closest_clusters_to(self, group_id, k, mode='LDA',
                                 sample=None, ax=None):
        """Visualize close clusters
        """
        ax = plt if ax is None else ax

        # get similar templates to template with id group_id
        groups = self.close_templates(group_id, k)

        # get the neighbors for the main channel in group_id
        main = self.main_channel_for_group(group_id)
        neighbors = self.recording_explorer.neighbors_for_channel(main)

        if mode == 'LDA':
            self.plot_lda(groups, neighbors, sample=sample, ax=ax)
        elif mode == 'PCA':
            self.plot_pca(groups, neighbors, sample=sample, ax=ax)
        else:
            raise ValueError('Only PCA and LDA modes are supported')

    @requires(_matplotlib, _matplotlib_message)
    def plot_closest_templates_to(self, group_id, k, ax=None,
                                  sharex=True, sharey=False):
        """Visualize close templates
        """
        ax = plt if ax is None else ax

        groups = self.close_templates(group_id, k)
        self.plot_templates(groups, ax=ax, sharex=sharex, sharey=sharey)

    @requires(_matplotlib, _matplotlib_message)
    def plot_all_clusters(self, k, mode='LDA', sample=None, ax=None,
                          sharex=True, sharey=False, max_cols=None):
        ax = plt if ax is None else ax

        fn = partial(self.plot_closest_clusters_to, k=k, mode=mode,
                     sample=sample)

        _make_grid_plot(fn, self.all_ids, ax, sharex, sharey, max_cols)

    @requires(_matplotlib, _matplotlib_message)
    def plot_waveforms_and_clusters(self, group_id, ax=None):
        """
        """
        ax = plt if ax is None else ax

        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=3)
        ax2 = plt.subplot2grid((6, 6), (0, 3), colspan=3)
        ax3 = plt.subplot2grid((6, 6), (1, 3), colspan=3, sharey=ax2)
        ax4 = plt.subplot2grid((6, 6), (2, 3), colspan=3, sharey=ax3)

        self.plot_closest_clusters_to(group_id, k=2, ax=ax1)

        close = self.close_templates(group_id, k=2)

        for ax, template in zip([ax2, ax3, ax4], close):
            self._plot_template(template, ax=ax)

    def _stats_for_group(self, group_id):
        """Return some summary statistics for a single group

        Returns
        -------
        dict
            Dict with id, range, max, min and main channel for the selected
            group
        """
        template = self.template_for_group(group_id)
        max_ = np.max(template)
        min_ = np.min(template)
        range_ = max_ - min_
        main_channel = self.main_channel_for_group(group_id)

        return dict(id=group_id, range=range_, max=max_, min=min_,
                    main_channel=main_channel)

    @ensure_iterator('group_ids')
    def stats_for_groups(self, group_ids):
        """Return some summary statistics for certain groups
        """
        stats = [self._stats_for_group(g) for g in group_ids]
        content = [s.values() for s in stats]
        header = stats[0].keys()
        return Table(content=content, header=header)

    def stats_for_closest_groups_to(self, group_id, k):
        """Return some summary statistics a given group and its k neighbors
        """
        groups = self.close_templates(group_id, k)
        return self.stats_for_groups(groups)


# TODO: documentation, proper errors for optional parameters, check
# plotting functions (matplotlib gca and all that stuff)
class RecordingExplorer(object):
    """Explore neural recordings: observations, geometry and waveforms

    Parameters
    ----------
    path_to_recordings: str
        Path to binary recordings file
    spike_size: int
        Spike size, this is used to get waveforms around points in the
        recordings
    path_to_geom: str, optional
        Path to geometry file (npy or csv)
    neighbor_radius: float
        Maximum radius to consider two channels as neighbors
    dtype: str
        Recordings dtype
    n_channels: int
        Number of channels in the recordings
    data_format: str
        Data format, it can be either 'long' (observations, channels) or
        'wide' (channels, observations)
    mmap: bool
        Whether to read the data using numpy.mmap, otherwise it reads
        the data using numpy.fromfile
    waveform_dtype: str, optional
        Waveform output type, defaults to 'default' which matches
        recordings dtype
    """

    def __init__(self, path_to_recordings, path_to_geom=None, spike_size=None,
                 neighbor_radius=None, dtype=None, n_channels=None,
                 data_format=None, mmap=True, waveform_dtype='float32'):
        self.data = RecordingsReader(path_to_recordings, dtype, n_channels,
                                     data_format, mmap, output_shape='long')

        if path_to_geom is not None:
            self.geom = geom.parse(path_to_geom, n_channels)
            self.neighbor_radius = neighbor_radius
            self.neigh_matrix = geom.find_channel_neighbors(self.geom,
                                                            neighbor_radius)
        self.n_channels = self.data.channels
        self.spike_size = spike_size

        if waveform_dtype == 'default':
            waveform_dtype = dtype

        self.waveform_dtype = waveform_dtype

        self.logger = logging.getLogger(__name__)

    def neighbors_for_channel(self, channel):
        """Get the neighbors for the channel

        Returns
        -------
        numpy.array
            An array containing the neighbors of the given channel
        """
        return np.where(self.neigh_matrix[channel])[0]

    def read_waveform(self, time, channels='all'):
        """
        Read a single waveform of size 2*spike_size + 1 observations around
        the given time

        Parameters
        ----------
        time: int
            Waveform center

        Returns
        -------
        numpy.array
            A (2 * spike_size + 1, channels) 2D array with the waveform around
            the given time
        """
        start = time - self.spike_size
        end = time + self.spike_size + 1

        if isinstance(channels, str) and channels == 'all':
            wf = self.data[start:end, :].astype(self.waveform_dtype)
        else:
            wf = self.data[start:end, channels].astype(self.waveform_dtype)

        if len(wf) != 2 * self.spike_size + 1:
            raise ValueError('Cannot read waveform at time {}, there is not '
                             'enough data to draw a complete waveform ({} '
                             'observations are needed to the left and to the '
                             'right)'.format(time, self.spike_size))

        return wf

    def read_waveforms(self, times, channels='all', flatten=False):
        """Read multiple waveforms around certain times

        Parameters
        ----------
        times: array-like
            1D array containing the times
        channels: array-like, optional
            Channels to be included in the waveforms, if 'all' it uses all
            channels
        flatten: bool, optional
            Whether to flat the first to dimensions, defaults to True

        Returns
        -------
        numpy.array
            A (times, 2 * spike_size + 1, channels) 3D array with the waveforms
            around the given times. If flatten is True, ir returns a
            (times * 2 * spike_size + 1, channels) 2D array
        """
        # TODO: may be faster by sending :?
        if isinstance(channels, str) and channels == 'all':
            channels = range(self.n_channels)

        total = len(times)
        wfs = np.empty((total, self.spike_size * 2 + 1, len(channels)),
                       dtype=self.waveform_dtype)

        for i, t in enumerate(times):
            wfs[i, :, :] = self.read_waveform(t, channels)

            if i % 10000 == 0 and i > 0:
                self.logger.info('Loaded {:,}/{:,} waveforms...'
                                 .format(i, total))

        self.logger.info('Loaded all {:,} waveforms...'.format(total))

        if flatten:
            self.logger.debug('Flattening waveforms...')
            wfs = wfs.reshape(wfs.shape[0], -1)

        return wfs

    def read_waveform_around_channel(self, time, channel):
        """
        Read a single waveform around a given time and using the neighbors of
        the given channel

        Parameters
        ----------
        time: int
            Waveform center
        channel: int
            The waveform will only contain the channels that are neighbors
            of this channel

        Returns
        -------
        numpy.array
            A (2 * spike_size + 1, neighbors) 2D array with the waveform around
            the given time where neighbors is the number of neighbors of the
            given channel
        """
        return self.read_waveform(time,
                                  channels=self.neighbors_for_channel(channel))

    def read_waveform_around_main_channel(self, time):
        """
        Read a single waveform around a given time and using the neighbors of
        the channel with maximum amplitude

        Parameters
        ----------
        time: int
            Waveform center

        Returns
        -------
        numpy.array
            A (2 * spike_size + 1, neighbors) 2D array with the waveform around
            the given time where neighbors is the number of neighbors of the
            channel with maximum amplitude
        """

        # read waveform along every channel and find the one with max amplitude
        wf = self.read_waveform(time=time, channels='all')
        channel = np.argmax(np.max(wf, axis=0))

        # read waveform around the max amplitude channel
        return self.read_waveform(time,
                                  channels=self.neighbors_for_channel(channel))

    def main_channel_for_waveforms(self, waveforms):
        """Get the main channel (biggest amplitude) for the given waveforms

        Parameters
        ----------
        waveforms: numpy.ndarray
            An array of size (n_temporal_features, n_channels) if only one
            waveform or one of size (n_waveforms, n_temporal_features,
            n_channels) if more than one waveform

        Returns
        -------
        int/numpy.ndarray
            The main channel for every waveform, int if only one waveform or
            a 1D array if more than one
        """
        if waveforms.ndim == 2:
            return np.argmax(np.max(waveforms, axis=0), axis=0)
        elif waveforms.ndim == 3:
            return np.argmax(np.max(waveforms, axis=1), axis=1)
        else:
            raise ValueError('waveforms must be an 2D or 3D array')

    @requires(_matplotlib, _matplotlib_message)
    def plot_waveform(self, time, channels, ax=None, line_at_t=False,
                      overlay=False):
        """
        Plot a waveform around a window size in selected channels
        """
        ax = ax if ax else plt

        n_channels = len(channels)
        formatter = FuncFormatter(lambda x, pos: time - self.spike_size +
                                  int(x))

        if overlay:
            axs = [ax] * n_channels
        else:
            f, axs = ax.subplots(n_channels, 1)

        for ch, ax in zip(channels, axs):
            waveform = self.read_waveform(time, ch)
            ax.plot(waveform)
            ax.set_title('Channel {}'.format(ch), fontsize=12)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', which='major', labelsize=10)

            if line_at_t:
                ax.axvline(x=time)

        plt.tight_layout()

    @requires(_matplotlib, _matplotlib_message)
    def plot_waveform_around_channel(self, time, channel, ax=None,
                                     line_at_t=False, overlay=False):
        return self.plot_waveform(time,
                                  channels=self.neighbors_for_channel(channel),
                                  ax=ax, line_at_t=line_at_t, overlay=overlay)

    @requires(_matplotlib, _matplotlib_message)
    def plot_waveform_around_main_channel(self, time, ax=None,
                                          line_at_t=False, overlay=False):
        # read waveform along every channel and find the one with max amplitude
        wf = self.read_waveform(time=time, channels='all')
        channel = np.argmax(np.max(wf, axis=0))

        return self.plot_waveform(time,
                                  channels=self.neighbors_for_channel(channel),
                                  ax=ax, line_at_t=line_at_t, overlay=overlay)

    @requires(_matplotlib, _matplotlib_message)
    def plot_geometry(self, channel_label=True, neighbor_radius=False,
                      ax=None):
        """Plot geometry file
        """
        ax = ax if ax else plt.gca()

        x, y = self.geom.T
        colors = range(len(x))

        plt.scatter(x, y, c=colors)

        if channel_label:
            for x, y, i in zip(x, y, range(self.n_channels)):
                ax.text(x, y, i, fontsize=15)

        if neighbor_radius:
            for x, y in zip(x, y):
                c = Circle((x, y), self.neighbor_radius, color='r',
                           fill=False)
                ax.add_artist(c)

    @requires(_matplotlib, _matplotlib_message)
    def plot_clusters(self, times, sample_percentage=None, ax=None):
        """
        """
        ax = ax if ax else plt.gca()

        if sample_percentage:
            times = sample(times, sample_percentage)

        wfs = self.read_waveforms(times, flatten=True)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(wfs)

        ax.scatter(reduced[:, 0], reduced[:, 1])

    @requires(_matplotlib, _matplotlib_message)
    def plot_series(self, from_time, to_time, channels='all', ax=None):
        """Plot observations in a selected number of channels
        """

        if channels == 'all':
            channels = range(self.n_channels)

        ax = ax if ax else plt

        f, axs = plt.subplots(len(channels), 1)

        formatter = FuncFormatter(lambda x, pos: from_time + int(x))

        for ax, ch in zip(axs, channels):
            ax.plot(self.data[from_time:to_time, ch])
            ax.set_title('Channel {}'.format(ch), fontsize=25)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', which='major', labelsize=25)

        plt.tight_layout()
