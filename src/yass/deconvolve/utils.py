"""Set of functions for visualization of mean waveform if action potentials."""

import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm



class Geometry(object):
    """Geometry Object for finidng closest channels."""
    def __init__(self, geometry):
        self.geom = geometry
        self.pdist = squareform(pdist(geometry))

    def neighbors(self, channel, size):
        return np.argsort(self.pdist[channel, :])[:size]


class WaveForms(object):

    def __init__(self, wave_forms, geometry=None):
        """Sets up and computes properties of wave forms.
        params:
        -------
        wave_forms: numpy.ndarray
            Shape of wave forms is (N, C, t). N is total number of wave forms
            C is number of channels and t is number of time points.
        geometry: numpy.ndarray
            Geometry of the probe that the wave forms belong to. Array has shape
            (N, 2) the coordinates of the probe.
        """
        self.wave_forms = wave_forms
        self.n_unit, self.n_channel, self.n_time = self.wave_forms.shape
        self.unit_overlap = None
        self.pdist = None
        self.geom = geometry
        self.main_chans = self.wave_forms.ptp(axis=2).argmax(axis=1)
        self.ptps = self.wave_forms.ptp(axis=2).max(axis=1)

    def pair_dist(self):
        """Pairwise distance of templates to each other."""
        if self.pdist is None: 
            # Align all waveforms to the one with largest peak to peak.
            self.pdist = np.zeros([self.n_unit, self.n_unit]) + np.inf
            max_ptp_unit = self.ptp().argmax()
            vis_chan = self.vis_chan()
            al_wf = self.align(
                    ref_wave_form=self.wave_forms[max_ptp_unit])
            for unit in range(self.n_unit):
                # Iterate over all units to find the best match.
                over_units = self.overlap()[unit]
                diff = al_wf[[unit]] - al_wf[over_units]
                diff = np.sqrt(np.square(diff).sum(axis=-1).sum(axis=-1))
                self.pdist[unit, over_units] = diff 

        return self.pdist

    def __getitem__(self, key):
        return self.wave_forms.__getitem__(key)

    def svd_reconstruct(self, temp_id, rank=3):
        """Reconstruct the wave forms by given id using SVD.
        params:
        -------
        temp_id: int or np.array
            template id(s) of the template to be reconstructed.
        rank: int
            Rank of the SVD reconstruction.
        returns:
        --------
        numpy.ndarray of shape (C, t) or (n, C, t) which is the SVD
        reconstructed version of the given wave forms.
        """
        u, h, v = np.linalg.svd(self.wave_forms[temp_id, :, :])
        if len(u.shape) == 3:
            # Multiple units at a time.
            return np.matmul(u[:, :, :rank] * h[:, None, :rank], v[:, :rank, :])

        return np.matmul(u[:, :rank] * h[:rank], v[:rank, :])

    def vis_chan(self, threshold=2.):
        """Computes boolean visibility matrix of the wave forms.
        params:
        -------
        threshold: float
            Threshold of visibility in terms of standard unit (SU).
        return:
        -------
        numpy.ndarray of shape (N, C).
        """
        return self.wave_forms.ptp(axis=-1) > threshold

    def overlap(self, threshold=2.):
        """Computes boolean spatial overlap of templates.
        params:
        -------
        threshold: float
            Threshold of visibility in terms of standard unit (SU).
        return:
        -------
        numpy.ndarray of shape (N, N).
        """
        if self.unit_overlap is None:
            vis = self.vis_chan()
            self.unit_overlap = np.sum(
                np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
            self.unit_overlap = self.unit_overlap > 0
        return self.unit_overlap
 
    def ptp(self, unit=None):
        """Returns ptp of wave forms in standard units.
        returns:
        --------
        numpy.array of size N.
        """
        if unit is None:
            return self.ptps
        return self.ptps[unit]

    def get_shifted_waveforms(self, shifts, clip_value):
        """Get shifted viersions of the wave forms given the amount of shifts.
        params:
        -------
        shifts: float or np.array.float
            List of shifts that indicated how much has to change.
        returns:
        --------
        numpy.ndarray of shifted wave forms.
        """
        unit_time_window = np.arange(
                self.n_time - 2 * clip_value) + shifts[:, None]
        default_range = np.arange(self.n_time - 2 * clip_value)
        sub_shifts = shifts - np.floor(shifts)
        shifts = np.floor(shifts).astype(np.int)

        def sub(i, shift, sub=None):
            if sub is None:
                return self.wave_forms[i, :, default_range + shift]
            return sub(i, shift) * sub + sub(i, shift + 1) * (1 - sub)

        if sub_shifts.sum() > 0.:
            # Linear interpolation.
            np.array(
                [sub(i, s, sub_shifts[i]) for i, s in enumerate(
                    shifts)]).transpose([0, 2, 1])

        return np.array(
                [sub(i, s) for i, s in enumerate(shifts)]).transpose([0, 2, 1])

    def align(self, ref_wave_form=None, jitter=3, upsample=1, return_shifts=False):
        """Aligns all the wave forms to the reference wave form.
        params:
        -------
        jitter: int
            How much jitter per wave form in subsample time is allowed.
        upsample: int
            Factor for interpolation of signals.
        """
        if jitter == 0:
            if return_shifts:
                return self.wave_forms + 0., np.zeros(self.n_unit, dtype=np.int32)
            else:
                return self.wave_forms + 0.
        if ref_wave_form is None:
            ref_wave_form = self.wave_forms.mean(axis=0)

        ptp = ref_wave_form.ptp(axis=1)
        max_chan = ptp.argmax()

        wf = self.wave_forms
        if upsample > 1:
            x_range = np.arange(0, self.n_time)
            f = interp1d(x_range, self.wave_forms)
            wf = f(x_range[:-1] + np.arange(0, 1, 1./upsample))

        # Upsample these guys
        ref = ref_wave_form[max_chan, jitter:-jitter]
        idx = np.arange(
                self.n_time - 2 * jitter) + np.arange(2 * jitter)[:, None]
        wf_ = self.wave_forms + 0.
        # wf_ /= wf_.ptp(-1)[..., None]
        all_shifts = wf_[:, max_chan, idx]
        all_dist = np.square(all_shifts - ref).sum(axis=-1)
        all_inv_dist = np.square(-all_shifts - ref).sum(axis=-1)
        inv_better_idx = np.where(all_inv_dist.min(axis=-1) < all_dist.min(axis=-1))[0]
        best_shift_idx = all_dist.argmin(axis=-1)
        best_inv_shift_idx = all_inv_dist.argmin(axis=-1)
        # 
        best_shift_idx[inv_better_idx] = best_inv_shift_idx[inv_better_idx]
        if return_shifts:
            return self.get_shifted_waveforms(best_shift_idx, clip_value=jitter), best_shift_idx
        return self.get_shifted_waveforms(best_shift_idx, clip_value=jitter)

    def generate_new_templates(self, base_rotation=3, scale_std=.1, translate_std=20.):
        """Creates new templates by rotation, scaling and translation.

        params:
        -------
        base_rotation: int
            Fraction of pi to be used for base rotation value. Multiples of
            this amount are used, up to 2 * pi to rotate templates spatially.
        """
        new_temps = np.zeros_like(self.wave_forms)
        scales = np.random.normal(1., scale_std, [self.n_unit, self.n_channel])
        max_rotation = base_rotation * 2
        rotations = np.random.randint(0, max_rotation, self.n_unit) * max_rotation
        translates = np.random.normal(0, translate_std, [self.n_unit, 2])
        for unit in range(self.n_unit):
            rot_matrix = Rotation.from_euler("x", rotations[unit]).as_dcm()[1:, 1:]
            x = np.matmul(self.geom.geom, rot_matrix) + translates[unit]
            # Find mapping of new geometry and the original geometry
            c_dist = cdist(x, self.geom.geom)
            new = np.array(c_dist.argmin(axis=1))
            seen = np.zeros(self.n_channel, dtype=bool)
            for i in c_dist.min(axis=1).argsort()[::-1]:
                if seen[new[i]]:
                    continue
                new_temps[unit, i] = self.wave_forms[unit, new[i]] * scales[unit, i] 
                seen[new[i]] = True
        return new_temps

    def main_channel(self, unit=None):
        """Returns the main channel (max ptp) of unit."""
        if unit is None:
            return self.main_chans
        return self.main_chans[unit]

    def generate_correlated_noise(self, time, n_filters=10, min_snr=10., dtype=np.float32):
        """Creates correlated background noise for synthetic datasets.

        params:
        -------
        n_filters: int
            The number of filters to create background noise with.
        min_snr: float
            Minimum SNR of filter that would be convovled to create correlated noise.
        """
        background_noise = []
        allwf = self.wave_forms.reshape([-1, self.n_time])
        allwf = allwf[allwf.ptp(1) > 10.]
        allwf = allwf / allwf.std(axis=1)[:, None] / 2.

        for it in tqdm(range(self.n_channel), "Generating correlated noise."):
            # Make noise for each channel
            cor_noise = 0.
            wf_idx = np.random.choice(range(len(allwf)), n_filters, replace='False')
            for idx in wf_idx:
                noise = np.random.normal(0, 1, time)
                cor_noise += np.convolve(noise, allwf[idx][::-1], 'same')
            cor_noise += np.random.normal(0, 3, len(cor_noise))
            cor_noise = (cor_noise - cor_noise.mean()) / cor_noise.std()
            background_noise.append(cor_noise.astype(dtype))
        return np.array(background_noise)


class SpikeTrain(object):

    def __init__(self, spike_train, num_unit=None, sort=False):
        """

        params:
        -------
        spike_train: np.ndarray (N, 2)
        """
        self.spt = spike_train + 0
        if sort:
            self.spt = self.spt[np.argsort(self.spt[:, 0])]
        # Holds spike counts per unit.
        self.count = []
        # Holds spike times lists per unit.
        self.times = []
        # Holds indices of spikes from unit.
        self.indices = []

        self.n_unit = num_unit
        if num_unit is None:
            # Based on spike train maximum id.
            self.n_unit = self.spt[:, 1].max() + 1
        self.n_spike = len(self.spt)

    def remove(self, idx, keep_idx=False):
        """Removes spikes given by indices from the spike train raster.

        params:
        -------
        keep: bool
        If True, instead of removing spikes with given indices, it will remove
        all spikes except the given indices.
        """
        bool_idx = idx
        if not idx.dtype == np.bool:
            # Change the idx to bool idx of the complenet set of spikes.
            bool_idx = np.ones(self.n_spike, dtype=np.bool)
            bool_idx[idx] = False
            if keep_idx:
                bool_idx = np.logical_not(bool_idx)
        else:
            # Complement the complement spikes idx.
            if not keep_idx:
                bool_idx = np.logical_not(bool_idx)
        self.spt = self.spt[bool_idx]
        # Reset saved attributes.
        self.times = []
        self.count = []
        self.indices = []
        self.n_spike = len(self.spt)

    def spike_times(self, unit):
        """A list of spike times for a given unit."""
        if len(self.times) > 0:
            return self.times[unit]
        for u in range(self.n_unit):
            self.times.append(
                self.spt[self.spike_indices(unit=u), 0])
        return self.spike_times(unit=unit)

    def spike_count(self, unit=None):
        """Number of total spikes for a given unit."""
        if len(self.count) > 0:
            if unit is None:
                return self.count
            return self.count[unit]
        for u in range(self.n_unit):
            t = self.spike_times(unit=u)
            self.count.append(len(t))
        self.count = np.array(self.count)
        return self.spike_count(unit=unit)

    def spike_indices(self, unit=None):
        """Number of total spikes for a given unit."""
        if len(self.indices) > 0:
            if unit is None:
                return self.indices
            return self.indices[unit]
        for u in range(self.n_unit):
            idx = np.where(self.spt[:, 1] == u)[0]
            self.indices.append(idx)
        return self.spike_indices(unit=unit)

    def match(self, sp, window=3):
        """Matches unit i to unit i of given SpikeTrain object."""
        mat = []
        for unit in tqdm(range(self.n_unit)):
            gt = self.spike_times(unit=unit)
            t = sp.spike_times(unit=unit)
            mat.append(
                    SpikeTrain.match_sorted_spike_times(gt, t, window=window))
        return np.array(mat)

    @staticmethod
    def match_sorted_spike_times(l1, l2, window=3, return_idx=True):
        """Matches spikes from first to second list that are sorted."""
        l1_match = np.zeros_like(l1) + np.nan
        l2_match = np.zeros_like(l2) + np.nan

        for i in np.arange(-window, window + 1):
            _, l1_idx, l2_idx = np.intersect1d(
                l1, l2 + i, return_indices=True)
            l1_match[l1_idx] = i
            l2_match[l2_idx] = i
        return l1_match, l2_match

class WaveFormVisualizer(object):

    def __init__(self, geom):
        """

        params:
        -------
        geom: numpy.ndarray of shape (C, 2) where C is total number of
        channels.
        """
        self.geom = geom
        self.n_chan = geom.shape[0]

    def vis_chan(self, wave_forms, threshold=2.):
        return np.where(
                np.max(wave_forms.ptp(axis=-1), axis=0) > threshold)[0]

    def plot_spatial(self, wave_forms, scale=10., squeeze=2., legends=[],
            vis_chan_only=0., jitter=0, ax=None, normalize=False,
            plot_chan_num=True, plot_zero_trace=True, **kwargs):
        """Spatial plot of the wave_forms."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        n_times = wave_forms.shape[-1]
        if not wave_forms.shape[-2] == self.n_chan:
            raise ValueError('Number of channels does not match geometry.')
        vis_chan = range(self.n_chan)
        if vis_chan_only > 0.:
            vis_chan = self.vis_chan(wave_forms, threshold=vis_chan_only)
        # Plot channel numbers.
        offset = 10
        if plot_chan_num:
            for c in vis_chan:
                plt.text(self.geom[c, 0] + offset, self.geom[c, 1], str(c),
                        size='large')
        # Plot Standard Unit for scale.
        # normalize if necessary
        wfvis = wave_forms[0, vis_chan]
        ptprank = (wfvis.ptp(1).argsort().argsort() + 1.) / wfvis.shape[0] + 1.
        norm_scale = ptprank / wfvis.ptp(1)
        if not normalize:
            norm_scale = norm_scale * 0 + 1.
        for i_, c in enumerate(vis_chan):
            ax.fill_between(
                np.arange(n_times) / squeeze + self.geom[c, 0],
                np.zeros(n_times) - (scale * norm_scale[i_]) + self.geom[c, 1],
                np.zeros(n_times) + (scale * norm_scale[i_]) + self.geom[c, 1],
                color='0', alpha=0.1)
            if plot_zero_trace:
                ax.plot(
                        np.arange(n_times) / squeeze + self.geom[c, 0],
                        np.zeros(n_times) + self.geom[c, 1], color='0')
        # Setting Legends
        legend_elements = [Line2D([0], [0], lw=15, color='0',
            alpha=0.2, label='2 Standard Unit')]
        for i, label in enumerate(legends):
            color = "C{}".format(i % 10)
            legend_elements.append(
                    Line2D([0], [0], color=color, label=label)) 
        # Plot channels per waveforms.
        if len(wave_forms.shape) == 2:
            wave_forms = [wave_forms]

        pass_color = False
        if "color" not in kwargs:
            pass_color = True

        for i, wf in enumerate(wave_forms):
            wf = wf + 0.
            wf[vis_chan] *= norm_scale[:, None]
            if pass_color:
                color = "C{}".format(i % 10)
                kwargs["color"] = color
            for c in vis_chan:
                ax.plot(
                    (np.arange(n_times) + i * jitter) / squeeze + self.geom[c, 0],
                    wf[c, :] * scale + self.geom[c, 1],
                    **kwargs)
        ax.legend(handles=legend_elements)
        if fig is not None:
            fig.set_size_inches(20, 10)

    def plot_examples(self, examples, ax=None, plus_raw=False, binary=True):
        time_length = examples.shape[1] - 1
        if plus_raw:
            time_length = time_length // 3
        else:
            time_length = time_length // 2
        def get_color(i):
            colors = ['C2', 'y', 'C1', 'C3']
            label = examples[i, -1, 0]
            if binary:
                if label == 1:
                    color = 'C2'
                else:
                    color = 'C3'
            else:
                if np.isnan(label):
                    color = '0'
                else:
                    color = colors[int(abs(label))]
            return color

        def plot_traces(i, ax=None):
            color = get_color(i)
            self.plot_spatial(examples[i, :time_length, :].T, ax=ax)
            if ax is None:
                ax = plt.gca()
            if plus_raw:
                self.plot_spatial(
                        examples[i, time_length:2*time_length, :].T,
                        ax=ax, color='C1')
            self.plot_spatial(
                    examples[i, -time_length-1:-1, :].T, ax=ax, color=color)

        if ax is None:
            for i in range(len(examples)):
                plot_traces(i)
                plt.gcf().set_size_inches(10, 6)
        else:
            ax = ax.flatten()
            for i, ax_ in enumerate(ax):
                plot_traces(i, ax=ax_)
                if not binary:
                    ax_.set_title('Shift {}'.format(label))


class SyntheticData(object):

    def __init__(self, templates, spike_train, time):
        """

        params:
        -------
        templates: np.ndarray (# units, # channels, # times)
        spike_train: np.ndarray (# spikes, 2)
        """
        self.temps = WaveForms(templates)
        self.spt = SpikeTrain(spike_train)
        self.time = time
        self.n_unit = min(self.temps.n_unit, self.spt.n_unit)
        self.orig_data = None
        self.data = None

    def generate(self, noise=True, dtype=np.float32):
        """Generates the synthetic data given the templates and spike_train."""
        if noise:
            self.data = self.temps.generate_correlated_noise(time=self.time, dtype=np.float32)
        else:
            self.data = np.zeros([self.temps.n_channel, self.time])
        for unit in tqdm(range(self.n_unit), "Generating Data."):
            idx = np.arange(self.temps.n_time)[:, None] + self.spt.spike_times(unit=unit)
            self.data[:, idx] += self.temps[unit][..., None].astype(dtype)
        # Keep a copy of the original data.
        self.orig_data = self.data + 0.
        return self.data

    def match_spike_train(self, spt, window=3):
        """Constructs examples for neural network deconvolution calssifer."""
        given_spt = SpikeTrain(spt, num_unit=self.n_unit, sort=True)
        match_res = self.spt.match(given_spt, window)
        return match_res

    def get_examples(self, spt, plus_raw=False, time_length=None, binary=True):
        """Constructs examples for neural network deconvolution calssifer."""
        given_spt = SpikeTrain(spt, num_unit=self.n_unit, sort=True)
        match_res = self.spt.match(given_spt)
        # Where around spikes should the algorithm grab spikes
        time_window = np.arange(0, self.temps.n_time) 
        if time_length is None:
            # Use full time sample size of the template
            time_length = self.temps.n_time
        # Left out of time window
        n_time_outside = self.temps.n_time - time_length
        time_window = np.arange(0, time_length) + n_time_outside // 2
        example_size = 2 * time_length + 1
        if plus_raw:
            example_size = 3 * time_length + 1
        examples = np.zeros(
            [given_spt.n_spike, example_size, 7])
        # Set labels to one by default
        if binary:
            examples[:, -1, :] =  1

        main_7_c = self.temps.wave_forms.ptp(2).argsort(
            axis=1)[:, -7:]

        for unit in tqdm(range(self.n_unit)):
            #grab_channels = chanidx[mainc[unit]
            grab_channels = main_7_c[unit]
            ex_idx = given_spt.spike_indices(unit)
            idx = time_window + given_spt.spike_times(unit)[:, None]
            examples[ex_idx, :time_length] = self.data.T[:, grab_channels][idx]
            broadcast_templates = self.temps.wave_forms[unit, grab_channels][..., time_window].T[None]
            # Add input templates
            if plus_raw:
                examples[ex_idx, time_length:2*time_length] = self.orig_data.T[:, grab_channels][idx]
                examples[ex_idx, 2*time_length:-1] += broadcast_templates
            else:
                examples[ex_idx, time_length:-1] += broadcast_templates
            # Set unmatched spikes' labels to zero
            # Unmatched spike indices from the given spike train
            if binary:
                unmatched_idx = np.where(np.isnan(match_res[unit, 1]))[0]
                examples[ex_idx[unmatched_idx], -1, :] = 0.
            else:
                examples[ex_idx, -1, 0] = match_res[unit, 1]
 
        return examples

    def remove_spike_train(self, spt):
        """Removes spikes given by indices from the data and the spike train.

        params:
        -------
        spt: np.ndarray (N, 2)
        """
        given_spt = SpikeTrain(spt, num_unit=self.n_unit)
        match_res = self.spt.match(given_spt)

        unmatched_idx = []
        for unit in range(self.n_unit):
            un_idx_ = np.where(np.isnan(match_res[unit, 0]))[0]
            unmatched_idx.append(self.spt.spike_indices(unit=unit)[un_idx_])
        unmatched_idx = np.concatenate(unmatched_idx)
        self.spt.remove(unmatched_idx, keep_idx=True)
        # Spikes that have been matched should be removed from synthetic data
        for unit in range(self.n_unit):
            idx = np.arange(self.temps.n_time)[:, None] + given_spt.spike_times(unit=unit)
            self.data[:, idx] -= self.temps[unit][..., None]
        return self.data

    def qqplot(self, subsample_size=1000):
        """Computes qqplot values.

        params:
        -------
        data: np.ndarray shape (# channel, # time)
        subsample_size: int
            Number of subsamples to be taken from data.
        """
        data = self.data
        n_chan, n_time = data.shape
        qq = np.zeros([n_chan, 2, subsample_size])
        for chan in tqdm(range(n_chan), "computing qqplot"):
            time_subset = np.random.choice(
                range(n_time), subsample_size, replace=False)
            qq_y = np.sort(data[chan, time_subset])
            qq_x = np.sort(np.random.normal(0, 1, subsample_size))
            qq[chan, 0] = qq_x
            qq[chan, 1] = qq_y
        return qq

####################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def metrics(labels, pred, logits=True, threshold=0.5):
    n_ex = len(labels)
    if logits:
        pred = sigmoid(pred)
        pred[pred >= threshold] = 1.
        pred[pred < threshold] = 0.
    metrics = {}
    idxs = [labels < 2, labels == 0, labels == 1]
    idxtype = ["Accuracy", "True Negative", "True Positive"]
    for idx, typ in zip(idxs, idxtype):
        metrics[typ] = (pred[idx] == labels[idx]).sum() * 1. / idx.sum()
    metrics["False Positive"] = 1 - metrics["True Negative"]
    metrics["False Negative"] = 1 - metrics["True Positive"]
    return metrics

def plot_loss(elbo, skip=10, epoch=None, offset=0, **kwargs):
    # Truncate pre offset
    elbo_ = np.array(elbo)[offset:]
    # Truncate a bit from the beginning so that average
    offset_2 = len(elbo_) % skip
    elbo_ = elbo_[offset_2:]
    elbo_avg = np.mean(np.reshape(elbo_, [-1, skip]), axis=-1)
    x = np.arange(offset + offset_2, len(elbo), skip)
    if epoch is not None:
        x = x / (epoch * 1.)
        plt.xlabel("Epochs")
    else:
        plt.xlabel("Iterations")

    plt.plot(x, elbo_avg, **kwargs)
    plt.ylabel("Cross Entropy")


def generate_spike_train(
        n_unit, time, rate_range, refractory_period):
    """

    params:
    -------
    n_unit: int
    time: int
    rate_range: tuple of int
    refractory_period: int

    returns:
    --------
    np.ndarray of shape (N, 2).
    """
    def generate_spt(time, rate):
        x = np.random.poisson(rate, time // rate)
        x[x < refractory_period] = refractory_period
        x = np.cumsum(x)
        x = x[np.logical_and(x > 100, x < time - 100)]
        return x
    spt = []
    min_rate, max_rate = rate_range
    for i in range(n_unit):
        rate = np.random.randint(min_rate, max_rate)
        spike_times = generate_spt(time=time, rate=rate)
        spt.append(
            np.array([spike_times, spike_times * 0 + i]))
    spt = np.concatenate(spt, axis=1).T
    spt = spt[spt[:, 0].argsort()]
    return spt

###################################################


def continuous_visible_channels(
    templates, geom, threshold=.5, neighb_threshold=1., spatial_neighbor_dist=70):
    """
    inputs:
    -------
    templates: np.ndarray with shape (#units, # channels, #time points)
    geom: np.ndarray with shape (# channel, 2)
    threshold: float
        Weaker channels threshold
    neighb_threshold: float
        Strong channel threshold
    spatial_neighbor_dist: float
        neighboring channel threshold (70 for 512 channels retinal probe)
    """
    ptps_ = templates.ptp(2)
    pdist_ = squareform(pdist(geom))
    vis_chan = (ptps_ >= neighb_threshold).astype(np.int32)
    neighbs = np.logical_and(
        pdist_ > 0,
        pdist_ < spatial_neighbor_dist).astype(np.int32)
    return np.logical_or(
        np.logical_and(
            np.matmul(vis_chan, neighbs) > 0,
            ptps_ >= threshold),
        ptps_ >= neighb_threshold)

def reverse_shifts(shifts):
    """Reverse the shifts so that all shifts are positive.

    params:
    -------
    shifts: np.ndarray of int
        All values should be non-negative

    returns:
    --------
    np.ndarray of non-negative integers.
    """
    return shifts.max() - shifts

def shift_channels(signal, shifts):
    """Shifts each channel of the signal according to given shifts.

    params:
    -------
    signal: np.ndarray with shape (#channels, #time)
    shifts: np.array with size #channels

    returns:
    --------
    a copy of the shifted signal according to the given shifts.
    """
    n_chan, size = signal.shape
    max_shift = shifts.max()
    shifted_signal_size = size + max_shift
    shifted_signal = np.zeros([n_chan, shifted_signal_size])
    # Getting shifted indices.
    ix1 = np.tile(np.arange(n_chan)[:, None], size)
    ix2 = np.arange(size) + shifts[:, None]
    shifted_signal[ix1, ix2] = signal
    return shifted_signal


def in_place_roll_shift(signal, shifts):
    """Shifts each channel of the signal according to given shifts.

    (IMPORTANT): This function is the equivalent of Ian's.
    params:
    -------
    signal: np.ndarray with shape (#channels, #time)
    shifts: np.array with size #channels

    returns:
    --------
    a copy of the shifted signal according to the given shifts.
    """
    idx = np.logical_not(shifts == 0)
    for i, s in zip(np.where(idx)[0], shifts[idx]):
        signal[i] = np.roll(signal[i], s)


def align_template_channels(temp, geom, zero_pad_len=30, jitter_len=50):
    """
        inputs:
        -------
        temp: np.ndarray with shape (#units, # channels, #time points)
        geom: np.ndarray with shape (# channel, 2)
        zero_pad_len: int
        jitter_len: int

        These default values are for when you have 101 (5ms)
        templates and want to end up with 61 (3ms) templates
        if spike_size is the original size then:
        spike_size + zero_pad_len * 2 - 2 * jitter_len
        is the new length
    """
    temp = np.pad(
        temp, ((0, 0), (0, 0), (zero_pad_len, zero_pad_len)), 'constant')

    n_unit, n_channel = temp.shape[:2]
    spike_size = 61

    # Maked and aligned and reconstructed templates.
    aligned_temp = np.zeros([n_unit, n_channel, spike_size], dtype=np.float32)
    align_shifts = np.zeros([n_unit, n_channel], dtype=np.int32)

    viscs = continuous_visible_channels(temp, geom)
    # Computes if units are spatially overlapping

    for unit in tqdm(range(n_unit)):
        # get vis channels only
        t = temp[unit, viscs[unit], :]
        # Instead of having 1 template with c channels
        # treat it as c teplates with 1 channels
        tobj = WaveForms(t[:, None])
        main_c = t.ptp(1).argmax()
        align, shifts_ = tobj.align(
            ref_wave_form=t[main_c][None], jitter=jitter_len, return_shifts=True)
        align = align[:, 0]
        # remove offset from shifts so that minimum is 0
        vis_chans = np.where(viscs[unit])[0]
        aligned_temp[unit, vis_chans] = align
    return aligned_temp

######################################################

class TempTempConv(object):

    def __init__(self, templates, geom, rank=5, sparse=True, temp_temp_fname="",
                 vis_threshold_strong=1., vis_threshold_weak=.5, pad_len=30, jitter_len=50):
        """

        params:
        -------
        templates: np.ndarray
            shape (n_unit, n_channel, n_time)
        geom: np.ndarray
            shape (n_channel, 2)
        rank: int
            Rank of SVD factorization
        sparse: boolean
            If true, sparse representation of temp_temp will be used. Otherwise,
            full tensor will be used for temp_temp and unit_overlap
        vis_threshold_strong: flaot
            Any channel with ptp > vis_threshold_strong will be visible
        vis_threshold_weak: float
            Any channel with ptp > vis_threshold_weak that has AT LEAST ONE neighbor
            with ptp > vis_threshold_strong, is visible
        pad_len: int
            Each channel will be zero-padded by pad_len on both size to allow
            for more jitter
        jitter_len: int
            Each channel will be jitter by a total of 2 * jitter_len to find
            best alignment
        """
        self.sparse = sparse
        temp = templates
        n_unit, n_channel, n_time = temp.shape
        self.n_unit = n_unit
        spike_size = temp.shape[2] + 2 * pad_len - 2 * jitter_len
        # We will need this information down the line when compute residual templates
        print("HOOSHMAND: {}".format(temp.shape))
        max_ptp_unit = temp.ptp(2).max(1).argmax()
        max_ptp_unit_main_chan = temp[max_ptp_unit].ptp(1).argmax()
        min_loc_orig = temp[max_ptp_unit, max_ptp_unit_main_chan].argmin()

        # Zero padding is done to allow a lot of jitter for alignment purposes
        temp = np.pad(temp, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant')

        # Maked and aligned and reconstructed templates.
        aligned_temp = np.zeros([n_unit, n_channel, spike_size], dtype=np.float32)
        align_shifts = np.zeros([n_unit, n_channel], dtype=np.int32)
        align_shifts_min = np.zeros(n_unit, dtype=np.int32)
        spat_comp = np.zeros([n_unit, n_channel, rank], dtype=np.float32)
        temp_comp = np.zeros([n_unit, rank, spike_size], dtype=np.float32)

        viscs = continuous_visible_channels(
            temp, geom,
            threshold=vis_threshold_weak, neighb_threshold=vis_threshold_strong)
        # Computes if units are spatially overlapping
        unit_unit_overlap = np.logical_and(viscs[None], viscs[:, None]).sum(-1) > 0

        for unit in tqdm(range(n_unit), "Aligning templates and computing SVD."):
            # get vis channels only
            t = temp[unit, viscs[unit], :]
            # Instead of having 1 template with c channels
            # treat it as c teplates with 1 channels
            tobj = WaveForms(t[:, None])
            main_c = t.ptp(1).argmax()
            align, shifts_ = tobj.align(
                ref_wave_form=t[main_c][None], jitter=jitter_len, return_shifts=True)
            align = align[:, 0]
            # remove offset from shifts so that minimum is 0
            vis_chans = np.where(viscs[unit])[0]
            align_shifts_min[unit] = shifts_.min()
            align_shifts[unit, vis_chans] = shifts_ - shifts_.min()
            # use reconstructed version of temp lates
            if len(align) <= rank:
                # The matrix rank is lower. Just pass
                # identity spatial component and the signal itself
                mat_rank = len(align)
                spat_comp[unit, vis_chans, :mat_rank] = np.eye(mat_rank)
                temp_comp[unit, :mat_rank] = align
                aligned_temp[unit, vis_chans] = align
                continue
            u, h, v = np.linalg.svd(align)
            spat_comp[unit, vis_chans] = u[:, :rank] * h[:rank]
            temp_comp[unit] = v[:rank]
            # Reconstructed version of the unit
            aligned_temp[unit, vis_chans] = np.matmul(u[:, :rank] * h[:rank], v[:rank])

        # computing template_norms
        self.temp_norms = np.square(aligned_temp).sum(-1).sum(-1)

        temp_temp = [[0. for i in range(n_unit)] for j in range(n_unit)]

        zero_padded_temp_temp = None
        global_argmax = None
        if not os.path.exists(temp_temp_fname):
            for unit in tqdm(range(n_unit), "Computing pairwise convolution of templates."):
                # Full temp is the unshifted reconstructed
                # templates for a unit that acts as the data
                # that other units get convolved by
                unshifted_temp = shift_channels(aligned_temp[unit], align_shifts[unit])
                for ounit in np.where(unit_unit_overlap[unit])[0]:
                    # For all spatially overlapping templates, convolve them with
                    # the outer loop template using the SVD trick
                    shifts = reverse_shifts(align_shifts[ounit])
                    shifted_data = shift_channels(unshifted_temp, shifts)
                    transformed_data = np.matmul(spat_comp[ounit][:, :rank].T, shifted_data)
                    temp_temp.append(0.)
                    for r in range(rank):
                        temp_temp[unit][ounit] += np.convolve(
                            transformed_data[r], temp_comp[ounit][r, ::-1])

            # zero pad and shift the temp temps so that everything all temp_temp[i][i] have
            # peak at the same time and they have the same size

            temp_temp_len = np.zeros([n_unit, n_unit], dtype=np.int32)
            temp_temp_argmax = np.zeros(n_unit, dtype=np.int32)
            for i in range(n_unit):
                temp_temp_argmax[i] = temp_temp[i][i].argmax()
                for j in range(n_unit):
                    if isinstance(temp_temp[i][j], np.ndarray):
                        temp_temp_len[i, j] = len(temp_temp[i][j])

            max_len = temp_temp_len.max()
            # (IMPORTANT): this variable is very important, later when you find
            # peaks, the time of each peak has to be subtracted by this value
            global_argmax = temp_temp_argmax.max()
            # Shift all temp_temps so that the peaks are aligned
            shifts_ = global_argmax - temp_temp_argmax
            zero_padded_temp_temp = np.zeros([n_unit, n_unit, max_len])
            for i in range(n_unit):
                u_shift = shifts_[i]
                for j in range(n_unit):
                    if isinstance(temp_temp[i][j], np.ndarray):
                        #temp temp exists
                        zero_padded_temp_temp[i, j, u_shift:u_shift+temp_temp_len[i, j]] = temp_temp[i][j]
            if len(temp_temp_fname) > 0:
                np.save(temp_temp_fname, zero_padded_temp_temp)
        else:
            print (".... loading temp-temp from disk")
            zero_padded_temp_temp = np.load(temp_temp_fname, allow_pickle=True)
            global_argmax = zero_padded_temp_temp[0][0].argmax()

        # Important step that gives the templates have the same length and shifted in a way
        # that spike trains for subtraction are synchronized

        temp_size = align_shifts.max(1) + spike_size
        new_temp_size = temp_size.max()


        # Shifts that were done do the main channel of each unit
        main_chan_shift = align_shifts[np.arange(n_unit), temp.ptp(-1).argmax(-1)]
        main_chan_shift = reverse_shifts(main_chan_shift)
        main_chan_shift = main_chan_shift - main_chan_shift.min()
        new_temp_size += (temp_size - temp_size.max() + main_chan_shift.max()).max()
        # These are the templates that have to be used for residual computation
        residual_computation_templates = np.zeros([n_unit, n_channel, new_temp_size], dtype=np.float32)
        for unit in range(n_unit):
            sh_ = main_chan_shift[unit]
            residual_computation_templates[unit, :, sh_:sh_+temp_size[unit]] = \
            shift_channels(aligned_temp[unit], align_shifts[unit])
        # let's make the templates the same size as the input templates.
        min_loc = residual_computation_templates[max_ptp_unit, max_ptp_unit_main_chan].argmin()
        cut_off_begin = min_loc - min_loc_orig
        self.residual_temps = residual_computation_templates[:, :, cut_off_begin:cut_off_begin+n_time] + 0.
        # This needs to be applied to every peak time
        self.peak_time_residual_offset = - temp_size + 1 - main_chan_shift
        self.peak_time_residual_offset += cut_off_begin

        # What the user needs from this class

        # integer
        self.spike_size = spike_size
        # np.ndarray, shape: (n_unit, n_channel, spike_size)
        self.aligned_templates = aligned_temp
        # np.ndarray, shape: (n_unit, n_channel)
        self.align_shifts = align_shifts
        # np.ndarray, shape: (n_unit, n_channel, new_spike_size)
        # new_spike_size > spike_size
        self.residual_templates = residual_computation_templates
        # np.ndarray, shape: (n_unit, n_channel, rank)
        self.spat_comp = spat_comp
        # np.ndarray, shape: (n_unit, rank, spike_size)
        self.temp_comp = temp_comp
        # np.ndarray, shape: (n_unit, n_unit, some len)
        # temp_size is large enough to account for shifts of largest template
        self.temp_temp = zero_padded_temp_temp
        # integer
        self.peak_time_temp_temp_offset = int(global_argmax)
        # integer
        self.rank = rank
        #
        self.unit_overlap = unit_unit_overlap
        # length of new templates

        if sparse:
            overlap_replacement = []
            temp_temp_replacement = []
            for u in range(n_unit):
                overlap_idx = np.where(self.unit_overlap[u])[0]
                overlap_replacement.append(overlap_idx)
                temp_temp_replacement.append(self.temp_temp[u][overlap_idx])
            self.temp_temp = temp_temp_replacement
            self.unit_overlap = overlap_replacement

    def set_offset(self, x):
        self.peak_time_temp_temp_offset = x

    def adjust_peak_times_for_temp_temp_subtraction(self, peak_time_spike_train):
        """
        inputs:
        -------
        peak_time_spike_train: np.ndarray
            This is the spike train whose time are objective function peaks.
            Shape is (n_spikes, 2), first column time and second 
        """
        new_spike_train = peak_time_spike_train + 0
        new_spike_train[:, 0] -= self.peak_time_temp_temp_offset
        return new_spike_train

    def adjust_peak_times_for_residual_computation(self, peak_time_spike_train):
        """
        inputs:
        -------
        peak_time_spike_train: np.ndarray
            This is the spike train whose time are objective function peaks.
            Shape is (n_spikes, 2), first column time and second 
        """
        new_spike_train = peak_time_spike_train + 0

        for unit in range(self.n_unit):
            new_spike_train[new_spike_train[:, 1] == unit, 0] += self.peak_time_residual_offset[unit]
        return new_spike_train
