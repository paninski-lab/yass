"""Set of functions for visualization of mean waveform if action potentials."""

import numpy as np
import os
import matplotlib.pyplot as plt
import parmap

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import argrelmin
from tqdm import tqdm

from yass.deconvolve._deconvolve_utils import shift_channels_cython

from yass.geometry import n_steps_neigh_channels

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
class TempAlign(object):

    def __init__(self, templates, geom, pad_len, jitter_len,
            vis_threshold_strong=2., vis_threshold_weak=1.):
        """

        params:
        -------
        templates: np.ndarray
            shape (n_unit, n_channel, n_time)
        geom: np.ndarray
            shape (n_channel, 2)
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
        temp = templates
        n_unit, n_channel, n_time = temp.shape
        self.n_unit = n_unit
        spike_size = temp.shape[2] + 2 * pad_len - 2 * jitter_len

        # Zero padding is done to allow a lot of jitter for alignment purposes
        temp = np.pad(temp, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant')    

        viscs = continuous_visible_channels(
            temp, geom,
            threshold=vis_threshold_weak, neighb_threshold=vis_threshold_strong)
        flatten = np.where(viscs.flatten())[0]
        # unit number of the signal
        signal_unit = flatten // n_channel
        # channel number of the signal
        signal_channel = flatten % n_channel

        # stack signals from all templates on their visible channels
        t = temp.reshape([-1, temp.shape[2]])[viscs.flatten()]
        tobj = WaveForms(t[:, None])
        main_c = t.ptp(1).argmax()
        align, shifts_ = tobj.align(
            ref_wave_form=t[main_c][None], jitter=jitter_len, return_shifts=True)
        
        self.align_shifts = shifts_ - shifts_.min()
        self.stacked_aligned_signals = align[:, 0]
        self.signal_unit = signal_unit
        self.signal_channel = signal_channel



class TempTempConv(object):

    def __init__(self, CONFIG, templates, geom, pad_len, jitter_len, rank=5,
                 sparse=True, #temp_temp_fname="",
                 vis_threshold_strong=1., vis_threshold_weak=0.5, parallel=True):
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
        max_ptp_unit = temp.ptp(2).max(1).argmax()
        max_ptp_unit_main_chan = temp[max_ptp_unit].ptp(1).argmax()
        min_loc_orig = temp[max_ptp_unit, max_ptp_unit_main_chan].argmin()

        # Zero padding is done to allow a lot of jitter for alignment purposes
        temp = np.pad(temp, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant')

        # Maked and aligned and reconstructed templates.
        aligned_temp = np.zeros([n_unit, n_channel, spike_size], dtype=np.float32)
        align_shifts = np.zeros([n_unit, n_channel], dtype=np.int32)
        #align_shifts_min = np.zeros(n_unit, dtype=np.int32)
        spat_comp = np.zeros([n_unit, n_channel, rank], dtype=np.float32)
        temp_comp = np.zeros([n_unit, rank, spike_size], dtype=np.float32)

        # visible channels for the purpose of finiding overlapping units
        viscs = temp.ptp(2) > vis_threshold_strong
        num_vis_chan = viscs.sum(1)
        # If a unit has no visible channel, make its main channel visible
        invis_units = np.where(num_vis_chan == 0)[0]
        viscs[invis_units, temp.ptp(2).argmax(1)[invis_units]] = True
        unit_unit_overlap = np.logical_and(viscs[None], viscs[:, None]).sum(-1) > 0

        # real visible channels using both strong and weak threshold
        viscs = continuous_visible_channels(
            temp, geom,
            threshold=vis_threshold_weak, neighb_threshold=vis_threshold_strong)
        num_vis_chan = viscs.sum(1)
        # If a unit has no visible channel, make its main channel visible
        invis_units = np.where(num_vis_chan == 0)[0]
        viscs[invis_units, temp.ptp(2).argmax(1)[invis_units]] = True
        # Computes if units are spatially overlapping

        # align and denoise
        #neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
        for unit in tqdm(range(n_unit), "....aligning templates and computing SVD."):
            # get vis channels only
            #t = temp[unit, viscs[unit], :]
            # Instead of having 1 template with c channels
            # treat it as c teplates with 1 channels
            #tobj = WaveForms(t[:, None])
            #main_c = t.ptp(1).argmax()
            #align, shifts_ = tobj.align(
            #    ref_wave_form=t[main_c][None], jitter=jitter_len, return_shifts=True)
            #align = align[:, 0]

            if np.sum(np.abs(temp[unit])) == 0:
                continue

            vis_chans = np.where(viscs[unit])[0]
            neigh_chans = CONFIG.neigh_channels[vis_chans][:, vis_chans]
            align, shifts_, vis_chan_keep = align_templates(
                temp[unit, viscs[unit]], jitter_len,
                neigh_chans, min_loc_ref=min_loc_orig+pad_len)

            # kill any unconnected vis chans
            align = align[vis_chan_keep]
            shifts_ = shifts_[vis_chan_keep]
            vis_chans = vis_chans[vis_chan_keep]

            # remove offset from shifts so that minimum is 0
            #align_shifts_min[unit] = shifts_.min()
            align_shifts[unit, vis_chans] = shifts_ - shifts_.min()

            center_spike_size = int(2 * CONFIG.recordings.sampling_rate / 1000.)
            align = monotonic_edge(align, CONFIG.center_spike_size)

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

        #temp_temp = [[0. for i in range(n_unit)] for j in range(n_unit)]
        temp_temp = None

        zero_padded_temp_temp = None
        global_argmax = None
        #if not os.path.exists(temp_temp_fname):
        if True:
            print (".... computing temp_temp ...")
            if parallel:
                # partition the units into 12 sub problems
                sub_size = int(np.ceil(n_unit / CONFIG.resources.n_processors))
                if sub_size == 0:
                    sub_size = 1
                sub_tasks = []
                i = 0
                while i < n_unit:
                    till = i + sub_size
                    if till > n_unit:
                        till = n_unit
                    sub_tasks.append(range(i, till))
                    i = i + sub_size

                temp_temp_list = parmap.map(
                    temp_temp_partial, sub_tasks,
                    aligned_temp_=aligned_temp, align_shifts_=align_shifts,
                    unit_unit_overlap_=unit_unit_overlap,
                    spat_comp_=spat_comp, temp_comp_=temp_comp,
                    rank_=rank, n_unit_=n_unit,
                    processes=CONFIG.resources.n_processors,
                    pm_pbar=True)
                temp_temp = []
                # Gather and combine all solutions
                for res in temp_temp_list:
                    temp_temp += res
            else:
                temp_temp = temp_temp_partial(
                    units=range(n_unit), aligned_temp_=aligned_temp, align_shifts_=align_shifts,
                    unit_unit_overlap_=unit_unit_overlap,
                    spat_comp_=spat_comp, temp_comp_=temp_comp, rank_=rank, n_unit_=n_unit)
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
            zero_padded_temp_temp = np.zeros([n_unit, n_unit, max_len], 'float32')
            for i in range(n_unit):
                if temp[i].ptp(1).max() == 0:
                    continue
                u_shift = shifts_[i]
                for j in range(n_unit):
                    if temp[j].ptp(1).max() == 0:
                        continue
                    if isinstance(temp_temp[i][j], np.ndarray):
                        #temp temp exists
                        zero_padded_temp_temp[i, j, u_shift:u_shift+temp_temp_len[i, j]] = temp_temp[i][j]
            #if len(temp_temp_fname) > 0:
            #    np.save(temp_temp_fname, zero_padded_temp_temp)
        #else:
        #    print (".... loading temp-temp from disk")
        #    zero_padded_temp_temp = np.load(temp_temp_fname, allow_pickle=True)
        #    global_argmax = zero_padded_temp_temp[0][0].argmax()

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
        if cut_off_begin < 0:
            left_ = -np.copy(cut_off_begin)
            right_ = max(0, n_time - left_ - residual_computation_templates.shape[2])
            residual_computation_templates = np.pad(
                residual_computation_templates, ((0, 0), (0, 0), (left_, right_)), 'constant')
            cut_off_begin = 0
        self.residual_temps = residual_computation_templates[:, :, cut_off_begin:cut_off_begin+n_time] + 0.
        # This needs to be applied to every peak time
        self.peak_time_residual_offset = - temp_size + 1 - main_chan_shift
        self.peak_time_residual_offset += (min_loc - min_loc_orig)

        # reduce the amount of overlapping units
        less_overlapping_units = True
        self.temp_norms = np.sum(np.square(self.residual_temps), (1,2))
        if less_overlapping_units:
            threshold = 0.05*self.temp_norms
            threshold[threshold > 50] = 50
            unit_unit_overlap = np.abs(zero_padded_temp_temp).max(2) > threshold[None]

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


def temp_temp_partial(
    units, aligned_temp_, align_shifts_, unit_unit_overlap_,
    spat_comp_, temp_comp_, rank_, n_unit_):
    # Helper class for computing temp temp that is used in parmap parallelization
    temp_temp = [[0. for i in range(n_unit_)] for j in range(len(units))]
    i_unit = 0
    #for unit in tqdm(units, "Computing pairwise convolution of templates."):
    for unit in units:
        # Full temp is the unshifted reconstructed
        # templates for a unit that acts as the data
        # that other units get aonvolved by
        unshifted_temp = shift_channels(aligned_temp_[unit], align_shifts_[unit])
        for ounit in np.where(unit_unit_overlap_[unit])[0]:
            # For all spatially overlapping templates, convolve them with
            # the outer loop template using the SVD trick
            shifts = reverse_shifts(align_shifts_[ounit])
            shifted_data = shift_channels(unshifted_temp, shifts)
            transformed_data = np.matmul(spat_comp_[ounit][:, :rank_].T, shifted_data)
            for r in range(rank_):
                temp_temp[i_unit][ounit] += np.convolve(
                    transformed_data[r], temp_comp_[ounit][r, ::-1])
        i_unit += 1
    return temp_temp


def align_templates(temp_, jitter, neigh_chans, ref=None, min_loc_ref=None):

    # get reference template if not given
    n_chans, n_time = temp_.shape

    # the temporal window of an aligned template
    n_time_small = n_time - 2 * jitter

    if np.sum(np.square(temp_)) == 0:
        return (np.zeros((n_chans, n_time_small), 'float32'),
                np.zeros(n_chans, 'int32'),
                np.arange(n_chans))

    if ref is None:
        main_c = temp_.ptp(1).argmax()
        ref = temp_[main_c]
        if min_loc_ref is not None:
            min_loc = ref.argmin()
            ref = np.roll(ref, min_loc_ref-min_loc)
        ref = ref[jitter:-jitter]

    # compute the distance between a template (for each channel)
    # and the ref template for every jitter
    idx = np.arange(n_time_small) + np.arange(2 * jitter)[:, None]
    all_shifts = temp_[:, idx]
    all_dist = np.square(all_shifts - ref).sum(axis=-1)
    all_inv_dist = np.square(-all_shifts - ref).sum(axis=-1)
    dist_ = np.min(np.stack((all_inv_dist, all_dist)), 0)

    # find argrelmin (all local minimums)
    cc, tt = argrelmin(dist_, axis=1, order=15)
    val = dist_[cc, tt]

    # keep only small enough ones
    idx_keep = np.zeros(len(cc), 'bool')
    for c in range(dist_.shape[0]):
        th = np.median(dist_[c])
        idx_ = np.where(cc == c)[0]
        idx_keep[idx_[val[idx_] < th]] = True

    # edge case: if all the local minima are not big enough,
    # just disregard
    if np.any(idx_keep):
        cc = cc[idx_keep]
        tt = tt[idx_keep]
        val = val[idx_keep]

    if len(val) == 0:
        cc = np.arange(n_chans)
        tt = np.argmin(dist_, axis=1)
        val = dist_[cc,tt]

    # do connecting
    #t_diff=10
    #keep = np.zeros(len(tt), 'bool')
    #while not np.all(np.in1d(chans_must_in, np.unique(cc[keep]))):
    #    not_keep_where = np.where(~keep)[0]
    #    index_start = not_keep_where[val[not_keep_where].argmin()]
    #    keep = connecting_points(
    #        np.vstack((tt, cc)).T,
    #        index_start,
    #        neigh_chans,
    #        t_diff,
    #        keep)

    high_recursion_limit = False
    if len(val) > 1000:
        import sys
        sys.setrecursionlimit(100000)
        high_recursion_limit = True

    t_diff=10
    index_start = val.argmin()
    keep = connecting_points(
        np.vstack((tt, cc)).T,
        index_start,
        neigh_chans,
        t_diff)
    cc = cc[keep]
    tt = tt[keep]
    val = val[keep]
    vis_chan_keep = np.unique(cc)

    if high_recursion_limit:
        sys.setrecursionlimit(1000)

    # include all channels with sufficiently large ptps
    chans_must_in = np.where(np.abs(temp_).max(1) > 1)[0]
    vis_chan_keep = np.unique(np.hstack((vis_chan_keep, chans_must_in)))

    # choose best among survived ones
    best_shifts = np.zeros(n_chans, 'int32')
    for c in vis_chan_keep:
        idx_ = np.where(cc == c)[0]
        if len(idx_) > 0:
            best_shifts[c] = tt[idx_][val[idx_].argmin()]
        else:
            best_shifts[c] = dist_[c].argmin()

    aligned_temp = np.zeros((n_chans, n_time_small), 'float32')
    for c in vis_chan_keep:
        aligned_temp[c] = temp_[c][best_shifts[c]:best_shifts[c]+n_time_small]

    return aligned_temp, best_shifts, vis_chan_keep


def connecting_points(points, index, neighbors, t_diff, keep=None):

    if keep is None:
        keep = np.zeros(len(points), 'bool')

    if keep[index] == 1:
        return keep
    else:
        keep[index] = 1
        spatially_close = np.where(neighbors[points[index, 1]][points[:, 1]])[0]
        close_index = spatially_close[np.abs(points[spatially_close, 0] - points[index, 0]) <= t_diff]

        for j in close_index:
            keep = connecting_points(points, j, neighbors, t_diff, keep)

        return keep


def make_it_monotonic(data):
    negs = data[:, 0] < 0

    data[negs] *= -1
    data[data < 0] = 0

    for j in range(data.shape[1]-1):
        idx = np.where(np.less(data[:, j], data[:, j+1]))[0]
        data[idx, j+1] = data[idx, j]

    data[negs] *= -1

    return data


def monotonic_edge(align, center_spike_size):

    n_chans, n_times = align.shape

    align_ = np.copy(align)

    edge_size = (n_times - center_spike_size)//2

    right_edge = align_[:, -edge_size-1:]
    right_edge = make_it_monotonic(right_edge)
    align_[:, -edge_size-1:] = right_edge

    left_edge = align_[:, :edge_size+1]
    left_edge = make_it_monotonic(left_edge[:, ::-1])
    align_[:, :edge_size+1] = left_edge[:, ::-1]

    return align_

def shift_svd_denoise(temp, CONFIG,
                      vis_threshold_strong, vis_threshold_weak,
                      rank, pad_len, jitter_len):

    temp = temp.transpose(0, 2, 1)

    n_unit, n_channel, n_time = temp.shape
    spike_size = temp.shape[2] + 2 * pad_len - 2 * jitter_len
    max_ptp_unit = temp.ptp(2).max(1).argmax()
    max_ptp_unit_main_chan = temp[max_ptp_unit].ptp(1).argmax()
    min_loc_orig = temp[max_ptp_unit, max_ptp_unit_main_chan].argmin()

    # Zero padding is done to allow a lot of jitter for alignment purposes
    temp = np.pad(temp, ((0, 0), (0, 0), (pad_len, pad_len)), 'constant')

    viscs = continuous_visible_channels(
        temp, CONFIG.geom,
        threshold=vis_threshold_weak, neighb_threshold=vis_threshold_strong)
    num_vis_chan = viscs.sum(1)
    # If a unit has no visible channel, make its main channel visible
    invis_units = np.where(num_vis_chan == 0)[0]
    viscs[invis_units, temp.ptp(2).argmax(1)[invis_units]] = True

    aligned_temp = np.zeros([n_unit, n_channel, spike_size], dtype=np.float32)
    align_shifts = np.zeros([n_unit, n_channel], dtype=np.int32)

    # get the min location of the largest unit and align all to that.
    # allowing two-step difference
    #neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
    # align and denoise
    for unit in range(n_unit):
        vis_chans = np.where(viscs[unit])[0]
        neigh_chans = CONFIG.neigh_channels[vis_chans][:, vis_chans]

        if np.sum(np.abs(temp[unit, vis_chans])) == 0:
            continue

        align, shifts_, vis_chan_keep = align_templates(
            temp[unit, vis_chans], jitter_len,
            neigh_chans, min_loc_ref=min_loc_orig+pad_len)

        # kill any unconnected vis chans
        align = align[vis_chan_keep]
        shifts_ = shifts_[vis_chan_keep]
        vis_chans = vis_chans[vis_chan_keep]

        # remove offset from shifts so that minimum is 0
        align_shifts[unit, vis_chans] = shifts_ - shifts_.min()
        align = monotonic_edge(align, CONFIG.center_spike_size)

        # use reconstructed version of temp lates
        if len(align) <= rank:
            # The matrix rank is lower. Just pass
            # identity spatial component and the signal itself
            mat_rank = len(align)
            aligned_temp[unit, vis_chans] = align
            continue

        u, h, v = np.linalg.svd(align)
        # Reconstructed version of the unit
        aligned_temp[unit, vis_chans] = np.matmul(u[:, :rank] * h[:rank], v[:rank])

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
    if cut_off_begin < 0:
        left_ = -np.copy(cut_off_begin)
        right_ = max(0, n_time - left_ - residual_computation_templates.shape[2])
        residual_computation_templates = np.pad(
            residual_computation_templates, ((0, 0), (0, 0), (left_, right_)), 'constant')
        cut_off_begin = 0

    residual_computation_templates = residual_computation_templates[
        :, :, cut_off_begin:cut_off_begin+n_time] + 0.

    return residual_computation_templates.transpose(0, 2, 1)
