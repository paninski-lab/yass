import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist, squareform

from yass.evaluate.util import *

def align_template(template, temp_len=40, mode='all'):
    window = np.arange(0, temp_len) - temp_len // 2
    n_chan = template.shape[1]
    main_chan = main_channels(template)[-1]
    base_trace = np.zeros(template.shape[0])
    base_trace[:] = template[:, main_chan]
    temp_norm = np.sum(template * template, axis=0)
    base_norm = temp_norm[main_chan]
    aligned_temp = np.zeros([temp_len, n_chan])
    
    if mode == 'neg':
        base_trace[base_trace > 0] = 0

    for c in range(n_chan):
        orig_filt = template[:, c]
        filt = np.zeros(orig_filt.shape)
        filt[:] = orig_filt
        if mode == 'neg':
            filt[filt > 0] = 0
        filt_norm = temp_norm[c]
        conv_dist = -2 * np.convolve(filt, np.flip(base_trace, axis=0), mode='same') + base_norm + filt_norm
        center = np.argmin(conv_dist)
        try:
            aligned_temp[:, c] = orig_filt[center + window]
        except:
            aligned_temp[:, c] = orig_filt[np.arange(0, temp_len)]
    return aligned_temp


def recon(template, rank=3):
    """SVD reconstruction of a template."""
    u, s, vh = np.linalg.svd(template)
    return np.matmul(u[:, :rank] * s[:rank], vh[:rank, :])

def recon_error(template, rank=3):
    """Reconstruction error of SVD with given rank."""
    temp_rec = recon(template, rank=rank)
    return np.linalg.norm((template - temp_rec))

class Geometry(object):
    """Geometry Object for finidng closest channels."""
    def __init__(self, geometry):
        self.geom = geometry
        self.pdist = squareform(pdist(geometry))

    def neighbors(self, channel, size):
        return np.argsort(self.pdist[channel, :])[:size]


def vis_chan(template, min_peak_to_peak=1):
    """Visible channels on a standardized template with given threshold."""
    return np.max(template, axis=0) - np.min(template, axis=0) > min_peak_to_peak


def conv_dist(ref, temp):
    """l2 distance of temp with all windows of ref."""
    return np.convolve((ref * ref), np.ones(len(temp)), mode='valid') - 2 * np.convolve(ref, np.flip(temp, axis=0), mode='valid') + np.sum(temp * temp)

def align_temp_to_temp(ref, temp):
    """Aligns temp with bigger window to ref with smaller window."""
    n_chan = ref.shape[1]
    shifts = np.zeros(n_chan)
    for c in range(n_chan):
        shifts[c] = np.argmin(conv_dist(temp[:, c], ref[:, c]))
        #plt.plot(conv_dist(temp[:, c], ref[:, c]))
    return shifts


def optimal_aligned_compress(template, upsample=5, rank=3, max_shift=6):
    """Greedy local search of alignments for best SVD compression error."""
    upsample = 5
    max_shift = max_shift * upsample
    half_max_shift = max_shift // 2
    
    n_chan = template.shape[1]
    n_times = template.shape[0]
    template = sp.signal.resample(template, n_times * upsample)
    new_times = upsample * n_times

    snip_win = (half_max_shift, -half_max_shift)
    snip_temp = copy.copy(template[snip_win[0]:snip_win[1], :])
    shifts = np.zeros(n_chan, dtype='int')

    #
    obj = recon_error(snip_temp, rank=rank)
    obj_list = []
    for i, k in enumerate(reversed(main_channels(template))):
        if i == 0:
            # main channel do nothing
            continue
        #cand_chan = np.random.randint(0, n_chan)
        cand_chan = k
        # obj of jitter -1, 0, 0 respectively
        new_obj = np.zeros(max_shift + 1)
        for j, jitter in enumerate(range(-half_max_shift, half_max_shift + 1)):
            snip_from, snip_to = snip_win[0] + jitter, snip_win[1] + jitter
            if snip_to == 0:
                snip_to = new_times
            snip_temp[:, cand_chan] = template[snip_from:snip_to, cand_chan]
            new_obj[j] = recon_error(snip_temp, rank=rank)
        #plt.plot(np.arange(- max_shift, max_shift + 1, 1), new_obj)
        # Optimal local jitterupsample
        opt_shift = np.argmin(new_obj) - half_max_shift
        shifts[cand_chan] = opt_shift
        snip_from, snip_to = snip_win[0] + opt_shift, snip_win[1] + opt_shift
        if snip_to == 0:
            snip_to = new_times
        snip_temp[:, cand_chan] = template[snip_from:snip_to, cand_chan]
        obj = min(new_obj)
        obj_list.append(obj)

    return snip_temp, obj_list


def optimal_svd_align(template, geometry, rank=3, upsample=5, chunk=7, max_shift=10):
    """Iterative svd then align approach to alignment."""
    max_shift = upsample * max_shift

    n_times = template.shape[0]
    n_chan = template.shape[1]
    main_chan = np.flip(main_channels(template), axis=0)
    win_len = n_times * upsample - max_shift

    # Upsample
    temp = sp.signal.resample(template, n_times * upsample)
    shifts = np.zeros(n_chan, dtype=int) + max_shift // 2
    #
    chunk_set = 0
    i = 1
    terminate = False
    while not terminate:
        if i * chunk > n_chan:
            cum_chan = main_chan
            terminate = True
        else:
            #cum_chan = main_chan[:i * chunk]
            cum_chan = geometry.neighbors(main_chan[0], size=chunk * i)
        for iteration in range(4):
            temp_ref = []
            for c in cum_chan:
                temp_ref.append(temp[shifts[c]:shifts[c] + win_len, c])
            temp_ref = np.array(temp_ref).T
            temp_ref_rec = recon(temp_ref, rank=rank)
            shifts[cum_chan] = align_temp_to_temp(temp_ref_rec, temp[:, cum_chan])
        i += 1
    aligned_temp = []
    for c in range(n_chan):
            aligned_temp.append(temp[shifts[c]:shifts[c] + win_len, c])
    return np.array(aligned_temp).T


def plot_spatial(geom, temp, ax, color='C0', alpha=0.7, scale=10., squeeze=8.):
    """Plots template spatially."""
    leng = temp.shape[0]
    for c in range(temp.shape[1]):
        ax.plot(
            np.arange(0, leng, 1) / squeeze + geom[c, 0],
            temp[:, c] * scale + geom[c, 1], alpha=alpha, color=color, lw=2)


def plot_spatial_fill(geom, temp, ax, color='C0', scale=10., squeeze=8.):
    """Plots standard error for each channel spatially."""
    temp_ = temp * 0
    leng = temp.shape[0]
    for c in range(temp.shape[1]):
        ax.fill_between(
            np.arange(0, leng, 1) / squeeze + geom[c, 0],
            temp_[:, c] - scale / 2  + geom[c, 1],
            temp_[:, c] + scale / 2 + geom[c, 1], color=color, alpha=0.3)


def plot_chan_numbers(geom, ax, offset=10):
    """Plots template spatially.77"""
    for c in range(geom.shape[0]):
        plt.text(geom[c, 0] + offset, geom[c, 1], str(c), size='large')



def fake_data(spt, temps, length, noise=True):
    """Given a spike train and templates creates a fake data."""
    n_time, n_chan, n_unit = temps.shape
    data = None
    if noise:
        data = np.random.normal(0, 1, [length, n_chan])
    else:
        data = np.zeros([length, n_chan])
    for u in range(n_unit):
        spt_u = spt[spt[:, 1] == u, 0]
        spt_u = spt_u[spt_u < length - n_time]
        idx = spt_u + np.arange(0, n_time)[:, np.newaxis]
        data[idx, :] += temps[:, :, u][:, np.newaxis, :]
    return data


def count_matches(array1, array2, admissible_proximity=40):
    """Finds the matches between two count process.

    Returns
    -------
    tuple of lists
        (M, U, M) where M is the list of indices of array2 where
        matched with array 1 happened and U contains a list of
        indices of array2 where no match with array1 happened.
    """
    # In time samples
    
    m, n = len(array1), len(array2)
    i, j = 0, 0
    count = 0
    matched_idx = []
    unmatched_idx = []
    while i < m and j < n:
        if abs(array1[i] - array2[j]) < admissible_proximity:
            matched_idx.append(j)
            i += 1
            j += 1
            count += 1
        elif array1[i] < array2[j]:
            i += 1
        else:
            unmatched_idx.append(j)
            j += 1
    return matched_idx, unmatched_idx


def compute_snr(temps):
    """Computes peak to peak SNR for given templates."""
    
    chan_peaks = np.max(temps, axis=0)
    chan_lows = np.min(temps, axis=0)
    peak_to_peak = chan_peaks - chan_lows
    return np.max(peak_to_peak, axis=0)


def enforce_refractory_period(spike_train, refractory_period):
    """Removes spike times that violate refractory period.
    
    Parameters:
    -----------
    spike_train: numpy.ndarray
        Shape (N, 2) where first column indicates spike times
        and second column unit identities. Should be sorted
        by times across all units.
    refractory_period: int

    Returns:
    --------
    np.ndarray of shape shape (N, 2).
    """
    n_unit = np.max(spike_train[:, 1])
    delete_idx = []
    for u in range(n_unit):
        sp_idx = np.where(spike_train[:, 1] == u)[0]
        
        sp = spike_train[sp_idx, 0]
        diffs = np.diff(sp)
        idx = diffs < refractory_period
        while np.sum(idx) > 0:
            # Remove violating spike times
            delete_idx += list(sp_idx[np.where(idx)[0] + 1])
            sp_idx = np.delete(sp_idx, np.where(idx)[0] + 1, axis=0)
            # Recompute
            sp = spike_train[sp_idx, 0]
            diffs = np.diff(sp)
            idx = diffs < refractory_period
    # Remove all the spike times from the original spike train
    return np.delete(spike_train, delete_idx, axis=0)