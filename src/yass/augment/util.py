"""Utility functions for augmenting data
"""
import random
import numpy as np
import os
import scipy
import logging
from sklearn.decomposition import PCA

from yass.template import align_get_shifts_with_ref, shift_chans
from yass.geometry import order_channels_by_distance


def crop_and_align_templates(fname_templates, save_dir, CONFIG):
    """Crop (spatially) and align (temporally) templates

    Parameters
    ----------

    Returns
    -------
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load templates
    templates = np.load(fname_templates)
    
    n_units, n_times, n_channels = templates.shape
    mcs = templates.ptp(1).argmax(1)
    spike_size = (CONFIG.spike_size_nn - 1)*2 + 1

    ########## TEMPORALLY ALIGN TEMPLATES #################
    
    # template on max channel only
    templates_max_channel = np.zeros((n_units, n_times))
    for k in range(n_units):
        templates_max_channel[k] = templates[k, :, mcs[k]]

    # align them
    ref = np.mean(templates_max_channel, axis=0)
    upsample_factor = 8
    nshifts = spike_size//2

    shifts = align_get_shifts_with_ref(
        templates_max_channel, ref, upsample_factor, nshifts)

    templates_aligned = shift_chans(templates, shifts)
    
    # crop out the edges since they have bad artifacts
    templates_aligned = templates_aligned[:, nshifts//2:-nshifts//2]

    ########## Find High Energy Center of Templates #################

    templates_max_channel_aligned = np.zeros((n_units, templates_aligned.shape[1]))
    for k in range(n_units):
        templates_max_channel_aligned[k] = templates_aligned[k, :, mcs[k]]

    # determin temporal center of templates and crop around it
    total_energy = np.sum(np.square(templates_max_channel_aligned), axis=0)
    center = np.argmax(np.convolve(total_energy, np.ones(spike_size//2), 'same'))
    templates_aligned = templates_aligned[:, (center-spike_size//2):(center+spike_size//2+1)]
    
    ########## spatially crop (only keep neighbors) #################

    neighbors = CONFIG.neigh_channels
    n_neigh = np.max(np.sum(CONFIG.neigh_channels, axis=1))
    templates_cropped = np.zeros((n_units, spike_size, n_neigh))

    for k in range(n_units):

        # get neighbors for the main channel in the kth template
        ch_idx = np.where(neighbors[mcs[k]])[0]

        # order channels
        ch_idx, _ = order_channels_by_distance(mcs[k], ch_idx, CONFIG.geom)

        # new kth template is the old kth template by keeping only
        # ordered neighboring channels
        templates_cropped[k, :, :ch_idx.shape[0]] = templates_aligned[k][:, ch_idx]

    fname_templates_cropped = os.path.join(save_dir, 'templates_cropped.npy')
    np.save(fname_templates_cropped, templates_cropped)

    return fname_templates_cropped


def denoise_templates(fname_templates, save_dir):

    # load templates
    templates = np.load(fname_templates)

    n_templates, n_times, n_chan = templates.shape

    # remove templates with ptp < 5 (if there are enough templates)
    ptps = templates.ptp(1).max(1)
    if np.sum(ptps > 5) > 100:
        templates = templates[ptps>5]
        n_templates = templates.shape[0]

    denoised_templates = np.zeros(templates.shape)

    #templates on max channels (index 0)
    templates_mc = templates[:, :, 0]
    ptp_mc = templates_mc.ptp(1)
    templates_mc = templates_mc/ptp_mc[:, None]

    # denoise max channel templates
    # bug fix = PCA(n_components=5); sometimes test dataset may have too few templates... not realistic though
    pca_mc = PCA(n_components=min(min(templates_mc.shape[0], 
                                      templates_mc.shape[1]),
                                  5))
    score = pca_mc.fit_transform(templates_mc)
    deno_temp = pca_mc.inverse_transform(score)
    denoised_templates[:, :, 0] = deno_temp*ptp_mc[:, None]

    # templates on neighboring channels
    templates_neigh = templates[:, :, 1:]
    templates_neigh = templates_neigh.transpose(0, 2, 1).reshape(-1, n_times)
    ptp_neigh = templates_neigh.ptp(1)
    idx_non_zero = ptp_neigh > 0

    # get pca trained
    pca_neigh = PCA(n_components=5)
    pca_neigh.fit(templates_neigh[idx_non_zero]/ptp_neigh[idx_non_zero][:, None])

    # denoise them
    for j in range(1, n_chan):
        temp = templates[:, :, j]
        temp_ptp = np.abs(temp.min(1))
        idx = temp_ptp > 0
        if np.any(idx):
            temp = (temp[idx]/temp_ptp[idx, None])
            denoised_templates[idx, :, j] = pca_neigh.inverse_transform(
                pca_neigh.transform(temp))*temp_ptp[idx, None]

    fname_out = os.path.join(save_dir, 'denoised_templates.npy')
    np.save(fname_out, denoised_templates)

    return fname_out


class Detection_Training_Data(object):
    
    def __init__(self,
                 fname_templates_cropped,
                 fname_spatial_sig,
                 fname_temporal_sig):

        self.spatial_sig = np.load(fname_spatial_sig)
        self.temporal_sig = np.load(fname_temporal_sig)
        self.templates = np.load(fname_templates_cropped)
        self.standardize_templates()

        self.spike_size = self.temporal_sig.shape[0]
    
    def standardize_templates(self):

        # standardize templates
        ptps = np.ptp(self.templates[: ,:, 0], 1)
        self.templates = self.templates/ptps[:, None, None]
        
    def make_training_data(self, n_data):
        
        n_templates, n_times, n_channels = self.templates.shape

        center = n_times//2
        t_idx_in = slice(center - self.spike_size//2,
                         center + (self.spike_size//2) + 1)

        # sample templates
        idx1 = np.random.choice(n_templates, n_data)
        idx2 = np.random.choice(n_templates, n_data)
        wf1 = self.templates[idx1]
        wf2 = self.templates[idx2]

        # sample scale
        s1 = np.exp(np.random.randn(n_data)*0.8 + 2)
        s2 = np.exp(np.random.randn(n_data)*0.8 + 2)

        # minimum scale
        min_scale = 3
        s1[s1 < min_scale] = min_scale
        s2[s2 < min_scale] = min_scale

        # turn off some
        c1 = np.random.binomial(1, 0.5, n_data)
        c2 = np.random.binomial(1, 0.7, n_data)

        # multiply them
        wf1 = wf1*s1[:, None, None]*c1[:, None, None]
        wf2 = wf2*s2[:, None, None]*c2[:, None, None]

        # choose shift amount
        shift = np.random.randint(low=3, high=self.spike_size-1, size=(n_data,))    
        shift *= np.random.choice([-1, 1], size=n_data, p=[0.2, 0.8])

        # make colliding wf    
        wf_col = np.zeros(wf2.shape)
        for j in range(n_data):
            temp = np.roll(wf2[j], shift[j], axis=0)
            chan_shuffle_idx = np.random.choice(
                n_channels, n_channels, replace=False)
            wf_col[j] = temp[:, chan_shuffle_idx]

        noise = make_noise(n_data, self.spatial_sig, self.temporal_sig)

        wf = wf1[:, t_idx_in] + wf_col[:, t_idx_in] + noise
        label = c1

        return wf, label, wf1[:, t_idx_in]

class Denoising_Training_Data(object):
    
    def __init__(self,
                 fname_templates_cropped,
                 fname_spatial_sig,
                 fname_temporal_sig):

        self.spatial_sig = np.load(fname_spatial_sig)
        self.temporal_sig = np.load(fname_temporal_sig)
        self.templates = np.load(fname_templates_cropped)
        self.templates = self.templates.transpose(0,2,1).reshape(
            -1, self.templates.shape[1])
        self.remove_small_templates()
        self.standardize_templates()
        self.jitter_templates()
        
        self.spike_size = self.temporal_sig.shape[0]

    def remove_small_templates(self):
        
        ptp = self.templates.ptp(1)
        self.templates = self.templates[ptp > 3]
    
    def standardize_templates(self):
    
        # standardize templates
        ptp = self.templates.ptp(1)
        self.templates = self.templates/ptp[:, None]

        ref = np.mean(self.templates, 0)
        shifts = align_get_shifts_with_ref(
            self.templates, ref)
        self.templates = shift_chans(self.templates, shifts)
        
    def jitter_templates(self, up_factor=8):
        
        n_templates, n_times = self.templates.shape

        # upsample best fit template
        up_temp = scipy.signal.resample(
            x=self.templates,
            num=n_times*up_factor,
            axis=1)
        up_temp = up_temp.T

        idx = (np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor))
        up_shifted_temps = up_temp[idx].transpose(2,0,1)
        up_shifted_temps = np.concatenate(
            (up_shifted_temps,
             np.roll(up_shifted_temps, shift=1, axis=1)),
            axis=2)
        self.templates = up_shifted_temps.transpose(0,2,1).reshape(-1, n_times)

        ref = np.mean(self.templates, 0)
        shifts = align_get_shifts_with_ref(
            self.templates, ref, upsample_factor=1)
        self.templates = shift_chans(self.templates, shifts)

    def make_training_data(self, n):

        n_templates, n_times = self.templates.shape

        center = n_times//2
        t_idx_in = slice(center - self.spike_size//2,
                         center + (self.spike_size//2) + 1)
    
        # sample templates
        idx1 = np.random.choice(n_templates, n)
        idx2 = np.random.choice(n_templates, n)
        wf1 = self.templates[idx1]
        wf2 = self.templates[idx2]

        # sample scale
        s1 = np.exp(np.random.randn(n)*0.8 + 2)
        s2 = np.exp(np.random.randn(n)*0.8 + 2)

        # turn off some
        c1 = np.random.binomial(1, 1-0.05, n)
        c2 = np.random.binomial(1, 1-0.05, n)

        # multiply them
        wf1 = wf1*s1[:, None]*c1[:, None]
        wf2 = wf2*s2[:, None]*c2[:, None]

        # choose shift amount
        shift = np.random.randint(low=0, high=3, size=(n,))

        # choose shift amount
        shift2 = np.random.randint(low=5, high=self.spike_size, size=(n,))

        shift *= np.random.choice([-1, 1], size=n)
        shift2 *= np.random.choice([-1, 1], size=n, p=[0.2, 0.8])

        # make colliding wf    
        wf_clean = np.zeros(wf1.shape)
        for j in range(n):
            temp = np.roll(wf1[j], shift[j])
            wf_clean[j] = temp

        # make colliding wf    
        wf_col = np.zeros(wf2.shape)
        for j in range(n):
            temp = np.roll(wf2[j], shift2[j])
            wf_col[j] = temp

        noise_wf = make_noise(n, self.spatial_sig, self.temporal_sig)[:, :, 0]

        wf_clean = wf_clean[:, t_idx_in]
        return (wf_clean + wf_col[:, t_idx_in] + noise_wf,
                wf_clean)


def make_noise(n, spatial_SIG, temporal_SIG):
    """Make noise

    Parameters
    ----------
    n: int
        Number of noise events to generate

    Returns
    ------
    numpy.ndarray
        Noise
    """
    n_neigh, _ = spatial_SIG.shape
    waveform_length, _ = temporal_SIG.shape

    # get noise
    noise = np.random.normal(size=(n, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n, waveform_length, n_neigh))

    return the_noise
