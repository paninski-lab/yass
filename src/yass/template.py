"""Class that provides some basic functions of wave forms."""

import numpy as np
import math
import logging

from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from scipy import signal

from yass.util import absolute_path_to_asset

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
 
    def ptp(self):
        """Returns ptp of wave forms in standard units.

        returns:
        --------
        numpy.array of size N.
        """
        return self.wave_forms.ptp(axis=-1).max(axis=-1)

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

    def align(self, ref_wave_form=None, jitter=3, upsample=1):
        """Aligns all the wave forms to the reference wave form.

        params:
        -------
        jitter: int
            How much jitter per wave form in subsample time is allowed.
        upsample: int
            Factor for interpolation of signals.
        """
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
        all_shifts = self.wave_forms[:, max_chan, idx]
        best_shift_idx = np.square(
                all_shifts - ref).sum(axis=-1).argmin(axis=-1)
        return self.get_shifted_waveforms(best_shift_idx, clip_value=jitter)

def run_template_computation(
    fname_spike_train,
    spike_size, reader, out_dir,
    multi_processing=None,
    n_processors=1):

    logger = logging.getLogger(__name__)
    
    logger.info("computing templates")

    # make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # make temp folder
    tmp_folder = os.path.join(out_dir, 'tmp_template')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # partition spike train per unit for multiprocessing
    fnames_spike_times, n_units = partition_spike_time(
        tmp_folder, fname_spike_train)

    # gather input arguments
    fnames_out = []
    for unit in range(n_units):
        fnames_out.append(os.path.join(
            tmp_folder,
            "template_unit_{}.npy".format(unit)))

    # run computing function
    if multi_processing:
        parmap.starmap(run_template_computation_parallel,
                   list(zip(fnames_spike_times, fnames_out)),
                   reader,
                   spike_size,
                   processes=n_processors,
                   pm_pbar=True)
    else:
        for ctr in range(n_units):
            run_template_computation_parallel(
                fnames_spike_times[ctr],
                fnames_out[ctr],
                reader,
                spike_size)

    # gather all info
    templates_new = np.zeros((n_units, spike_size, n_channels))
    for ctr, unit in enumerate(range(n_units)):
        if os.path.exists(fnames_out[ctr]):
            templates_new[unit] = np.load(fnames_out[ctr])

    fname_templates = os.path.join(out_dir, 'templates.npy')
    np.save(fname_templates, templates_new)

    return fname_templates

def run_template_computation_parallel(
    fname_spike_times, fname_out, reader, spike_size):

    # load spike times
    spike_times = np.load(fname_spike_times)

    template = compute_a_template(spike_times,
                                  reader,
                                  spike_size)

    # save result
    np.save(fname_out, template)

def compute_a_template(spike_times, reader, spike_size):

    # subsample upto 1000
    max_spikes = 1000
    if len(spike_times) > max_spikes:
        spike_times = np.random.choice(a=spike_times,
                                       size=max_spikes,
                                       replace=False)

    # get waveforms
    wf, _ = reader.read_waveforms(spike_times,
                                  spike_size)

    # max channel
    mc = np.mean(wf, axis=0).ptp(0).argmax()

    # load reference template
    ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))


    wf = align_waveforms(wf=wf,
                         ref=ref_template,
                         upsample_factor=5,
                         nshifts=3)

    return np.median(wf, axis=0)

def partition_spike_time(save_dir,
                         fname_spike_index):

    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # load data
    spike_index = np.load(fname_spike_index)

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    spike_index_list = [[] for ii in range(n_units)]
    for j in range(len(spike_index)):
        tt, ii = spike_index[j]
        spike_index_list[ii].append(tt)

    # save them
    fnames = []
    for unit in range(n_units):

        fname = os.path.join(save_dir, 'partition_{}.npy'.format(unit))
        np.save(fname,
                spike_index_list[unit])
        fnames.append(fname)
        
    return fnames, n_units

def align_waveforms(wf, ref=None, upsample_factor=5, nshifts=7):

    # get shapes
    n_spikes, n_times, n_channels = wf.shape

    # if no reference, make a reference using mean
    if ref is None:
        mean_wf = np.mean(wf, axis=0)
        mc = mean_wf.armgax(1)
        ref = mean_wf[:, mc]

    n_times_ref = len(ref)

    # if length of waveforms is different than ref,
    # either pad zeros to ref or cut edges of ref
    len_diff = n_times - n_times_ref
    if len_diff > 0:
        ref = np.hstack((np.zeros(len_diff//2),
                         ref,
                         np.zeros(len_diff - len_diff//2)))
    elif len_diff < 0:
        ref = ref[len_diff//2:-(len_diff-len_diff//2)]
    
    shifts = align_get_shifts_with_ref(
        wf, ref, upsample_factor, nshifts)
    
    wf_aligned = shift_chans(wf, shifts)
    
    return wf_aligned, shifts
    
def align_get_shifts_with_ref(wf, ref, upsample_factor = 5, nshifts = 7):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces

def shift_chans(wf, best_shifts):
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    # Cat: TODO read this from CNOFIG
    wfs_final= np.zeros(wf.shape, 'float32')
    for k, shift_ in enumerate(best_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final[k] = temp
    
    return wfs_final
