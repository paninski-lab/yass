"""Class that provides some basic functions of wave forms."""

import os
import numpy as np
import math
import logging
import parmap

from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from scipy import signal

from yass import read_config
from yass.reader import READER
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


def update_templates(
    fname_templates,
    fname_spike_train,
    recordings_filename,
    recording_dtype,
    output_directory,
    rate=0.002,
    unit_ids=None):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_templates_updated = os.path.join(
        output_directory, 'templates_updated.npy')
    if os.path.exists(fname_templates_updated):
        return fname_templates_updated, None

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG)

    # max channel for each unit
    max_channels = np.load(fname_templates).ptp(1).argmax(1)
    fname_templates_new = run_template_computation(
        fname_spike_train,
        reader,
        output_directory,
        max_channels=max_channels,
        unit_ids=unit_ids,
        multi_processing=CONFIG.resources.multi_processing,
        n_processors=CONFIG.resources.n_processors)

    # load templates
    templates_orig = np.load(fname_templates)
    templates_new = np.load(fname_templates_new)

    n_units, n_times, n_channels = templates_orig.shape
    n_units_new = templates_new.shape[0]
    
    if unit_ids is None:
        unit_ids = np.arange(n_units)

    # if last few units have no spikes deconvovled, the length of new templates
    # can be shorter. then, zero pad it
    if n_units_new < n_units:
        zero_pad = np.zeros((n_units-n_units_new, n_times, n_channels), 'float32')
        templates_new = np.concatenate(
            (templates_new, zero_pad), axis=0)

    # number of deconvolved spikes
    n_spikes = np.zeros(n_units)
    units_unique, n_spikes_unique = np.unique(
        np.load(fname_spike_train)[:, 1], return_counts=True)
    n_spikes[units_unique] = n_spikes_unique

    # update rule if it will be updated
    weight_to_update = np.power((1 - rate), n_spikes)

    # only update for units in unit_ids 
    weight = np.ones(n_units)
    weight[unit_ids] = weight_to_update[unit_ids]
    weight = weight[:, None, None]

    # align templates
    templates_orig, templates_new = align_two_set_of_templates(
        templates_orig, templates_new)

    # update and save
    templates_updated = weight*templates_orig + (1-weight)*templates_new
    np.save(fname_templates_updated, templates_updated)

    # check the difference
    max_diff = np.zeros(n_units)
    max_diff[unit_ids] = np.max(
        np.abs(templates_new[unit_ids] - templates_orig[unit_ids]),
        axis=(1,2))
    max_diff = max_diff/templates_orig.ptp(1).max(1)

    return fname_templates_updated, max_diff


def align_two_set_of_templates(templates1, templates2, ref_set=0):
    
    n_units = templates1.shape[0]
    
    for unit in range(n_units):
        temp = np.concatenate((templates1[[unit]],
                               templates2[[unit]]),
                              axis=0)
        aligned_temp, _ = align_templates(temp, ref_unit=ref_set)
        templates1[unit] = aligned_temp[0]
        templates2[unit] = aligned_temp[1]
    
    return templates1, templates2
                              

def align_templates(templates, ref_unit=None):

    if ref_unit is None:
        max_idx = templates.ptp(1).max(1).argmax(0)
        ref_template = templates[max_idx]
    else:
        ref_template = templates[ref_unit]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]
       

    temps = templates[:, :, max_chan]

    best_shifts = align_get_shifts_with_ref(
                    temps, ref_template)

    aligned_templates = shift_chans(templates, best_shifts)
    
    return aligned_templates, best_shifts


def run_template_computation(
    out_dir,
    fname_spike_train,
    reader,
    spike_size=None,
    unit_ids=None,
    multi_processing=False,
    n_processors=1):

    logger = logging.getLogger(__name__)
    
    logger.info("computing templates")

    # make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    #fname_templates = os.path.join(out_dir, 'templates.npy')
    #if os.path.exists(fname_templates):
    #    return fname_templates

    # make temp folder
    tmp_folder = os.path.join(out_dir, 'tmp_template')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # partition spike train per unit for multiprocessing
    #fname_spike_times, n_units = partition_spike_time(
    #    tmp_folder, fname_spike_train)

    n_units = np.max(np.load(fname_spike_train)[:, 1]) + 1

    if unit_ids is None:
        unit_ids = np.arange(n_units)

    # gather input arguments
    fnames_out = []
    for unit in unit_ids:
        fnames_out.append(os.path.join(
            tmp_folder,
            "template_unit_{}.npy".format(unit)))

    #
    if spike_size is None:
        spike_size = reader.spike_size
    
    # run computing function
    if multi_processing:
        parmap.starmap(run_template_computation_parallel,
                       list(zip(unit_ids, fnames_out)),
                       fname_spike_train,
                       reader,
                       spike_size,
                       processes=n_processors,
                       pm_pbar=True)
    else:
        for ctr in unit_ids:
            run_template_computation_parallel(
                fname_spike_train,
                fnames_out[ctr],
                reader,
                spike_size)

    # gather all info
    templates_new = np.zeros((n_units, spike_size, reader.n_channels),
                             'float32')
    for ctr, unit in enumerate(unit_ids):
        if os.path.exists(fnames_out[ctr]):
            templates_new[unit] = np.load(fnames_out[ctr])

    fname_templates = os.path.join(out_dir, 'templates.npy')
    np.save(fname_templates, templates_new)

    return fname_templates


def run_template_computation_parallel(
    unit_id, fname_out, fname_spike_train, reader, spike_size):

    if os.path.exists(fname_out):
        return

    # load spike times
    spike_train = np.load(fname_spike_train)
    spike_times = spike_train[spike_train[:, 1] == unit_id, 0]

    if len(spike_times) > 0:
        template = compute_a_template(spike_times,
                                      reader,
                                      spike_size)
    else:
        template = np.zeros(
            (spike_size, reader.n_channels), 'float32')

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
    wf = reader.read_waveforms(spike_times, spike_size)[0]

    max_channel = np.mean(wf, axis=0).ptp(0).argmax()

    wf, _ = align_waveforms(wf=wf,
                            max_channel=max_channel,
                            upsample_factor=5,
                            nshifts=3)

    return np.mean(wf, axis=0).astype('float32')

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
    fname = os.path.join(save_dir, 'spike_times.npy')
    np.save(fname, spike_index_list)

    return fname, n_units

def align_waveforms(wf, max_channel=None, upsample_factor=5, nshifts=7):

    # get shapes
    n_spikes, n_times, n_channels = wf.shape

    # mean shape and max channel
    mean_wf = np.mean(wf, axis=0)
    if max_channel is None:
        max_channel = mean_wf.ptp(0).argmax()

    shifts = align_get_shifts_with_ref(
        wf[:, :, max_channel], None, upsample_factor, nshifts)
    
    wf_aligned = shift_chans(wf, shifts)
    
    return wf_aligned, shifts
    
def align_get_shifts_with_ref(wf, ref=None, upsample_factor=5, nshifts=7):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    # Cat: TODO: Peter's fix to clip/lengthen loaded waveforms to match reference templates    
    n_data, n_time = wf.shape

    if ref is None:
        ref = np.mean(wf, axis=0)

    #n_time_rf = len(ref)
    #if n_time > n_time_rf:
    #    left_cut = (n_time - n_time_rf)//2
    #    right_cut = n_time - n_time_rf - left_cut
    #    wf = wf[:, left_cut:-right_cut]
    #elif n_time < n_time_rf:
    #    left_buffer = np.zeros((n_data, (n_time_rf - n_time)//2))
    #    right_buffer = np.zeros((n_data,n_time_rf - n_time - left_buffer))
    #    wf = np.concatenate((left_buffer, wf, right_buffer), axis=1)
      
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1

    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = nshifts//2
    wf_end = -nshifts//2
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-(nshifts//2), (nshifts//2)+1)):
        ref_shifted[:,i] = ref_upsampled[s + wf_start: s + wf_end]

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


def fix_template_edges(templates, w=None):
    """zero pads around the template edges"""

    n_unit, n_chan, n_time = templates.shape
    temps = np.pad(templates, ((0, 0), (0, 0), (w, w)), 'constant')
    idx = temps.argmin(axis=2)

    # shift idx relative to the center of the original templates
    ptps = templates.ptp(2)
    mcs = ptps.argmax(1)
    templates_mc = np.zeros((n_unit, n_time))
    for j in range(n_unit):
        templates_mc[j] = templates[j,mcs[j]]
    min_time_all = int(np.median(np.argmin(templates_mc, 1)))
    shift = n_time//2 - min_time_all
    idx += shift

    # Get a window outside of the minimum and set it to zero
    idx_dim_2 = np.arange(n_chan).repeat(n_time)
    for k in range(n_unit):
        idx_dim_3 = (idx[k] + np.arange(w, w + n_time)[:, None]).T.flatten() % (n_time + 2 * w)
        temps[k][idx_dim_2, idx_dim_3] = 0

    return temps[..., w:-w]


def fix_template_edges_by_file(fname_templates, center_length, perm=[0, 2, 1]):
    """
    Given a template .npy file it fixes the edges.

    input:
    fname_template: str
        Template .npy file name that has to be altered to fix the edges.
    center_length: the size of template not to be zeroed-out
    perm: list of int size 3
        How to numpy.ndarray.permute the template to get the order
        #units, #channels, #timesteps. The default mode handles many of the
        current formats.
    """

    templates = np.load(fname_templates)

    if templates.shape[1] <= center_length:
        # does nothing
        return
    
    if perm is not None:
        templates = templates.transpose(perm)

    templates = fix_template_edges(templates, w=center_length//2)
    if perm is not None:
        templates = templates.transpose(perm)

    np.save(fname_templates, templates)
