from yass.empty import empty
import numpy as np

def make_CONFIG2(CONFIG):
    ''' Makes a copy of several attributes of original config parameters
        to be sent into parmap function; original CONFIG can't be pickled;
    '''
    
    # make a copy of the original CONFIG object;
    # multiprocessing doesn't like the methods in original CONFIG        
    CONFIG2 = empty()
    CONFIG2.recordings=empty()
    CONFIG2.resources=empty()
    CONFIG2.deconvolution=empty()
    CONFIG2.data=empty()
    CONFIG2.neuralnetwork=empty()
    
    CONFIG2.recordings.sampling_rate = CONFIG.recordings.sampling_rate
    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    CONFIG2.recordings.spike_size_ms = CONFIG.recordings.spike_size_ms
    
    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing
    CONFIG2.resources.n_sec_chunk = CONFIG.resources.n_sec_chunk
    CONFIG2.resources.n_gpu_processors = CONFIG.resources.n_gpu_processors
    
    try:
        CONFIG2.resources.n_sec_chunk_gpu_deconv = CONFIG.resources.n_sec_chunk_gpu_deconv
        CONFIG2.resources.n_sec_chunk_gpu = CONFIG2.resources.n_sec_chunk_gpu_deconv
    except:
        CONFIG2.resources.n_sec_chunk_gpu = CONFIG.resources.n_sec_chunk_gpu
        print ("older config")
    
    try:
        CONFIG2.resources.gpu_id = CONFIG.resources.gpu_id
    except:
        CONFIG2.resources.gpu_id = 0
        print ("older config")


    CONFIG2.data.root_folder = CONFIG.data.root_folder
    CONFIG2.data.geometry = CONFIG.data.geometry
    CONFIG2.geom = CONFIG.geom

    CONFIG2.neigh_channels = CONFIG.neigh_channels

    CONFIG2.spike_size = CONFIG.spike_size
    CONFIG2.center_spike_size = CONFIG.center_spike_size
    CONFIG2.spike_size_nn = CONFIG.spike_size_nn
    
    CONFIG2.deconvolution.threshold = CONFIG.deconvolution.threshold
    CONFIG2.deconvolution.deconv_gpu = CONFIG.deconvolution.deconv_gpu
    CONFIG2.deconvolution.update_templates = CONFIG.deconvolution.update_templates
    CONFIG2.deconvolution.template_update_time = CONFIG.deconvolution.template_update_time
    CONFIG2.deconvolution.neuron_discover_time = CONFIG.deconvolution.neuron_discover_time
    CONFIG2.deconvolution.drift_model = CONFIG.deconvolution.drift_model
    CONFIG2.deconvolution.min_split_spikes = CONFIG.deconvolution.min_split_spikes
    CONFIG2.deconvolution.neuron_discover = CONFIG.deconvolution.neuron_discover
    
    CONFIG2.rec_len = CONFIG.rec_len
    
    CONFIG2.torch_devices = CONFIG.torch_devices

    CONFIG2.neuralnetwork.apply_nn = CONFIG.neuralnetwork.apply_nn
    
    CONFIG2.neuralnetwork.training = empty()
    CONFIG2.neuralnetwork.training.spike_size_ms = CONFIG.neuralnetwork.training.spike_size_ms
    
    CONFIG2.neuralnetwork.detect = empty()
    CONFIG2.neuralnetwork.detect.filename = CONFIG.neuralnetwork.detect.filename
    CONFIG2.neuralnetwork.detect.n_filters = CONFIG.neuralnetwork.detect.n_filters
    
    CONFIG2.neuralnetwork.denoise = empty()
    CONFIG2.neuralnetwork.denoise.n_filters = CONFIG.neuralnetwork.denoise.n_filters
    CONFIG2.neuralnetwork.denoise.filename = CONFIG.neuralnetwork.denoise.filename
    CONFIG2.neuralnetwork.denoise.filter_sizes = CONFIG.neuralnetwork.denoise.filter_sizes
    
    
    CONFIG2.cluster = empty()
    CONFIG2.cluster.prior = empty()
    CONFIG2.cluster.prior.beta = CONFIG.cluster.prior.beta
    CONFIG2.cluster.prior.a = CONFIG.cluster.prior.a
    CONFIG2.cluster.prior.lambda0 = CONFIG.cluster.prior.lambda0
    CONFIG2.cluster.prior.nu = CONFIG.cluster.prior.nu
    CONFIG2.cluster.prior.V = CONFIG.cluster.prior.V
    

    return CONFIG2


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
