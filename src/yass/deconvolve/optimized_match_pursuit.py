import copy
import numpy as np
import scipy
import matplotlib.pyplot as plt


class OptimizedMatchPursuit(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, data, temps, refrac_period=20, threshold=2.,
            conv_approx_rank=3, upsample=1, vis_su=2., keep_iterations=False):
        """Sets up the deconvolution object.

        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        conv_approx_rank: int
            Rank of SVD decomposition for approximating convolution
            operations for templates.
        threshold: float
            amount of energy differential that is admissible by each
            spike. The lower this threshold, more spikes are recovered.
        upsample: int
            Upsampling factor to be used for fitting templates. If 1, no
            upsampling is done. If non-positive, dynamic upsampling will be used.
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        keep_iterations: boolean
            Keeps the spike train per iteration if True. Otherwise,
            does not keep the history.
        """
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps.astype(np.float32)
        self.orig_temps = temps.astype(np.float32)
        # Dynamic Upsampling Setup.
        if upsample < 1:
            self.unit_up_factor = np.power(
                    2, np.floor(np.log2(np.max(temps.ptp(axis=0), axis=0))))
            self.up_factor = min(32, int(np.max(self.unit_up_factor)))
            self.unit_up_factor[self.unit_up_factor > 32] = 32
            self.up_up_map = np.zeros(
                    self.n_unit * self.up_factor, dtype=np.int32)
            for i in range(self.n_unit):
                u_idx = i * self.up_factor
                u_factor = self.unit_up_factor[i]
                skip = self.up_factor // u_factor
                self.up_up_map[u_idx:u_idx + self.up_factor] = u_idx  + np.arange(
                        0, self.up_factor, skip).repeat(skip)
        else:
            # Upsample and downsample time shifted versions
            self.up_factor = upsample
            self.up_up_map = range(self.n_unit * self.up_factor)
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.vis_su_threshold = vis_su
        self.vis_chan = None
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        # Upsample the templates
        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor
        self.orig_template_idx = np.arange(0, self.n_unit, self.up_factor)
        # Computing SVD for each template.
        self.compress_templates()
        # Compute pairwise convolution of filters
        self.pairwise_filter_conv()
        # compute norm of templates
        self.norm = np.zeros([self.orig_n_unit, 1], dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                    np.square(self.temps[:, self.vis_chan[:, i], i]))
        # Setting up data properties
        self.keep_iterations = keep_iterations
        self.update_data(data)
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        # Energey reduction for assigned spikes.
        self.dist_metric = np.array([])
        # Single time preperation for high resolution matches
        # matching indeces of peaks to indices of upsampled templates
        factor = self.up_factor
        radius = factor // 2 + factor % 2
        self.up_window = np.arange(-radius - 1, radius + 1)[:, None]
        self.up_window_len = len(self.up_window)
        off = (factor + 1) % 2
        # Indices of single time window the window around peak after upsampling
        self.zoom_index = (radius + 1) * factor + np.arange(-factor // 2, radius)
        peak_to_template_idx = np.append(
                np.arange(radius + off, factor),
                np.arange(radius + off))
        self.peak_to_template_idx = np.pad(
                peak_to_template_idx, (1, 0), 'edge')
        if off:
            self.peak_to_template_idx[0] -= 1
        peak_time_jitter = np.array([1, 0]).repeat(radius)
        peak_time_jitter[radius - 1] = 0
        self.peak_time_jitter = np.pad(peak_time_jitter, (1, 0), 'edge')
        # Refractory Perios Setup.
        self.refrac_radius = refrac_period
        # Account for upsampling window so that np.inf does not fall into the
        # window around peak for valid spikes.
        self.adjusted_refrac_radius = max(
                1, self.refrac_radius - self.up_factor // 2)

    def update_data(self, data):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = data.astype(np.float32)
        self.data_len = data.shape[0]
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros(
                [self.orig_n_unit, self.obj_len],
                dtype=np.float32)
        # Indicator for computation of the objective.
        self.obj_computed = False
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.iter_spike_train = []

    def visible_chans(self):
        if self.vis_chan is None:
            a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
            self.vis_chan = a > self.vis_su_threshold
        return self.vis_chan

    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        self.unit_overlap = self.unit_overlap > 0
        self.unit_overlap = np.repeat(self.unit_overlap, self.up_factor, axis=0)

    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        idx = np.logical_xor(
                np.ones(self.temps.shape, dtype=bool), self.vis_chan)
        self.temps[idx] = 0.

    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        self.temporal, self.singular, self.spatial = np.linalg.svd(
            np.transpose(np.flipud(self.temps), (2, 0, 1)))
        # Keep only the strongest components
        self.temporal = self.temporal[:, :, :self.approx_rank]
        self.singular = self.singular[:, :self.approx_rank]
        self.spatial = self.spatial[:, :self.approx_rank, :]
        # Upsample the temporal components of the SVD
        # in effect, upsampling the reconstruction of the
        # templates.
        if self.up_factor == 1:
            # No upsampling is needed.
            self.temporal_up = self.temporal
            return
        self.temporal_up = scipy.signal.resample(
                self.temporal, self.n_time * self.up_factor, axis=1)
        idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(self.up_factor)[:, None]
        self.temporal_up = np.reshape(
                self.temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank]).astype(np.float32)

    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""
        conv_res_len = self.n_time * 2 - 1
        self.pairwise_conv = []
        for i in range(self.n_unit):
            self.pairwise_conv.append(None)
        available_upsampled_units = np.unique(self.up_up_map)
        for unit2 in available_upsampled_units:
            # Set up the unit2 conv all overlaping original units.
            n_overlap = np.sum(self.unit_overlap[unit2, :])
            self.pairwise_conv[unit2] = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
            orig_unit = unit2 // self.up_factor
            masked_temp = np.flipud(np.matmul(
                    self.temporal_up[unit2] * self.singular[orig_unit][None, :],
                    self.spatial[orig_unit, :, :]))
            for j, unit1 in enumerate(np.where(self.unit_overlap[unit2, :])[0]):
                u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1] 
                vis_chan_idx = self.vis_chan[:, unit1]

                mat_mul_res = np.matmul(
                        masked_temp[:, vis_chan_idx], vh[:self.approx_rank, vis_chan_idx].T)
                for i in range(self.approx_rank):
                    self.pairwise_conv[unit2][j, :] += np.convolve(
                            mat_mul_res[:, i],
                            s[i] * u[:, i].flatten(), 'full')
        self.pairwise_conv = np.array(self.pairwise_conv)

    def get_reconstructed_upsampled_templates(self):
        """Get the reconstructed upsampled versions of the original templates.

        If no upsampling was requested, returns the SVD reconstructed version
        of the original templates.
        """
        rec = np.matmul(
                self.temporal_up * np.repeat(self.singular, self.up_factor, axis=0)[:, None, :],
                np.repeat(self.spatial, self.up_factor, axis=0))
        return np.fliplr(rec).transpose([1, 2, 0])

    def get_sparse_upsampled_templates(self):
        """Returns the fully upsampled sparse version of the original templates.

        returns:
        --------
        Tuple of numpy.ndarray. First element is of shape (t, C, M) is the set
        updampled shifted templates that have been used in the dynamic
        upsampling approach. Second is an array of lenght K (number of original
        units) * maximum upsample factor. Which maps cluster ids that are result
        of deconvolution to 0,...,M-1 that corresponds to the sparse upsampled
        templates.
        """
        down_sample_idx = np.arange(
                0, self.n_time * self.up_factor, self.up_factor)
        down_sample_idx = down_sample_idx + np.arange(
                0, self.up_factor)[:, None]
        result = []
        # Reordering the upsampling. This is done because we upsampled the time
        # reversed temporal components of the SVD reconstruction of the
        # templates. This means That the time-reveresed 10x upsampled indices
        # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
        # the original templates.
        all_temps = []
        reorder_idx = np.append(
                np.arange(0, 1),
                np.arange(self.up_factor - 1, 0, -1))
        # Sequentialize the number of up_up_map. For instance,
        # [0, 0, 0, 0, 4, 4, 4, 4, ...] turns to [0, 0, 0, 0, 1, 1, 1, 1, ...].
        deconv_id_sparse_temp_map = []
        tot_temps_so_far = 0
        for i in range(self.orig_n_unit):
            up_temps = scipy.signal.resample(
                    self.orig_temps[:, :, i],
                    self.n_time * self.up_factor)[down_sample_idx, :]
            up_temps = up_temps.transpose([1, 2, 0])
            up_temps = up_temps[:, :, reorder_idx]
            skip = self.up_factor // self.unit_up_factor[i]
            keep_upsample_idx = np.arange(0, self.up_factor, skip).astype(np.int32)
            deconv_id_sparse_temp_map.append(np.arange(
                    self.unit_up_factor[i]).repeat(skip) + tot_temps_so_far)
            tot_temps_so_far += self.unit_up_factor[i]
            all_temps.append(up_temps[:, :, keep_upsample_idx])

        deconv_id_sparse_temp_map = np.concatenate(
                deconv_id_sparse_temp_map, axis=0)
        return np.concatenate(all_temps, axis=2), deconv_id_sparse_temp_map

    def get_upsampled_templates(self):
        """Returns the fully upsampled version of the original templates."""
        down_sample_idx = np.arange(0, self.n_time * self.up_factor, self.up_factor)
        down_sample_idx = down_sample_idx + np.arange(0, self.up_factor)[:, None]
        up_temps = scipy.signal.resample(
                self.orig_temps, self.n_time * self.up_factor)[down_sample_idx, :, :]
        up_temps = up_temps.transpose(
            [2, 3, 0, 1]).reshape([self.n_chan, -1, self.n_time]).transpose([2, 0, 1]) 
        # Reordering the upsampling. This is done because we upsampled the time
        # reversed temporal components of the SVD reconstruction of the
        # templates. This means That the time-reveresed 10x upsampled indices
        # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
        # the original templates.
        reorder_idx = np.tile(
                np.append(
                    np.arange(0, 1),
                    np.arange(self.up_factor - 1, 0, -1)),
                self.orig_n_unit)
        reorder_idx += np.arange(
                0, self.up_factor * self.orig_n_unit,
                self.up_factor).repeat(self.up_factor)
        return up_temps[:, :, reorder_idx]

    def correct_shift_deconv_spike_train(self):
        """Get time shift corrected version of the deconvovled spike train.

        This corrected version only applies if you consider getting upsampled
        templates with get_upsampled_templates() method.
        """
        correct_spt = copy.copy(self.dec_spike_train)
        correct_spt[correct_spt[:, 1] % self.up_factor > 0, 0] += 1
        return correct_spt

    def compute_objective(self):
        """Computes the objective given current state of recording."""
        if self.obj_computed:
            return self.obj
        n_rows = self.orig_n_unit * self.approx_rank
        matmul_result = np.matmul(
                self.spatial.reshape([n_rows, -1]) * self.singular.reshape([-1, 1]),
                self.data.T)
        conv_result = np.zeros(
                [n_rows, self.data_len + self.n_time - 1], dtype=np.float32)
        filters = self.temporal.transpose([0, 2, 1]).reshape([n_rows, -1])
        for i in range(n_rows):
            conv_result[i, :] = np.convolve(
                    matmul_result[i, :], filters[i, :], mode='full')
        for i in range(1, self.approx_rank):
            conv_result[np.arange(0, n_rows, self.approx_rank), :] +=\
                    conv_result[np.arange(i, n_rows, self.approx_rank), :]
        self.obj = 2 * conv_result[np.arange(0, n_rows, self.approx_rank), :] - self.norm
        # Set indicator to true so that it no longer is run
        # for future iterations in case subtractions are done
        # implicitly.
        self.obj_computed = True
        return self.obj

    def high_res_peak(self, times, unit_ids):
        """Finds best matching high resolution template.

        Given an original unit id and the infered spike times
        finds out which of the shifted upsampled templates of
        the unit best matches at that time to the residual.

        Parameters:
        -----------
        times: numpy.array of numpy.int
            spike times for the unit.
        unit_ids: numpy.array of numpy.int
            Respective to times, id of each spike corresponding
            to the original units.

        Returns:
        --------
            tuple in the form of (numpy.array, numpy.array, numpy.array)
            respectively the offset of shifted templates and a necessary time
            shift to correct the spike time, and the index of spike times that
            do not violate refractory period.
        """
        if self.up_factor == 1 or len(times) < 1:
            return 0, 0, range(len(times))
        idx = times + self.up_window
        peak_window = self.obj[unit_ids, idx]
        # Find times that the window around them do not inlucde np.inf.
        # In other words do not violate refractory period.
        invalid_idx = np.logical_or(
            np.isinf(peak_window[0, :]), np.isinf(peak_window[-1, :]))
        # Turn off the invlaid units for next iterations.
        turn_off_idx = times[invalid_idx] + np.arange(
                - self.refrac_radius, 1)[:, None]
        self.obj[unit_ids[invalid_idx], turn_off_idx] = - np.inf
        valid_idx = np.logical_not(invalid_idx)
        peak_window = peak_window[:, valid_idx]
        if peak_window.shape[1]  == 0:
            return np.array([]), np.array([]), valid_idx 
        high_resolution_peaks = scipy.signal.resample(
                peak_window, self.up_window_len * self.up_factor, axis=0)
        shift_idx = np.argmax(
                high_resolution_peaks[self.zoom_index, :], axis=0)
        return self.peak_to_template_idx[shift_idx], self.peak_time_jitter[shift_idx], valid_idx

    def find_peaks(self):
        """Finds peaks in subtraction differentials of spikes."""
        max_across_temp = np.max(self.obj, axis=0)
        spike_times = scipy.signal.argrelmax(
                max_across_temp, order=self.refrac_radius)[0]
        spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        dist_metric = max_across_temp[spike_times]
        # TODO(hooshmand): this requires a check of the last element(s)
        # of spike_times only not of all of them since spike_times
        # is sorted already.
        valid_idx = spike_times < self.data_len - self.n_time
        dist_metric = dist_metric[valid_idx]
        spike_times = spike_times[valid_idx]
        # Upsample the objective and find the best shift (upsampled)
        # template.
        spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        upsampled_template_idx, time_shift, valid_idx = self.high_res_peak(
                spike_times, spike_ids)
        # The spikes that had NAN in the window and could not be updampled
        # should fall-back on default value.
        spike_ids *= self.up_factor
        if np.sum(valid_idx) > 0:
            spike_ids[valid_idx] += upsampled_template_idx
            spike_times[valid_idx] -= time_shift
        # Note that we shift the discovered spike times from convolution
        # Space to actual raw voltate space by subtracting self.n_time
        result = np.append(
            spike_times[:, None] - self.n_time + 1,
            spike_ids[:, None], axis=1)

        return result, dist_metric[valid_idx]

    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        window = np.arange(- self.adjusted_refrac_radius, self.adjusted_refrac_radius)
        n_spikes = spike_train.shape[0]
        win_len = len(window)
        # The offset self.n_time - 1 is necessary to revert the spike times
        # back to objective function indices which is the result of convoultion
        # operation.
        time_idx = (spike_train[:, 0:1] + self.n_time - 1) + window
        # Re-adjust cluster id's so that they match
        # with the original templates
        unit_idx = spike_train[:, 1:2] // self.up_factor
        self.obj[unit_idx, time_idx[:, 1:-1]] = - np.inf

    def enforce_regularization(self, spike_train):
        """Adds sparsity regularization term to the objective function.

        Assuming that rates
        """
        cosnt = 10
        self.obj -= spike_train.shape[0] * const

    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        present_units = np.unique(spt[:, 1])
        for i in present_units:
            conv_res_len = self.n_time * 2 - 1
            unit_sp = spt[spt[:, 1] == i, :]
            spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1] 
            # Grid idx of subset of channels and times
            unit_idx = self.unit_overlap[i]
            idx = np.ix_(unit_idx, spt_idx.ravel())
            self.obj[idx] -= np.tile(
                    2 * self.pairwise_conv[self.up_up_map[i]], len(unit_sp))

        self.enforce_refractory(spt)
        # self.enforce_regularization(spt)

    def get_iteration_spike_train(self):
        return self.iter_spike_train

    def run(self, max_iter):
        ctr = 0
        tot_max = np.inf
        self.compute_objective()
        while tot_max > self.threshold and ctr < max_iter:
            spt, dist_met = self.find_peaks()
            print "Iteration {0} Found {1} spikes with {2:.2f} energy reduction.".format(
                ctr, spt.shape[0], np.sum(dist_met))
            if len(spt) == 0:
                break
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            self.subtract_spike_train(spt)
            if self.keep_iterations:
                self.iter_spike_train.append(spt)
            self.dist_metric = np.append(self.dist_metric, dist_met)
            ctr += 1
        return self.dec_spike_train, self.dist_metric

