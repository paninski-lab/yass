import numpy as np
import scipy
import time, os

import copy
from tqdm import tqdm
import time
from yass.cluster.util import (binary_reader, load_waveforms_from_memory)


# ********************************************************
# ********************************************************
# ********************************************************

class MatchPursuit_objectiveUpsample(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, temps, deconv_chunk_dir, standardized_filename, 
                 max_iter, refrac_period=20, upsample=1, threshold=10., 
                 conv_approx_rank=5,
                 vis_su=2., 
                 keep_iterations=False):
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
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        keep_iterations: boolean
            Keeps the spike train per iteration if True. Otherwise,
            does not keep the history.
        """
        
        self.deconv_dir = deconv_chunk_dir
        self.standardized_filename = standardized_filename
        self.max_iter = max_iter


        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps.astype(np.float32)
        self.orig_temps = temps.astype(np.float32)
        print ("  inside match pursuit: templates: ", self.temps.shape)
        
        
        # Upsample and downsample time shifted versions
        self.up_factor = upsample
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
        print ("  # upsample tempaltes: ", self.n_unit)

        # Computing SVD for each template.
        self.compress_templates()
        
        # Compute pairwise convolution of filters
        print ("  computing temp_temp (todo: parallelize)")
        self.pairwise_filter_conv()
        
        # compute norm of templates
        self.norm = np.zeros([self.orig_n_unit, 1], dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                    np.square(self.temps[:, self.vis_chan[:, i], i]))
        
        # Setting up data properties
        self.keep_iterations = keep_iterations
        #self.update_data(data)
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
        # Stack for turning on invalud units for next iteration
        self.turn_off_stack = []
        
    def update_data(self):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = self.data.astype(np.float32)
        self.data_len = self.data.shape[0]
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
                self.temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank])


    # Cat: TODO: Parallelize this function
    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""

        if os.path.exists(self.deconv_dir+"/pairwise_conv.npy") == False:

            conv_res_len = self.n_time * 2 - 1
            self.pairwise_conv = np.zeros(
                    [self.n_unit, self.orig_n_unit, conv_res_len], dtype=np.float32)
            for unit1 in range(self.orig_n_unit):
                u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1]
                vis_chan_idx = self.vis_chan[:, unit1]
                for unit2 in np.where(self.unit_overlap[:, unit1])[0]:
                    orig_unit = unit2 // self.up_factor
                    masked_temp = np.flipud(np.matmul(
                            self.temporal_up[unit2] * self.singular[orig_unit],
                            self.spatial[orig_unit, :, vis_chan_idx].T))
                    for i in range(self.approx_rank):
                        self.pairwise_conv[unit2, unit1, :] += np.convolve(
                            np.matmul(masked_temp, vh[i, vis_chan_idx].T),
                            s[i] * u[:, i].flatten(), 'full')
            
            np.save(self.deconv_dir+"/pairwise_conv.npy", self.pairwise_conv)
        else:
            self.pairwise_conv = np.load(self.deconv_dir+"/pairwise_conv.npy")

        # Cat: Set this to None as it is reloaded by each core separately, shoulnd't be
        #       duplicated during the object parallelization step
        self.pairwise_conv = None
                               
    
    def get_reconstructed_upsampled_templates(self):
        """Get the reconstructed upsampled versions of the original templates.
        If no upsampling was requested, returns the SVD reconstructed version
        of the original templates.
        """
        rec = np.matmul(
                self.temporal_up * np.repeat(self.singular, self.up_factor, axis=0)[:, None, :],
                np.repeat(self.spatial, self.up_factor, axis=0))
        return np.fliplr(rec).transpose([1, 2, 0])
        
        
    def get_upsampled_templates(self):
        """Returns the fully upsampled version of the original templates."""
        down_sample_idx = np.arange(0, self.n_time * self.up_factor, self.up_factor)
        down_sample_idx = down_sample_idx + np.arange(0, self.up_factor)[:, None]
        up_temps = scipy.signal.resample(
                self.orig_temps, self.n_time * self.up_factor)[down_sample_idx, :, :]
        up_temps = up_temps.transpose(
            [2, 3, 0, 1]).reshape([self.n_chan, -1, self.n_time]).transpose([2, 0, 1])
        self.n_unit = self.n_unit * self.up_factor
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
                     

    def approx_conv_filter(self, unit):
        """Approximation of convolution of a template with the data.
        Parameters:
        -----------
        unit: int
            Id of the unit whose filter will be convolved with the data.
        """
        conv_res = 0.
        u, s, vh = self.temporal[unit], self.singular[unit], self.spatial[unit]
        for i in range(self.approx_rank):
            vis_chan_idx = self.vis_chan[:, unit]
            conv_res += np.convolve(
                np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T),
                s[i] * u[:, i].flatten(), 'full')
        return conv_res


    def compute_objective(self):
        """Computes the objective given current state of recording."""
        if self.obj_computed:
            return self.obj
        for i in range(self.orig_n_unit):
            self.dot[i, :] = self.approx_conv_filter(i)
        self.obj = 2 * self.dot - self.norm
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
        spike_ids = spike_ids[valid_idx] * self.up_factor + upsampled_template_idx
        spike_times = spike_times[valid_idx] - time_shift
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
            self.obj[idx] -= np.tile(2 * self.pairwise_conv[i, unit_idx, :], len(unit_sp))

        self.enforce_refractory(spt)
        
        
    def get_iteration_spike_train(self):
        return self.iter_spike_train

        
    def run(self, data_in):

        start_time = time.time()
        verbose = False
        self.idx_list = data_in[0][0]
        self.seg_ctr = data_in[0][1]
        self.chunk_ctr = data_in[1]
        self.buffer_size = data_in[2]
        
        # ********* run deconv ************
        fname_out = (self.deconv_dir+"/seg_{}_deconv.npz".format(
                                            str(self.seg_ctr).zfill(6)))

        # read raw data for segment using idx_list vals
        #self.load_data_from_memory()
        data = binary_reader(self.idx_list, self.buffer_size, 
                             self.standardized_filename,
                             self.n_chan)

        self.data = data.astype(np.float32)
        self.data_len = self.data.shape[0]
        
        # load pairwise conv filter
        self.pairwise_conv = np.load(self.deconv_dir+"/pairwise_conv.npy")

        # update data
        self.update_data()
        
        # compute objective function
        self.compute_objective()
            
        ctr = 0
        tot_max = np.inf
        while tot_max > self.threshold and ctr < self.max_iter:
            spt, dist_met = self.find_peaks()
            
            if len(spt) == 0:
                break
            
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            
            self.subtract_spike_train(spt)
            
            if self.keep_iterations:
                self.iter_spike_train.append(spt)
            self.dist_metric = np.append(self.dist_metric, dist_met)
                        
            if verbose: 
                print ("Iteration {0} Found {1} spikes with {2:.2f} energy reduction.".format(
                ctr, spt.shape[0], np.sum(dist_met)))

            ctr += 1

        print ('  deconv chunk {0}, seg {1}, # iter: {2}, tot_spikes: {3}, tot_time: {4:.2f}'.
                format(
                self.chunk_ctr,self.seg_ctr, ctr, self.dec_spike_train.shape[0],
                time.time()-start_time))

        # ******** ADJUST SPIKE TIMES TO REMOVE BUFFER AND OFSETS *******
        # order spike times
        idx = np.argsort(self.dec_spike_train[:,0])
        self.dec_spike_train = self.dec_spike_train[idx]

        # find spikes inside data block, i.e. outside buffers
        idx = np.where(np.logical_and(self.dec_spike_train[:,0]>=self.idx_list[2],
                                      self.dec_spike_train[:,0]<self.idx_list[3]))[0]
        self.dec_spike_train = self.dec_spike_train[idx]

        # offset spikes to start of index
        self.dec_spike_train[:,0]+= self.idx_list[0] - self.idx_list[2]
        
        np.savez(fname_out, spike_train = self.dec_spike_train, 
                            dist_metric = self.dist_metric)

        #return self.dec_spike_train
        
# ********************************************************
# ********************************************************
# ********************************************************

class MatchPursuit3(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, temps, deconv_chunk_dir, standardized_filename, 
                 max_iter, dynamic_templates, upsample=1, threshold=10, 
                 conv_approx_rank=3,
                 implicit_subtraction=True, obj_energy=False, vis_su=2., 
                 keep_iterations=False, sparse_subtraction=True,
                 broadcast_subtraction=False):
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
        obj_energy: boolean
            Whether to include ||V||^2 term in the objective.
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        keep_iterations: boolean
            Keeps the spike train per iteration if True. Otherwise,
            does not keep the history.
        sparse_subtraction: bool
            Only units that have visible channel overlap affect
            each other's computation in subtraction phase.
        broadcast_subtraction: bool
            If true, the subtraction step is done in linear algebraic
            operations instead of loops. This mode of operation is
            not support sparse subtraction.
        """
        
        #global recording_chunk_raw
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps.astype(np.float32)
        print ("  inside match pursuit: templates: ", self.temps.shape)
        
        self.deconv_dir = deconv_chunk_dir
        self.standardized_filename = standardized_filename
        self.max_iter = max_iter
        
        # Upsample and downsample time shifted versions
        self.up_factor = upsample
        #if self.up_factor > 1:
        if True:
            #self.upsample_templates()
            self.upsample_templates_dynamic()
            print ("  upsample tempaltes: ", self.temps.shape)
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.implicit_subtraction = implicit_subtraction
        self.vis_su_threshold = vis_su
        self.vis_chan = None
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        
        # Computing SVD for each template.
        self.temporal, self.singular, self.spatial = np.linalg.svd(
            np.transpose(np.flipud(self.temps), (2, 0, 1)))        
        
        # Compute pairwise convolution of filters
        print ("  computing pairwise filter (TODO: Parallelize)")
        self.pairwise_filter_conv()
        
        # compute norm of templates
        self.norm = np.zeros([self.n_unit, 1], dtype=np.float32)
        for i in range(self.n_unit):
            self.norm[i] = np.sum(np.square(self.temps[:, self.vis_chan[:, i], i]))

        # Setting up data properties - but don't load data here
        self.keep_iterations = keep_iterations
        self.obj_energy = obj_energy
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.sparse_sub = sparse_subtraction
        self.broadcast_sub = broadcast_subtraction

    def update_data(self):
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros([self.n_unit, self.obj_len], dtype=np.float32)

        # Compute v_sqaured if it is included in the objective.
        if self.obj_energy:
            self.update_v_squared()
        
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


    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        idx = np.logical_xor(np.ones(self.temps.shape, dtype=bool), self.vis_chan)
        self.temps[idx] = 0.


    def upsample_templates(self):
        """Computes downsampled shifted upsampled of templates."""
        down_sample_idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(0, self.up_factor)[:, None]
        self.up_temps = scipy.signal.resample(self.temps, self.n_time * self.up_factor)[down_sample_idx, :, :]
        self.up_temps = self.up_temps.transpose(
            [2, 3, 0, 1]).reshape([self.n_chan, -1, self.n_time]).transpose([2, 0, 1])
        self.temps = self.up_temps
        
        self.n_unit = self.n_unit * self.up_factor
    
    def upsample_templates_dynamic(self):
        
        """Computes downsampled shifted upsampled of templates."""
        
        # compute ptp of templates; 10-20SU: 10 upsample; 30SU+ 20 upsample
        ptps = self.temps.ptp(0).max(0)

        # loop over templates and make multiple upsample + shifted versions
        # of larger templates
        ctr_temp = 0
        up_temps = []
        self.temps_ids = []
        for t in range(ptps.shape[0]):
            if ptps[t]<5:
                up_factor = 1
                for k in range(up_factor): 
                    self.temps_ids.append(t)
            elif (ptps[t]>=5) and (ptps[t]<10):
                up_factor = 3
                for k in range(up_factor):
                    self.temps_ids.append(t)
            elif (ptps[t]>=10) and (ptps[t]<20):
                up_factor = 5
                for k in range(up_factor):
                    self.temps_ids.append(t)
            elif (ptps[t]>=20) and (ptps[t]<40):
                up_factor = 10
                for k in range(up_factor):
                    self.temps_ids.append(t)
            elif (ptps[t]>=40) and (ptps[t]<60):
                up_factor = 15
                for k in range(up_factor):
                    self.temps_ids.append(t)
            elif (ptps[t]>=60):
                up_factor = 25
                for k in range(up_factor):
                    self.temps_ids.append(t)
            
            down_sample_idx = np.arange(0, self.n_time * up_factor, up_factor) + np.arange(0, up_factor)[:, None]
            temp_temp = scipy.signal.resample(self.temps[:,:, t], self.n_time * up_factor)[down_sample_idx, :]
            up_temps.append(temp_temp)

            #print (t, ptps[t], temp_temp.shape)

        self.temps_ids = np.int32(self.temps_ids)
        self.temps = np.vstack(up_temps)
        self.temps = self.temps.transpose([1,2,0])

        self.n_unit = self.temps_ids.shape[0]

        np.savez(self.deconv_dir+'/templates_upsampled.npz', 
                 temps = self.temps,
                 temps_ids = self.temps_ids)

        print (np.unique(self.temps_ids).shape)


    # Cat: TODO: Parallelize this function
    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""
        
        if os.path.exists(self.deconv_dir+"/pairwise_conv.npy") == False:
            conv_res_len = self.n_time * 2 - 1
            self.pairwise_conv = np.zeros([self.n_unit, self.n_unit, conv_res_len], dtype=np.float32)
            for unit1 in range(self.n_unit):
                u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1]
                vis_chan_idx = self.vis_chan[:, unit1]
                for unit2 in np.where(self.unit_overlap[unit1])[0]:
                    for i in range(self.approx_rank):
                        self.pairwise_conv[unit2, unit1, :] += np.convolve(
                            np.matmul(self.temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                            s[i] * u[:, i].flatten(), 'full')

            np.save(self.deconv_dir+"/pairwise_conv.npy", self.pairwise_conv)
        else:
            self.pairwise_conv = np.load(self.deconv_dir+"/pairwise_conv.npy")

        self.pairwise_conv = None

    def update_v_squared(self):
        """Updates the energy of consecutive windows of data."""
        one_pad = np.ones([self.n_time, self.n_chan])
        self.v_squared = self.conv_filter(np.square(self.data), one_pad, approx_rank=None)

    def approx_conv_filter(self, unit):
        """Approximation of convolution of a template with the data.

        Parameters:
        -----------
        unit: int
            Id of the unit whose filter will be convolved with the data.
        """
        conv_res = 0.
        u, s, vh = self.temporal[unit], self.singular[unit], self.spatial[unit]
        for i in range(self.approx_rank):
            vis_chan_idx = self.vis_chan[:, unit]
            
            conv_res += np.convolve(
                np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T),
                s[i] * u[:, i].flatten(), 'full')
                
            #print (self.data[:, vis_chan_idx].shape)
            #print (vh[i, vis_chan_idx].T.shape)
            #print (np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T).shape)
            #print (s[i])
            #print (u[:, i].flatten().shape)
            #print ((s[i] * u[:, i].flatten()).shape)
        
            #quit()

        return conv_res

    def conv_filter(data, temp, approx_rank=None, mode='full'):
        """Convolves multichannel filter with multichannel data.
        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temp: numpy array of shape (t, C)
            Where t is number of time samples and C is the number of
            channels.
        Returns:
        --------
        numpy.array
        result of convolving the filter with the recording.
        """
        n_chan = temp.shape[1]
        conv_res = 0.
        if approx_rank is None or approx_rank > n_chan:
            for c in range(n_chan):
                conv_res += np.convolve(data[:, c], temp[:, c], mode)
        # Low rank approximation of convolution
        else:
            u, s, vh = np.linalg.svd(temp)
            for i in range(approx_rank):
                conv_res += np.convolve(
                    np.matmul(data, vh[i, :].T),
                    s[i] * u[:, i].flatten(), mode)
        return conv_res


    def compute_objective(self):
        """Computes the objective given current state of recording."""
        fname_out = (self.deconv_dir+"/seg_{}_obj_matrix.npy".format(
                                            str(self.seg_ctr).zfill(6)))
        if os.path.exists(fname_out)==False:

            if self.obj_computed and self.implicit_subtraction:
                return self.obj
            for i in range(self.n_unit):
                self.dot[i, :] = self.approx_conv_filter(i)
            self.obj = 2 * self.dot - self.norm
            if self.obj_energy:
                self.obj -= self.v_squared

            # Set indicator to true so that it no longer is run
            # for future iterations in case subtractions are done
            # implicitly.
            
            # Cat: stop saving obj_matrix, data is too large over time;
            #np.save(fname_out,self.obj)
        else: 
            self.obj = np.load(fname_out)

        self.obj_computed = True
        return self.obj

    def find_peaks(self):
        """Finds peaks in subtraction differentials of spikes."""
        refrac_period = self.n_time
        max_across_temp = np.max(self.obj, 0)
        spike_times = scipy.signal.argrelmax(max_across_temp, order=refrac_period)[0]
        spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        dist_metric = max_across_temp[spike_times]
        # TODO(hooshmand): this requires a check of the last element(s)
        # of spike_times only not of all of them since spike_times
        # is sorted already.
        valid_idx = spike_times < self.data_len - self.n_time
        dist_metric = dist_metric[valid_idx]
        spike_times = spike_times[valid_idx]
        spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        result = np.append(
            spike_times[:, np.newaxis] - self.n_time + 1,
            spike_ids[:, np.newaxis], axis=1)
        return result, dist_metric

    def enforce_refractory(self, spike_train, present_units=None):
        """Enforces refractory period for units."""
        radius = self.n_time // 2
        window = np.arange(- radius, radius)
        n_spikes = spike_train.shape[0]
        if present_units is None:
            present_units = np.unique(spike_train[:, 1])
        time_idx = spike_train[:, 0:1] + window
        unit_idx = spike_train[:, 1:2]
        if self.up_factor > 1:
            # Trace unit number back to all the shifted
            # versions of the same unit.
            unit_idx -= unit_idx % self.up_factor
            unit_idx = unit_idx.repeat(self.up_factor, axis=0)
            unit_idx += np.arange(0, self.up_factor)[None, :].repeat(n_spikes, axis=0).ravel()[:, None]
            self.obj[unit_idx, time_idx.repeat(self.up_factor, axis=0)] = -np.inf
            # Alternative way of doing it
            #unit_idx = (np.tile(unit_idx, self.up_factor) + np.arange(self.up_factor)).T.ravel()
            #self.obj[unit_idx[:, None], np.tile(time_idx.T, self.up_factor).T] = -np.inf
        else:
            self.obj[unit_idx, time_idx] = -np.inf


    def enforce_refractory_dynamic(self, spike_train, present_units=None):
        """Enforces refractory period for units."""
        radius = self.n_time // 2
        window = np.arange(- radius, radius)
        n_spikes = spike_train.shape[0]
        if present_units is None:
            present_units = np.unique(spike_train[:, 1])
        time_idx = spike_train[:, 0:1] + window
        unit_idx = spike_train[:, 1:2]
        if self.up_factor > 1:
            # Trace unit number back to all the shifted
            # versions of the same unit.
            unit_idx -= unit_idx % self.up_factor
            unit_idx = unit_idx.repeat(self.up_factor, axis=0)
            unit_idx += np.arange(0, self.up_factor)[None, :].repeat(n_spikes, axis=0).ravel()[:, None]
            self.obj[unit_idx, time_idx.repeat(self.up_factor, axis=0)] = -np.inf
            # Alternative way of doing it
            #unit_idx = (np.tile(unit_idx, self.up_factor) + np.arange(self.up_factor)).T.ravel()
            #self.obj[unit_idx[:, None], np.tile(time_idx.T, self.up_factor).T] = -np.inf
        else:
            self.obj[unit_idx, time_idx] = -np.inf
            
            
    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        present_units = np.unique(spt[:, 1])
        if not self.implicit_subtraction:
            for i in present_units:
                unit_sp = spt[spt[:, 1] == i, :]
                self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]
            # There is no need to update v_squared if it is not included in objective.
            if self.obj_energy:
                self.update_v_squared()
            # recompute objective function
            self.compute_objective()
            # enforce refractory period for the current
            # global spike train.
            #self.enforce_refractory(self.dec_spike_train)
            self.enforce_refractory_dynamic(self.dec_spike_train)
        elif self.broadcast_sub:
            conv_res_len = self.n_time * 2 - 1
            spt_idx = np.arange(0, conv_res_len) + spt[:, :1]
            self.obj[:, spt_idx] -= 2 * self.pairwise_conv[spt[:, 1], :, :].transpose([1, 0, 2])
            #self.enforce_refractory(spt, present_units)
            self.enforce_refractory_dynamic(spt, present_units)
        else:
            for i in present_units:
                conv_res_len = self.n_time * 2 - 1
                unit_sp = spt[spt[:, 1] == i, :]
                spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1]
                if self.sparse_sub:
                    # Grid idx of subset of channels and times
                    unit_idx = self.unit_overlap[i]
                    idx = np.ix_(unit_idx, spt_idx.ravel())
                    self.obj[idx] -= np.tile(2 * self.pairwise_conv[i, unit_idx, :], len(unit_sp))
                else:
                    self.obj[:, spt_idx] -= 2 * self.pairwise_conv[i, :, :][:, None, :]
            #self.enforce_refractory(spt, present_units)
            self.enforce_refractory_dynamic(spt, present_units)

        
    def snr_main_chans(temps):
        """Computes peak to peak SNR for given templates."""
        chan_peaks = np.max(temps, axis=0)
        chan_lows = np.min(temps, axis=0)
        peak_to_peak = chan_peaks - chan_lows
        return np.argsort(peak_to_peak, axis=0)


    def load_data_from_memory(self):
        ''' Index into global variable recording_chunk and select required
            data.
        ''' 
        print ("Loading from memoery...")
        arr = np.frombuffer(self.toShare_new)
        recording_chunk = arr.reshape(-1, self.n_chan)
        
        #print (self.recording_chunk.shape)
        start = self.idx_list[0]
        end = self.idx_list[1]
        buffer_ = self.idx_list[2]
        
        # if first chunk, append buffer at beginning
        if start<buffer_:
            self.data = recording_chunk[start:end+buffer_,:]
            temp_data = np.zeros((buffer_, recording_chunk.shape[1]), 
                                    'float32')
            self.data = np.concatenate((temp_data, self.data))

        # if last chunk, append buffer at end
        elif end > (recording_chunk.shape[0]-buffer_):
            self.data = recording_chunk[start-buffer_:end,:]
            temp_data = np.zeros((buffer_, recording_chunk.shape[1]), 
                                    'float32')
            self.data = np.concatenate((self.data, temp_data),'float32')            
           
        # mid chunks
        else:
            self.data = recording_chunk[start-buffer_:end+buffer_,:]
       
    def run(self, data_in):

        start_time = time.time()

        self.idx_list = data_in[0][0]
        self.seg_ctr = data_in[0][1]
        self.chunk_ctr = data_in[1]
        self.buffer_size = data_in[2]
        
        # ********* run deconv ************
        fname_out = (self.deconv_dir+"/seg_{}_deconv.npz".format(
                                            str(self.seg_ctr).zfill(6)))

        # read raw data for segment using idx_list vals
        #self.load_data_from_memory()
        data = binary_reader(self.idx_list, self.buffer_size, 
                             self.standardized_filename,
                             self.n_chan)

        self.data = data.astype(np.float32)
        self.data_len = self.data.shape[0]
        
        # load pairwise conv filter
        self.pairwise_conv = np.load(self.deconv_dir+"/pairwise_conv.npy")

        # run inside run - function
        self.update_data()
        
        # compute objective function
        self.compute_objective()
            
        ctr = 0
        tot_max = np.inf
        while tot_max > self.threshold and ctr < self.max_iter:
            spt, dist_met = self.find_peaks()
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)

            self.subtract_spike_train(spt)

            if self.keep_iterations:
                self.iter_spike_train.append(spt)
            self.dist_metric = np.append(self.dist_metric, dist_met)
            ctr += 1
            #print ("Iteration {0} Found {1} spikes with {2:.2f} energy reduction, time: {3:.2f}".format(
            #    ctr, spt.shape[0], np.sum(dist_met), time.time()-start_time))
            if len(spt) == 0:
                break
        
        print ("  finished chunk {0}, seg {1}, # iter: {2}, tot_time: {3:.2f}".format(
                            self.chunk_ctr,self.seg_ctr, ctr, time.time()-start_time))

        # ******** ADJUST SPIKE TIMES TO REMOVE BUFFER AND OFSETS *******
        # order spike times
        idx = np.argsort(self.dec_spike_train[:,0])
        self.dec_spike_train = self.dec_spike_train[idx]

        # find spikes inside data block, i.e. outside buffers
        idx = np.where(np.logical_and(self.dec_spike_train[:,0]>=self.idx_list[2],
                                      self.dec_spike_train[:,0]<self.idx_list[3]))[0]
        self.dec_spike_train = self.dec_spike_train[idx]

        # offset spikes to start of index
        self.dec_spike_train[:,0]+= self.idx_list[0] - self.idx_list[2]
        
        np.savez(fname_out, spike_train = self.dec_spike_train, 
                            dist_metric = self.dist_metric)

        #return self.dec_spike_train
    
    
# ********************************************************************
# ********************* RESIDUAL FUNCTIONS ***************************
# ********************************************************************

class MatchPursuitWaveforms(object):
    
    def __init__(self, data, temps, dec_spike_train, buffer_size):

        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        self.data = data
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps
        self.dec_spike_train = dec_spike_train
    
    def compute_residual(self):
        for i in tqdm(range(self.n_unit), '  computing residual postdeconv'):
            unit_sp = self.dec_spike_train[self.dec_spike_train[:, 1] == i, :]
            self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]

    def get_unit_spikes(self, unit, unit_sp):
        """Gets clean spikes for a given unit."""
        #unit_sp = dec_spike_train[dec_spike_train[:, 1] == unit, :]
        
        # Add the spikes of the current unit back to the residual
        temp = self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] + self.temps[:, :, unit]
        return temp

