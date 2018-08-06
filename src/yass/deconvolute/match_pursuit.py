import numpy as np
import scipy
import time, os

import copy
from tqdm import tqdm

# ******************* MATCH PURSUIT FUNCTION ****************
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

class MatchPursuit2(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, temps, deconv_dir, vis_chan, 
                 threshold=10, conv_approx_rank=3,
                 implicit_subtraction=True, obj_energy=True):
        """Sets up the deconvolution object.
        Parameters:
        -----------
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        obj_energy: boolean
            Whether to include ||V||^2 term in the objective.
        """
        self.vis_chan = vis_chan
        self.deconv_dir = deconv_dir
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.implicit_subtraction = implicit_subtraction
        #self.vis_chan = None
        #self.visible_chans()
        self.obj_energy=obj_energy
        
        self.up_factor = 1
        if self.up_factor > 1:
            self.upsample_templates()
            
        # compute norm of templates
        print ("norm of templates")
        self.norm = np.zeros([self.n_unit, 1])
        for i in range(self.n_unit):
            self.norm[i] = np.sum(np.square(self.temps[:, self.vis_chan[:, i], i]))
        
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])


    def upsample_templates(self):
        """Computes downsampled shifted upsampled of templates."""
        down_sample_idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(0, self.up_factor)[:, None]
        self.up_temps = scipy.signal.resample(self.temps, self.n_time * 3)[down_sample_idx, :, :]
        self.up_temps = self.up_temps.transpose(
            [2, 3, 0, 1]).reshape([self.n_chan, -1, self.n_time]).transpose([2, 0, 1])
        self.temps = self.up_temps
        self.n_unit = self.n_unit * self.up_factor
        

    def update_data(self, data):
        """Updates the data for the deconv to be run on with same templates."""
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.data = data
        self.data_len = data.shape[0]
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros([self.n_unit, self.obj_len])
        # Compute v_sqaured if it is included in the objective.
        if obj_energy:
            self.update_v_squared()
        # Indicator for computation of the objective.
        self.obj_computed = False
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])

    #def visible_chans(self):
        #if self.vis_chan is None:
            #a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
            #self.vis_chan = a > 1
        #return self.vis_chan

    #def visible_chans_new(self):
        #if self.vis_chan is None:
            ##a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
            #max_chan_ptp = self.temps[self.temps.ptp(0).argmax(0)]
            #a = self.temps.ptp(0)
            #self.vis_chan = a > max_chan_ptp*0.5
        #return self.vis_chan


    #def pairwise_filter_conv(self):
        #"""Computes pairwise convolution of templates using SVD approximation."""

        #if os.path.exists(self.deconv_dir+"/parwise_conv.npy") == False:
            #conv_res_len = self.n_time * 2 - 1
            #self.pairwise_conv = np.zeros([self.n_unit, self.n_unit, conv_res_len])
            #for unit1 in range(self.n_unit):
                #u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1]
                #vis_chan_idx = self.vis_chan[:, unit1]
                #for unit2 in range(self.n_unit):
                    #for i in range(self.approx_rank):
                        #self.pairwise_conv[unit2, unit1, :] += np.convolve(
                            #np.matmul(self.temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                            #s[i] * u[:, i].flatten(), 'full')

            #np.save(self.deconv_dir+"/parwise_conv.npy", self.pairwise_conv)
            
        #else:
            #self.pairwise_conv = np.load(self.deconv_dir+"/parwise_conv.npy")


    def update_v_squared(self):
        """Updates the energy of consecutive windows of data."""
        one_pad = np.ones([self.n_time, self.n_chan])
        self.v_squared = conv_filter(np.square(self.data), one_pad, approx_rank=None)



    #def approx_conv_filter(self, unit):
        #conv_res = 0.
        #u, s, vh = self.temporal[unit], self.singular[unit], self.spatial[unit]
        #for i in range(self.approx_rank):
            #vis_chan_idx = self.vis_chan[:, unit]
            #conv_res += np.convolve(
                #np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T),
                                    #s[i] * u[:, i].flatten(), 'full')
                                    
            ##print (self.data[:, vis_chan_idx].shape)
            ##print (vh[i, vis_chan_idx].T.shape)
            ##print (np.matmul(self.data[:, vis_chan_idx], vh[i, vis_chan_idx].T).shape)
            ##print ((s[i] * u[:, i].flatten()).shape)
            ##quit()
        #return conv_res

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
        
        # Cat: TODO: eventually don't want to save this to disk as it's 
        #            ~500MB for 49 chans @ 10sec; and ~5GB for 512 chans
        self.obj_chunk_file = (os.path.split(self.standardized_filename)[0]+
                            "/deconv/chunk_"+ str(self.chunk_ctr).zfill(6)+
                            "/objective_seg_"+str(self.proc_index).zfill(6)+".npy")
        if os.path.exists(self.obj_chunk_file)==False:
            if self.obj_computed and self.implicit_subtraction:
                return self.obj
            for i in range(self.n_unit):
                self.dot[i, :] = self.approx_conv_filter(i)
                    
            self.obj = 2 * self.dot - self.norm
            if self.obj_energy:
                self.obj -= self.v_squared

            # Enforce refrac period
            radius = self.n_time // 2
            window = np.arange(- radius, radius)
            
            for i in range(self.n_unit):
                unit_sp = self.dec_spike_train[self.dec_spike_train[:, 1] == i, 0]
                refrac_idx = unit_sp[:, np.newaxis] + window
                self.obj[i, refrac_idx] = - np.inf

            # Set indicator to true so that it no longer is run
            # for future iterations in case subtractions are done
            # implicitly.
            
            
            np.save(self.obj_chunk_file, self.obj)
        else:
            self.obj = np.load(self.obj_chunk_file)
            
        self.obj_computed = True
        return self.obj


    #def find_peaks(self):
        #refrac_period = self.n_time
        #max_across_temp = np.max(self.obj, 0)
        #spike_times = scipy.signal.argrelmax(max_across_temp, order=refrac_period)[0]
        #spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        #dist_metric = max_across_temp[spike_times]
        ## TODO(hooshmand): this requires a check of the last element(s)
        ## of spike_times only not of all of them since spike_times
        ## is sorted already.
        #valid_idx = spike_times < self.data_len - self.n_time
        #dist_metric = dist_metric[valid_idx]
        #spike_times = spike_times[valid_idx]
        #spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        #result = np.append(
            #spike_times[:, np.newaxis] - self.n_time + 1,
            #spike_ids[:, np.newaxis], axis=1)
        #return result, dist_metric


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
        

    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        radius = self.n_time // 2
        window = np.arange(- radius, radius)
        for i in range(self.n_unit):
            refrac_idx = []
            for j in range(self.up_factor):
                unit_sp = spike_train[spike_train[:, 1] == i * self.up_factor + j, 0]
                refrac_idx.append(unit_sp[:, np.newaxis] + window)
            for idx in refrac_idx:
                self.obj[i:i + self.up_factor, idx] = - np.inf
                
                
    #def subtract_spike_train(self, spt):
        #"""Substracts a spike train from the original spike_train."""
        #if not self.implicit_subtraction:
            #for i in range(self.n_unit):
                #unit_sp = spt[spt[:, 1] == i, :]
                #self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]
            ## There is no need to update v_squared if it is not included in objective.
            #if self.obj_energy:
                #self.update_v_squared()
        #else:
            #for i in range(self.n_unit):
                #conv_res_len = self.n_time * 2 - 1
                #unit_sp = spt[spt[:, 1] == i, :]
                #spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1]
                #temp = 2 * self.pairwise_conv[i, :, :][:, None, :]
                #self.obj[:, spt_idx] -= temp


    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        if not self.implicit_subtraction:
            for i in range(self.n_unit):
                unit_sp = spt[spt[:, 1] == i, :]
                self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]
            # There is no need to update v_squared if it is not included in objective.
            if self.obj_energy:
                self.update_v_squared()
            # recompute objective function
            self.compute_objective()
            # enforce refractory period for the current
            # global spike train.
            self.enforce_refractory(self.dec_spike_train)
        else:
            for i in range(self.n_unit):
                conv_res_len = self.n_time * 2 - 1
                unit_sp = spt[spt[:, 1] == i, :]
                spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1]
                self.obj[:, spt_idx] -= 2 * self.pairwise_conv[i, :, :][:, None, :]
            self.enforce_refractory(spt)
            
            



    def binary_reader(self, idx_list, buffer_size, n_channels):
        
        # New indexes
        data_start = idx_list[0]
        data_end = idx_list[1]
        offset = idx_list[2]

        # ***** LOAD RAW RECORDING *****
        with open(self.standardized_filename, "rb") as fin:
            if data_start == 0:
                # Seek position and read N bytes
                recordings_1D = np.fromfile(
                    fin,
                    dtype='float32',
                    count=(data_end + buffer_size) * n_channels)
                recordings_1D = np.hstack((np.zeros(
                    buffer_size * n_channels, dtype='float32'), recordings_1D))
            else:
                fin.seek((data_start - buffer_size) * 4 * n_channels, os.SEEK_SET)
                recordings_1D = np.fromfile(
                    fin,
                    dtype='float32',
                    count=((data_end - data_start + buffer_size * 2) * n_channels))

            if len(recordings_1D) != (
                  (data_end - data_start + buffer_size * 2) * n_channels):
                recordings_1D = np.hstack((recordings_1D,
                                           np.zeros(
                                               buffer_size * n_channels,
                                               dtype='float32')))
        fin.close()

        # Convert to 2D array
        recording = recordings_1D.reshape(-1, n_channels)
        
        return recording
    
    
    def get_iteration_spike_train(self):
        return self.iter_spike_train
    
        
    def run(self, data_temp):
        
        ''' data_in is a bunch of indexes (not raw data)
        '''                    

        data_in = data_temp[0]
        chunk_ctr = data_temp[1]
        max_iter = data_temp[2]
        buffer_size = data_temp[3]
        standardized_filename = data_temp[4]
        n_channels = data_temp[5]
               
        self.chunk_ctr = chunk_ctr
        self.standardized_filename = standardized_filename
        self.n_channels = n_channels
        
        idx_list = data_in[0]
        self.proc_index = data_in[1]
        #print ("deconv chunk: ", self.proc_index)
        
        #self.chunk_dir = (os.path.split(self.standardized_filename)[0]+
                            #"/deconv/chunk_"+ str(self.proc_index).zfill(6))
        #if not os.path.isdir(self.chunk_dir):
            #os.makedirs(self.chunk_dir)
        
        start_time = time.time()
        data = self.binary_reader(idx_list, buffer_size,
                                    n_channels)
        print ("chunk: ", self.proc_index, "  file read time: ", 
                    np.round(time.time()-start_time,3))
        
        # post-initializzation steps
        # initialize dot product matrix
        start_time = time.time()
        self.data = data
        self.data_len = data.shape[0]
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros([self.n_unit, self.obj_len])
        #print ("initializing time: ", time.time()-start_time)
                
        # Compute v_sqaured if it is included in the objective.
        start_time = time.time()
        if self.obj_energy:
            self.update_v_squared()
        #print ("update_v_squared time: ", time.time()-start_time)
        
        # Indicator for computation of the objective.
        self.obj_computed = False

        thresh_list = []
        ctr = 0
        tot_max = np.inf
        self.compute_objective()
        while tot_max > self.threshold and ctr<5:
            start_time = time.time()
            spt, dist_met = self.find_peaks()

            start_time = time.time()
            self.subtract_spike_train(spt)

            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            self.dist_metric = np.append(self.dist_metric, dist_met)
            
            tot_max = np.max(self.obj[:,300:-300])

            ctr += 1
            print (" chunk: ", str(chunk_ctr), "  segment: ", self.proc_index, 
                   "  Iteration ", ctr, " # spikes: ", spt.shape[0], 
                   " max obj: ", tot_max)
            thresh_list.append(spt.shape[0])


        return self.dec_spike_train, self.dist_metric


# **********************************************************


def snr_main_chans(temps):
    """Computes peak to peak SNR for given templates."""
    chan_peaks = np.max(temps, axis=0)
    chan_lows = np.min(temps, axis=0)
    peak_to_peak = chan_peaks - chan_lows
    return np.argsort(peak_to_peak, axis=0)


class MatchPursuitAnalyze(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, data, spike_train, temps, n_channels, n_features):
        """Sets up the deconvolution object.
        Parameters:
        -----------
        data: numpy.ndarray of shape (T, C)
            Where T is number of time samples and C number of channels.
        spike_train: numpy.ndarray of shape(T, 2)
            First column represents spike times and second column
            represents cluster ids.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        """
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps
        self.spike_train = spike_train
        self.data = data
        self.data_len = data.shape[0]
        #
        self.n_main_chan = n_channels
        self.n_feat = n_features
        #
        self.features = None
        self.cid = None
        self.means = None
        self.snrs = snr_main_chans(temps)
        self.n_clusters = 0
        #
        self.residual = None
        self.get_residual()

    def get_residual(self):
        """Returns the residual or computes it the first time."""
        if self.residual is None:         
            self.residual = copy.copy(self.data)
            for i in tqdm(range(self.n_unit), 'Computing Residual'):
                unit_sp = self.spike_train[self.spike_train[:, 1] == i, :]
                self.residual[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]

        return self.residual

    def get_unit_spikes(self, unit):
        """Gets clean spikes for a given unit."""
        unit_sp = self.spike_train[self.spike_train[:, 1] == unit, :]
        # Add the spikes of the current unit back to the residual
        return self.residual[np.arange(0, self.n_time) + unit_sp[:, :1], :] + self.temps[:, :, unit]
