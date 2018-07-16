import numpy as np
import scipy


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


class MatchPursuit(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, data, temps, threshold=10, conv_approx_rank=3,
                 implicit_subtraction=True, obj_energy=True):
        """Sets up the deconvolution object.

        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        obj_energy: boolean
            Whether to include ||V||^2 term in the objective.
        """
        
        #print ("recordings shape: ", data.shape)
        #print ("templates: ", temps.shape)
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps
        self.data = data
        self.data_len = data.shape[0]
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.implicit_subtraction = implicit_subtraction
        # Computing SVD for each template.
        self.temporal, self.singular, self.spatial = np.linalg.svd(
            np.transpose(np.flipud(temps), (2, 0, 1)))
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros([self.n_unit, self.obj_len])
        # Compute pairwise convolution of filters
        self.pairwise_filter_conv()
        # compute norm of templates
        self.norm = np.zeros([self.n_unit, 1])
        for i in range(self.n_unit):
            self.norm[i] = np.sum(np.square(self.temps[:, :, i]))
        # Compute v_sqaured if it is included in the objective.
        self.obj_energy = obj_energy
        if obj_energy:
            self.update_v_squared()
        # Indicator for computation of the objective.
        self.obj_computed = False
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])

    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""
        conv_res_len = self.n_time * 2 - 1
        self.pairwise_conv = np.zeros([self.n_unit, self.n_unit, conv_res_len])
        for unit1 in range(self.n_unit):
            u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1]
            for unit2 in range(self.n_unit):
                for i in range(self.approx_rank):
                    self.pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(self.temps[:, :, unit2], vh[i, :].T),
                        s[i] * u[:, i].flatten(), 'full')

    def update_v_squared(self):
        one_pad = np.ones([self.n_time, self.n_chan])
        self.v_squared = conv_filter(np.square(self.data), one_pad, approx_rank=None)

    def approx_conv_filter(self, unit):
        conv_res = 0.
        u, s, vh = self.temporal[unit], self.singular[unit], self.spatial[unit]
        for i in range(self.approx_rank):
            conv_res += np.convolve(
                np.matmul(self.data, vh[i, :].T),
                s[i] * u[:, i].flatten(), 'full')
        return conv_res

    def compute_objective(self):
        """Computes the objective given current state of recording."""
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
        self.obj_computed = True
        return self.obj

    def find_peaks(self):
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

    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        if not self.implicit_subtraction:
            for i in range(self.n_unit):
                unit_sp = spt[spt[:, 1] == i, :]
                self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]
            # There is no need to update v_squared if it is not included in objective.
            if self.obj_energy:
                self.update_v_squared()
        else:
            for i in range(self.n_unit):
                conv_res_len = self.n_time * 2 - 1
                unit_sp = spt[spt[:, 1] == i, :]
                spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1]
                self.obj[:, spt_idx] -= 2 * self.pairwise_conv[i, :, :][:, None, :]

    def run(self, max_iter=3):
        ctr = 0
        tot_max = np.inf
        while tot_max > self.threshold and ctr < max_iter:
            
            self.compute_objective()
            spt, dist_met = self.find_peaks()
            self.subtract_spike_train(spt)
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            self.dist_metric = np.append(self.dist_metric, dist_met)
            tot_max = np.max(self.obj)
            ctr += 1
            #print ("Iteration {} Found {} spikes with Max Obj {}.".format(
            #    ctr, spt.shape[0], tot_max))
        return self.dec_spike_train, self.dist_metric
