import numpy as np
import torch
from tqdm import tqdm

import cudaSpline as deconv
from scipy.interpolate import splrep

def fit_spline(curve, knots=None, prepad=0, postpad=0, order=3):
    if knots is None:
        knots = np.arange(len(curve) + prepad + postpad)
    return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)


def transform_template(template, knots=None, prepad=7, postpad=3, order=3):

    if knots is None:
        knots = np.arange(len(template.data[0]) + prepad + postpad)
    splines = [
        fit_spline(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
        for curve in template.data.cpu().numpy()
    ]
    coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
    return deconv.Template(torch.from_numpy(coefficients).cuda(), template.indices)


class SOFTNOISEASSIGNMENT(object):
    def __init__(self, fname_spike_train, fname_templates, fname_shifts, fname_scales,
                 reader_residual, detector, channel_index, large_unit_threshold):
        
        self.templates = np.load(fname_templates).astype('float32')
        self.spike_train = np.load(fname_spike_train)
        self.shifts = np.load(fname_shifts)
        self.scales = np.load(fname_scales)

        self.reader_residual = reader_residual
        self.channel_index = channel_index
        
        detector.temporal_filter1[0].padding = [0, 0]
        self.detector = detector
        
        self.n_units, self.n_times_templates, self.n_channels = self.templates.shape
        self.n_times_nn = self.detector.temporal_filter1[0].weight.shape[2]
        
        self.n_times_extra = 3
        self.n_neigh_chans = self.channel_index.shape[1]
        self.n_total_spikes = self.spike_train.shape[0]
        
        
        self.preprocess_templates_and_spike_times()
        self.exclude_large_units(large_unit_threshold)
        self.get_bspline_coeffs()
        self.move_to_torch()

    def preprocess_templates_and_spike_times(self):
        
        # templates on neighboring channels
        self.mcs = self.templates.ptp(1).argmax(1)
        templates_neigh = np.zeros((self.n_units, self.n_times_templates, self.n_neigh_chans))
        for k in range(self.n_units):
            neigh_chans = self.channel_index[self.mcs[k]]
            neigh_chans = neigh_chans[neigh_chans<self.n_channels]

            templates_neigh[k, :, :len(neigh_chans)] = self.templates[k,:,neigh_chans].T

        # get alignment shift
        min_points = templates_neigh[:,:,0].argmin(1)
        ref_min = int(np.median(min_points))
        self.temp_shifts = min_points - ref_min

        # add buffer for alignment
        buffer_size = np.max((self.n_times_nn//2 +
                              self.n_times_extra +
                              np.max(np.abs(self.temp_shifts)) -
                              self.n_times_templates//2, 0))

        if buffer_size > 0:
            buffer = np.zeros((self.n_units, buffer_size, self.n_neigh_chans))
            templates_neigh = np.concatenate((buffer, templates_neigh, buffer), axis=1)

        # get ailgned templates
        t_in = np.arange(-(self.n_times_nn//2)-self.n_times_extra,
                         self.n_times_nn//2+self.n_times_extra+1) + templates_neigh.shape[1]//2

        templates_aligned = np.zeros((self.n_units,
                                      self.n_times_nn+self.n_times_extra*2,
                                      self.n_neigh_chans), 'float32')
        for k in range(self.n_units):
            temp = templates_neigh[k]
            t_in_temp = t_in + self.temp_shifts[k]
            templates_aligned[k] = templates_neigh[k,t_in_temp]

        self.templates_aligned = templates_aligned

        # shift spike times according to the alignment
        self.spike_train[:, 0] += self.temp_shifts[self.spike_train[:, 1]]
        
    def exclude_large_units(self, threshold):
        
        norms = np.zeros(self.n_units)
        for j in range(self.n_units):
            temp = self.templates[j]
            vis_chan = np.where(temp.ptp(0) > 1)[0]
            norms[j] = np.sum(np.square(temp[:, vis_chan]))
            
        units_in = np.where(norms < threshold)[0]

        self.idx_included = np.where(np.in1d(self.spike_train[:,1], units_in))[0]
        self.spike_train = self.spike_train[self.idx_included]
        self.shifts = self.shifts[self.idx_included]
        self.scales = self.scales[self.idx_included]

    def move_to_torch(self):
        self.templates_aligned = torch.from_numpy(self.templates_aligned).float().cuda()
        self.spike_train = torch.from_numpy(self.spike_train).long().cuda()
        self.shifts = torch.from_numpy(self.shifts).float().cuda()
        self.scales = torch.from_numpy(self.scales).float().cuda()
        
        self.mcs = torch.from_numpy(self.mcs).cuda()
        self.channel_index = torch.from_numpy(self.channel_index).cuda()
        
    def get_bspline_coeffs(self):

        n_data, n_times, n_channels = self.templates_aligned.shape

        channels = torch.arange(n_channels).cuda()
        temps_torch = torch.from_numpy(-(self.templates_aligned.transpose(0, 2, 1))).cuda().contiguous()

        temp_cpp = deconv.BatchedTemplates([deconv.Template(temp, channels) for temp in temps_torch])
        self.coeffs = deconv.BatchedTemplates([transform_template(template) for template in temp_cpp])

    def get_shifted_templates(self, temp_ids, shifts, scales):
        
        temp_ids = torch.from_numpy(temp_ids.cpu().numpy()).long().cuda()
        shifts = torch.from_numpy(shifts.cpu().numpy()).float().cuda()
        scales = torch.from_numpy(scales.cpu().numpy()).float().cuda()

        n_sample_run = 1000
        n_times = self.templates_aligned.shape[1]

        idx_run = np.hstack((np.arange(0, len(shifts), n_sample_run), len(shifts)))

        shifted_templates = torch.cuda.FloatTensor(len(shifts), n_times, self.n_neigh_chans).fill_(0)
        for j in range(len(idx_run)-1):
            ii_start = idx_run[j]
            ii_end = idx_run[j+1]
            obj = torch.zeros(self.n_neigh_chans, (ii_end-ii_start)*n_times + 10).cuda()
            times = torch.arange(0, (ii_end-ii_start)*n_times, n_times).long().cuda() + 5
            deconv.subtract_splines(obj,
                                    times,
                                    shifts[ii_start:ii_end],
                                    temp_ids[ii_start:ii_end],
                                    self.coeffs, 
                                    scales[ii_start:ii_end])
            obj = obj[:, 5:-5].reshape((self.n_neigh_chans, (ii_end-ii_start), n_times))
            shifted_templates[ii_start:ii_end] = obj.transpose(0,1).transpose(1,2)
    
        return shifted_templates

    def compute_soft_assignment(self):
        
        probs = torch.zeros(len(self.spike_train)).cuda()

        # batch offsets
        offsets = torch.from_numpy(self.reader_residual.idx_list[:, 0]
                                   - self.reader_residual.buffer).cuda().long()

        t_range = torch.arange(-(self.n_times_nn//2), self.n_times_nn//2+1).cuda()
        with tqdm(total=self.reader_residual.n_batches) as pbar:

            for batch_id in range(self.reader_residual.n_batches):

                # load residual data
                resid_dat = self.reader_residual.read_data_batch(batch_id, add_buffer=True)
                resid_dat = torch.from_numpy(resid_dat).cuda()

                # relevant idx
                idx_in = torch.nonzero(
                    (self.spike_train[:, 0] > self.reader_residual.idx_list[batch_id][0]) & 
                    (self.spike_train[:, 0] < self.reader_residual.idx_list[batch_id][1]))[:,0]

                spike_train_batch = self.spike_train[idx_in] 
                spike_train_batch[:, 0] -= offsets[batch_id]

                shift_batch = self.shifts[idx_in]
                scale_batch = self.scales[idx_in]

                # get residual snippets
                t_index = spike_train_batch[:, 0][:, None] + t_range
                c_index = self.channel_index[self.mcs[spike_train_batch[:,1]]].long()

                resid_dat = torch.cat((resid_dat, torch.zeros((resid_dat.shape[0], 1)).cuda()), 1)
                resid_snippets = resid_dat[t_index[:,:,None], c_index[:,None]]


                # get shifted templates
                shifted_templates = self.get_shifted_templates(
                    spike_train_batch[:,1], shift_batch, scale_batch)
                shifted_templates = shifted_templates[:, self.n_times_extra:-self.n_times_extra]

                # get clean wfs
                clean_wfs = resid_snippets + shifted_templates

                n_sample_run = 1000

                idx_list = np.hstack((
                    np.arange(0, clean_wfs.shape[0], n_sample_run), clean_wfs.shape[0]))
                probs_batch = torch.zeros(len(clean_wfs)).cuda()
                for j in range(len(idx_list)-1):
                    probs_batch[idx_list[j]:idx_list[j+1]] = self.detector(
                        clean_wfs[idx_list[j]:idx_list[j+1]])[0][:, 0]

                probs[idx_in] = probs_batch.data
                
                pbar.update()
                
            del probs_batch

        probs_included = probs.cpu().numpy()

        probs = np.ones(self.n_total_spikes, 'float32')
        probs[self.idx_included] = probs_included

        return probs
