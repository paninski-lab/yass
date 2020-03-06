import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d

from yass import read_config
from yass.reader import READER
from yass.residual.residual_gpu import RESIDUAL_GPU2

def run_template_update(output_directory,
                        fname_templates, fname_spike_train,
                        fname_shifts, fname_scales,
                        fname_residual, residual_dtype, residual_offset=0,
                        update_weight=50, units_to_update=None):
    
    fname_templates_out = os.path.join(output_directory, 'templates.npy')

    if not os.path.exists(fname_templates_out):
        
        print('updating templates')

        CONFIG = read_config()

        # output folder
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # reader
        if CONFIG.deconvolution.deconv_gpu:
            n_sec_chunk = CONFIG.resources.n_sec_chunk_gpu_deconv
        else:
            n_sec_chunk = CONFIG.resources.n_sec_chunk
        reader_residual = READER(fname_residual,
                                 residual_dtype,
                                 CONFIG,
                                 n_sec_chunk,
                                 offset=residual_offset)

        # residual obj that can shift templates in gpu
        residual_comp = RESIDUAL_GPU2(
            None, CONFIG, None, None, None,
            None, None, None, None, None, True)
        residual_comp.load_templates(fname_templates)
        residual_comp.make_bsplines_parallel()

        avg_min_max_vals, weights = get_avg_min_max_vals(
        fname_templates, fname_spike_train,
        fname_shifts, fname_scales,
        reader_residual, residual_comp, units_to_update)

        templates_updated = update_templates(
            fname_templates, weights,
            avg_min_max_vals, update_weight, units_to_update)

        np.save(fname_templates_out, templates_updated)

    return fname_templates_out


def get_avg_min_max_vals(fname_templates, fname_spike_train,
                         fname_shifts, fname_scales,
                         reader_residual, residual_comp,
                         units_to_update=None):
    # load input data
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)
    shifts = np.load(fname_shifts)
    scales = np.load(fname_scales)

    n_units, n_times, n_channels = templates.shape
    
    if units_to_update is None:
        units_to_update = np.arange(n_units)
    idx_in = np.in1d(spike_train[:, 1], units_to_update)
    spike_train = spike_train[idx_in]
    shifts = shifts[idx_in]
    scales = scales[idx_in]

    # get vis chan
    ptp_temps = templates.ptp(1)
    ptp_max = ptp_temps.max(1)
    vis_chans = [None]*n_units
    for k in range(n_units):
        vis_chans[k] = np.where(ptp_temps[k] > 0)[0]

    min_max_loc = np.stack((templates.argmin(1),
                            templates.argmax(1))).transpose(1, 0, 2)
    min_max_loc[min_max_loc < 2] = 2
    min_max_loc[min_max_loc > n_times -3] = n_times - 3

    # move to gpu
    templates = torch.from_numpy(templates).float().cuda()
    spike_train = torch.from_numpy(spike_train).long().cuda()
    shifts = torch.from_numpy(shifts).float().cuda()
    scales = torch.from_numpy(scales).float().cuda()
    min_max_loc = torch.from_numpy(min_max_loc).long().cuda()
    ptp_temps = torch.from_numpy(ptp_temps).float().cuda()

    min_max_vals_avg = [None]*n_units
    weights = [None]*n_units
    for batch_id in tqdm(range(reader_residual.n_batches)):

        t_start, t_end = reader_residual.idx_list[batch_id]
        batch_offset = int(t_start - reader_residual.buffer)
        idx_in = torch.nonzero((spike_train[:, 0] > t_start)&(spike_train[:, 0] <= t_end))[:, 0]

        spike_times_batch = spike_train[idx_in, 0]
        spike_times_batch -= batch_offset
        neuron_ids_batch = spike_train[idx_in, 1]

        shifts_batch = shifts[idx_in]
        scales_batch = scales[idx_in]

        # get residual batch
        residual = reader_residual.read_data_batch(batch_id, add_buffer=True)
        residual = torch.from_numpy(residual).cuda()

        # get min/max values
        # due to memory constraint, find the values 10,000 spikes at a time
        max_spikes = 5000
        counter = torch.cat((torch.arange(0, len(spike_times_batch), max_spikes),
                             torch.tensor([len(spike_times_batch)]))).cuda()

        # get the values near template min/max locations for each channel
        # min_max_vals_spikes: # spikes x 2 (min/max) x # channels x 5 (-2 to 2 of argmin/argmax)
        # this will get min/max values of cleaned spikes.
        # it will first get residual values and then add templates
        min_max_vals_spikes = torch.cuda.FloatTensor(len(spike_times_batch), 2, 5, n_channels).fill_(0)
        for j in range(len(counter)-1):
            ii_start = counter[j]
            ii_end = counter[j+1]

            min_max_loc_spikes = (min_max_loc[neuron_ids_batch[ii_start:ii_end]] + 
                                  spike_times_batch[ii_start:ii_end, None, None] - n_times//2)
            min_max_loc_spikes = min_max_loc_spikes[:, :, None] + torch.arange(-2, 3).cuda()[None, None, :, None]
            min_max_vals_spikes[ii_start:ii_end] = torch.gather(
                residual, 0, min_max_loc_spikes.reshape(-1, n_channels)).reshape(
                -1, 2, 5, n_channels)

            shifted_templates = residual_comp.get_shifted_templates(
                    neuron_ids_batch[ii_start:ii_end], 
                    shifts_batch[ii_start:ii_end], 
                    scales_batch[ii_start:ii_end])
            min_max_loc_spikes = min_max_loc[
                neuron_ids_batch[ii_start:ii_end]][
                :, :, None] + torch.arange(-2, 3).cuda()[None, None, :, None]
            min_max_vals_spikes[ii_start:ii_end] += torch.gather(
                shifted_templates,
                1,
                min_max_loc_spikes.reshape(-1, 10, n_channels)
            ).reshape(-1, 2, 5, n_channels)
        del min_max_loc_spikes
        del shifted_templates
        del residual

        ptps_spikes = torch.max(min_max_vals_spikes[:,1], 1)[0] - torch.min(min_max_vals_spikes[:,0], 1)[0]
        diffs = torch.abs(ptps_spikes - ptp_temps[neuron_ids_batch])
        weights_batch = diffs < ptp_temps[neuron_ids_batch]*0.2
        weights_batch[diffs < 3] = 1

        del ptps_spikes
        del diffs

        for k in units_to_update:
            idx_ = neuron_ids_batch == k
            if min_max_vals_avg[k] is not None:
                min_max_vals_avg[k] += torch.sum(
                    min_max_vals_spikes[idx_]*weights_batch[idx_][:, None, None], 0)[:,:,vis_chans[k]]
                weights[k] += torch.sum(weights_batch[idx_], 0)[vis_chans[k]]
            else:
                min_max_vals_avg[k] = torch.sum(
                    min_max_vals_spikes[idx_]*weights_batch[idx_][:, None, None], 0)[:,:,vis_chans[k]]
                weights[k] = torch.sum(weights_batch[idx_], 0)[vis_chans[k]]

        del min_max_vals_spikes
        del weights_batch
        del spike_times_batch
        del neuron_ids_batch
        del shifts_batch
        del scales_batch
        torch.cuda.empty_cache()

    min_max_vals_avg_cpu = [None]*n_units
    weights_cpu = [None]*n_units
    for k in units_to_update:
        weights_ = weights[k].cpu().data.numpy().astype('float32')
        weights_[weights_==0] = 0.0000001
        min_max_vals_avg_ = min_max_vals_avg[k].cpu().data.numpy()
        min_max_vals_avg_cpu[k] = min_max_vals_avg_/weights_
        weights_cpu[k] = weights_

    del residual_comp
    torch.cuda.empty_cache()

    return min_max_vals_avg_cpu, weights_cpu


def quad_interp_loc(pts):
    ''' find x-shift after fitting quadratic to 3 points
        Input: [n_peaks, 3] which are values of three points centred on obj_func peak
        Assumes: equidistant spacing between sample times (i.e. the x-values are hardcoded below)
    '''

    num = ((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/2
    denom = -2*((pts[0]-pts[1])-(((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/(2)))
    num[denom==0] = 1
    denom[denom==0] = 1
    return (num/denom)-1    


def quad_interp_val(vals, shift):
    a = 0.5*vals[0] + 0.5*vals[2] - vals[1]
    b = -0.5*vals[0] + 0.5*vals[2]
    c = vals[1]
    
    return a*shift**2 + b*shift + c 


def update_templates(fname_templates, weights, avg_min_max_vals,
                     update_weight=50, units_to_update=None):

    templates = np.load(fname_templates)

    n_units, n_times, n_chans = templates.shape

    if units_to_update is None:
        units_to_update = np.arange(n_units)

    ptp_temps = templates.ptp(1)
    ptp_max = ptp_temps.max(1)
    vis_chans = [None]*n_units
    for k in range(n_units):
        vis_chans[k] = np.where(ptp_temps[k] > 0)[0]

    templates_updated = np.copy(templates)
    for k in units_to_update:
        # get geometric update weights
        weight_new = 1 - np.exp(-weights[k]/update_weight)

        # avg ptp of this batch
        ptp_avg = np.max(avg_min_max_vals[k][1], 0) - np.min(avg_min_max_vals[k][0], 0) 

        # min location in this batch
        min_vals = avg_min_max_vals[k][0]
        peak_loc_integer = min_vals.argmin(0)
        peak_loc = np.copy(peak_loc_integer).astype('float32')
        for j in range(5):
            idx_ = peak_loc_integer == j
            if j == 0:
                subsample_shift = quad_interp_loc(
                    min_vals[j:j+3, idx_])
                subsample_shift[subsample_shift < -1] = -1
                subsample_shift[subsample_shift > -0.5] = -1
                subsample_shift += 1
            elif j > 0 and j < 4:
                subsample_shift = quad_interp_loc(
                    min_vals[j-1:j+2, idx_])
            elif j == 4:
                subsample_shift = quad_interp_loc(
                    min_vals[j-2:j+1, idx_])
                subsample_shift[subsample_shift > 1] = 1
                subsample_shift[subsample_shift < 0.5] = 1
                subsample_shift -= 1
            peak_loc[idx_] += subsample_shift
        peak_loc -= 2

        # min location of templates (subsample shift)
        temp_k = templates[k][:, vis_chans[k]]
        ptp_temp = temp_k.ptp(0)
        temp_min_loc = temp_k.argmin(0)
        loc_3pts = temp_min_loc[:, None] + np.arange(-1, 2)
        loc_3pts[loc_3pts < 0] = 0
        loc_3pts[loc_3pts > n_times-1] = n_times - 1
        chan_idx = np.tile(np.arange(temp_k.shape[1])[:, None], (1, 3))
        val_3pts = temp_k[loc_3pts, chan_idx].T
        temp_peak_loc = quad_interp_loc(val_3pts)
        temp_peak_loc[np.isnan(temp_peak_loc)] = 0


        # update peak location
        peak_loc_updated = (1-weight_new)*temp_peak_loc + weight_new*peak_loc
        # and determine how much has shifted
        shifts = peak_loc_updated - temp_peak_loc
        shifts[ptp_temp.argmax()] = 0
        shifts[np.abs(shifts) < 0.05] = 0
        # update ptp
        ptps_updated = (1-weight_new)*ptp_temp + weight_new*ptp_avg


        # scale templates
        scale = ptps_updated/ptp_temp
        temp_k_scaled = temp_k*scale[None]

        # shift templates
        temp_k_updated = np.zeros_like(temp_k_scaled)
        t_range = np.arange(n_times)
        for c in range(temp_k.shape[1]):

            if shifts[c] == 0:
                temp_k_updated[:, c] = temp_k_scaled[:, c]
            else:
                t_range_new = t_range - shifts[c]
                #f = interp1d(t_range, templates_scaled[unit,:,c], 'cubic', fill_value='extrapolate')
                f = interp1d(t_range, temp_k_scaled[:,c],
                             'cubic', bounds_error=False, fill_value=0.0)
                temp_k_updated[:, c] = f(t_range_new)

        templates_updated[k,: ,vis_chans[k]] = temp_k_updated.T

    return templates_updated
