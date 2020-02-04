import logging
import os
import numpy as np
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from diptest import diptest as dp

import yass
from yass import read_config
from yass.reader import READER
from yass.residual.residual_gpu import RESIDUAL_GPU2
from yass import postprocess

def run_post_deconv_split(output_directory,
                          fname_templates,
                          fname_spike_train,
                          fname_shifts,
                          fname_scales,
                          fname_raw,
                          raw_dtype,
                          fname_residual,
                          residual_dtype,
                          residual_offset=0,
                          update_original_templates=False):

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

    reader_raw = READER(fname_raw,
                        raw_dtype,
                        CONFIG)

    # load input data
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)
    shifts = np.load(fname_shifts)
    scales = np.load(fname_scales)

    # get cleaned ptp
    fname_cleaned_ptp = os.path.join(output_directory, 'cleaned_ptp.npy')
    fname_spike_times = os.path.join(output_directory, 'spike_times.npy')
    fname_shifts = os.path.join(output_directory, 'shifts_list.npy')
    fname_scales = os.path.join(output_directory, 'scales_list.npy')
    fname_vis_chans = os.path.join(output_directory, 'vis_chans.npy')
    if os.path.exists(fname_cleaned_ptp) and os.path.exists(fname_spike_times):
        
        cleaned_ptp = np.load(fname_cleaned_ptp, allow_pickle=True)
        spike_times_list = np.load(fname_spike_times, allow_pickle=True)
        shifts_list = np.load(fname_shifts, allow_pickle=True)
        scales_list = np.load(fname_scales, allow_pickle=True)
        vis_chans = np.load(fname_vis_chans, allow_pickle=True)
    else:
        print('get cleaned ptp')
        (cleaned_ptp, spike_times_list,
         shifts_list, scales_list, vis_chans) = get_cleaned_ptp(
            templates, spike_train, shifts, scales,
            reader_residual, fname_templates, CONFIG)

        np.save(fname_shifts, shifts_list, allow_pickle=True)
        np.save(fname_scales, scales_list, allow_pickle=True)
        np.save(fname_vis_chans, vis_chans, allow_pickle=True)
        np.save(fname_cleaned_ptp, cleaned_ptp, allow_pickle=True)
        np.save(fname_spike_times, spike_times_list, allow_pickle=True)

    # split units
    fname_templates_updated = os.path.join(
        output_directory, 'templated_updated.npy')
    fname_spike_train_updated = os.path.join(
        output_directory, 'spike_train_updated.npy')
    fname_shifts_updated = os.path.join(
        output_directory, 'shifts_updated.npy')
    fname_scales_updated = os.path.join(
        output_directory, 'scales_updated.npy')
    if not(os.path.exists(fname_templates_updated) and 
           os.path.exists(fname_spike_train_updated)):

        print('run split')
        (templates_updated, spike_train_updated,
         shifts_updated, scales_updated) = run_split(
            cleaned_ptp, spike_times_list,
            shifts_list, scales_list, vis_chans,
            templates, spike_train, reader_raw,
            CONFIG, update_original_templates)

        ## add new templates and spike train to the existing one
        #templates_updated = np.concatenate((templates, new_temps), axis=0)
        #spike_train_new[:, 1] += templates.shape[0]
        #spike_train_updated = np.concatenate((spike_train, spike_train_new), axis=0)
        idx_sort = np.argsort(spike_train_updated[:, 0])
        spike_train_updated = spike_train_updated[idx_sort]
        shifts_updated = shifts_updated[idx_sort]
        scales_updated = scales_updated[idx_sort]
        
        np.save(fname_templates_updated, templates_updated)
        np.save(fname_spike_train_updated, spike_train_updated)
        np.save(fname_shifts_updated, shifts_updated)
        np.save(fname_scales_updated, scales_updated)

        # can be used to find gpu memory not freed
        # import gc
        #n_objects = 0
        #for obj in gc.get_objects():
        #    try:
        #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #            print(obj, type(obj), obj.size())
        #
        #            n_objects += 1
        #    except:
        #        pass
        #print(n_objects)

        #units_to_process = np.arange(templates.shape[0], templates_updated.shape[0])

    #else:
    #    templates_updated = np.load(fname_templates_updated)
    #    units_to_process = np.arange(templates.shape[0], templates_updated.shape[0])

    ## kill duplicate templates
    #methods = ['low_ptp', 'duplicate']
    #(fname_templates_out, fname_spike_train_out, 
    # _, _, _)  = postprocess.run(
    #    methods,
    #    os.path.join(output_directory,
    #                 'duplicate_remove'),
    #    None,
    #    None,
    #    fname_templates_updated,
    #    fname_spike_train_updated,
    #    None,
    #    None,
    #    None,
    #    None,
    #    units_to_process)

    #return fname_templates_out, fname_spike_train_out

    return (fname_templates_updated, fname_spike_train_updated,
            fname_shifts_updated, fname_scales_updated)

def get_cleaned_ptp(templates, spike_train, shifts, scales,
                    reader_residual, fname_templates, CONFIG):

    n_units, n_times, n_channels = templates.shape

    ptp = templates.ptp(1)
    ptp_max = ptp.max(1)
    vis_chans = [None]*n_units
    for k in range(n_units):
        vis_chans[k] = np.where(ptp[k] > 0)[0]

    min_max_loc = np.stack((templates.argmin(1),
                            templates.argmax(1))).transpose(1, 0, 2)
    
    # move to gpu
    templates = torch.from_numpy(templates).float().cuda()
    spike_train = torch.from_numpy(spike_train).long().cuda()
    shifts = torch.from_numpy(shifts).float().cuda()
    scales = torch.from_numpy(scales).float().cuda()
    min_max_loc = torch.from_numpy(min_max_loc).long().cuda()
    
    residual_comp = RESIDUAL_GPU2(
        None, CONFIG, None, None, None,
        None, None, None, None, None, True)
    residual_comp.load_templates(fname_templates)
    residual_comp.make_bsplines_parallel()
    
    cleaned_ptp = [None]*n_units
    spike_times_list = [None]*n_units
    shifts_list = [None]*n_units
    scales_list = [None]*n_units
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
        #min_max_vals_spikes = torch.zeros((len(spike_times), 2, self.N_CHAN, 5)).float().cuda()
        min_max_vals_spikes = torch.cuda.FloatTensor(len(spike_times_batch), 2, n_channels).fill_(0)
        for j in range(len(counter)-1):
            ii_start = counter[j]
            ii_end = counter[j+1]

            min_max_loc_spikes = (min_max_loc[neuron_ids_batch[ii_start:ii_end]] + 
                                  spike_times_batch[ii_start:ii_end, None, None] - n_times//2)
            min_max_vals_spikes[ii_start:ii_end] = torch.gather(
                residual, 0, min_max_loc_spikes.reshape(-1, n_channels)).reshape(
                -1, 2, n_channels)

            shifted_templates = residual_comp.get_shifted_templates(
                    neuron_ids_batch[ii_start:ii_end], 
                    shifts_batch[ii_start:ii_end], 
                    scales_batch[ii_start:ii_end])
            min_max_vals_spikes[ii_start:ii_end] += torch.gather(
                shifted_templates, 1, min_max_loc[neuron_ids_batch[ii_start:ii_end]])
        ptp_batch = min_max_vals_spikes[:,1] - min_max_vals_spikes[:,0]

        del min_max_loc_spikes
        del min_max_vals_spikes
        del shifted_templates
        del residual
        torch.cuda.empty_cache()

        for k in range(n_units):
            idx_ = neuron_ids_batch == k
            if cleaned_ptp[k] is not None:
                cleaned_ptp[k] = np.concatenate(
                    (cleaned_ptp[k], ptp_batch[idx_].cpu().numpy()[:, vis_chans[k]]),
                    axis = 0)
                spike_times_list[k] = np.hstack((spike_times_list[k],
                                                 spike_times_batch[idx_].cpu().numpy() + batch_offset))
                shifts_list[k] = np.hstack((shifts_list[k],
                                            shifts_batch[idx_].cpu().numpy()))
                scales_list[k] = np.hstack((scales_list[k],
                                            scales_batch[idx_].cpu().numpy()))
            else:
                cleaned_ptp[k] = ptp_batch[idx_].cpu().numpy()[:, vis_chans[k]]
                spike_times_list[k] = spike_times_batch[idx_].cpu().numpy() + batch_offset
                shifts_list[k] = shifts_batch[idx_].cpu().numpy()
                scales_list[k] = scales_batch[idx_].cpu().numpy()

        del ptp_batch
        del spike_times_batch
        del shifts_batch
        del scales_batch
        torch.cuda.empty_cache()

    del templates
    del spike_train
    del shifts
    del scales
    del min_max_loc
    del residual_comp
    torch.cuda.empty_cache()

    return cleaned_ptp, spike_times_list, shifts_list, scales_list, vis_chans


def pca_gmm_and_dip(data, min_data):
    
    if data.shape[0] < min_data:
        return np.zeros(len(data), 'int32'), 1

    pca = PCA(n_components=1)
    score = pca.fit_transform(data)

    label = np.zeros(len(score), 'int32')

    p_val = dp(score[:, 0])[1]
    if p_val < 0.05:
        g = GaussianMixture(n_components=2)
        label_ = g.fit_predict(score)
        idx_0 = label_==0
        if np.sum(idx_0) == 0 or np.sum(~idx_0) == 0:
            return label, 1
        else:
            return label_, p_val

        label_0 = pca_gmm_and_dip(data[idx_0], min_data)[0]
        label_1 = pca_gmm_and_dip(data[~idx_0], min_data)[0]
        label_1 += np.max(label_0) + 1
        label[idx_0] = label_0
        label[~idx_0] = label_1

    return label, p_val


def run_split(cleaned_ptp, spike_times_list,
              shifts_list, scales_list, vis_chans,
              templates, spike_train, reader_raw, CONFIG,
              update_original_templates=False,
              min_ptp_split=5, min_fr_accept=2, min_ptp_accept=100):
    
    n_units, n_times, n_channels = templates.shape
    
    ptp_max = templates.ptp(1).max(1)

    min_fr = CONFIG.cluster.min_fr
    len_rec = spike_train[:,0].ptp()/CONFIG.recordings.sampling_rate
    min_data = int(min_fr*len_rec)
    
    new_temps = np.zeros((0, n_times, n_channels), 'float32')
    new_spts = []
    new_shifts = []
    new_scales = []
    for k in tqdm(range(len(cleaned_ptp))):

        if ptp_max[k] > min_ptp_split:

            spt_ = spike_times_list[k]
            shifts_ = shifts_list[k]
            scales_ = scales_list[k]

            idx_sort = np.argsort(spt_)
            cleaned_ptp_unit = cleaned_ptp[k][idx_sort]
            spt_ = spt_[idx_sort]
            shifts_ = shifts_[idx_sort]
            scales_ = scales_[idx_sort]

            isi_ = np.diff(np.hstack((-1000, spt_)))
            cleaned_ptp_unit = cleaned_ptp_unit[isi_ > 300]
            spt_ = spt_[isi_ > 300]
            shifts_ = shifts_[isi_ > 300]
            scales_ = scales_[isi_ > 300]
            #std_ = np.std(cleaned_ptp_unit, 0)

            #if std_.max() > 1.2:
            label, pval = pca_gmm_and_dip(cleaned_ptp_unit, min_data)
            if pval < 0.05:
                unique_label, n_counts = np.unique(label, return_counts=True)
                unique_label = unique_label[n_counts > min_data]
                n_counts = n_counts[n_counts > min_data]

                if len(unique_label) <= 1:
                    continue

                # estimate templates from raw
                new_temps_k = np.zeros((len(unique_label), n_times, n_channels))
                for ii in range(len(unique_label)):
                    spt_temp = spt_[label == unique_label[ii]]
                    new_temps_k[ii] = reader_raw.read_waveforms(spt_temp)[0].mean(0)

                ptp_new = new_temps_k.ptp(1).max(1)
                idx_keep = np.logical_or(ptp_new > min_ptp_accept,
                                         n_counts > min_fr_accept*len_rec)
                unique_label = unique_label[idx_keep]

                if len(unique_label) <= 1:
                    continue

                # find out what matches the orignal unit
                # get ptp of each split unit
                split_units_ptp = np.zeros((len(unique_label), cleaned_ptp_unit.shape[1]))
                for ii in range(len(unique_label)):
                    split_units_ptp[ii] = cleaned_ptp_unit[label == unique_label[ii]].mean(0)
                # compare to the original unit's ptp
                original_unit_ptp = templates[k].ptp(0)[vis_chans[k]]
                split_unit_closest = unique_label[np.sum(np.square(
                    split_units_ptp - original_unit_ptp), 1).argmin()]

                # add spike times
                for ii in unique_label[unique_label != split_unit_closest]:
                    new_spts.append(spt_[label == ii])
                    new_shifts.append(shifts_[label == ii])
                    new_scales.append(scales_[label == ii])

                # save all units other than the closest on as new templates
                new_temps = np.concatenate(
                    (new_temps, new_temps_k[unique_label != split_unit_closest]), axis=0)

                # if the split unit closest to the orignal has enough spikes,
                # the new template replaces the original one
                #if np.sum(label == split_unit_closest) > 300:
                if update_original_templates:
                    templates[k] = new_temps_k[unique_label == split_unit_closest][0]

                spike_times_list[k] = spt_[label == split_unit_closest]
                shifts_list[k] = shifts_[label == split_unit_closest]
                scales_list[k] = scales_[label == split_unit_closest]


    # updated templates
    templates_updated = np.concatenate((templates, new_temps), 0).astype('float32')

    # updated spike train of original units
    spike_train_updated = np.zeros((0, 2), 'int32')
    shifts_updated = np.zeros(0, 'float32')
    scales_updated = np.zeros(0, 'float32')
    for k in range(n_units):
        spike_train_temp = np.vstack(
            (spike_times_list[k], k*np.ones(len(spike_times_list[k]), 'int32'))).T
        spike_train_updated = np.concatenate(
            (spike_train_updated, spike_train_temp), axis=0)
        shifts_updated = np.hstack((shifts_updated, shifts_list[k]))
        scales_updated = np.hstack((scales_updated, scales_list[k]))

    # add new spike train
    spike_train_new = np.zeros((0, 2), 'int32')
    shifts_new = np.zeros(0, 'float32')
    scales_new = np.zeros(0, 'float32')
    for k in range(new_temps.shape[0]):
        spike_train_temp = np.vstack((new_spts[k], k*np.ones(len(new_spts[k]), 'int32'))).T
        spike_train_new = np.concatenate((spike_train_new, spike_train_temp), axis=0)
        shifts_new = np.hstack((shifts_new, new_shifts[k]))
        scales_new = np.hstack((scales_new, new_scales[k]))
    spike_train_new[:, 1] += n_units
    spike_train_updated = np.concatenate((spike_train_updated, spike_train_new), axis=0)
    shifts_updated = np.concatenate((shifts_updated, shifts_new), axis=0)
    scales_updated = np.concatenate((scales_updated, scales_new), axis=0)
        
    return templates_updated, spike_train_updated, shifts_updated, scales_updated
