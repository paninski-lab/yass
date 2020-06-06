import numpy as np
import torch
import os
from tqdm import tqdm

# from yass import read_config, set_config
# output_dir='tmp/'
# set_config(config, output_dir)

# CONFIG = read_config()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)

class GETPTP(object):
    def __init__(self, fname_spike_index, reader, CONFIG, denoiser=None):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
        
        self.spike_index = np.load(fname_spike_index)
        self.spike_index = torch.from_numpy(self.spike_index).long().cuda()
        self.reader = reader
        self.denoiser = denoiser
        
        if denoiser is not None:
            self.n_times = denoiser.out.weight.shape[0]
        else:
            self.n_times = reader.spike_size
        
    def compute_ptps(self):

        t_range = torch.arange(-(self.n_times//2), self.n_times//2+1).cuda()
        
        ptps_raw = torch.zeros(self.spike_index.shape[0]).float().cuda()
        if self.denoiser is not None:
            ptps_denoised = torch.zeros(self.spike_index.shape[0]).float().cuda()
        else:
            ptps_denoised = None
        
        # batch offsets
        offsets = torch.from_numpy(self.reader.idx_list[:, 0] - self.reader.buffer).cuda().long()

        with tqdm(total=self.reader.n_batches) as pbar:

            for batch_id in range(self.reader.n_batches):

                # load residual data
                dat = self.reader.read_data_batch(batch_id, add_buffer=True)
                dat = torch.from_numpy(dat).cuda()

                # relevant idx
                idx_in = torch.nonzero(
                    (self.spike_index[:, 0] > self.reader.idx_list[batch_id][0]) & 
                    (self.spike_index[:, 0] < self.reader.idx_list[batch_id][1]))[:,0]

                spike_index_batch = self.spike_index[idx_in]
                spike_index_batch[:, 0] -= offsets[batch_id]

                # skip if no spikes
                if len(spike_index_batch) == 0:
                    continue

                # get residual snippets
                t_index = spike_index_batch[:, 0][:, None] + t_range
                c_index = spike_index_batch[:,1].long()

                dat = torch.cat((dat, torch.zeros((dat.shape[0], 1)).cuda()), 1)
                wfs = dat[t_index, c_index[:,None]]
                ptps_raw[idx_in] = (torch.max(wfs, 1)[0] - torch.min(wfs, 1)[0])

                if self.denoiser is not None:
                    n_sample_run = 1000

                    idx_list = np.hstack((
                        np.arange(0, wfs.shape[0], n_sample_run), wfs.shape[0]))
                    denoised_wfs = torch.zeros_like(wfs).cuda()
                    #print ("denoised_wfs; ", denoised_wfs.shape)
                    #print ("wfs; ", wfs.shape)
                    for j in range(len(idx_list)-1):
                        #print ("idx_list[j], j+1: ", idx_list[j], idx_list[j+1])
                        denoised_wfs[idx_list[j]:idx_list[j+1]] = self.denoiser(
                            wfs[idx_list[j]:idx_list[j+1]])[0].data
                    ptps_denoised[idx_in] = (torch.max(denoised_wfs, 1)[0] - torch.min(denoised_wfs, 1)[0])

                pbar.update()

        ptps_raw_cpu = ptps_raw.cpu().numpy()

        del dat, idx_in, spike_index_batch, t_index, c_index, wfs, ptps_raw

        if self.denoiser is not None:
            ptps_denoised_cpu = ptps_denoised.cpu().numpy()
            del denoised_wfs, ptps_denoised
        else:
            ptps_denoised_cpu = np.copy(ptps_raw_cpu)

        torch.cuda.empty_cache()

        return ptps_raw_cpu, ptps_denoised_cpu
    
    
    def compute_wfs(self, idx):

        t_range = torch.arange(-(self.n_times_nn//2), self.n_times_nn//2+1).cuda()
        
        wfs_raw = torch.zeros(len(idx), self.n_times_nn).float().cuda()
        wfs_denoised = torch.zeros(len(idx), self.n_times_nn).float().cuda()
        
        spike_index_in = self.spike_index[idx]
        with tqdm(total=self.reader.n_batches) as pbar:

            for batch_id in range(self.reader.n_batches):

                # load residual data
                dat = self.reader.read_data_batch(batch_id, add_buffer=True)
                dat = torch.from_numpy(dat).cuda()

                # relevant idx
                idx_in = torch.nonzero(
                    (spike_index_in[:, 0] > self.reader.idx_list[batch_id][0]) & 
                    (spike_index_in[:, 0] < self.reader.idx_list[batch_id][1]))[:,0]

                spike_index_batch = spike_index_in[idx_in] 
                spike_index_batch[:, 0] -= (self.reader.idx_list[batch_id][0] - 
                                            self.reader.buffer)

                # get residual snippets
                t_index = spike_index_batch[:, 0][:, None] + t_range
                c_index = spike_index_batch[:,1].long()

                dat = torch.cat((dat, torch.zeros((dat.shape[0], 1)).cuda()), 1)
                wfs = dat[t_index, c_index[:,None]]

                n_sample_run = 1000

                idx_list = np.hstack((
                    np.arange(0, wfs.shape[0], n_sample_run), wfs.shape[0]))
                denoised_wfs = torch.zeros_like(wfs).cuda()
                for j in range(len(idx_list)-1):
                    denoised_wfs[idx_list[j]:idx_list[j+1]] = self.denoiser(
                        wfs[idx_list[j]:idx_list[j+1]])[0]

                wfs_raw[idx_in] = wfs.data
                wfs_denoised[idx_in] = denoised_wfs.data
                
                pbar.update()

        return wfs_raw.cpu().numpy(), wfs_denoised.cpu().numpy()

    
class GETCLEANPTP(object):
    def __init__(self, fname_spike_index, fname_labels,
                 fname_templates, fname_shifts, fname_scales,
                 reader_residual, denoiser=None):

        self.spike_index = np.load(fname_spike_index)
        self.spike_index = torch.from_numpy(self.spike_index).long().cuda()
        
        self.labels = np.load(fname_labels)
        self.labels = torch.from_numpy(self.labels).long().cuda()
        
        templates = np.load(fname_templates)
        mcs = templates.ptp(1).argmax(1)
        n_units, n_times, n_channels = templates.shape
        self.templates = np.zeros((n_units, n_times))
        for k in range(n_units):
            self.templates[k] = templates[k, :, mcs[k]]

        self.shifts = np.load(fname_shifts)
        self.shifts = torch.from_numpy(self.shifts).float().cuda()

        self.scales = np.load(fname_scales)
        self.scales = torch.from_numpy(self.scales).float().cuda()

        self.reader_residual = reader_residual
        self.denoiser = denoiser

        if self.denoiser is not None:
            self.n_times = denoiser.out.weight.shape[0]
            self.crop_templates()
        else:
            self.n_times = n_times

        self.templates = torch.from_numpy(self.templates).float().cuda()

    def crop_templates(self):
        
        n_times_templates = self.templates.shape[1]
        if n_times_templates > self.n_times:
            n_diff = (n_times_templates - self.n_times)//2
            self.templates = self.templates[:, n_diff:-n_diff]
            
        elif n_times_templates < self.n_times:
            n_diff = (self.n_times - n_times_templates)//2
            buffer = np.zeros((self.templates.shape[0], n_diff), 'float32')
            self.templates = np.concatenate((buffer, self.templates, buffer), axis=1)

    def compute_ptps(self):

        t_range = torch.arange(-(self.n_times//2), self.n_times//2+1).cuda()
        
        ptps_raw = torch.zeros(self.spike_index.shape[0]).float().cuda()
        if self.denoiser is not None:
            ptps_denoised = torch.zeros(self.spike_index.shape[0]).float().cuda()
        else:
            ptps_denoised = None

        # batch offsets
        offsets = torch.from_numpy(self.reader_residual.idx_list[:, 0]
                                   - self.reader_residual.buffer).cuda().long()

        with tqdm(total=self.reader_residual.n_batches) as pbar:

            for batch_id in range(self.reader_residual.n_batches):

                # load residual data
                dat = self.reader_residual.read_data_batch(batch_id, add_buffer=True)
                dat = torch.from_numpy(dat).cuda()

                # relevant idx
                idx_in = torch.nonzero(
                    (self.spike_index[:, 0] > self.reader_residual.idx_list[batch_id][0]) & 
                    (self.spike_index[:, 0] < self.reader_residual.idx_list[batch_id][1]))[:,0]
                
                if len(idx_in) == 0:
                    continue

                spike_index_batch = self.spike_index[idx_in] 
                spike_index_batch[:, 0] -= offsets[batch_id]

                # get residual snippets
                t_index = spike_index_batch[:, 0][:, None] + t_range
                c_index = spike_index_batch[:,1].long()

                dat = torch.cat((dat, torch.zeros((dat.shape[0], 1)).cuda()), 1)
                residuals = dat[t_index, c_index[:,None]]
                
                # TODO: align residuals
                #shifts_batch = self.shifts[idx_in]
                #residuals = shift_chans(residuals, -shifts_batch)

                # make clean wfs
                wfs = residuals + self.scales[idx_in][:, None]*self.templates[self.labels[idx_in]]

                ptps_raw[idx_in] = (torch.max(wfs, 1)[0] - torch.min(wfs, 1)[0])

                if self.denoiser is not None:
                    n_sample_run = 1000

                    idx_list = np.hstack((
                        np.arange(0, wfs.shape[0], n_sample_run), wfs.shape[0]))
                    denoised_wfs = torch.zeros_like(wfs).cuda()
                    for j in range(len(idx_list)-1):
                        denoised_wfs[idx_list[j]:idx_list[j+1]] = self.denoiser(
                            wfs[idx_list[j]:idx_list[j+1]])[0].data
                    ptps_denoised[idx_in] = (torch.max(denoised_wfs, 1)[0] - torch.min(denoised_wfs, 1)[0])

                pbar.update()

        ptps_raw_cpu = ptps_raw.cpu().numpy()

        del dat, idx_in, spike_index_batch, t_index, c_index, residuals, wfs, ptps_raw

        torch.cuda.empty_cache()

        if self.denoiser is not None:
            ptps_denoised_cpu = ptps_denoised.cpu().numpy()
            del denoised_wfs, ptps_denoised
        else:
            ptps_denoised_cpu = np.copy(ptps_raw_cpu)

        return ptps_raw_cpu, ptps_denoised_cpu
