import os
import logging
import numpy as np
import parmap

class RESIDUAL(object):
    
    def __init__(self, 
                 fname_up,
                 reader,
                 fname_out,
                 dtype_out):
        
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        
        up_data = np.load(fname_up)
        self.templates = up_data['templates_up']
        self.spike_train = up_data['spike_train_up']

        self.n_time, self.n_chan, self.n_unit = self.templates.shape

        self.reader = reader
        self.reader.buffer = self.n_time
        self.fname_out = fname_out
        self.dtype_out = dtype_out

        self.partition_spike_time()
    
    def partition_spike_time(self):

        spike_times = []
        for ctr in range(len(self.reader.idx_list)):
            start, end = self.reader.idx_list[ctr]
            start -= self.reader.buffer
            
            # assign spikes in each chunk; offset 
            idx_in_chunk = np.where(
                np.logical_and(self.spike_train[:,0]>=start,
                               self.spike_train[:,0]<end))[0]
            spikes_in_chunk = self.spike_train[idx_in_chunk]

            # reset spike times to zero for each chunk but add in bufer_size
            #  that will be read in with data 
            spikes_in_chunk[:,0] -=  start
            spike_times.append(spikes_in_chunk)
        
        self.spike_train = spike_times

    def compute_residual(self, save_dir,
                         multi_processing=False, n_processors=1):
        '''
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        batch_ids = []
        fnames_seg = []
        for batch_id in range(self.reader.n_batches):
            batch_ids.append(batch_id)
            fnames_seg.append(
                os.path.join(save_dir,
                             'residual_seg{}.npy'.format(batch_id)))

        if multi_processing:
            parmap.starmap(self.subtract_parallel, 
                         list(zip(batch_ids, fnames_seg)),
                         processes=n_processors,
                         pm_pbar=True)
        else:
            for ctr in range(len(batch_ids)):
                self.subtract_parallel(
                    batch_ids[ctr], fname_seg[ctr])

        self.fnames_seg = fnames_seg

    def subtract_parallel(self, batch_id, fname_out):
        '''
        '''
        if os.path.exists(fname_out):
            return

        # note only derasterize up to last bit, don't remove spikes from 
        # buffer_size end because those will be looked at by next chunk
        # load indexes and then index into original data
        data = self.reader.read_data_batch(batch_id, add_buffer=True)

        # loop over units and subtract energy
        local_spike_train = self.spike_train[batch_id]
        time_idx = np.arange(0, self.n_time)
        for j in range(local_spike_train.shape[0]):
            tt, ii = local_spike_train[j]

            data[time_idx + tt, :] -= self.templates[:, :, ii]

        # remove buffer
        data = data[self.reader.buffer:-self.reader.buffer]

        # save
        np.save(fname_out, data)


    def save_residual(self):
    
        f = open(self.fname_out,'wb')
        for fname in self.fnames_seg:

            res = np.load(fname).astype(self.dtype_out)
            f.write(res)
        f.close()
        
        # delete residual chunks after successful merging/save
        for fname in self.fnames_seg:
            os.remove(fname)
