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
        
        # load necessary data
        up_data = np.load(fname_up)
        self.n_unit, self.n_time, self.n_chan = up_data['templates_up'].shape
        self.spike_train = up_data['spike_train_up']

        # shift spike times for easy indexing
        self.spike_train[:,0] -= self.n_time//2

        self.fname_up = fname_up
        self.reader = reader
        self.reader.buffer = self.n_time
        self.fname_out = fname_out
        self.dtype_out = dtype_out

    def partition_spike_time(self, fname_partitioned):

        if not os.path.exists(fname_partitioned):
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

            # save it
            np.save(fname_partitioned, spike_times)
        else:
            spike_times = np.load(fname_partitioned)

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
            batches_in = np.array_split(batch_ids, n_processors)
            fnames_in = np.array_split(fnames_seg, n_processors)
            parmap.starmap(self.subtract_parallel, 
                         list(zip(batches_in, fnames_in)),
                         processes=n_processors,
                         pm_pbar=True)

        else:
            for ctr in range(len(batch_ids)):
                self.subtract_parallel(
                    [batch_ids[ctr]], [fnames_seg[ctr]])

        self.fnames_seg = fnames_seg


    def subtract_parallel(self, batch_ids, fnames_out):
    #def subtract_parallel(self, data_in):
        '''
        '''
                
        self.templates = None
        
        for batch_id, fname_out in zip(batch_ids, fnames_out):
            if os.path.exists(fname_out):
                continue
            
            # load pairwise conv filter only once per core:
            if self.templates is None:
                #self.logger.info("loading upsampled templates")

                up_data = np.load(self.fname_up)
                self.templates = up_data['templates_up']
                # do not read spike train again here
                #self.spike_train = up_data['spike_train_up']
            
            #print ("SPIKE train: ", self.spike_train.shape)
            #print ("templates: ", self.templates.shape)
            # note only derasterize up to last bit, don't remove spikes from 
            # buffer_size end because those will be looked at by next chunk
            # load indexes and then index into original data
            #print ("batch_id: ", batch_id)
            data = self.reader.read_data_batch(batch_id, add_buffer=True)
            #print ("data: ", data.shape)

            # loop over units and subtract energy
            local_spike_train = self.spike_train[batch_id]
            #print ("local spike train: ", local_spike_train.shape)
            time_idx = np.arange(0, self.n_time)
            #print ("self.n_time: ", self.n_time)
            #print ("local spiketrain: ", local_spike_train)
            for j in range(local_spike_train.shape[0]):
                tt, ii = local_spike_train[j]

                data[time_idx + tt, :] -= self.templates[ii]

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
