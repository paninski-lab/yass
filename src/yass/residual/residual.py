import os
import logging
import numpy as np
import parmap

class RESIDUAL(object):
    
    def __init__(self, 
                 fname_templates,
                 fname_spike_train,
                 reader,
                 fname_out,
                 dtype_out):
        
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        #self.logger = logging.getLogger(__name__)

        # keep templates and spike train filname
        # will be loaded during each prallel process
        self.fname_templates = fname_templates
        self.fname_spike_train = fname_spike_train

        self.reader = reader

        # save output name and dtype
        self.fname_out = fname_out
        self.dtype_out = dtype_out

    def compute_residual(self, save_dir,
                         multi_processing=False,
                         n_processors=1):
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

        #self.logger.info("computing residuals")
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
        '''
        '''
                
        templates = None
        spike_train = None

        for batch_id, fname_out in zip(batch_ids, fnames_out):
            if os.path.exists(fname_out):
                continue
            
            # load upsampled templates only once per core:
            if templates is None or spike_train is None:
                #self.logger.info("loading upsampled templates")
                templates = np.load(self.fname_templates)
                spike_train = np.load(self.fname_spike_train)

                # do not read spike train again here
                #self.spike_train = up_data['spike_train_up']
                n_time = templates.shape[1]
                time_idx = np.arange(0, n_time)
                self.reader.buffer = n_time
                
                # shift spike time so that it is aligned at
                # time 0
                spike_train[:, 0] -= n_time//2

            # get relevantspike times
            start, end = self.reader.idx_list[batch_id]
            start -= self.reader.buffer 
            idx_in_chunk = np.where(
                np.logical_and(spike_train[:,0]>=start,
                               spike_train[:,0]<end))[0]
            spikes_in_chunk = spike_train[idx_in_chunk]
            # offset
            spikes_in_chunk[:,0] -=  start

            #print ("SPIKE train: ", self.spike_train.shape)
            #print ("templates: ", self.templates.shape)
            # note only derasterize up to last bit, don't remove spikes from 
            # buffer_size end because those will be looked at by next chunk
            # load indexes and then index into original data
            #print ("batch_id: ", batch_id)
            data = self.reader.read_data_batch(batch_id, add_buffer=True)
            #print ("data: ", data.shape)

            #print ("local spike train: ", local_spike_train.shape)
            #print ("self.n_time: ", self.n_time)
            #print ("local spiketrain: ", local_spike_train)
            for j in range(spikes_in_chunk.shape[0]):
                tt, ii = spikes_in_chunk[j]
                data[time_idx + tt, :] -= templates[ii]

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
