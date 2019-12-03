import logging
import os
import numpy as np

def gather_result(fname_save, batch_files_dir):

    logger = logging.getLogger(__name__)
    logger.info('gather detected spikes')

    fnames = os.listdir(batch_files_dir)
    spike_index = []
    spike_index_prekill = []
    n_spikes_prekill = 0
    n_spikes_postkill = 0
    for batch_id in range(len(fnames)):

        # detection index
        fname = os.path.join(batch_files_dir, fnames[batch_id])
        detect_data =  np.load(fname, allow_pickle=True)
        spike_index_prekill_list = detect_data['spike_index']
        spike_index_list = detect_data['spike_index_dedup']
        minibatch_loc = detect_data['minibatch_loc']

        # kill edge spikes and gather results
        for ctr in range(len(spike_index_list)):

            t_start, t_end = minibatch_loc[ctr]
            spike_index_temp = spike_index_list[ctr]

            idx_keep = np.where(np.logical_and(
                spike_index_temp[:, 0] >= t_start,
                spike_index_temp[:, 0] < t_end))[0]
            spike_index.append(spike_index_temp[idx_keep])

            spike_index_prekill.append(spike_index_prekill_list[ctr])

            n_spikes_prekill += spike_index_prekill_list[ctr].shape[0]
            n_spikes_postkill += len(idx_keep)

    logger.info('Total {} spikes detected'.format(
        n_spikes_prekill))
    logger.info('Total {} spikes survived after deduplication'.format(
        n_spikes_postkill))

    spike_index = np.vstack(spike_index)
    spike_index_prekill = np.vstack(spike_index_prekill)
    
    idx_sort = np.argsort(spike_index[:,0])
    spike_index = spike_index[idx_sort]

    idx_sort = np.argsort(spike_index_prekill[:,0])
    spike_index_prekill = spike_index_prekill[idx_sort]

    np.save(fname_save, spike_index)
    np.save(fname_save[:fname_save.rfind('.')]+'_prekill.npy',
            spike_index_prekill)


def gather_result_orig(fname_save, batch_files_dir, dedup_dir, output_directory):
    
    logger = logging.getLogger(__name__)
    
    n_batches = len(os.listdir(batch_files_dir))
    spike_index_postkill = []
    spike_index_prekill = []
    n_spikes_detected = 0
    for batch_id in range(n_batches):

        # detection index 
        fname_index = os.path.join(
            batch_files_dir,
            "detect_"+str(batch_id).zfill(5)+'.npz')
        detect_data =  np.load(fname_index,allow_pickle=True)
        spike_index = detect_data['spike_index']
        minibatch_loc = detect_data['minibatch_loc']

        # dedup index
        fname_dedup = os.path.join(
            dedup_dir,
            "dedup_"+str(batch_id).zfill(5)+'.npy')
        dedup_idx = np.load(fname_dedup,allow_pickle=True)

        for ctr in range(len(spike_index)):
            t_start, t_end = minibatch_loc[ctr]
            
            spike_index_temp = spike_index[ctr]

            idx_keep = np.where(np.logical_and(
                spike_index_temp[:, 0] >= t_start,
                spike_index_temp[:, 0] < t_end))[0]

            spike_index_prekill.append(spike_index_temp[idx_keep])
            spike_index_postkill.append(spike_index_temp[
                np.intersect1d(dedup_idx[ctr], idx_keep)])

    spike_index_postkill = np.vstack(spike_index_postkill)
    spike_index_prekill = np.vstack(spike_index_prekill)
    
    logger.info('{} spikes detected'.format(len(spike_index_prekill)))
    logger.info('{} spikes after deduplication'.format(len(spike_index_postkill)))

    np.save(fname_save, spike_index_postkill)
    np.save(os.path.join(output_directory, 'spike_index_prekill.npy'), spike_index_prekill)
        
    
