import os
import numpy as np

def gather_result(fname_save, batch_files_dir, dedup_dir):
    
    
    n_batches = len(os.listdir(batch_files_dir))
    spike_index_postkill = []
    for batch_id in range(n_batches):

        # detection index 
        fname_index = os.path.join(
            batch_files_dir,
            "detect_"+str(batch_id).zfill(5)+'.npz')
        detect_data =  np.load(fname_index)
        spike_index = detect_data['spike_index']
        minibatch_loc = detect_data['minibatch_loc']

        # dedup index
        fname_dedup = os.path.join(
            dedup_dir,
            "dedup_"+str(batch_id).zfill(5)+'.npy')
        dedup_idx = np.load(fname_dedup)

        for ctr in range(len(spike_index)):
            spike_index_temp = spike_index[ctr][dedup_idx[ctr]]
            t_start, t_end = minibatch_loc[ctr]

            idx_keep = np.logical_and(
                spike_index_temp[:, 0] >= t_start,
                spike_index_temp[:, 0] < t_end)
            spike_index_temp = spike_index_temp[idx_keep]

            spike_index_postkill.append(spike_index_temp)

    np.save(fname_save, np.vstack(spike_index_postkill))
        
    