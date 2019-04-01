import os
import numpy as np

def gather_result(batch_files_dir, dedup_dir, fname_save):
    
    
    n_batches = len(os.listdir(batch_files_dir))
    spike_index_postkill = []
    for batch_id in range(n_batches):

        # detection index 
        fname_index = os.path.join(
            batch_files_dir,
            "detect_"+str(batch_id).zfill(5)+'.npz')
        spike_index = np.load(fname_index)['spike_index']

        # dedup index
        fname_dedup = os.path.join(
            dedup_dir,
            "dedup_"+str(batch_id).zfill(5)+'.npy')
        dedup_idx = np.load(fname_dedup)
        
        spike_index_postkill.append(spike_index[dedup_idx])

    np.save(fname_save, np.vstack(spike_index_postkill))
        
    