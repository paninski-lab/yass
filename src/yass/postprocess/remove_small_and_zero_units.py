import numpy as np
import os
import parmap
import scipy


def remove_small_and_zero_units(fname_templates, fname_spike_train, CONFIG):
    
    #
    try:
        threshold = CONFIG.clean_up.min_ptp
    except:
        threshold = 3
    
    # 
    #templates = np.load(os.path.join(fname_templates,"templates_init.npy"))
    templates = np.load(fname_templates)
    
    #
    ptps = templates.ptp(1).max(1)

    # 
    idx_small_units = np.where(ptps<threshold)[0]
    print ("  ... deleting units and units with ptp < : ", 
            threshold, " total: ", idx_small_units.shape[0])
    
    # delete small units
    templates_clean = np.delete(templates, idx_small_units, axis=0)
    
    #
    spike_train = np.load(fname_spike_train)
    
    # delete small neurons and neurons without spikes:
    spike_train_clean = np.zeros((0,2), 'int32')
    ctr=0
    for k in range(templates.shape[0]):
        idx = np.where(spike_train[:,1]==k)[0]

        if idx.shape[0]>0 and k not in idx_small_units:
            times = spike_train[idx,0]
            ids = times*0+ctr
            temp_train = np.vstack((times, ids)).T
            
            spike_train_clean = np.vstack((spike_train_clean,
                                           temp_train))
            ctr+=1
            
    # reorder by time
    idx = np.argsort(spike_train_clean[:,0])
    spike_train_clean = spike_train_clean[idx]
    
    # save
    np.save(fname_templates, templates_clean)
    np.save(fname_spike_train, spike_train_clean)
    
