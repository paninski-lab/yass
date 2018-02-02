import numpy as np

from .util import upsample_templates, make_spt_list, get_longer_spt_list
from .match import make_tf_tensors, template_match

def greedy_deconvolve(recording, templates, spike_index, 
                      n_explore, n_rf, upsample_factor, 
                      threshold_a, threshold_dd):
    
    # writable recording
    rec = np.copy(recording)
    
    # get useful parameters
    T = rec.shape[0]
    n_channels, n_timebins, n_templates = templates.shape
    R = int((n_timebins - 1)/2)
    #n_explore = 1
    #n_rf = int(1.5*cfg.recordings.sampling_rate/1000)
    #upsample_factor = 5

    # determine principal channels for each template
    # and order templates by it energy
    template_max_energy = np.max(np.abs(templates),1)
    principal_channels = np.argmax(template_max_energy,0)
    templates_order = np.argsort(np.max(
        template_max_energy,0))[::-1]

    
    # make tensorflow tensors in advance so that we don't
    # have to create multiple times in a loop
    (rec_local_tf, template_local_tf,
        spt_tf, result) = make_tf_tensors(T, n_timebins, 
                                       upsample_factor, 
                                       R, threshold_a, 
                                       threshold_dd)
    
    
    # change the format of spike index for easier access
    spt_list = make_spt_list(spike_index, n_channels)
    
    
    # do template matching in a greedy way from biggest template
    # to the smallest
    spike_train = np.zeros((0,2), 'int32')
    for j in range(n_templates):
        
        # cluster and its main channel
        k = templates_order[j]
        mainc = principal_channels[k]
        
        # channels with big enough energy relative to energy 
        # in the principal channel
        channels_big = np.where(
            template_max_energy[:, k] > \
            template_max_energy[mainc, k]*0.7)[0]
        
        # upsample and localize template
        upsampled_template = upsample_templates(templates[:,:,k], 
                                                upsample_factor)
        upsampled_template_local = upsampled_template[:,channels_big,:]

        # localize recording
        rec_local = rec[:,channels_big]

        # localize spike times
        spt_interest = np.zeros(0, 'int32')
        for c in range(channels_big.shape[0]):    
            spt_interest = np.concatenate((
                spt_interest, spt_list[channels_big[c]]))
        # for each spike time t, add t-n_explore : t+n_explore to
        # spike times of interest too
        spt_interest = get_longer_spt_list(spt_interest, n_explore)

        
        # run template match
        spt_good, ahat_good, max_idx_good = template_match(
            rec_local, spt_interest, upsampled_template_local, 
            n_rf, rec_local_tf, template_local_tf, spt_tf, result)


        # subtract off deconvolved spikes from the recording
        for j in range(spt_good.shape[0]):
            rec[spt_good[j]-R:spt_good[j]+R+1
               ] -= ahat_good[j]*upsampled_template[max_idx_good[j]].T

        # collect detected spike times
        spike_train = np.vstack((spike_train, np.vstack((
                spt_good, np.ones(spt_good.shape[0], 'int32')*k)).T))

        
    return spike_train