# Hard coded function to test data1_set1 yass results against
# kilosort results and gold standard

import numpy as np

def compare_templates(templates):
    
    """ Compare templates 

    """
    
    # get kilosort spike_trains  
    ks_spike_train = np.load('/media/cat/1TB/liam/49channels/data1_allset/kilosort/spike_times.npy')
    
    # make ks templates
    
    
    # get gold standard spike trains and templates
    gold_spike_train = np.load('/media/cat/1TB/liam/49channels/ej_49_gold_standard/groundtruth_ej49_data1_allset.npy')
    
    
    # make gold standard templates
    
    
    

    # get size
    n_channels, temporal_size, n_templates = templates.shape

    # get energy
    energy = np.ptp(templates, axis=1)
    mainc = np.argmax(energy, axis=0)



def compare_spiketrains(spikes):
    
    
    pass
