import os
import logging
import numpy as np
import parmap
import scipy

import datetime as dt
from tqdm import tqdm

from yass import read_config

from yass.visual.util import binary_reader_waveforms

#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(CONFIG):
            
    """Generate phy2 visualization files
    """

    logger = logging.getLogger(__name__)

    # set root directory for output
    root_dir = CONFIG.data.root_folder
 
    # output folder
    output_directory = os.path.join(root_dir, 'phy')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # cluster id for each spike; [n_spikes]
    spike_train = np.load(root_dir + '/tmp/spike_train.npy')
    spike_clusters = spike_train[:,1]
    np.save(root_dir+'/phy/spike_clusters.npy', spike_clusters)

    # spike times for each spike: [n_spikes]
    spike_times = spike_train[:,0]
    np.save(root_dir+'/phy/spike_times.npy', spike_times)

    # save templates; not sure why this is required?!
    np.save(root_dir+'/phy/spike_templates.npy', spike_clusters)

    # save geometry
    chan_pos = np.loadtxt(root_dir+"geom.txt")
    np.save(root_dir+'/phy/channel_positions.npy', chan_pos)

    # sequential channel order
    channel_map = np.arange(chan_pos.shape[0])
    np.save(root_dir + '/phy/channel_map.npy', channel_map)

    # unit templates [n_units, times, n_chans]
    temps = np.load(root_dir + '/tmp/templates.npy').transpose(2,0,1)
    np.save(root_dir + "/phy/templates.npy",temps)

    # pick largest SU channels for each unit; [n_templates x n_channels_loc]; 
    # gives # of channels of the corresponding columns in pc_features, for each spike.
    n_idx_chans = 7
    templates = np.load(root_dir+'/tmp/templates.npy')
    ptps = templates.ptp(0)
    pc_feature_ind = ptps.argsort(0)[::-1][:n_idx_chans].T
    np.save(root_dir+'/phy/pc_feature_ind.npy',pc_feature_ind)

    # *********************************************
    # ************** GET PCA OBJECTS **************
    # *********************************************
    pc_projections = get_pc_objects(root_dir)
    
    
    # *********************************************
    # ******** GENERATE PC PROJECTIONS ************
    # *********************************************
    pc_projections = compute_pc_projections(root_dir, templates, spike_train, 
                           pc_feature_ind, fname_standardized, n_channels, 
                           n_times, pc_features) 
        
    return 
    
def get_pc_objects(root_dir):
    
    ''' Grabs 10% of the spikes on each channel and makes PCA objects for each channel 
        The PCA object is used in another function to project all spikes
    ''' 

    # load templates from spike trains
    templates = np.load(root_dir + '/tmp/templates.npy')
    print (templates.shape)

    # standardized filename
    fname_standardized = root_dir+'/tmp/preprocess/standardized.bin'

    # spike_train
    spike_train = np.load(root_dir + '/tmp/spike_train.npy')

    # 
    n_channels = templates.shape[1]
    n_times = templates.shape[0]
    units = np.arange(templates.shape[2])

    # ********************************************
    # ***** APPROXIMATE PROJ MATRIX EACH CHAN ****
    # ********************************************
    # grab 10k spikes from each neuron and populate some larger array n_events x n_channels
    wfs_array = [[] for x in range(n_channels)]
    for unit in units:
        # load data only on max chans
        load_chans = pc_feature_ind[unit]
        
        idx1 = np.where(spike_train[:,1]==unit)[0]
        if idx1.shape[0]==0: continue
        spikes = np.int32(spike_train[idx1][:,0])-30

        idx3 = np.random.choice(np.arange(spikes.shape[0]),spikes.shape[0]//10)
        spikes = spikes[idx3]
            
        wfs = binary_reader_waveforms_allspikes(fname_standardized, n_channels, n_times, spikes, load_chans)
        print ("loaded unit: ", unit, spikes.shape, '/', idx1.shape, wfs.shape, "chans: ", load_chans)

        # make the waveform array 
        for ctr, chan in enumerate(load_chans):
            wfs_array[chan].extend(wfs[:,:,ctr])
            
    wfs_array = np.array(wfs_array)

    # compute PCA object on each channel using every 10th spike on that channel
    n_components = 3
    pc_projections = []
    for c in range(len(wfs_array)):
        _,_,pca = PCA(np.array(wfs_array[c]), n_components)
        pc_projections.append(pca)
    
    return (pc_projections)

def compute_pc_projections(root_dir, templates, spike_train, pc_feature_ind,
                           fname_standardized, n_channels, n_times,
                           pc_features):
    
    ''' Use PCA objects to compute projection for each spike on each channel
    '''
    
    # find max chan of each template; 
    max_chans = templates.ptp(0).argmax(0)

    # argmin and armgax locations
    locs = []
    for unit in units:
        min_loc = templates[:,max_chans[unit],unit].argmin(0)
        max_loc = templates[:,max_chans[unit],unit].argmax(0)
        locs.append([min_loc,max_loc])
        
    # pc_features; [n_spikes x n_channels_loc x n_pcs]
    # loop over all spikes and grab pc_features and amplitudes at the same time
    amplitudes = np.zeros(spike_train.shape[0],'float32')
    pc_features = np.zeros((spike_train.shape[0], n_idx_chans, n_components), 'float32')
    #print ("Length of recording: ", rec_len, ", in sec: ", rec_len/20000.)

    for unit in units:
    #for unit in [50]:
        # load data only on nearest channels as computed above
        load_chans = pc_feature_ind[unit]
        
        idx1 = np.where(spike_train[:,1]==unit)[0]
        if idx1.shape[0]==0: 
            # no need to change pc_features or amplitudes if no spikes
            continue
            
        # shift
        spikes = np.int32(spike_train[idx1][:,0])-30

        wfs = binary_reader_waveforms(fname_standardized, n_channels, n_times, spikes, load_chans)
        print ("loaded unit: ", unit, spikes.shape, '/', idx1.shape, wfs.shape, "chans: ", load_chans)

        # compute amplitudes
        temp = wfs[:,locs[unit][1],0]-wfs[:,locs[unit][0],0]
        amplitudes[idx1] = temp

        # compute PCA projections
        for ctr, chan in enumerate(pc_feature_ind[unit]):
            pc_features[idx1,ctr] = pc_projections[chan].transform(wfs[:,:,ctr])

    print ("Done reading")
    # transpose last 2 dimensions of pc_features
    np.save(root_dir + '/phy/amplitudes.npy',amplitudes)

    # transpose features to correct format expected by phy
    pc_features = pc_features.transpose(0,2,1)
    np.save(root_dir + '/phy/pc_features.npy',pc_features)

    
    
def binary_reader_waveforms_allspikes(filename, n_channels, n_times, spikes, channels=None, data_type='float32'):
    ''' Reader for loading raw binaries
    
        standardized_filename:  name of file contianing the raw binary
        n_channels:  number of channels in the raw binary recording 
        n_times:  length of waveform 
        spikes: 1D array containing spike times in sample rate of raw data
        channels: load specific channels only
        data_type: float32 for standardized data
        
        NOTE: this function returns zero arrays if outside the file boundaries
    
    '''

    # ***** LOAD RAW RECORDING *****
    wfs=[]
    data_empty = np.zeros((n_times,n_channels),'float32')
    with open(filename, "rb") as fin:
        for ctr,s in enumerate(spikes):
            # index into binary file: time steps * 4  4byte floats * n_channels
            try:
                fin.seek(s * 4 * n_channels, os.SEEK_SET)
                data = np.fromfile(
                fin,
                dtype='float32',
                count=(n_times * n_channels)).reshape(n_times, n_channels)[:,channels]
                wfs.append(data)
            except:
                wfs.append(data_empty[:,channels])
    
    wfs=np.array(wfs)

    fin.close()
    return wfs


def PCA(X, n_components):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components)
    pca.fit(X)
    X = pca.transform(X)
    Y = pca.inverse_transform(X)
    return X, Y, pca
