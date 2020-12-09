import os
import logging
import numpy as np
import parmap
import scipy

import datetime as dt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from yass import read_config

from yass.visual.util import binary_reader_waveforms

#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(CONFIG, fname_spike_train, fname_templates):
            
    """Generate phy2 visualization files
    """

    logger = logging.getLogger(__name__)

    logger.info('GENERATTING PHY files')

    # set root directory for output
    root_dir = CONFIG.data.root_folder
    fname_standardized = os.path.join(os.path.join(os.path.join(
                        root_dir,'tmp'),'preprocess'),'standardized.bin')

    #
    n_channels = CONFIG.recordings.n_channels
    n_times = CONFIG.recordings.sampling_rate//1000 * CONFIG.recordings.spike_size_ms +1

    # output folder
    output_directory = os.path.join(root_dir, 'phy')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # pca # of components    
    n_components = 3

    # cluster id for each spike; [n_spikes]
    #spike_train = np.load(root_dir + '/tmp/spike_train.npy')
    #spike_train = np.load(root_dir + '/tmp/final_deconv/deconv/spike_train.npy')
    spike_train = np.load(fname_spike_train)
    spike_clusters = spike_train[:,1]
    np.save(os.path.join(root_dir,'phy','spike_clusters.npy'), spike_clusters)

    # spike times for each spike: [n_spikes]
    spike_times = spike_train[:,0]
    np.save(os.path.join(root_dir,'phy','spike_times.npy'), spike_times)

    # save templates; not sure why this is required?!
    np.save(os.path.join(root_dir,'phy','spike_templates.npy'), spike_clusters)

    # save geometry
    chan_pos = np.loadtxt(os.path.join(root_dir,CONFIG.data.geometry))
    np.save(os.path.join(root_dir,'phy','channel_positions.npy'), chan_pos)

    # sequential channel order
    channel_map = np.arange(chan_pos.shape[0])
    np.save(os.path.join(root_dir,'phy','channel_map.npy'), channel_map)

    # pick largest SU channels for each unit; [n_templates x n_channels_loc]; 
    # gives # of channels of the corresponding columns in pc_features, for each spike.
    n_idx_chans = 7
    templates = np.load(fname_templates).transpose(1,2,0)
    print ("PHY loaded templates: ", templates.shape)
    ptps = templates.ptp(0)
    pc_feature_ind = ptps.argsort(0)[::-1][:n_idx_chans].T
    np.save(os.path.join(root_dir,'phy','pc_feature_ind.npy'),pc_feature_ind)

    # 
    n_channels = templates.shape[1]
    n_times = templates.shape[0]
    units = np.arange(templates.shape[2])

    # unit templates [n_units, times, n_chans]
    temps = templates.transpose(2,0,1)
    np.save(os.path.join(root_dir,"phy","templates.npy"),temps)

    # *********************************************
    # ************** SAVE params.py file **********
    # *********************************************
    fname_out = os.path.join(output_directory, 'params.py')
    fname_bin = os.path.join(root_dir,CONFIG.data.recordings)
    #
    f= open(fname_out,"w+")
    f.write("dat_path = '%s'\n" % fname_bin)
    f.write("n_channels_dat = %i\n" % n_channels)
    f.write("dtype = 'int16'\n")
    f.write("offset = 0\n")
    f.write("sample_rate = %i\n" % CONFIG.recordings.sampling_rate)
    f.write("hp_filtered = False")
    f.close()
    
    # *********************************************
    # ************** GET PCA OBJECTS **************
    # *********************************************
    fname_out = os.path.join(output_directory,'pc_objects.npy')
    if os.path.exists(fname_out)==False:
        pc_projections = get_pc_objects(root_dir, pc_feature_ind, n_channels,
                            n_times, units, n_components, CONFIG, spike_train)
        np.save(fname_out, pc_projections)
    else:
        pc_projections = np.load(fname_out,allow_pickle=True)
    
    
    # *********************************************
    # ******** GENERATE PC PROJECTIONS ************
    # *********************************************
    fname_out = os.path.join(output_directory, 'pc_features.npy')
    if os.path.exists(fname_out)==False:
        pc_projections = compute_pc_projections(root_dir, templates, spike_train, 
                           pc_feature_ind, fname_standardized, n_channels, 
                           n_times, units, pc_projections, n_idx_chans,
                           n_components, CONFIG) 
  
  
    # *********************************************
    # ******** GENERATE SIMILARITY MATRIX *********
    # *********************************************
    print ("... making similarity matrix")
    # Cat: TODO: better similarity algorithms/metrics available in YASS


    similar_templates = np.zeros((temps.shape[0],temps.shape[0]),'float32')
    
    fname_out = os.path.join(os.path.join(root_dir,'phy'),'similar_templates.npy')
    if os.path.exists(fname_out)==False:

        if CONFIG.resources.multi_processing==False:
            for k in tqdm(range(temps.shape[0])):
                for p in range(k,temps.shape[0]):
                    temp1 = temps[k].T.ravel()
                    results=[]
                    for z in range(-1,2,1):
                        temp_temp = np.roll(temps[p].T,z,axis=0).ravel()
                        results.append(cos_sim(temps[k].T.ravel(),temp_temp))

                    similar_templates[k,p] = np.max(results)
        else:
            units_split = np.array_split(np.arange(temps.shape[0]), CONFIG.resources.n_processors)
            res = parmap.map(similarity_matrix_parallel, units_split, temps, similar_templates,
                                processes=CONFIG.resources.n_processors,
                                pm_pbar=True)
            
            print (res[0].shape)
            similar_templates = res[0]
            for k in range(1, len(res),1):
                similar_templates+=res[k]
            
    similar_templates = symmetrize(similar_templates)
    np.save(fname_out,similar_templates)

    return 
    
def cos_sim(a, b):
    # Takes 2 vectors a, b and returns the cosine similarity according 
    # to the definition of the dot product
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

    #temps = np.load(os.path.join(root_dir, 'tmp'),'templates.npy').transpose(2,0,1)

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def similarity_matrix_parallel(units, temps, similar_templates):
    
    for k in units:
        for p in range(k,temps.shape[0]):
            temp1 = temps[k].T.ravel()
            results=[]
            for z in range(-1,2,1):
                temp_temp = np.roll(temps[p].T,z,axis=0).ravel()
                results.append(cos_sim(temps[k].T.ravel(),temp_temp))

            similar_templates[k,p] = np.max(results)

    return similar_templates
    

def get_pc_objects_parallel(units, n_channels, pc_feature_ind, spike_train,
                fname_standardized, n_times, phy_percent):
    
    ''' Function that reads 10% of spikes on top 7 channels
        Data is then used to make PCA objects/rot matrices for each channel
    ''' 
    
    # grab 10% spikes from each neuron and populate some larger array n_events x n_channels
    wfs_array = [[] for x in range(n_channels)]
    
    for unit in units:
        # load data only on max chans
        load_chans = pc_feature_ind[unit]
        
        idx1 = np.where(spike_train[:,1]==unit)[0]
        if idx1.shape[0]==0: continue
        spikes = np.int32(spike_train[idx1][:,0])-30

        idx3 = np.random.choice(np.arange(spikes.shape[0]),spikes.shape[0]//int(1./phy_percent))
        spikes = spikes[idx3]
            
        wfs = binary_reader_waveforms_allspikes(fname_standardized, n_channels, n_times, spikes, load_chans)
        #print(wfs.shape)
        
        # make the waveform array 
        for ctr, chan in enumerate(load_chans):
            wfs_array[chan].extend(wfs[:,:,ctr])
            
    return (wfs_array)


def get_pc_objects(root_dir,pc_feature_ind, n_channels, n_times, units, n_components, CONFIG,
                  spike_train):
    
    ''' First grab 10% of the spikes on each channel and makes PCA objects for each channel 
        Then generate PCA object for each channel using spikes
    ''' 

    # load templates from spike trains
    # templates = np.load(root_dir + '/tmp/templates.npy')
    # print (templates.shape)

    # standardized filename
    fname_standardized = os.path.join(os.path.join(os.path.join(root_dir,'tmp'),
                                'preprocess'),'standardized.bin')

    # spike_train
    #spike_train = np.load(os.path.join(os.path.join(root_dir, 'tmp'),'spike_train.npy'))
    #spike_train = np.load(os.path.join(os.path.join(root_dir, 'tmp'),'spike_train.npy'))


    # ********************************************
    # ***** APPROXIMATE PROJ MATRIX EACH CHAN ****
    # ********************************************
    print ("...reading sample waveforms for each channel")
    fname_out = os.path.join(os.path.join(root_dir, 'phy'),'wfs_array.npy')
    
    try:
        phy_percent = CONFIG.resources.phy_percent_spikes
    except:
        phy_percent = 0.1

    print ("... generating phy output for ", phy_percent *100, " % of total spikes")

    if os.path.exists(fname_out)==False:
        if CONFIG.resources.multi_processing==False:
            wfs_array = get_pc_objects_parallel(units, n_channels, pc_feature_ind, 
                                      spike_train, fname_standardized, n_times, phy_percent)
        else:
            unit_list = np.array_split(units, CONFIG.resources.n_processors)
            res = parmap.map(get_pc_objects_parallel, unit_list, n_channels, pc_feature_ind, 
                                spike_train, fname_standardized, n_times, phy_percent,
                                processes=CONFIG.resources.n_processors,
                                pm_pbar=True)
                       
            # make the waveform array 
            wfs_array = [[] for x in range(n_channels)]
            for k in range(len(res)):
                for c in range(n_channels):
                    #print ("res[k][c]: ", res[k][c].shape)
                    wfs_array[c].extend(res[k][c])
            
            #for k in range(len(wfs_array)):
            #    wfs_array[c] = np.vstack(wfs_array[c])
        wfs_array = np.array(wfs_array)
        np.save(fname_out, wfs_array)
    else:
        #print ("loading from disk")
        wfs_array = np.load(fname_out,allow_pickle=True)

    # compute PCA object on each channel using every 10th spike on that channel
    print ("...making projection objects for each chan...")
    pc_projections = []
    for c in tqdm(range(len(wfs_array))):
        #print ("chan: ", c, " wfs_array: ", np.array(wfs_array[c]).shape)
        if (len(wfs_array[c])>2):
            _,_,pca = PCA(np.array(wfs_array[c]), n_components)
            pc_projections.append(pca)
        else:
            # add noise waveforms; should eventually fix to just turn these channesl off
            wfs_noise = np.random.rand(100, CONFIG.recordings.spike_size_ms* 
                                            CONFIG.recordings.sampling_rate//1000+1)
            #print ("inserting noise: ", wfs_noise.shape)
            _,_,pca = PCA(wfs_noise, n_components)
            pc_projections.append(pca)
            
    return (pc_projections)
    
    
def compute_pc_projections(root_dir, templates, spike_train, pc_feature_ind,
                           fname_standardized, n_channels, n_times, units,
                           pc_projections, n_idx_chans, n_components, CONFIG):
    
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

    print ("...getting PCA features for each spike...")
    if CONFIG.resources.multi_processing==False:
        (pc_features, amplitudes) = get_final_features_amplitudes(units, pc_feature_ind, 
                                        spike_train, fname_standardized,
                                        n_channels, n_times, amplitudes, pc_features, locs,
                                        pc_projections)
    else:
        unit_list = np.array_split(units, CONFIG.resources.n_processors)
        res = parmap.map(get_final_features_amplitudes, unit_list, pc_feature_ind, 
                            spike_train, fname_standardized, n_channels, n_times, 
                            locs, pc_projections, n_idx_chans, n_components,
                            processes=CONFIG.resources.n_processors,
                            pm_pbar=True)

        # reconcile the results:
        amplitudes_final = res[0][0]
        pc_features_final = res[0][1]
        for k in range(1,len(res),1):
            amplitudes_final+= res[k][0]
            pc_features_final+= res[k][1]

    # transpose last 2 dimensions of pc_features
    np.save(os.path.join(os.path.join(root_dir, 'phy'),'amplitudes.npy'), amplitudes_final)

    # transpose features to correct format expected by phy
    pc_features_final = pc_features_final.transpose(0,2,1)
    np.save(os.path.join(os.path.join(root_dir,'phy'),'pc_features.npy'), pc_features_final)

def get_final_features_amplitudes(units, pc_feature_ind, spike_train, 
                    fname_standardized, n_channels, n_times, locs, 
                    pc_projections, n_idx_chans, n_components):

    amplitudes = np.zeros(spike_train.shape[0], 'float32')
    pc_features = np.zeros((spike_train.shape[0], n_idx_chans, n_components), 'float32')
    
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

        wfs = binary_reader_waveforms_allspikes(fname_standardized, n_channels, n_times, spikes, load_chans)
        #print ("loaded unit: ", unit, spikes.shape, '/', idx1.shape, wfs.shape, "chans: ", load_chans)

        # compute amplitudes
        temp = wfs[:,locs[unit][1],0]-wfs[:,locs[unit][0],0]
        amplitudes[idx1] = temp

        # compute PCA projections
        for ctr, chan in enumerate(pc_feature_ind[unit]):
            pc_features[idx1,ctr] = pc_projections[chan].transform(wfs[:,:,ctr])
    
    return (amplitudes, pc_features)
    
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
