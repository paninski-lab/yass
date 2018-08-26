import os
import logging
import numpy as np
import parmap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from statsmodels import robust

from yass.deconvolute.util import (svd_shifted_templates,
                                   small_shift_templates,
                                   make_spt_list_parallel, clean_up,
                                   calculate_temp_temp_parallel)
                                   
from yass.deconvolute.deconvolve import (deconvolve_new_allcores_updated,
                                         deconvolve_match_pursuit)
                                         
from yass.deconvolute.match_pursuit import (MatchPursuit_objectiveUpsample, 
                                            MatchPursuitWaveforms)
from yass.cluster.util import (binary_reader, RRR3_noregress_recovery,
                               global_merge_all_ks_deconv, PCA)
from yass import read_config

import multiprocessing as mp
colors = np.asarray(["#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"])
        
colors = np.concatenate([colors,colors])
def run2(spike_train_cluster,
        templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_cluster.shape: {}'.format(templates.shape,
                                                 spike_train_cluster.shape))

    # ******************************************************************
    # *********************** SET PARAMETERS ***************************
    # ******************************************************************

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates, (2, 1, 0))

    sampling_rate = CONFIG.recordings.sampling_rate
    n_shifts = CONFIG.deconvolution.upsample_factor
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd
    n_features = CONFIG.deconvolution.n_features
    max_spikes = CONFIG.deconvolution.max_spikes
    n_processors = CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk
    
    
    # Cat: TODO: to read from CONFIG
    upsample = 10
    
    TMP_FOLDER = CONFIG.data.root_folder
    
    # Cat: TODO: read from CONFIG file
    buffer_size = 200

    # Grab length of .dat file to compute chunk indexes below
    standardized_filename = os.path.join(CONFIG.data.root_folder, 
                                    output_directory, recordings_filename)
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # Generate indexes in chunks (units of seconds); usually 10-60sec
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    # Make the 4 parameter list to be sent to deconvolution algorithm
    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list)) #[:2]
    proc_indexes = np.arange(len(idx_list))
    
    print("# of chunks for deconvolution: ", len(idx_list), " verbose mode: ",
          CONFIG.deconvolution.verbose)
            
    # need to transpose axes for match_pursuit 
    templates = np.swapaxes(templates,0,1)
    templates = np.swapaxes(templates,1,2)
    
    # make deconv directory
    deconv_dir = os.path.join(CONFIG.data.root_folder,
                              'tmp/deconv')
    if not os.path.isdir(deconv_dir):
        os.makedirs(deconv_dir)

    
    # read recording chunk and share as global variable
    # Cat: TODO: recording_chunk should be a shared variable in 
    #            multiprocessing module;
    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                        'tmp', 'standarized.bin')
    n_channels = CONFIG.recordings.n_channels
    root_folder = CONFIG.data.root_folder
    
    # remove templates < 3SU
    # Cat: TODO: read this from CONFIG
    template_threshold = 3
    ptps = templates.ptp(0).max(0)
    idx = np.where(ptps>template_threshold)[0]
    templates = templates[:,:,idx]
    
    # also remove 3SU units from spike train
    spike_train_cluster_new = []
    for ctr,k in enumerate(idx):
        temp = np.where(spike_train_cluster[:,1]==k)[0]
        temp_train = spike_train_cluster[temp]
        temp_train[:,1]=ctr
        spike_train_cluster_new.append(temp_train)
        
    spike_train_cluster_new = np.vstack(spike_train_cluster_new)
    
    #print (np.unique(spike_train_cluster_new[:,1]))
    #print (np.unique(spike_train_cluster_new[:,1]).shape)
    #quit()
    
    # reset templates to 61 time points
    # Cat: TODO: Load this from CONFIG; careful as clustering preamble
    #            extends template by flexible amount
    templates = templates[15:-15]
    
    
    ''' ****************************************************************
        ****************************************************************
        ********************** DECONV **********************************
        ****************************************************************
        ****************************************************************
    ''' 
    # *******************************
    # ******* INITIALIZE DATA *******
    # *******************************

    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    initial_chunk = 120 # 120 x 10 sec = 20mins chunk
    chunk_ctr = 0
    max_iter = 5000
    
    # select segments to be processed in current chunk
    idx_list_local = idx_list[:initial_chunk]
    
    # make deconv chunk directory
    deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                      'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))
    if not os.path.isdir(deconv_chunk_dir):
        os.makedirs(deconv_chunk_dir)
        
    ''' 
    # *******************************************
    # ***** INITIALIZE & RUN MATCH PURSUIT  *****
    # *******************************************
    '''
    # initialize object
    print ("")
    print ("Initializing Match Pursuit for chunk: ", chunk_ctr,
            " # segments: ", idx_list_local.shape[0], 
            " start: ", idx_list_local[0][0], " end: ", 
            idx_list_local[-1][1], " start(sec): ", 
            round(idx_list_local[0][0]/float(CONFIG.recordings.sampling_rate),1),
            " end(sec): ", 
            round(idx_list_local[-1][1]/float(CONFIG.recordings.sampling_rate),1))
            
    # initialize match pursuit
    mp_object = MatchPursuit_objectiveUpsample(
                              temps=templates,
                              deconv_chunk_dir=deconv_chunk_dir,
                              standardized_filename=standardized_filename,
                              max_iter=max_iter,
                              upsample=upsample)
    
    # run match pursuit
    print ("  running Match Pursuit...")
    # find which sections within current chunk not complete
    args_in = []
    for k in range(len(idx_list_local)):
        fname_out = (deconv_chunk_dir+
                     "/seg_{}_deconv.npz".format(
                     str(k).zfill(6)))
        if os.path.exists(fname_out)==False:
            args_in.append([[idx_list_local[k], k],
                            chunk_ctr,
                            buffer_size])

    if len(args_in)>0: 
        if CONFIG.resources.multi_processing:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            p.map_async(mp_object.run, args_in).get(988895)
            p.close()
        else:
            for k in range(len(args_in)):
                mp_object.run(args_in[k])
    
    # collect spikes
    res = []
    for k in range(len(idx_list_local)):
        fname_out = (deconv_chunk_dir+
                     "/seg_{}_deconv.npz".format(
                     str(k).zfill(6)))
                     
        data = np.load(fname_out)
        res.append(data['spike_train'])

    dec_spike_train = np.vstack(res)
    
    upsampled_templates = mp_object.get_reconstructed_upsampled_templates()
    print ("  match pursuit templates: ", upsampled_templates.shape)
    print ("  orig temps: ", templates.shape)
    
    '''
    # *****************************************
    # ********** RESIDUAL COMP STEP ***********
    # *****************************************
    
    '''
    # re-read entire block to get waveforms 
    # get indexes for entire chunk from local chunk list
    idx_chunk = [idx_list_local[0][0], idx_list_local[-1][1], 
                 idx_list_local[0][2], idx_list_local[0][3]]
                 
    # read data block using buffer
    recording_chunk = binary_reader(idx_chunk, buffer_size, 
                                    standardized_filename, 
                                    n_channels)
    
    # compute residual for data chunk and save to disk
    # Cat TODO: parallelize this and also figure out a faster way to 
    # process this data
    # Note: offset spike train to account for recording_chunk buffer size
    # this also enables working with spikes that are near the edges
    dec_spike_train_offset = dec_spike_train
    dec_spike_train_offset[:,0]+=buffer_size
    wf_object = MatchPursuitWaveforms(recording_chunk,
                                      upsampled_templates,
                                      dec_spike_train_offset,
                                      buffer_size)
        
    # cmopute residual using templates above
    # Cat: TODO: can this be parallelized?
    fname = (deconv_chunk_dir+"/residual.npy")
    if os.path.exists(fname)==False:
        wf_object.compute_residual()
        np.save(fname, wf_object.data)
    else:
        wf_object.data = np.load(fname)
    
    # recompute unit ids using the expanded templates above
    # (i.e. some templates were upsampled by 3 x or 5 x factor
    # so need to reset unit ids back to original templates
    print ("  deduplicating shifted templates for deconv...")
    fname = (deconv_chunk_dir + "/dec_spike_train_precluster.npy")
    
    #print (mp_object.temps_ids)
    if os.path.exists(fname)==False:
        new_spike_train = []
        for k in range(0, upsampled_templates.shape[2], upsample):
            idx_ids = np.arange(k,k+upsample, 1)
            idx_spikes = np.isin(dec_spike_train[:,1], idx_ids)
            dec_spike_train[idx_spikes,1]=k
            
            # Cat: TODO: pythonize this
            temp = dec_spike_train[idx_spikes]
            temp[:,1]=int(k/upsample)
            new_spike_train.append(temp)
            #print ("unit: ", int(k/upsample), " # spikes: ", temp.shape)
        
        dec_spike_train_offset = np.vstack(new_spike_train)
        np.save(fname, dec_spike_train_offset)
    else:
        dec_spike_train_offset = np.load(fname)

    '''
    # *****************************************
    # ************** RECLUSTERING *************
    # *****************************************   
    '''
    
    # make lists of arguments to be passed to 
    print ("Reclustering initial deconv chunk output...")
    from yass.cluster.util import (make_CONFIG2)
    CONFIG2 = make_CONFIG2(CONFIG)
    
    args_in = []
    units = np.arange(templates.shape[2])
    for unit in units:
    #for unit in [48,49]:
        fname_out = (deconv_chunk_dir+
                     "/unit_{}.npz".format(
                     str(k).zfill(6)))
        if os.path.exists(fname_out)==False:
            args_in.append([unit, 
                            dec_spike_train_offset,
                            spike_train_cluster_new,
                            idx_chunk,
                            templates[:,:,unit],
                            CONFIG2,
                            deconv_chunk_dir
                            ])

    # run residual-reclustering function
    if len(args_in)>0: 
        if CONFIG.resources.multi_processing:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            res = p.map_async(deconv_residual_recluster, args_in).get(988895)
            p.close()
            
        else:
            for unit in range(len(args_in)):
                res = deconv_residual_recluster(args_in[unit])
    
    print ("  completed deconv chunk: ", chunk_ctr)

    # run template merge
    spike_train, tmp_loc, templates_first_chunk = global_merge_all_ks_deconv(
                                          deconv_chunk_dir, recording_chunk,
                                          units,
                                          CONFIG2)
                                          
    np.savez(deconv_chunk_dir+"/deconv_results.npz", 
            spike_train=spike_train, 
            tmp_loc=tmp_loc, 
            templates=templates_first_chunk)

    ''' 
    ***********************************************************
    *********** MATCH PURSUIT SECONDARY CHUNKS ****************
    ***********************************************************
    '''

    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    chunk_ctr += 1
    chunk_size = initial_chunk
    
    # need to transpose axes coming out of global merge
    templates = np.swapaxes(templates_first_chunk,0,1)
    
    for c in range(initial_chunk, len(idx_list), chunk_size):
 
        # select segments to be processed in current chunk
        idx_list_local = idx_list[c:c+chunk_size]
        
        # make deconv chunk directory
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))

        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
        
       
        ''' *******************************************
            ****** INITIALIZE/RUN MATCH PURSUIT *******
            *******************************************
        ''' 
        print ("\n\nInitializing Match Pursuit for chunk: ", chunk_ctr,
            " # segments: ", idx_list_local.shape[0], 
            " start: ", idx_list_local[0][0], " end: ", 
            idx_list_local[-1][1], " start(sec): ", 
            round(idx_list_local[0][0]/float(CONFIG.recordings.sampling_rate),1),
            " end(sec): ", 
            round(idx_list_local[-1][1]/float(CONFIG.recordings.sampling_rate),1))

        # Cat: TODO: read this from CONFIG
        max_iter = 5000
        #dynamic_templates=True
        #mp_object = MatchPursuit3(templates, 
                                  #deconv_chunk_dir,
                                  #standardized_filename,
                                  #max_iter,
                                  #upsample,
                                  #dynamic_templates)
        # re-initialize match pursuit
        mp_object = MatchPursuit_objectiveUpsample(
                              temps=templates,
                              deconv_chunk_dir=deconv_chunk_dir,
                              standardized_filename=standardized_filename,
                              max_iter=max_iter,
                              upsample=upsample)
      
        # run match pursuit
        print (" running Match Pursuit...")
        # find which sections within current chunk not complete
        args_in = []
        for k in range(len(idx_list_local)):
            fname_out = (deconv_chunk_dir+
                         "/seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
            if os.path.exists(fname_out)==False:
                args_in.append([[idx_list_local[k], k],
                                chunk_ctr,
                                buffer_size])

        if len(args_in)>0: 
            if CONFIG.resources.multi_processing:
                p = mp.Pool(processes = CONFIG.resources.n_processors)
                p.map_async(mp_object.run, args_in).get(988895)
                p.close()
            else:
                for k in range(len(args_in)):
                    mp_object.run(args_in[k])
        
        # collect spikes
        res = []
        for k in range(len(idx_list_local)):
            fname_out = (deconv_chunk_dir+
                         "/seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
                         
            data = np.load(fname_out)
            res.append(data['spike_train'])

        dec_spike_train = np.vstack(res)
        
        upsampled_templates = mp_object.get_reconstructed_upsampled_templates()
        print ("  match pursuit templates: ", upsampled_templates.shape)
        print ("  orig temps: ", templates.shape)
        


        # recompute unit ids using the expanded templates above
        # (i.e. some templates were upsampled by 3 x or 5 x factor
        # so need to reset unit ids back to original templates
        print ("  deduplicating shifted templates for deconv...")
        fname = (deconv_chunk_dir + "/dec_spike_train_precluster.npy")
        
        #print (mp_object.temps_ids)
        if os.path.exists(fname)==False:
            new_spike_train = []
            for k in range(0, upsampled_templates.shape[2], upsample):
                idx_ids = np.arange(k,k+upsample, 1)
                idx_spikes = np.isin(dec_spike_train[:,1], idx_ids)
                dec_spike_train[idx_spikes,1]=k
                
                # Cat: TODO: pythonize this
                temp = dec_spike_train[idx_spikes]
                temp[:,1]=int(k/upsample)
                new_spike_train.append(temp)
                #print ("unit: ", int(k/upsample), " # spikes: ", temp.shape)
            
            dec_spike_train_offset = np.vstack(new_spike_train)
            np.save(fname, dec_spike_train_offset)
        else:
            dec_spike_train_offset = np.load(fname)
            
                                              
        np.savez(deconv_chunk_dir+"/deconv_results.npz", 
                spike_train=dec_spike_train, 
                tmp_loc=tmp_loc, 
                templates=templates)

        print ("  completed deconv chunk: ", chunk_ctr)
        chunk_ctr+=1



    '''
         REMERGE ALL SPIKE TRAIN CHUNKS
    '''


    # reload all spike trains and concatenate them:
    spike_train = np.zeros((0,2),'int32')
    for chunk_ctr, c in enumerate(range(0, len(idx_list), chunk_size)):
 
        # make deconv chunk directory
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))

        deconv_results = np.load(deconv_chunk_dir+'/deconv_results.npz')
        spike_train = np.vstack((spike_train, deconv_results['spike_train']))
    
    print (spike_train.shape)

    # Cat: TODO: reorder spike train by time
    print ("Final deconv spike train: ", spike_train.shape)

    logger.info('spike_train.shape: {}'.format(spike_train.shape))

    return spike_train


def deconv_residual_recluster(data_in): 
    
    unit = data_in[0]
    dec_spike_train_offset = data_in[1]
    spike_train_cluster_new = data_in[2]
    idx_chunk = data_in[3]
    template = data_in[4]
    CONFIG = data_in[5]
    deconv_chunk_dir = data_in[6]
    
    # Cat: TODO: read this from CONFIG
    n_dim_pca_compression = 5
    
    deconv_filename = (deconv_chunk_dir+"/unit_"+str(unit).zfill(6)+'.npz')

    if os.path.exists(deconv_filename)==False:
        
        # select deconv spikes and read waveforms
        unit_sp = dec_spike_train_offset[dec_spike_train_offset[:, 1] == unit, :]
        if unit_sp.shape[0]!= np.unique(unit_sp[:,0]).shape[0]:
            idx_unique = np.unique(unit_sp[:,0], return_index = True)[1]
            unit_sp = unit_sp[idx_unique]

        # Cat: TODO: load wider waveforms just as in clustering
        # Cat TODO: Need to load from CONFIG; careful as the templates are
        #           now being extended during cluster preamble using flexible val
        n_times = 61
        wf = get_wfs_from_residual(unit_sp, template, deconv_chunk_dir,
                                            n_times)
        
        # PCA denoise waveforms before processing
        wf_PCA = np.zeros(wf.shape)
        for ch in range(wf.shape[2]):
            _, wf_PCA[:,:,ch] = PCA(wf[:,:,ch], n_dim_pca_compression)
            
            
        channel = wf.mean(0).ptp(0).argmax(0)
        print ("unit: ", unit, wf.shape, "maxchan: ", channel, 
                    unit_sp.shape[0], "unique: ", np.unique(unit_sp[:,0]).shape[0])

        # run mfm
        scale = 10 

        triageflag = False
        alignflag = True
        plotting = True
        n_feat_chans = 5
        n_dim_pca = 3
        wf_start = 0
        wf_end = 40
        mfm_threshold = 0.90
        upsample_factor = 5
        nshifts = 15
                
        chans = [] 
        gen = 0     #Set default generation for starting clustering stpe
        assignment_global = []
        spike_index = []
        feat_chans_cumulative = []
        
        # plotting parameters
        if plotting:
            #x = np.zeros(100, dtype = int)            
            #fig = plt.figure(figsize =(50,25))
            #grid = plt.GridSpec(10,5,wspace = 0.0,hspace = 0.2)
            #ax_t = fig.add_subplot(grid[13:, 6:])
            
            x = np.zeros(100, dtype = int)
            fig = plt.figure(figsize =(100,100))
            grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
            ax_t = fig.add_subplot(grid[13:, 6:])
            
        else:
            fig = []
            grid = []
            ax_t = []
            x = []
            
        RRR3_noregress_recovery(unit, wf_PCA, unit_sp, gen, fig, grid, x,
            ax_t, triageflag, alignflag, plotting, n_feat_chans, 
            n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
            upsample_factor, nshifts, assignment_global, spike_index, scale)


        # finish plotting 
        if plotting: 
            #ax_t = fig.add_subplot(grid[13:, 6:])
            for i in range(CONFIG.recordings.n_channels):
                ax_t.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4, 
                                                                fontsize=30)
                # fill bewteen 2SUs on each channel
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-61,0,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                    
                # plot original templates
                ax_t.plot(CONFIG.geom[:,0]+
                    np.arange(-template.shape[0],0)[:,np.newaxis]/3., 
                    CONFIG.geom[:,1] + template*scale, 'r--', c='red')
                        
            # plot max chan with big red dot                
            ax_t.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], s = 2000, 
                                                    color = 'red')
            
            #print ("len spike_index: ", len(spike_index))
            # if at least 1 cluster is found:
            labels=[]
            if len(spike_index)>0: 
                sic_temp = np.concatenate(spike_index, axis = 0)
                assignment_temp = np.concatenate(assignment_global, axis = 0)
                idx = sic_temp[:,1] == unit
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)
                if len(clusters)>1: 
                    print ("Multiple clusters unit: ", unit)

                chans.extend(channel*np.ones(clusters.size))

                
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = colors[clust%100], label = "deconv = {}".format(sizes[i]))
                    labels.append(patch_j)
            
            idx3 = np.where(spike_train_cluster_new[:,1]==unit)[0]
            spikes_in_chunk = np.where(np.logical_and(spike_train_cluster_new[idx3][:,0]>idx_chunk[0], 
                                                      spike_train_cluster_new[idx3][:,0]<=idx_chunk[1]))[0]

            patch_original = mpatches.Patch(color = 'red', label = 
                             "cluster in chunk/total: "+ 
                             str(spikes_in_chunk.shape[0])+"/"+
                             str(idx3.shape[0]))
            labels.append(patch_original)
            
            ax_t.legend(handles = labels, fontsize=30)

            # plto title
            fig.suptitle("Unit: "+str(unit), fontsize=25)
            fig.savefig(deconv_chunk_dir+"/unit{}.png".format(unit))
            plt.close(fig)

        
        # Save weighted templates also:
        # Cat: TODO: save wider waveforms just as in clustering
        spike_train = unit_sp
        temp = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
        temp_std = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
        for k in range(len(spike_index)):
            idx = np.in1d(spike_train[:,0], spike_index[k][:,0])
            temp[k] = np.mean(wf[idx],axis=0)
            temp_std[k] = robust.mad(wf[idx],axis=0)

        # save all clustered data
        np.savez(deconv_filename, spike_index=spike_index, 
                        templates=temp,
                        templates_std=temp_std,
                        weights=np.asarray([sic.shape[0] for sic in spike_index]))

    else:
        print ("unit: ", unit, " deconv output already reclustered...")
                        
                    
    return


def get_wfs_from_residual(unit_sp, template, deconv_chunk_dir, n_times=61):
    """Gets clean spikes for a given unit."""
    
    data = np.load(deconv_chunk_dir+'/residual.npy')

    # Add the spikes of the current unit back to the residual
    temp = data[np.arange(0, n_times) + unit_sp[:, :1], :] + template
    return temp


def visible_chans(temps):
    a = temps.ptp(0) #np.max(temps, axis=0) - np.min(temps, 0)
    vis_chan = a > 1

    return vis_chan

        
def pairwise_filter_conv_local(deconv_chunk_dir, n_time, n_unit, temporal, 
                         singular, spatial, approx_rank, vis_chan, temps):
    
    #print (deconv_chunk_dir+"/parwise_conv.npy")
    if os.path.exists(deconv_chunk_dir+"/pairwise_conv.npy")==False:
        print ("IN LOOP")
        conv_res_len = n_time * 2 - 1
        pairwise_conv = np.zeros([n_unit, n_unit, conv_res_len])
        for unit1 in range(n_unit):
            u, s, vh = temporal[unit1], singular[unit1], spatial[unit1]
            vis_chan_idx = vis_chan[:, unit1]
            for unit2 in range(n_unit):
                for i in range(approx_rank):
                    pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                        s[i] * u[:, i].flatten(), 'full')

        np.save(deconv_chunk_dir+"/pairwise_conv.npy", pairwise_conv)
    else:
        pairwise_conv = np.load(deconv_chunk_dir+"/pairwise_conv.npy")
        
    return pairwise_conv
    
        

    
