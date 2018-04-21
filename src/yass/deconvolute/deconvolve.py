import numpy as np
import logging
from scipy.signal import argrelmax
import os
import time


def deconvolve_new_allcores_updated(data_in, output_directory, TMP_FOLDER,
                    filename_bin, filename_spt_list, filename_temp_temp, 
                    filename_shifted_templates, buffer_size, n_channels, 
                    temporal_features, spatial_features, n_explore, 
                    threshold_d, verbose):
    
    #start time counter 
    start_time_chunk = time.time()

    #Load indexes from zipped data_in variable
    idx_list, chunk_idx = data_in[0], data_in[1]

    #Convert indexes to required start/end indexes including buffering
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]
   
    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    offset = idx_local   #idx_local[0].start


    #******************************************************************
    #******************************************************************
    #********* STAGE 1: LOAD RAW DATA / SETUP ARRAYS ******************
    #******************************************************************
    #******************************************************************

    #Load spt_list; Note: Peter thinks about removing need for spt_list
    spt_list = np.load(filename_spt_list)
    temp_temp = np.load(filename_temp_temp)
    shifted_templates = np.load(filename_shifted_templates)

    #Load spike_train clear after templates; 
    #Cat: eventually change from hardcoded filename
    spike_train_clear_fname = TMP_FOLDER + \
              output_directory+ '/spike_train_clear_after_templates.npy'
    spike_train_clear = np.load(spike_train_clear_fname)
    
    with open(filename_bin, "rb") as fin:
	if data_start==0:
	    # Seek position and read N bytes
	    recordings_1D = np.fromfile(fin, dtype='float32', count=(
                                       data_end+buffer_size)*n_channels)
	    recordings_1D = np.hstack((np.zeros(buffer_size*n_channels,
                                        dtype='float32'),recordings_1D))
	else:
	    fin.seek((data_start-buffer_size)*4*n_channels, os.SEEK_SET)         
        #Grab 2 x template_width x 2 buffers
        recordings_1D =  np.fromfile(fin, dtype='float32', count=(
                       (data_end-data_start+buffer_size*2)*n_channels))	

	if len(recordings_1D)!=(
                        (data_end-data_start+buffer_size*2)*n_channels):
	    recordings_1D = np.hstack((recordings_1D,
                      np.zeros(buffer_size*n_channels,dtype='float32')))

    fin.close()

    #Convert to 2D array
    recordings = recordings_1D.reshape(-1,n_channels)

    #**** SETUP ARRAYS ****
    n_templates, n_shifts, waveform_size, n_channels = \
                                                shifted_templates.shape
    R = int((waveform_size-1)/2)
    R2 = int((waveform_size-1)/4)
    principal_channels = np.argmax(
                            np.max(np.abs(shifted_templates),(1,2)), 1)
    
    #||V|| L2 norms (square of this) for the shifted templates 
    norms = np.sum(np.square(shifted_templates),(2,3))		
    visible_channels = np.max(np.abs(spatial_features), (1,2)) > \
                  np.min(np.max(np.abs(spatial_features), (1,2,3)))*0.5
    temporal_features = temporal_features[:, :, R2:3*R2+1]
    
    #**** MAKE D_MATRIX ****
    #d_matrix = np.ones((recordings.shape[0],n_templates,n_shifts))*-np.Inf
    #d_matrix = np.empty((recordings.shape[0],n_templates,n_shifts))*-np.Inf
    d_matrix = np.full((recordings.shape[0],n_templates,n_shifts),-np.Inf)
    #print ("...d_matrix make time: ", time.time()-start_dmatrix)



    #******************************************************************
    #******************************************************************
    #************** STAGE 2:  REMOVE CLEAR SPIKES *********************
    #******************************************************************
    #******************************************************************
   
    #Identify clear spikes in time window
    indexes = np.logical_and(spike_train_clear[:,0]>=data_start, 
                                       spike_train_clear[:,0]<=data_end)
    clear_spiketimes = spike_train_clear[indexes]

    #******* DE-DUPLICATE SPIKES *******
    #Deduplicate spikes: remove spikes from same template within 20 steps
    deduplication_time = time.time()
    indexes=[]      #list to hold duplicate spike time indexes
    ctr=0           #loop ctr; probably can write this loop simpler 
    t_steps = 20    #window of time-steps to remove spikes if they occur
    for k in range(len(clear_spiketimes)):
        nearby_indexes = np.where(np.logical_and(
                clear_spiketimes[:,0]>=clear_spiketimes[k,0]-t_steps, 
                clear_spiketimes[:,0]<=clear_spiketimes[k,0]+t_steps))[0]
        
        matches = np.where(
            clear_spiketimes[nearby_indexes,1]==clear_spiketimes[k,1])[0]
        
        if len(matches)>1:
            if ctr!=0:
                if indexes[ctr-1][0]!=nearby_indexes[matches][0]:
                    indexes.append(nearby_indexes[matches])
                    ctr+=1
            else:
                indexes.append(nearby_indexes[matches])
                ctr+=1

    #Make list of duplicated events and delete all but one
    duplicated_event_indexes = []
    for k in range(len(indexes)):
        duplicated_event_indexes.append(indexes[k][:-1])
    
    if len(duplicated_event_indexes)!=0:
        duplicated_event_indexes = np.hstack(duplicated_event_indexes)
        clear_spiketimes_unique = np.delete(
                       clear_spiketimes,duplicated_event_indexes,axis=0)
    else: 
        clear_spiketimes_unique = clear_spiketimes

    clear_spiketimes_unique = np.int32(clear_spiketimes_unique)

    #***** REMOVING CLEAR SPIKES *********
    window = 30 #Window in time steps; #Cat: Can use even smaller chunks
    recordings_copy = recordings.copy()
    #clearing_time = time.time()
    for p in range(len(clear_spiketimes_unique)):
        #Load correct templates_shifted - but only chunk in centre 
        templates_shifted = shifted_templates[
            clear_spiketimes_unique[p,1],:,:,
            principal_channels[clear_spiketimes_unique[p,1]]] \
            [:,30-window:30+window+1]
        
        #Select times and expand into window
        times = (clear_spiketimes_unique[p,0,np.newaxis] + \
                              np.arange(-R2-n_explore, n_explore+R2+1))

        #Select entire channel trace on max channel
        recording = recordings[:, 
                       principal_channels[clear_spiketimes_unique[p,1]]]

        #Select X number of recoding chunks that are time_shifted 
        #Cat: try to pythonize this step
        recording_chunks = []
        for time_ in times:
            recording_chunks.append(recording[
                        buffer_size+time_-window-data_start:  \
                        buffer_size+time_+window+1-data_start])
        recording_chunks = np.array(recording_chunks)

        #Dot product of shifted recording chunks and shifted raw data
        dot_product = np.matmul(templates_shifted,recording_chunks.T)
        index = np.unravel_index(dot_product.argmax(), dot_product.shape)

        recordings_copy[buffer_size+time_+index[1]-window- 
              2*(R2+n_explore)-data_start:buffer_size+time_+index[1]-  
              2*(R2+n_explore)+window+1-data_start,:]-= \
              shifted_templates[clear_spiketimes_unique[p,1],
              index[0],30-window:30+window+1]

    #Save spike_cleared recordings into recordings
    recordings=recordings_copy

    #******* CLEAN SPT_LIST ******
    #First make spt_list for local analysis only.
    spt_list_local = []
    for k in range(len(spt_list)):
        spt = spt_list[k]
        #Find spikes on each channel within time window and offset to 0
        spt_list_local.append(spt[np.logical_and(spt >= 
                                data_start,spt < data_end)]-data_start)			

    #Remove clear spikes and save to spt_list_local        
    #Select spike times as 1st column of array 
    clear_spike_times = clear_spiketimes_unique[:,0]          
    #Loop over all spt_list, i.e. channels
    for k in range(len(spt_list_local)):                      
        # find intersetion of clear spikes and spt_list on all channels
        common = np.intersect1d(clear_spike_times+data_start, 
                                                    spt_list_local[k])      
        if len(common)>0:
            indexes = np.hstack(
                  [(spt_list_local[k]==i).nonzero()[0] for i in common])    
            spt_list_local[k] = np.delete(spt_list_local[k],indexes)                           


    #******************************************************************
    #******************************************************************
    #***************** STAGE 3:  DOT PRODUCT LOOP *********************
    #******************************************************************
    #******************************************************************
    

    ctr_times=0	
    for k in range(n_templates):
        #Select spikes on the max channel and offset to t=0 + buffer
        if False: 
            spt = spt_list[principal_channels[k]]
            spt = spt[np.logical_and(spt >= data_start,spt < data_end)]		
            spt = np.int32(spt - data_start + offset)
        
        #Cat use cleared_spike spt_list, also index locally only
        ##Cat: todo talk to Peter re: eliminating spt_list
        else: 
            spt = spt_list_local[principal_channels[k]]                         
            spt = np.int32(spt + offset)
        
        #Pick channels around template 
        ch_idx = np.where(visible_channels[k])[0]	

        times = (spt[:, np.newaxis] + \
                               np.arange(-R2-n_explore, n_explore+R2+1))	
        if len(times)==0: continue		

        #This step is equivalent to: wf = recordings[times][ch_idx]; 
        #In sum, wf contains n_spikes X n_sample_pts X n_channels
        wf = ((recordings.ravel()[(ch_idx + 
             (times * recordings.shape[1]).reshape((-1,1))).ravel()]). \
                            reshape(times.size, ch_idx.size)). \
                            reshape((spt.shape[0], -1, ch_idx.shape[0]))	
        
        spatial_dot = np.matmul(spatial_features[k]   \
                      [np.newaxis, np.newaxis, :, :, ch_idx],  \
                      wf[:, :, np.newaxis, :, np.newaxis])  \
                      [:,:,:,:,0].transpose(0, 2, 1, 3)

        dot = np.zeros((spt.shape[0], 2*n_explore+1, n_shifts))
        for j in range(2*n_explore+1):
            dot[:, j] = np.sum(spatial_dot[:, :, j:j+2*R2+1]*  \
                        temporal_features[k][np.newaxis], (2, 3))
        
        d_matrix[spt[:, np.newaxis] + np.arange(-n_explore,n_explore+1), 
                           k] = 2*dot - norms[k][np.newaxis, np.newaxis]
        
        ctr_times+=1
    
	
	#Skip this chunk of time entirely
    if ctr_times==0: 
        return None
	    

    #******************************************************************
    #******************************************************************
    #***************** STAGE 4:  THRESHOLD LOOP ***********************
    #******************************************************************
    #******************************************************************
    
    spike_train = np.zeros((0, 2), 'int32')
   
    max_d = np.max(d_matrix, (1,2))
    max_d_array = []
    max_d_array.append(max_d.copy())

    spike_times = []

    max_val = np.max(max_d)
    threshold_ctr=0
    
    while max_val > threshold_d:

        # find spike time and template from objective function
        peaks = argrelmax(max_d)[0]
        
        idx_good = peaks[np.argmax(max_d[peaks[:, np.newaxis] + \
                                             np.arange(-R,R+1)],1) == R]
        
        spike_time = idx_good[max_d[idx_good] > threshold_d]
        spike_times.append(spike_time)
        
        # find the template_id and max_shift for each peak detected
        template_id, max_shift = np.unravel_index(np.argmax(
                                      np.reshape(d_matrix[spike_time],
                                      (spike_time.shape[0], -1)),1),
                                      [n_templates, n_shifts])	

        ## prevent refractory period violation
        rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
        rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
        d_matrix[rf_area, rf_area_t] = -np.Inf

        #****************BOTTLENECK #1************************
        ## update nearby times
        for j in range(spike_time.shape[0]):

            # return times and templates 
            t_neigh, k_neigh = np.where(
                 d_matrix[spike_time[j]-2*R : spike_time[j]+2*R, :, 0] > 
                 -np.Inf)		

            # new; may not be used
            #t_neigh, k_neigh2 = np.where(d_matrix[spike_time[j]-2*R :
            #  spike_time[j]+2*R, overlap_channels[template_id[j]], 0] > 
            # -np.Inf)
            #k_neigh = overlap_channels[template_id[j]][k_neigh2]
            
            #Shift the absolute times back to real-time (in specific block)
            t_neigh_abs = spike_time[j] + t_neigh - 2*R			
            
            d_matrix[t_neigh_abs, k_neigh] -= temp_temp[template_id[j], 
                                      k_neigh, max_shift[j], :, t_neigh]		


        # old: update d_matrix at specific times only
        #time_affected = np.reshape(spike_time[:, np.newaxis] + np.arange(-2*R,2*R+1), -1)
        #time_affected = time_affected[max_d[time_affected] > -np.Inf]
        #max_d[time_affected] = np.max(d_matrix[time_affected], (1,2))
        
        # new: #Cat: seems to run faster than above
        max_d = np.max(d_matrix, (1,2))					

        max_val = np.max(max_d)
        
        spike_train_temp = np.hstack((spike_time[:, np.newaxis],
                          template_id[:, np.newaxis]))
        spike_train = np.concatenate((spike_train, spike_train_temp), 0)         
        
        #Keep track of # of times looped in threshold loop
        threshold_ctr+=1
        
        max_d_array.append(max_d.copy())
        
        # do not erase: pretty print update during threshold loop
        ##sys.stdout.write("...# events removed: %d,   max_val: %f,   
        ## threshold counter: %d   \r" % (spike_time.shape[0], 
        ## max_val, threshold_ctr) )
        ##sys.stdout.flush()

    #Fix indexes explicitly carried out here
    spike_times = spike_train[:, 0]
    # get only observations outside the buffer
    train_not_in_buffer = spike_train[np.logical_and(spike_times>=offset,
						     spike_times <= idx_local_end)]
    # offset spikes depending on the absolute location
    train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + data_start
				 - buffer_size)

    if len(clear_spiketimes_unique)>0: 
        final_spikes = np.concatenate((clear_spiketimes_unique,
                                           train_not_in_buffer),axis=0)
    else:
        final_spikes = train_not_in_buffer

    if verbose: 
        prRed("Chunk: "+str(chunk_idx)), prGreen(" time: "+ 
                                   str(time.time() - start_time_chunk)),
        print("#decon spks: ", len(train_not_in_buffer), 
              " # clear spks: ", len(clear_spiketimes_unique), 
              " Total: ", len(final_spikes), "  # threshold loops: ",
              threshold_ctr)

    return final_spikes


def prRed(prt): print("\033[91m {}\033[00m" .format(prt)),
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt)),
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt)),
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
