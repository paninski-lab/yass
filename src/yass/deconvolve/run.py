import os
import logging
import numpy as np
import parmap
from sklearn import mixture
import scipy

import datetime as dt
from tqdm import tqdm
import torch
import torch.multiprocessing as mp

from yass import read_config
from yass.reader import READER
from yass.deconvolve.match_pursuit import MatchPursuit_objectiveUpsample
from yass.deconvolve.match_pursuit_gpu import deconvGPU, deconvGPU2
from yass.template import shift_chans, align_get_shifts_with_ref
from yass.util import absolute_path_to_asset
from scipy import interpolate
from yass.deconvolve.util import make_CONFIG2
from yass.neuralnetwork import Denoise
from yass import mfm
from yass.merge.merge import (test_merge, run_ldatest, run_diptest)
from yass.cluster.cluster import knn_triage
from yass.residual.residual_gpu import RESIDUAL_DRIFT

from yass.visual.util import binary_reader_waveforms

#from yass.deconvolve.soft_assignment import get_soft_assignments

def run(fname_templates_in,
        output_directory,
        recordings_filename,
        recording_dtype,
        threshold=None,
        update_templates=False,
        run_chunk_sec='full',
        save_up_data=True):
            
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
        standardized.bin

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
    CONFIG = make_CONFIG2(CONFIG)

    print("... deconv using GPU device: ", torch.cuda.current_device())
    
    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_templates = os.path.join(
        output_directory, 'templates.npy')
    fname_spike_train = os.path.join(
        output_directory, 'spike_train.npy')
    fname_shifts = os.path.join(
        output_directory, 'shifts.npy')
    fname_scales = os.path.join(
        output_directory, 'scales.npy')
                               
    # Cat: TODO: use Peter's conditional (below) instead of single file check
    # if (os.path.exists(fname_templates) and
        # os.path.exists(fname_spike_train) and
        # os.path.exists(fname_templates_up) and
        # os.path.exists(fname_spike_train_up)):
        # return (fname_templates, fname_spike_train,
                # fname_templates_up, fname_spike_train_up)

    if (os.path.exists(fname_templates) and
        os.path.exists(fname_spike_train) and
        os.path.exists(fname_shifts) and
        os.path.exists(fname_scales)):
        return (fname_templates, fname_spike_train,
                fname_shifts, fname_scales)
    # parameters
    # TODO: read from CONFIG
    if threshold is None:
        threshold = CONFIG.deconvolution.threshold
    elif threshold == 'low_fp':
        threshold = 150

    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    if CONFIG.deconvolution.deconv_gpu:
        n_sec_chunk = CONFIG.resources.n_sec_chunk_gpu
    else:
        n_sec_chunk = CONFIG.resources.n_sec_chunk

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    n_sec_chunk,
                    chunk_sec=chunk_sec)
 
         
    # deconv using GPU
    if CONFIG.deconvolution.deconv_gpu:
        
        deconv_ONgpu2(fname_templates_in,
                      output_directory,
                      reader,
                      threshold,
                      update_templates,
                      CONFIG,
                      run_chunk_sec)
        
    # deconv using CPU
    else:
        deconv_ONcpu(fname_templates_in,
                     output_directory,
                     reader,
                     threshold,
                     save_up_data,
                     fname_spike_train,
                     fname_templates,
                     CONFIG)

    return (fname_templates, fname_spike_train,
            fname_shifts, fname_scales)



def deconv_ONgpu2(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 update_templates,
                 CONFIG,
                 run_chunk_sec):

    # **************** MAKE DECONV OBJECT *****************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)

    #print (kfadfa)
    # Cat: TODO: gpu deconv requires own chunk_len variable
    #root_dir = '/media/cat/1TB/liam/49channels/data1_allset'
    root_dir = CONFIG.data.root_folder
    d_gpu.root_dir = root_dir
    
    # Cat: TODO: read from CONFIG
    d_gpu.max_iter = 1000
    d_gpu.deconv_thresh=threshold

    # Cat: TODO: make sure svd recomputed for higher rank etc.
    d_gpu.svd_flag = True

    # Cat: TODO read from CONFIG file 
    d_gpu.RANK = 10
    d_gpu.vis_chan_thresh = 1.0

    d_gpu.fit_height = False
    d_gpu.max_height_diff = 0.1
    d_gpu.fit_height_ptp = 20

    # debug/printout parameters
    # Cat: TODO: read all from CONFIG
    d_gpu.save_objective = False
    d_gpu.verbose = False
    d_gpu.print_iteration_counter = 1
    
    # Turn on refactoriness
    d_gpu.refractoriness = True
    
    # Stochastic gradient descent option
    # Cat: TODO: move these and other params to CONFIG
    d_gpu.scd = True
    
    if d_gpu.scd==False:
        print (" ICD TURNED OFFF.....")
    else:
        print (" ICD TUREND ON .....")
        
    # Cat: TODO: move to CONFIG; # of times to run scd inside the chunk
    # Cat: TODO: the number of stages need to be a fuction of # of channels; 
    #      around 1 stage per 20-30 channels seems to work; 
    #      but for 100s of chans this many need to be scaled further
    # d_gpu.n_scd_stages = self.CONFIG.recordings.n_channels // 24 
    d_gpu.n_scd_stages = 2

    # Cat: TODO move to CONFIG; # of addition steps each time
    d_gpu.n_scd_iterations = 10
    
    # Cat: TODO: parameters no longer used, to remove;
    #d_gpu.scd_max_iteration = 1000  # maximum iteration number from which to grab spikes
    #                                # smaller means grabbing spikes from earlier (i.e. larger SNR units)
    #d_gpu.scd_n_additions = 3       # number of addition steps to be done for every loop
    
    # this can turn off the superresolution alignemnt as an option
    d_gpu.superres_shift = True
    
    # parameter allows templates to be updated forward (i.e. templates
    #       are updated based on spikes in previous chunk)
    # Cat: TODO read from CONFIG
    d_gpu.update_templates = update_templates
    # min difference allowed (in terms of ptp of templates)
    d_gpu.min_bad_diff = 0.3
    # max difference with the max weights 
    d_gpu.max_good_diff = 3

    if d_gpu.update_templates:
        print ("   templates being updated every ", 
                CONFIG.deconvolution.template_update_time, " sec")
    else:
        print ("   templates NOT being updated ...")

    # update template time chunk; in seconds
    # Cat: TODO: read from CONFIG file
    d_gpu.template_update_time = CONFIG.deconvolution.template_update_time
    
    # set forgetting factor to 5Hz (i.e. 5 spikes per second of chunk)
    # Cat: TODO: read from CONFIG
    d_gpu.nu = 1 * d_gpu.template_update_time 
        
    # time to try and split deconv-based spikes
    d_gpu.neuron_discover_time = CONFIG.deconvolution.neuron_discover_time
    print ("    d_gpu.neuron_discover_time: ", d_gpu.neuron_discover_time)
        
    # add reader
    d_gpu.reader = reader
    
    # enforce broad buffer
    d_gpu.reader.buffer=1000

    # *********************************************************
    # *********************** RUN DECONV **********************
    # *********************************************************
    begin=dt.datetime.now().timestamp()
    if update_templates:
        d_gpu, fname_templates_out = run_deconv_with_templates_update(
                                                d_gpu, 
                                                CONFIG, 
                                                output_directory)
        templates_post_deconv = np.load(fname_templates_out)

    else:
        d_gpu = run_deconv_no_templates_update(d_gpu, CONFIG)
        
        templates_post_deconv = d_gpu.temps.transpose(2, 1, 0)

    # ****************************************************************
    # *********************** GATHER SPIKE TRAINS ********************
    # ****************************************************************
    subtract_time = np.round((dt.datetime.now().timestamp()-begin),4)

    print ("-------------------------------------------")
    total_length_sec = int((d_gpu.reader.end - d_gpu.reader.start)/d_gpu.reader.sampling_rate)
    print ("Total Deconv Speed ", np.round(total_length_sec/(subtract_time),2), " x Realtime")

    # ************* DEBUG MODE *****************
    if d_gpu.save_objective:
        fname_obj_array = os.path.join(d_gpu.out_dir, 'obj_array.npy')
        np.save(fname_obj_array, d_gpu.obj_array)

    # ************** SAVE SPIKES & SHIFTS **********************
    print ("  gathering spike trains and shifts from deconv (todo: parallelize)")
    batch_size = d_gpu.reader.batch_size
    buffer_size = d_gpu.reader.buffer
    temporal_size = (CONFIG.recordings.sampling_rate/1000*
                     CONFIG.recordings.spike_size_ms)
    
    # loop over chunks and add spikes;
    spike_train = [np.zeros((0,2),'int32')]
    shifts = []
    scales = []
    for chunk_id in tqdm(range(reader.n_batches)):
        #fname = os.path.join(d_gpu.seg_dir,str(chunk_id).zfill(5)+'.npz')
        time_index = (chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv
        fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
        data = np.load(fname, allow_pickle=True)

        spike_array = data['spike_array']
        neuron_array = data['neuron_array']
        offset_array = data['offset_array']
        shift_list = data['shift_list']
        scale_list = data['height_list']
        for p in range(len(spike_array)):
            spike_times = spike_array[p].cpu().data.numpy()
            idx_keep = np.logical_and(spike_times >= buffer_size,
                                      spike_times < batch_size+buffer_size)
            idx_keep = np.where(idx_keep)[0]
            temp=np.zeros((len(idx_keep),2), 'int32')
            temp[:,0]=spike_times[idx_keep]+offset_array[p]
            temp[:,1]=neuron_array[p].cpu().data.numpy()[idx_keep]

            # Cat: TODO: is it faster to make list and then array?
            #            or make array on the fly?
            spike_train.extend(temp)
            shifts.append(shift_list[p].cpu().data.numpy()[idx_keep])
            scales.append(scale_list[p].cpu().data.numpy()[idx_keep])

    # Cat; TODO: sepped this up.
    print ("   vstacking spikes (TODO initalize large array and then try to fill it...): ")
    spike_train = np.vstack(spike_train)
    shifts = np.hstack(shifts)
    scales = np.hstack(scales)

    # add half the spike time back in to get to centre of spike
    spike_train[:,0] = spike_train[:,0]+temporal_size//2
    spike_train = d_gpu.ttc.adjust_peak_times_for_residual_computation(spike_train)


    # sort spike train by time
    print ("   ordering spikes: ")
    idx = spike_train[:,0].argsort(0)
    spike_train = spike_train[idx]
    shifts = shifts[idx]
    scales = scales[idx]

    np.save(os.path.join(d_gpu.out_dir,
                         'spike_train_prededuplication.npy'),
            spike_train)

    # remove duplicates
    # Cat: TODO: are there still duplicates in spike trains!?
    print ("  skipping spike deduplication step ")
    if False:
        print ("removing duplicates... (TODO: remove this requirement eventually...)")
        for k in np.unique(spike_train[:,1]):
            idx = np.where(spike_train[:,1]==k)[0]
            _,idx2 = np.unique(spike_train[idx,0], return_index=True)
            idx3 = np.delete(np.arange(idx.shape[0]),idx2)
            if idx3.shape[0]>0:
                print ("unit: ", k, " has duplicates: ", idx3.shape[0])
                spike_train[idx[idx3],0]=-1E6
        
    # quit()
    idx = np.where(spike_train[:,0]==-1E6)[0]
    spike_train = np.delete(spike_train, idx, 0)
    shifts = np.delete(shifts, idx, 0)
    scales = np.delete(scales, idx, 0)

    # save spike train
    print ("  saving spike_train: ", spike_train.shape)
    fname_spike_train = os.path.join(d_gpu.out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)

    # save shifts
    fname_shifts = os.path.join(d_gpu.out_dir, 'shifts.npy')
    np.save(fname_shifts, shifts)

    # save scales
    fname_scales = os.path.join(d_gpu.out_dir, 'scales.npy')
    np.save(fname_scales, scales)
    
    # save templates
    USE_RECON_TEMPLATE = True
    if USE_RECON_TEMPLATE:
        templates_post_deconv = d_gpu.ttc.residual_temps.transpose(0, 2, 1)
    fname_templates = os.path.join(d_gpu.out_dir, 'templates.npy')
    np.save(fname_templates, templates_post_deconv)


def run_deconv_no_templates_update(d_gpu, CONFIG):

    chunk_ids = np.arange(d_gpu.reader.n_batches)
    n_sec_chunk_gpu = CONFIG.resources.n_sec_chunk_gpu

    processes = []
    if len(CONFIG.torch_devices) == 1:
        run_deconv_no_templates_update_parallel(d_gpu,
                                                chunk_ids,
                                                n_sec_chunk_gpu,
                                                CONFIG.torch_devices[0])
    else:
        chunk_ids_split = np.split(chunk_ids,
                               len(CONFIG.torch_devices))
        for ii, device in enumerate(CONFIG.torch_devices):
            p = mp.Process(target=run_deconv_no_templates_update_parallel,
                           args=(d_gpu, chunk_ids_split[ii],
                                 n_sec_chunk_gpu, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    return d_gpu


def run_deconv_no_templates_update_parallel(d_gpu, chunk_ids, n_sec_chunk_gpu, device):

    torch.cuda.set_device(device)
    d_gpu.initialize()

    for chunk_id in chunk_ids:
        time_index = (chunk_id+1)*n_sec_chunk_gpu
        fname = os.path.join(d_gpu.seg_dir, str(time_index).zfill(6)+'.npz')

        if os.path.exists(fname)==False:

            print ("Forward deconv only ", time_index, " sec, ", chunk_id, "/", len(chunk_ids))

            # run deconv
            d_gpu.run(chunk_id)

            # save deconv results
            np.savez(fname,
                     spike_array = d_gpu.spike_array,
                     offset_array = d_gpu.offset_array,
                     neuron_array = d_gpu.neuron_array,
                     shift_list = d_gpu.shift_list,
                     height_list = d_gpu.height_list
                    )

def run_deconv_with_templates_update2(d_gpu):

    n_sec_chunk = d_gpu.reader.n_sec_chunk
    n_chunks_update = int(d_gpu.template_update_time/n_sec_chunk)
    update_chunk = np.hstack((np.arange(0, d_gpu.reader.n_batches,
                             n_chunks_update), d_gpu.reader.n_batches))

    d_gpu.initialize()

    for batch_id in range(len(update_chunk)-1):

        ##################
        ## Forward pass ##
        ##################

        fnames_forward = []
        for chunk_id in range(update_chunk[batch_id], update_chunk[batch_id+1]):

            # output name
            time_index = (chunk_id+1)*n_sec_chunk
            fname = os.path.join(d_gpu.seg_dir,
                                 str(time_index).zfill(6)+'_forward.npz')
            fnames_forward.append(fname)

            if os.path.exists(fname)==False:

                print ("Forward deconv", time_index, " sec, ",
                       chunk_id, "/", d_gpu.reader.n_batches)

                # run deconv
                d_gpu.run(chunk_id)

                # get ptps
                avg_ptps, weights = d_gpu.compute_average_ptps()

                # save deconv results
                np.savez(fname,
                         spike_array = d_gpu.spike_array,
                         offset_array = d_gpu.offset_array,
                         neuron_array = d_gpu.neuron_array,
                         shift_list = d_gpu.shift_list,
                         height_list = d_gpu.height_list,
                         avg_ptps = avg_ptps.cpu(),
                         weights = weights.cpu()
                        )

            else:
                d_gpu.chunk_id = chunk_id

        ######################
        ## update templates ##
        ######################

        fname_templates_updated = os.path.join(
            d_gpu.out_dir,
            'template_updates',
            'templates_{}sec.npy'.format(n_chunks_update*n_sec_chunk*(batch_id+1)))
        update_templates(fnames_forward,
                         d_gpu.fname_templates,
                         fname_templates_updated,
                         update_weight = 50)

        # re initialize with updated templates
        d_gpu.fname_templates = fname_templates_updated
        # make sure that it is at the right chunk location
        d_gpu.chunk_id = chunk_id
        d_gpu.initialize()

        ###################
        ## Backward pass ##
        ###################
        for chunk_id in range(update_chunk[batch_id], update_chunk[batch_id+1]):

            # output name
            time_index = (chunk_id+1)*n_sec_chunk
            fname = os.path.join(d_gpu.seg_dir,
                                 str(time_index).zfill(6)+'.npz')

            if os.path.exists(fname)==False:

                print ("Backward deconv", time_index, " sec, ",
                       chunk_id, "/", d_gpu.reader.n_batches)

                # run deconv
                d_gpu.run(chunk_id)

                # save deconv results
                np.savez(fname,
                         spike_array = d_gpu.spike_array,
                         offset_array = d_gpu.offset_array,
                         neuron_array = d_gpu.neuron_array,
                         shift_list = d_gpu.shift_list,
                         height_list = d_gpu.height_list)

            else:
                d_gpu.chunk_id = chunk_id

    return fname_templates_updated


def update_templates(fnames_forward,
                     fname_templates,
                     fname_templates_updated,
                     update_weight = 30):

    # get all ptps sufficent stats
    avg_ptps_all = [None]*len(fnames_forward)
    weights_all = [None]*len(fnames_forward)

    for ii, fname in enumerate(fnames_forward):
        temp = np.load(fname, allow_pickle=True)
        avg_ptps_all[ii] = temp['avg_ptps']
        weights_all[ii] = temp['weights']

    avg_ptps_all = np.stack(avg_ptps_all)
    weights_all = np.stack(weights_all)
    avg_ptps = np.average(avg_ptps_all, axis=0, weights=weights_all)
    n_spikes = np.sum(weights_all, axis=0)

    # laod templates
    templates = np.load(fname_templates)

    # get template ptp
    temp_ptps = templates.ptp(1)
    temp_ptps[temp_ptps==0] = 0.01

    # do geometric update
    weight_old = np.exp(-n_spikes/update_weight)
    ptps_updated = temp_ptps*weight_old + (1-weight_old)*avg_ptps
    scale = ptps_updated/temp_ptps
    updated_templates = templates*scale[:, None]
    np.save(fname_templates_updated, updated_templates)


def run_deconv_with_templates_update(d_gpu, CONFIG,
                                    output_directory):

    begin=dt.datetime.now().timestamp()

    # loop over chunks and run sutraction step
    #templates_old = None
    wfs_array = []
    n_spikes_array = []    
    ptp_array = []
    ptp_time_array = []

    # this is a place holder; gets returned to main wrapper to save templates
    #           post deconv; 
    #       - it is updated during deconv to contain latest updated templates;
    fname_updated_templates = d_gpu.fname_templates
    
    #***********************************************************
    #********************* MAIN DECONV LOOP ********************
    #***********************************************************
    # main idea: 2 loops; outer checks for backward/updated deconv
    #                     inner does the forward deconv
    chunk_id = 0
    neuron_discovery_flag = CONFIG.deconvolution.neuron_discover 
    new_neuron_len = CONFIG.deconvolution.neuron_discover_time
    batch_len = CONFIG.deconvolution.template_update_time
    chunk_len = CONFIG.resources.n_sec_chunk_gpu
        
    verbose = False
    while True:
        # keep track of chunk being deconved and time_index
        time_index = (chunk_id+1)*chunk_len

        #if d_gpu.update_templates_backwards:
        fname_forward = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'_forward.npz')
        fname_updated = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')

        if verbose:
            print (" searching for finalized chunnk: ", fname_updated)
        if os.path.exists(fname_updated):
            #print ("             found it")
            chunk_id+=1
            continue
        
        # exit when finished reading;
        if chunk_id>=d_gpu.reader.n_batches:
            break
            
        # if updated file missing; check which block we're in
        updated_temp_time = ((chunk_id*chunk_len)//batch_len+1)*batch_len
        previous_temp_time = ((chunk_id*chunk_len)//batch_len)*batch_len

        # print ("")
        # print ("")
        # print ("")
        
        # check if the batch templates have already been updated;
        # if yes, then do backward step; if not finish the forward batch
        fname_updated_templates = os.path.join(output_directory,'template_updates',
                                                'templates_' + str(updated_temp_time)+'sec.npy')
        if verbose:
            print ("searching for updated tempaltes fname: ", fname_updated_templates)
        
            
        # BACKWARD PASS
        if os.path.exists(fname_updated_templates) and (d_gpu.update_templates):
            
            if verbose:
                print ("")
                print ("")
                print ("")
                print (" >>>>>>>>>>>>>>>> BACKWARD PASS <<<<<<<<<<<<<<<< ")
                
            # reinitialize 
            # initialize deconv at the right location;
            # forward pass need last set of updates
            d_gpu.chunk_id = (updated_temp_time)//chunk_len-1
            d_gpu.fname_templates = fname_updated_templates
            d_gpu.initialize()
            
            for k in range(batch_len//chunk_len):
                time_index = (updated_temp_time-batch_len+chunk_len+k*chunk_len)
                fname_forward = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
                #print (" searching for backward/updated deconv file: ", fname_forward)
                
                if os.path.exists(fname_forward):
                    chunk_id+=1
                    continue
                
                # exit when getting to last file
                if chunk_id>=d_gpu.reader.n_batches:
                    break
                
                #if verbose:
                print (" Backward pass time ", time_index)

                
                # run deconv
                #chunk_id = 
                if verbose:
                    print (" chunk_id passed to deconv: ", chunk_id)
                d_gpu.run(chunk_id)
      
                # save deconv results
                fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
                np.savez(fname, 
                         spike_array = d_gpu.spike_array,
                         offset_array = d_gpu.offset_array,
                         neuron_array = d_gpu.neuron_array,
                         shift_list = d_gpu.shift_list,
                         height_list = d_gpu.height_list)

                chunk_id+=1
            
            print ("  DONE BACKWARD PASS: ")
        else:
            
            if verbose:
                print ("")
                print ("")
                print ("")
                print (" >>>>>>>>>>>>>>>> FORWARD PASS <<<<<<<<<<<<<<<< ")
            # initialize deconv at the right location;
            # forward pass need last set of updates
            fname_previous_templates = os.path.join(output_directory,'template_updates',
                                                'templates_' + str(previous_temp_time)+'sec.npy')

            if verbose:
                print (" fname rpev templates ", fname_previous_templates)
            
            if chunk_id == 0:
                d_gpu.chunk_id = (previous_temp_time)//chunk_len
            else:
                d_gpu.chunk_id = (previous_temp_time)//chunk_len-1
                
            d_gpu.fname_templates = fname_previous_templates
            d_gpu.initialize()

            # loop over batch forward steps
            chunks = []
            # loop over chunks in eatch batch; 
            for k in range(batch_len//chunk_len):
                
                # Note this entire wrapper assumes templates are bing updated; no need to check;
                time_index = (updated_temp_time - batch_len + chunk_len + k*chunk_len)
                fname_forward = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'_forward.npz')
                
                if os.path.exists(fname_forward):
                    if verbose:
                        print ("  time index: ", time_index, " already completed (TODO: make sure metadata is there")

                    # exit when getting to last file
                    if chunk_id>=d_gpu.reader.n_batches:
                        break

                    chunks.append(chunk_id)
                    
                    chunk_id+=1
                    continue

                # exit when getting to last file
                if chunk_id>=d_gpu.reader.n_batches:
                    break
                
                chunks.append(chunk_id)
                
                #if verbose:
                print (" Forward pass time ", time_index, ", chunk : ", chunk_id, " / ", d_gpu.reader.n_batches)
               
                # run deconv
                d_gpu.run(chunk_id)
      
                # save deconv results
                fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'_forward.npz')
                np.savez(fname, 
                         spike_array = d_gpu.spike_array,
                         offset_array = d_gpu.offset_array,
                         neuron_array = d_gpu.neuron_array,
                         shift_list = d_gpu.shift_list,
                         height_list = d_gpu.height_list)

                track_spikes_post_deconv(d_gpu, 
                                        CONFIG,
                                        time_index, 
                                        output_directory,
                                        chunk_id
                                        )
                chunk_id+=1

            # after batch is complete, run template update   
            # Cat; TODO: is this flag redundant?  This entire wrapper is for updating templates
            if d_gpu.update_templates:
                templates_new = update_templates_forward_backward(d_gpu,
                                              CONFIG,
                                              chunks,
                                              time_index)

            # check if new neurons are found
            # Cat; TODO: is this flag redundant?  This entire wrapper is for updating templates
            if d_gpu.update_templates:
                if ((time_index%new_neuron_len==0) and (time_index>chunk_len) and
                    (neuron_discovery_flag)):
                    n_temps = templates_new.shape[2]
                    templates_new = split_neurons(templates_new, 
                                                  d_gpu, 
                                                  CONFIG, 
                                                  time_index
                                                  )
                
                    # if finding new neurons backup all the way to beginning of 
                    #        the new neuron_batch (note we also step back below)
                    if n_temps!=templates_new.shape[2]:
                        chunk_id -= (new_neuron_len-batch_len)//chunk_len                
                        
                        if verbose:
                            print ("  New neurons found: ", templates_new.shape[2]-n_temps, 
                                    ", reseeting chunk_id to: ", chunk_id)
                
                    neuron_discovery_flag = False
                
            # finalize and reinitialize deconvolution with new templates
            # Cat; TODO: is this flag redundant?  This entire wrapper is for updating templates
            if d_gpu.update_templates:
                finish_templates(templates_new, d_gpu, CONFIG, time_index, 
                                 chunk_id, updated_temp_time)
        
            # reset the chunk ID back to initialize the updated/backward template pass
            # Cat; TODO: is this flag redundant?  This entire wrapper is for updating templates
            if d_gpu.update_templates:
                print ("time index: ", time_index)
                if (time_index-chunk_len)<= CONFIG.deconvolution.template_update_time:
                    chunk_id= 0
                else:
                    chunk_id-= (batch_len//chunk_len)

                if verbose:
                    print ("  tempaltes updated, resetting chunk_id to: ", chunk_id)
            
        print (" chunk_id: ", chunk_id)
        print (" d_gpu.reader.n_batche: ", d_gpu.reader.n_batches)
        print (" time_index: ", time_index)
                        
        # exit when finished reading;
        if chunk_id>=d_gpu.reader.n_batches:
            break

    return d_gpu, fname_updated_templates
    
    
def compute_residual_drift(d_gpu):

    RESIDUAL_DRIFT(d_gpu)
    
        
    
def track_spikes_post_deconv(d_gpu, 
                             CONFIG, 
                             time_index, 
                             output_directory,
                             chunk_id):

    wfs_array = []
    n_spikes_array = []
    ptp_array = []
    ptp_time_array = []

    # load deconvolution shifts, ids, spike times
    if len(d_gpu.neuron_array)>0: 
        ids = torch.cat(d_gpu.neuron_array)
        spike_array = torch.cat(d_gpu.spike_array)
        shifts_array = torch.cat(d_gpu.shift_list)
    else:
        ids = torch.empty(0,dtype=torch.double)
        spike_array = torch.empty(0,dtype=torch.double)
        shifts_array = torch.empty(0,dtype=torch.double).cpu().data.numpy()

    units = np.arange(d_gpu.temps.shape[2])

    # Cat: TODO: move these values to CONFIG;
    save_flag = False
    verbose = False
    super_res_align= True
    nn_denoise = False
    debug = True
    
    # arrays for tracking ptp data for split step
    # Cat: TODO move this to CONFIG files
    resplit = True
    ptp_all_array = []
    wfs_all_array = []
    
    # DEBUG arrays:
    raw_wfs_array = []
    idx_ptp_array = []
    wfs_temp_original_array = []
    wfs_temp_aligned_array = []
    
    # Cat: TODO remove this try: code into wrapper code
    try:
        os.mkdir(d_gpu.out_dir + '/wfs/')
    except:
        pass
    try:
        os.mkdir(d_gpu.out_dir + '/resplit/')
    except:
        pass
    


    # *********************************************************
    # **************** COMPUTE RESIDUALS **********************
    # *********************************************************

    # print ("Calling residual computation during drift: ") 
    # res = compute_residual_drift(d_gpu)






    # *********************************************************
    # **************** SAVE SPIKE SHAPES **********************
    # *********************************************************

    spike_train_array = []

    max_percent_update = d_gpu.max_percent_update

    # set to large value when not debugging    
    unit_test = 53400  

    # GPU version of weighted computation
    d_gpu.temps_cuda = torch.from_numpy(d_gpu.temps).cuda()  #torch(d_gpu.temps).cuda
    data = d_gpu.data
    snipit = torch.arange(0,d_gpu.temps.shape[1],1).cuda() 
    wfs_empty = np.zeros((d_gpu.temps.shape[1],d_gpu.temps.shape[0]),'float32')
    
    # Cat: TODO read from config; 
    multi_chan_rank1 = False
    
    # minimum amount in SU units that spike can be different at peak/trough before being
    #   triaged/rejected for template update
    max_diff_peaks = 1.0
    
    # minimum amount in SU units that spike ptp can be different at fixed peak/trough before
    #   being rejected
    max_diff_ptp = 3.0 
    
    #print ("... processing template update...")
    #for unit in tqdm(units):
    for unit in units:
        # 
        idx = torch.where(ids==unit, ids*0+1, ids*0)
        idx = torch.nonzero(idx)[:,0]

        times = spike_array[idx]
        shifts = shifts_array[idx]
        
        # get indexes of spikes that are not too close to end; 
        #     note: spiketimes are relative to current chunk; i.e. start at 0
        idx2 = torch.where(times<(data.shape[1]-d_gpu.temps.shape[1]), times*0+1, times*0)
        idx2 = torch.nonzero(idx2)[:,0]
        times = times[idx2]
        shifts = shifts[idx2].cpu().data.numpy()

        # save spiketrains for resplit step
        spike_train_array.append(times.cpu().data.numpy())
        
        # grab waveforms; 
        # Cat: TODO: decide if median or mean need to be used here;
        if idx2.shape[0]>0:
            #wfs = torch.median(data[:,times[idx2][:,None]+
            #                        snipit-d_gpu.temps.shape[1]+1].
            #                        transpose(0,1).transpose(1,2),0)[0]
            wfs_temp_original = data[:,times[:,None]+
                                    snipit-d_gpu.temps.shape[1]+1]. \
                                    transpose(0,1).transpose(1,2)[:,None]
            
            # select max channel spikes only from 4D tensor
            if multi_chan_rank1:
                # keep all channels
                wf_torch = wfs_temp_original[:,0,:,:]
            else:
                wf_torch = wfs_temp_original[:,0,:,d_gpu.max_chans[unit]]

            
            # *********************************************************
            # **************** NN DENOISE STEP ************************
            # *********************************************************
            # not used at this time;
            # convert data to cpu 
            denoised_wfs = wf_torch.cpu().data.numpy()

            # save data for post-run debugging; 
            # Cat: TODO this is a large file, can eventually erase it;
            # saving only max channel data -not whole file
            #print ("denoised_wfs: ", denoised_wfs.shape)
            if multi_chan_rank1:
                wfs_temp_original_array.append(denoised_wfs[:,:,d_gpu.max_chans[unit]])
            else:
                wfs_temp_original_array.append(denoised_wfs)
            
            if multi_chan_rank1:
                template_original_denoised = d_gpu.temps_cuda[:,:,unit].cpu().data.numpy()
            else:
                template_original_denoised = d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy()
                

            # *********************************************************
            # ************ USE DECONV SHIFTS TO ALIGN SPIKES **********
            # *********************************************************
            # work with denoised waveforms
            if super_res_align: 
                if shifts.shape[0]==1:
                    shifts = shifts[:,None]
                wfs_temp_aligned = shift_chans(denoised_wfs, -shifts)

                #print ("wfs_temp_aligned: ", wfs_temp_aligned.shape)
                wfs_temp_aligned_array.append(wfs_temp_aligned)
            else:
                wfs_temp_aligned = denoised_wfs
                wfs_temp_aligned_array.append(wfs_temp_aligned)

            
            # *********************************************************
            # ************ COMPUTE THRESHOLDS FOR TRIAGE **************
            # *********************************************************
            # Grab ptps from wfs using previously computed ptp_max and ptp_min values in ptp_locs
            # exclude ptps that are > or < than 20% than the current template ptp
            
            # STEP 1: compute the boundaries on thresholds
            # select +/- 10% of waveform or +/- 1SU whichever is larger
            if multi_chan_rank1:
                template_at_peak = d_gpu.max_temp_array[unit] 
                template_at_trough = d_gpu.min_temp_array[unit] 
                # print ("template_at_peak: ", template_at_peak.shape)
                # print ("template_at_peak: ", template_at_peak)

                # define the max_threshold based on peak locations
                max_thresh_dynamic = [] # np.zeros(d_gpu.temps.shape[2])
                min_thresh_dynamic = [] # np.zeros(d_gpu.temps.shape[2])
                for c in range(d_gpu.temps.shape[0]):
                    max_thresh_dynamic.append([max(template_at_peak[c]*(1+max_percent_update),
                                                 template_at_peak[c]+max_diff_peaks),
                                             min(template_at_peak[c]*(1-max_percent_update),
                                                 template_at_peak[c]-max_diff_peaks)
                                            ])

                    min_thresh_dynamic.append([min(template_at_trough[c]*(1+max_percent_update),
                                              template_at_trough[c]+max_diff_peaks),
                                             max(template_at_trough[c]*(1-max_percent_update),
                                               template_at_trough[c]-max_diff_peaks)
                                            ])
                    
                # 
                max_thresh_dynamic = np.vstack(max_thresh_dynamic)
                min_thresh_dynamic = np.vstack(min_thresh_dynamic)
                
                #                 
                
                print ("template_original_denoised: ", template_original_denoised.shape)
                
                # this will give ptp on each of the channels
                ptp_template_original_denoised = template_original_denoised.ptp(1)
            
                # select +/- 10% of waveform or +/- 3SU whichever is larger
                ptp_thresh_dynamic = []
                for c in range(d_gpu.temps.shape[0]):
                    ptp_thresh_dynamic.append([max(ptp_template_original_denoised[c]*(1+max_percent_update),
                                                  ptp_template_original_denoised[c]+max_diff_ptp),
                                              min(ptp_template_original_denoised[c]*(1-max_percent_update),
                                                  ptp_template_original_denoised[c]-max_diff_ptp)
                                             ])
                ptp_thresh_dynamic = np.array(ptp_thresh_dynamic)
                print ("ptp_thresh_dynamic: ", ptp_thresh_dynamic)
                
            else:
                template_at_peak = template_original_denoised[d_gpu.ptp_locs[unit][0]]
                template_at_trough = template_original_denoised[d_gpu.ptp_locs[unit][1]]
                
                max_thresh_dynamic = [max(template_at_peak*(1+max_percent_update),template_at_peak+max_diff_peaks),
                                      min(template_at_peak*(1-max_percent_update),template_at_peak-max_diff_peaks)
                                     ]
                                            
                min_thresh_dynamic = [min(template_at_trough*(1+max_percent_update),template_at_trough+max_diff_peaks),
                                      max(template_at_trough*(1-max_percent_update),template_at_trough-max_diff_peaks)
                                     ]
                
                # threshold also using ptp of template
                # note this makes either a single channel template or multi-chan template;
                ptp_template_original_denoised = template_original_denoised.ptp(0)
            
                # select +/- 10% of waveform or +/- 3SU whichever is larger
                ptp_thresh_dynamic = [max(ptp_template_original_denoised*(1+max_percent_update),
                                         ptp_template_original_denoised+max_diff_ptp),
                                      min(ptp_template_original_denoised*(1-max_percent_update),
                                         ptp_template_original_denoised-max_diff_ptp)
                                      ]
                                     

            # *************************************************************
            # **** OPTION #2: Maxes/Mins at fixed poitns + PTP LIMITS *****
            # *************************************************************
            # search the immediate vicinity of the peaks: -1..+1 (not just exact peak)
            # this makes it more noisy - but helps avoid alignment/denoising steps
            # if False:
                # maxes = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]-1:d_gpu.ptp_locs[unit][0]+2].max(1)
                # mins = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]-1:d_gpu.ptp_locs[unit][1]+2].min(1)
            
            # search just specific peak
            
            if multi_chan_rank1:
                maxes = []
                mins = []
                print ("d_gpu.max_temp_array: ", d_gpu.max_temp_array.shape)
                for c in range(d_gpu.temps.shape[0]):
                    maxes.append(wfs_temp_aligned[:,c,d_gpu.max_temp_array[unit,c]])
                    mins.append(wfs_temp_aligned[:,c,d_gpu.min_temp_array[unit,c]])
            else:
                maxes = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]]
                mins = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]]
            
            #print ("maxes: ", maxes)
            
            # save ptps for all spike waveforms at selected time points                    
            ptp_all = (maxes-mins)
            ptp_all_array.append(ptp_all)

            # also compute ptps blindly over all waveforms at all locations
            #       - this could be improved by limiting to a window between through and peak...
            ptps_dumb = wfs_temp_aligned.ptp(1)


            # *************************************************************
            # ************** FIND POINTS WITHIN THRESHOLDS ****************
            # *************************************************************

            # old method that uses relative threshold only:
            idx_maxes = np.where(np.logical_and(maxes>=max_thresh_dynamic[1],
                                                maxes<=max_thresh_dynamic[0])
                                )[0]
                                                
            idx_mins = np.where(np.logical_and(mins<=min_thresh_dynamic[1],
                                               mins>=min_thresh_dynamic[0])
                                )[0]

            idx_ptp_dumb = np.where(np.logical_and(ptps_dumb>=ptp_thresh_dynamic[1],
                                                   ptps_dumb<=ptp_thresh_dynamic[0])
                                )[0]     

            # find intersection of ptp_maxes and ptp_mins
            idx_max_min, _, _ = np.intersect1d(idx_maxes,idx_mins,return_indices=True)
            
            # find intersection with ptp_dumb
            idx_ptp, idx_max_min_ptp, _ = np.intersect1d(idx_max_min,
                                            idx_ptp_dumb,return_indices=True)
            
            # index into the original ptp array;
            idx_ptp_final = idx_max_min[idx_max_min_ptp]
            
            # save indexes where all 3 conditions are met: max, min and ptp fall within boudnaries
            idx_ptp_array.append(idx_ptp_final)
            
        
            # DEBUG PRINTOUT FOR DRIFT MODEL for a particular unit;
            #  do not remove
            if unit==unit_test:

                print ("UNIT: ", unit)
                print ("template at peak: ", template_at_peak)
                print ("template at trough: ", template_at_trough)
                print ("max threshold dynamic: ", max_thresh_dynamic)
                print ("min threshold dynamic: ", min_thresh_dynamic)
                print ("ptp threshold dynamic: ", ptp_thresh_dynamic)
                
                
                #print ("template_original_denoised: ", template_original_denoised)
                print ("template_original_denoised[d_gpu.ptp_locs[unit][0]]: ", template_original_denoised[d_gpu.ptp_locs[unit][0]])
                print ("template_original_denoised[d_gpu.ptp_locs[unit][1]]: ", template_original_denoised[d_gpu.ptp_locs[unit][1]])
                print ("idx_maxes: ", idx_maxes[:10])
                print ("idx_mins: ", idx_mins[:10])
                
                print ("maxes: ", maxes[:10])
                print ("maxes average: ", maxes.mean(0))
                print ("mins: ", mins[:10])
                print ("mins average: ", mins.mean(0))

                print ("maxes[idx_ptp]: ", maxes[idx_ptp_final][:10])
                print ("maxes average[idx_ptp]: ", maxes[idx_ptp_final].mean(0))
                print ("mins:[idx_ptp] ", mins[idx_ptp_final][:10])
                print ("mins average:[idx_ptp] ", mins[idx_ptp_final].mean(0))
                
                print ("idx_ptp_max_min: ", idx_max_min[:10])
                print ("idx_ptp_dumb: ", idx_ptp_dumb[:10])
                print ("idx_ptp_final: ", idx_ptp_final[:10])
                
                # this uses ptps of the raw waveforms
                ptp_temp1 = (maxes-mins)
                ptp_temp1 = ptp_temp1[idx_ptp_final].mean(0)

                print ("ptp value: ", ptp_temp1)
                print ("original Template ptp without denoise: ",
                        d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy().ptp(0))
                        
                print ("ptp[triaged spikes]: ", ptp_temp1)
                print ("all spikes ptp(mean): ", wf_torch.cpu().data.numpy().mean(0).ptp(0))
                print ("all spikes mean(ptp): ", wf_torch.cpu().data.numpy().ptp(1).mean(0))

                print ("")
                print ("")
                print ("")


            # if at least 1 spike survived
            if idx_ptp.shape[0]>0:

                # compute mean ptp at non-triaged events
                ptp = ptp_all[idx_ptp_final].mean(0)

                # Cat: TODO: THIS IS NOT CORRECT FOR THE FULL TEMPLATE MODEL!!!
                wfs = np.mean(wfs_temp_aligned[idx_ptp_final],0)[0]
                wfs_array.append(wfs)
                
                n_spikes_array.append(idx_ptp_final.shape[0])
                ptp_array.append(ptp)
                
                # save this as metadata; not really required
                ptp_time_array.append(times[idx_ptp_final].cpu().data.numpy())

            else:
                # default save zeros
                wfs_array.append(d_gpu.temps[:,:,unit].transpose(1,0))
                                        #d_gpu.temps.shape[1], d_gpu.temps.shape[0])
                n_spikes_array.append(0)
                ptp_array.append(0)
                ptp_time_array.append(0)
        
        
        # THERE ARE NO SPIKES IN TIME CHUNK for unit
        else:
            # default save zeros
            wfs_array.append(d_gpu.temps[:,:,unit].transpose(1,0))
                                    #d_gpu.temps.shape[1], d_gpu.temps.shape[0])
            n_spikes_array.append(0)
            ptp_array.append(0)
            ptp_time_array.append(0)
            
            wfs_temp_original_array.append([])
            idx_ptp_array.append([])
            
            # save meta data for split information below
            wfs_temp_aligned_array.append([])
            ptp_all_array.append([])

    # *************************************
    # ***** POST PROCESSING SAVES *********
    # *************************************
    np.savez(d_gpu.out_dir + '/template_updates/chunk_data_'+
                    str((chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu)+'.npz',
            wfs_array=wfs_array,
            n_spikes_array=n_spikes_array,
            ptp_array=ptp_array,
            ptp_time_array=ptp_time_array
            )
    
    if debug:
        np.savez(d_gpu.out_dir + '/wfs/'+
                str((chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu)+'.npz',
                wfs_temp_original_array = wfs_temp_original_array,
                idx_ptp_array = idx_ptp_array
                )
        
    # save meta information to do split;
    if resplit:
        np.savez(d_gpu.out_dir + '/resplit/'+
                str((chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu)+'.npz',
                ptp_all_array = ptp_all_array,
                raw_wfs_array_aligned = wfs_temp_aligned_array,
                spike_train_array = spike_train_array
                )
    
    # return if in middle of forward pass
    # Cat: TODO read length of time to update from CONFIG
    #if (((time_index%d_gpu.template_update_time)!=0) or (time_index==0)):
        
    # Cat: TODO not sure this is necessary/needed
    del d_gpu.temps_cuda
    torch.cuda.empty_cache()


def update_templates_forward_backward(d_gpu, CONFIG, chunks, time_index):
    
    print ("    UPDATING TEMPLATES   <<<<<<<<<<<<<")
    # find max chans of templates
    max_chans = d_gpu.temps.ptp(1).argmax(0)

    # make new templates
    templates_new = np.zeros(d_gpu.temps.shape,'float32')
    units = np.arange(d_gpu.temps.shape[2])
    
    verbose = False
    
    n_batches_per_chunk = d_gpu.template_update_time//CONFIG.resources.n_sec_chunk_gpu

    wfs_local_array = []
    n_spikes_local_array = []
    ptp_local_array = []
    for k in range(len(chunks)):
        fname = (d_gpu.out_dir + '/template_updates/chunk_data_'+
                       str((chunks[k]+1)*CONFIG.resources.n_sec_chunk_gpu)+'.npz')
        
        if verbose:
            print ("Loadin gchunks: ", fname)
        data = np.load(fname, allow_pickle=True)
        wfs_local_array.append(data['wfs_array'])
        n_spikes_local_array.append(data['n_spikes_array'])
        ptp_local_array.append(data['ptp_array'])
             
    # Cat: TODO: This is in CONFIG file already
    ptp_flag = True
    for unit in units:
        ptp_local = []
        wfs_local = []
        n_spikes_local = []
        for c in range(len(chunks)):
            #wfs_local[c] = wfs_array[batch_offset+c][unit]
            #n_spikes_local[c] = n_spikes_array[batch_offset+c][unit]
            #ptp_local.append(ptp_array[batch_offset+c][unit])
            wfs_local.append(wfs_local_array[c][unit])
            n_spikes_local.append(n_spikes_local_array[c][unit])
            ptp_local.append(ptp_local_array[c][unit])
            
        n_spikes = np.hstack(n_spikes_local).sum(0)
        
        # if there are no spikes at all matched, just use previous template 
        if n_spikes==0:
            templates_new[:,:,unit]=d_gpu.temps[:,:,unit]
            continue

        
        # *******************************************************
        # **************** COMPUTE DRIFT MODEL SCALING **********
        # *******************************************************
        # DRIFT MODEL 0; PARTIAL template update using scaling of neurons
        if ptp_flag:
            ptp_temp = np.average(np.float32(ptp_local), weights=np.int32(n_spikes_local), axis=0)
            scale = ptp_temp/d_gpu.temps[:,:,unit].ptp(1).max(0)
        
        # DRIFT MODEL 1; FULL template update using raw spikes 
        else:
            # compute weighted average of the template
            template = np.average(np.float32(wfs_local), weights=np.int32(n_spikes_local), axis=0).T
        
        
        # *******************************************************
        # **************** UPDATE TEMPLATE **********************
        # *******************************************************
        # # first chunk of data; just scale starting template/or keep original     
        if time_index==d_gpu.template_update_time:
            if ptp_flag:
                templates_new[:,:,unit]=d_gpu.temps[:,:,unit]*scale
            else:
                print ("   NOTE: First chunk of time FULL TEMPLATE UPDATE"+
                        " (<<<<< NOT CORRECTLY UPDATED >>>>)")
                templates_new[:,:,unit]=template

        else:
            # # use KS eq (6)
            # else:
            if ptp_flag:
                #templates_new[:,:,unit] = d_gpu.temps[:,:,unit]*scale
                # exponential updates
                t1 = d_gpu.temps[:,:,unit]*np.exp(-n_spikes/d_gpu.nu)
                t2 = (1-np.exp(-n_spikes/d_gpu.nu))*d_gpu.temps[:,:,unit]*scale
                templates_new[:,:,unit] = (t1+t2)
                
            else:
                #print ("temps: ", d_gpu.temps.shape, "Unit: ", unit, ", scaling factors: ", 
                #    np.exp(-n_spikes/d_gpu.nu), (1-np.exp(-n_spikes/d_gpu.nu)))
                t1 = d_gpu.temps[:,:,unit]*np.exp(-n_spikes/d_gpu.nu)
                t2 = (1-np.exp(-n_spikes/d_gpu.nu))*template
                templates_new[:,:,unit] = (t1+t2)
        
        # if unit==unit_test:
            # print ("  FINAL SCALE: ", scale)
            # print ("  templates_new max ptp: ", templates_new[:,:,unit].ptp(1).max(0))
            # print ("  ptp_local: ", ptp_local)
    if verbose:

        if time_index==d_gpu.template_update_time:
            print ("  FIRST DECONV STEP... updating existing template forward only")
        else:
            print ("  SECONDARY DECONV STEPs... updating existing template forward and backward")
        
        
    return templates_new

def split_neurons(templates_new, d_gpu, CONFIG, time_index):

    print ("   CHECKING FOR NEW NEURONS (in development...)")
    #         
    standardized_filename = os.path.join(os.path.join(os.path.join(d_gpu.root_dir, 'tmp'),
                                                            'preprocess'), 
                                                            'standardized.bin')

    units = np.arange(d_gpu.temps.shape[2])

    # load relevant data; verify how many steps of data to load for split
    n_steps_back = CONFIG.deconvolution.neuron_discover_time//CONFIG.resources.n_sec_chunk_gpu
    print ("   # of steps backwards: ", n_steps_back)

    ptps = []
    raw_wfs = []
    spike_train = []
    for unit in units:
        ptps.append([])
        raw_wfs.append([])
        spike_train.append([])
        
    # Cat: TODO: this is a bit hacky; should speed it up.
    sample_rate = CONFIG.recordings.sampling_rate
    for k in range(time_index-(n_steps_back-1)*CONFIG.resources.n_sec_chunk_gpu, 
                   time_index+1, CONFIG.resources.n_sec_chunk_gpu):
        fname_resplit = d_gpu.out_dir + '/resplit/'+str(k)+'.npz'
        print ("FNAMe resplit: ", fname_resplit)
        data = np.load(fname_resplit)
        ptps_temp = data['ptp_all_array']
        raw_temp = data['raw_wfs_array_aligned']
        spikes = data['spike_train_array']
        for unit in units:
            ptps[unit].append(ptps_temp[unit])
            raw_wfs[unit].append(raw_temp[unit])
            spikes_temp = (spikes[unit]
                           + (k-CONFIG.resources.n_sec_chunk_gpu)*sample_rate
                           -d_gpu.reader.buffer
                           -d_gpu.temps.shape[1]
                           )
            spike_train[unit].append(spikes_temp)

    # loop over units and find splits:
    print (".... Searching for new neurons...")
    if CONFIG.resources.multi_processing:
        #batches_in = np.array_split(units, CONFIG.resources.n_processors)
        new_templates = parmap.map(new_neuron_search, units, ptps, CONFIG, spike_train,
                         standardized_filename, 
                         d_gpu.temps,
                         d_gpu.out_dir,
                         processes=CONFIG.resources.n_processors,
                         pm_pbar=True)
    
    else:
        new_templates = []
        for unit in units:
            temp = new_neuron_search(unit, ptps, CONFIG, spike_train,
                         standardized_filename, 
                         d_gpu.temps,
                         d_gpu.out_dir)
            
            #if temp is not None: 
             #   print ("new_temp: ", temp.shape)
            new_templates.append(temp)
            
    print ("   TODO: delete all intermediate resplit files saved...")
    # append the new tempaltes to 
    print ("  STARTING TEMPLATES: ", templates_new.shape)
    for k in range(len(new_templates)):
        if new_templates[k] is not None:
            templates_new = np.concatenate((templates_new, 
                              new_templates[k][:,:,None].transpose(1,0,2)),axis=2)
    print ("  FINAL TEMPLATES: ", templates_new.shape)
    
    return templates_new

def finish_templates(templates_new, d_gpu, CONFIG, time_index, chunk_id, 
                     updated_temp_time):
    
    verbose = False
    if verbose:
        print ("")
        print ("")
        print ("   FINISHING TEMPLATES   ")

    # save updated templates
    out_file = os.path.join(d_gpu.out_dir,'template_updates',
                    'templates_'+
                    str(time_index)+
                    'sec.npy')
    
    # also save updated templates for end of file with name as a muitple of the update_template
    # time so that the backward step can find this file;
    # Cat: TODO: this can probably be done better/more elegantly
    time_index_extended = str(updated_temp_time)
    out_file = os.path.join(d_gpu.out_dir,'template_updates',
                    'templates_'+
                    str(time_index_extended)+
                    'sec.npy')                                                
    
    if verbose:
        print (" TEMPS being saved: ", out_file)
    np.save(out_file, templates_new.transpose(2,1,0))
        
    # re-initialize d_gpu
    # change fname-templates
    # Cat: TODO: is this the best way to pass on template?  probably name is fine;
    d_gpu.fname_templates = out_file
    d_gpu.chunk_id = time_index//CONFIG.resources.n_sec_chunk_gpu-1
    d_gpu.initialize()    


    # pass reinitialized object for following pass
    return d_gpu 


def new_neuron_search(unit, ptps, CONFIG, spike_train,
                     standardized_filename, 
                     d_gpu_temps,
                     d_gpu_out_dir):

    features = np.hstack(ptps[unit])

    if features.shape[0]<CONFIG.deconvolution.min_split_spikes:
        return None

    # triage 5% of spikes
    idx_keep = knn_triage(95,features[:,None]) # return boolean
    idx_keep = np.where(idx_keep)[0]
    features_triaged = features[idx_keep]

    # screen distribution for bimodalities using diptest before running MFM
    pval = run_diptest_resplit(features_triaged, assignment=None)
    if pval>0.1:
        return None

    # ************************
    # ******* SPLIT **********
    # ************************
    # we just do 2comp gmm now (rather than MFM + cc)
    assignments = em_test(features_triaged[:,None])
    
    pvals=[]
    for k in np.unique(assignments):
        idx = np.where(assignments==k)[0]
        pvals.append(run_diptest_resplit(features_triaged[idx]))
  
    # if neither split is stable skip unit
    #       this is a bit too conservative; might need to retweak
    # Cat: TODO: export to CONFIG
    pval_thresh = 0.95
    if max(pvals)<pval_thresh:
        return None
    
    # if unit survived diptest; need to load raw waveforms to compute 
    temp_spikes = np.hstack(spike_train[unit])[idx_keep]
    temp_ptps = np.hstack(ptps[unit])[idx_keep]
    wfs, skipped_idx = binary_reader_waveforms(standardized_filename, 
                                  CONFIG.recordings.n_channels, 
                                  d_gpu_temps.shape[1],
                                  temp_spikes)
    
    #np.save('/home/cat/wfs.npy', wfs)
    #np.save('/home/cat/spike_train.npy', spike_train)
    #print ("idx_keep: ", .shape)            
    #print ("WFS: ", wfs.shape)

    # if reader misses spikes; delete them from assignments also
    if len(skipped_idx)>0:
        assignments = np.delete(assignments, skipped_idx)
        temp_ptps = np.delete(temp_ptps, skipped_idx)
        
    # generate new templates for particular unit
    wfs_resplit = []
    new_templates = []
    temps = []
    # loop over the 2 assignments
    for k in np.unique(assignments):
        idx = np.where(assignments==k)[0]
        temp = wfs[idx]
        temp = temp.mean(0)
        temps.append(temp)

    # compute cosine-similarty check to see which neuron already is present in recording;
    #       and which is not
    res = check_matches(d_gpu_temps, temps)

    match_vals1, match_vals2 = res[0], res[1]
    match1 = match_vals1.max(0)
    match2 = match_vals2.max(0)
    
    fname_newneuron = d_gpu_out_dir + '/resplit/new_'+str(unit)+'.npz'
    if match1<=match2:
        match_new = 0
        match_old = 1
        
        np.savez(fname_newneuron, 
                new_neuron = temps[match_new],
                old_neuron = temps[match_old]
                )

        return temps[match_new] 
        
    else:
        match_new = 1
        match_old = 0

        np.savez(fname_newneuron, 
                new_neuron = temps[match_new],
                old_neuron = temps[match_old]
                )

        return temps[match_new] 
            

def check_matches(templates, temp):
    
    templates_local = templates.transpose(1,0,2)
    #temp = temp.transpose(0,1)
    res = []
    
    for p in range(2):
        match_vals = []
        units = np.arange(templates_local.shape[2])
        data1 = temp[p].T
        for unit in units:
            data2 = templates_local[:,:,unit].T.ravel()
            best_result = 0
            for k in range(-10,10,1):
                data_test = np.roll(data1,k,axis=1).ravel()
                result = 1 - scipy.spatial.distance.cosine(data_test,data2)
                if result>best_result:
                    best_result = result

            match_vals.append(best_result)
        res.append(np.hstack(match_vals))
    
    return (res)
    

def em_test(features):
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(features)
    
    return gmm.predict(features)
    

def mfm_resplit(features, CONFIG):

    mask = np.ones((features.shape[0], 1))
    group = np.arange(features.shape[0])
    vbParam = mfm.spikesort(features[:,:,None],
                            mask,
                            group,
                            CONFIG)
    return vbParam

def run_diptest_resplit(features, assignment=None):

    from diptest import diptest as dp

    if assignment is not None:
        lda = LDA(n_components = 1)
        lda_feat = lda.fit_transform(features, assignment).ravel()
        pval = dp(lda_feat)[1]
    else:
        pval = dp(features)[1]

    return pval
    
    
    
def template_calculation_parallel(unit, ids, spike_array, temps, data, snipit):

    idx = np.where(ids==unit)[0]
    times = spike_array[idx]
    
    # get indexes of spikes that are not too close to end; 
    #       note: spiketimes are relative to current chunk; i.e. start at 0
    idx2 = np.where(times<(data.shape[1]-temps.shape[1]))[0]

    # grab waveforms; 
    if idx2.shape[0]>0:
        wfs = np.median(data[:,times[idx2][:,None]+
                                snipit-temps.shape[1]+1].
                                transpose(1,2,0),0)
        return (wfs, idx2.shape[0])
    else:
        return (wfs_empty, 0)


def update_templates_GPU(d_gpu, 
                         CONFIG, 
                         time_index, 
                         wfs_array, 
                         n_spikes_array,
                         output_directory):

    # ******************************************************
    # ***************** SAVE WFS FIRST *********************
    # ******************************************************

    # make new entry in array
    wfs_array.append([])
    
    # is this array redundant?
    n_spikes_array.append([])
    
    iter_ = len(wfs_array)-1
    # raw data:
    
    # original templates needed for array generation
    #templates_old = d_gpu.temps
    ids = torch.cat(d_gpu.neuron_array)
    units = np.arange(d_gpu.temps.shape[2])

    # GPU version + weighted computation
    data = d_gpu.data        
    spike_array = torch.cat(d_gpu.spike_array)
    snipit = torch.arange(0,d_gpu.temps.shape[1],1).cuda() #torch.from_numpy(coefficients).cuda()
    wfs_empty = np.zeros((d_gpu.temps.shape[1],d_gpu.temps.shape[0]),'float32')
    for unit in units:
        # 
        idx = torch.where(ids==unit, ids*0+1, ids*0)
        idx = torch.nonzero(idx)[:,0]
        #print ("spike array: ", spike_array.shape)
        times = spike_array[idx]
        
        # exclude buffer; technically should exclude both start and ends
        #       note: spiketimes are relative to current chunk; i.e. start at 0
        idx2 = torch.where(times < data.shape[1], times*0+1, times*0)
        idx2 = torch.nonzero(idx2)[:,0]
        
        # grab waveforms; need to add a time point to data
        if idx2.shape[0]>0:
            wfs = torch.median(data[:,times[idx2][:,None]+
                                    snipit-d_gpu.temps.shape[1]+1].
                                    transpose(0,1).transpose(1,2),0)[0]

            # Cat: TODO: maybe save only vis chans to save space? 
            wfs_array[iter_].append(wfs.cpu().data.numpy())
            n_spikes_array[iter_].append(idx2.shape[0])
        else:
            wfs_array[iter_].append(wfs_empty)
            n_spikes_array[iter_].append(0)
            
        
    # ******************************************************
    # ********** UPDATE TEMPLATES EVERY 60SEC **************
    # ******************************************************

    # update only every 60 seconds
    #print ("time_index: ", time_index)
    # Cat: TODO read length of time to update from CONFIG
    #if (templates_in is None) or (time_index%60!=0):
    if (((time_index%d_gpu.template_update_time)!=0) or (time_index==0)):
        #and not d_gpu.update_templates_recursive):
        return (d_gpu, wfs_array)
                #d_gpu, templates_old, wfs_array
    # forgetting factor ~ number of spikes
    # Cat: TODO: read from CONFIG file
    nu = 10
    
    # find max chans of templates
    max_chans = d_gpu.temps.ptp(1).argmax(0)

    # make new templates
    templates_new = np.zeros(d_gpu.temps.shape,'float32')
    units = np.arange(d_gpu.temps.shape[2])
    
    #print ("wfs_array going into computation: ", wfs_array[0][0].shape)
    
    # Weighted template computation over chunks of data...
    n_chunks = len(wfs_array)
    wfs_local = np.zeros((n_chunks, d_gpu.temps.shape[1], d_gpu.temps.shape[0]),'float32')
    n_spikes_local = np.zeros((n_chunks), 'int32')
    # Cat: TODO: this code might crash if we don't have enough spikes overall
    #           or even within a single window
    for unit in units:

        # get saved waveforms and number of spikes
        for c in range(len(wfs_array)):
            wfs_local[c] = wfs_array[c][unit]#.cpu().data.numpy()
            n_spikes_local[c] = n_spikes_array[c][unit]
        
        #print ("wfs_local: ", wfs_local.shape)
        #print ("n_spikes_local: ", n_spikes_local.shape)
        
        n_spikes = n_spikes_local.sum(0)

        # if there are no spikes at all matched, just use previous template shape
        if n_spikes==0:
            templates_new[:,:,unit]=template
            continue
            
        template = np.average(wfs_local, weights=n_spikes_local,axis=0).T
        #print ("Wfs local weighted averages: ", wfs_local.shape)
        
        # first chunk of data just use without weight.        
        if time_index==d_gpu.template_update_time:
            templates_new[:,:,unit]=template
        # use KS eq (6)
        else:
            t1 = d_gpu.temps[:,:,unit]*np.exp(-n_spikes/nu)
            t2 = (1-np.exp(-n_spikes/nu))*template
            templates_new[:,:,unit] = (t1+t2)

    # # save template chunk for offline analysis only
    # np.save('/media/cat/1TB/liam/49channels/data1_allset/tmp/block_2/deconv/'+
            # str(time_index)+'.npy', templates_new)

    out_file = os.path.join(output_directory,'template_updates',
                    'templates_'+
                    str((d_gpu.chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu)+
                    'sec.npy')
    # print (" TEMPS DONE: ", templates_in.shape, templates_new.shape)
    np.save(out_file, templates_new.transpose(2,1,0))
    
    # re-initialize d_gpu
    # change fname-templates
    d_gpu.fname_templates = out_file
    d_gpu.initialize()    

    # reset wfs_array to empty
    return (d_gpu, [])
    

def deconv_ONgpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 fname_spike_train,
                 fname_spike_train_up,
                 fname_templates,
                 fname_templates_up,
                 fname_shifts,
                 CONFIG,
                 run_chunk_sec):

    # *********** CONSTRUCT DECONV OBJECT ************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)
    
    #print (kfadfa)
    # Cat: TODO: gpu deconv requires own chunk_len variable
    n_sec=CONFIG.resources.n_sec_chunk_gpu
    #root_dir = '/media/cat/1TB/liam/49channels/data1_allset'
    root_dir = CONFIG.data.root_folder

    # Cat: TODO: read from CONFIG
    d_gpu.max_iter=1000
    d_gpu.deconv_thresh=threshold

    # Cat: TODO: make sure svd recomputed for higher rank etc.
    d_gpu.svd_flag = True

    # Cat: TODO read from CONFIG file 
    d_gpu.RANK = 49
    d_gpu.vis_chan_thresh = 1.0
    d_gpu.superres_shift = True
    
    # debug/printout parameters
    # Cat: TODO: read all from CONFIG
    d_gpu.save_objective = False
    d_gpu.verbose = False
    d_gpu.print_iteration_counter = 50
    d_gpu.save_state = True
    
    # add reader
    d_gpu.reader = reader

    # *********** INIT DECONV ****************
    begin=dt.datetime.now().timestamp()
    d_gpu.initialize()
    setup_time = np.round((dt.datetime.now().timestamp()-begin),4)
    print ("-------------------------------------------")
    print ("Total init time ", setup_time, 'sec')
    print ("-------------------------------------------")
    print ("")

    # ************ RUN DECONV ***************
    print ("Subtraction step...")
    begin=dt.datetime.now().timestamp()
    #if True:
    #    chunks = []
    #    for k in range(0, CONFIG.rec_len//CONFIG.recordings.sampling_rate, 
    #                    CONFIG.resources.n_sec_chunk_gpu_deconv):
    #        chunks.append([k,k+n_sec])
    ## run data on small chunk only
    #else:
    #    chunks = [run_chunk_sec]

    # Cat: TODO : last chunk of data may be skipped if this doesn't work right.
    print ("  (TODO: Make sure last bit is added if rec_len not multiple of n_sec_gpu_chnk)")

    # loop over chunks and run sutraction step
    for chunk_id in tqdm(range(reader.n_batches)):
        fname = os.path.join(d_gpu.seg_dir,str(chunk_id).zfill(5)+'.npz')
        if os.path.exists(fname)==False:
            # rest lists for each segment of time
            d_gpu.offset_array = []
            d_gpu.spike_array = []
            d_gpu.neuron_array = []
            d_gpu.shift_list = []
            
            # run deconv
            d_gpu.run(chunk_id)
            
            # save deconv
            np.savez(fname, 
                     spike_array = d_gpu.spike_array,
                     offset_array = d_gpu.offset_array,
                     neuron_array = d_gpu.neuron_array,
                     shift_list = d_gpu.shift_list)
                            
    subtract_time = np.round((dt.datetime.now().timestamp()-begin),4)

    print ("-------------------------------------------")
    total_length_sec = int((d_gpu.reader.end - d_gpu.reader.start)/d_gpu.reader.sampling_rate)
    print ("Total Deconv Speed ", np.round(total_length_sec/(setup_time+subtract_time),2), " x Realtime")

    # ************* DEBUG MODE *****************
    if d_gpu.save_objective:
        fname_obj_array = os.path.join(d_gpu.out_dir, 'obj_array.npy')
        np.save(fname_obj_array, d_gpu.obj_array)
        

    # ************** SAVE SPIKES & SHIFTS **********************
    print ("  gathering spike trains and shifts from deconv ")
    batch_size = d_gpu.reader.batch_size
    buffer_size = d_gpu.reader.buffer
    temporal_size = CONFIG.spike_size
    
    # loop over chunks and run sutraction step
    spike_train = [np.zeros((0,2),'int32')]
    shifts = []
    for chunk_id in tqdm(range(reader.n_batches)):
        fname = os.path.join(d_gpu.seg_dir,str(chunk_id).zfill(5)+'.npz')
        data = np.load(fname)
        
        spike_array = data['spike_array']
        neuron_array = data['neuron_array']
        offset_array = data['offset_array']
        shift_list = data['shift_list']
        for p in range(len(spike_array)):
            spike_times = spike_array[p].cpu().data.numpy()
            idx_keep = np.logical_and(spike_times >= buffer_size,
                                      spike_times < batch_size+buffer_size)
            idx_keep = np.where(idx_keep)[0]
            temp=np.zeros((len(idx_keep),2), 'int32')
            temp[:,0]=spike_times[idx_keep]+offset_array[p]
            temp[:,1]=neuron_array[p].cpu().data.numpy()[idx_keep]

            spike_train.extend(temp)
            shifts.append(shift_list[p].cpu().data.numpy()[idx_keep])
            
    spike_train = np.vstack(spike_train)
    shifts = np.hstack(shifts)
    # add half the spike time back in to get to centre of spike
    spike_train[:,0] = spike_train[:,0]-temporal_size//2

    # sort spike train by time
    idx = spike_train[:,0].argsort(0)
    spike_train = spike_train[idx]
    shifts = shifts[idx]

    # save spike train
    print ("  saving spike_train: ", spike_train.shape)
    fname_spike_train = os.path.join(d_gpu.out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)
    np.save(fname_spike_train_up, spike_train)

    # save shifts
    fname_shifts = os.path.join(d_gpu.out_dir, 'shifts.npy')
    np.save(fname_shifts,shifts)

    # save templates and upsampled templates
    templates_in_original = np.load(fname_templates_in)
    np.save(fname_templates, templates_in_original)
    np.save(fname_templates_up, templates_in_original)


def deconv_ONcpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 save_up_data,
                 fname_spike_train,
                 fname_spike_train_up,
                 fname_templates,
                 fname_templates_up,
                 CONFIG):

    logger = logging.getLogger(__name__)

    conv_approx_rank = 5
    upsample_max_val = 8
    max_iter = 1000

    mp_object = MatchPursuit_objectiveUpsample(
        fname_templates=fname_templates_in,
        save_dir=output_directory,
        reader=reader,
        max_iter=max_iter,
        upsample=upsample_max_val,
        threshold=threshold,
        conv_approx_rank=conv_approx_rank,
        n_processors=CONFIG.resources.n_processors,
        multi_processing=CONFIG.resources.multi_processing)

    logger.info('Number of Units IN: {}'.format(mp_object.temps.shape[2]))

    # directory to save results for each segment
    seg_dir = os.path.join(output_directory, 'seg')
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # skip files/batches already completed; this allows more even distribution
    # across cores in case of restart
    # Cat: TODO: if cpu is still being used by endusers, may wish to implement
    #       dynamic file assignment here to deal with slow cores etc.
    fnames_out = []
    batch_ids = []
    for batch_id in range(reader.n_batches):
        fname_temp = os.path.join(seg_dir,
                          "seg_{}_deconv.npz".format(
                              str(batch_id).zfill(6)))
        if os.path.exists(fname_temp):
            continue
        fnames_out.append(fname_temp)
        batch_ids.append(batch_id)
    logger.info("running deconvolution on {} batches of {} seconds".format(
        len(batch_ids), CONFIG.resources.n_sec_chunk))

    if len(batch_ids)>0: 
        if CONFIG.resources.multi_processing:
            logger.info("running deconvolution with {} processors".format(
                CONFIG.resources.n_processors))
            batches_in = np.array_split(batch_ids, CONFIG.resources.n_processors)
            fnames_in = np.array_split(fnames_out, CONFIG.resources.n_processors)
            parmap.starmap(mp_object.run,
                           list(zip(batches_in, fnames_in)),
                           processes=CONFIG.resources.n_processors,
                           pm_pbar=True)
        else:
            logger.info("running deconvolution")
            for ctr in range(len(batch_ids)):
                mp_object.run([batch_ids[ctr]], [fnames_out[ctr]])

    # collect result
    res = []
    logger.info("gathering deconvolution results")
    for batch_id in range(reader.n_batches):
        fname_out = os.path.join(seg_dir,
                  "seg_{}_deconv.npz".format(
                      str(batch_id).zfill(6)))
        res.append(np.load(fname_out)['spike_train'])
    res = np.vstack(res)

    logger.info('Number of Spikes deconvolved: {}'.format(res.shape[0]))

    # save templates and upsampled templates
    np.save(fname_templates, np.load(fname_templates_in))
    #np.save(fname_templates,
    #        mp_object.temps.transpose(2,0,1))

    # since deconv spike time is not centered, get shift for centering
    shift = CONFIG.spike_size // 2

    # get spike train and save
    spike_train = np.copy(res)
    # map back to original id
    spike_train[:, 1] = np.int32(spike_train[:, 1]/mp_object.upsample_max_val)
    spike_train[:, 0] += shift
    # save
    np.save(fname_spike_train, spike_train)

    if save_up_data:
        # get upsampled templates and mapping for computing residual
        (templates_up,
         deconv_id_sparse_temp_map) = mp_object.get_sparse_upsampled_templates()

        np.save(fname_templates_up,
                templates_up.transpose(2,0,1))

        # get upsampled spike train
        spike_train_up = np.copy(res)
        spike_train_up[:, 1] = deconv_id_sparse_temp_map[
                    spike_train_up[:, 1]]
        spike_train_up[:, 0] += shift
        np.save(fname_spike_train_up, spike_train_up)

    # Compute soft assignments
    #soft_assignments, assignment_map = get_soft_assignments(
    #        templates=templates.transpose([2, 0, 1]),
    #        templates_upsampled=templates.transpose([2, 0, 1]),
    #        spike_train=spike_train,
    #        spike_train_upsampled=spike_train,
    #        filename_residual=deconv_obj.residual_fname,
    #        n_similar_units=2)

    #np.save(deconv_obj.root_dir + '/soft_assignment.npy', soft_assignments)
    #np.save(deconv_obj.root_dir + '/soft_assignment_map.npy', assignment_map)



def update_templates_CPU(d_gpu, templates_in, CONFIG, ref_template,
                     time_index, wfs_array, n_spikes_array):


    # need to load reference template for super-res alignment OPTION
    # ref template
    ref_template = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'ref_template.npy')))
    
    spike_size = CONFIG.spike_size
    x = np.arange(ref_template.shape[0])
    xnew = np.linspace(0, x.shape[0]-1, num=spike_size, endpoint=True)

    y = ref_template
    tck = interpolate.splrep(x, y, s=0)
    ref_template = interpolate.splev(xnew, tck, der=0)
    
    
    
    # ******************************************************
    # ***************** SAVE WFS FIRST *********************
    # ******************************************************

    # make new entry in array
    wfs_array.append([])
    
    # is this array redundant?
    n_spikes_array.append([])
    
    iter_ = len(wfs_array)-1
    # raw data:
    
    # original templates needed for array generation
    templates_old = d_gpu.temps
    ids = torch.cat(d_gpu.neuron_array)
    units = np.arange(templates_old.shape[2])

    # CPU Version
    if True:
        data = d_gpu.data_cpu        
        # Cat: TODO: save only vis channels data
        for unit in units:
            # cpu version
            idx = np.where(ids.cpu().data.numpy()==unit)[0]
            times = torch.cat(d_gpu.spike_array).cpu().data.numpy()[idx]
            
            # exclude buffer; technically should exclude both start and ends
            idx2 = np.where(times<data.shape[1]-200)[0]
            wfs = (data[:,times[idx2]
                    [:,None]+
                    np.arange(61)-60].transpose(1,2,0))
            
            # Cat: TODO: save only vis chans
            wfs_array[iter_].append(wfs)
            n_spikes_array[iter_].append(idx2.shape[0])

    # GPU version + weighted computation
    else: 
        data = d_gpu.data        
        spike_array = torch.cat(d_gpu.spike_array)
        snipit = torch.arange(0,61,1).cuda() #torch.from_numpy(coefficients).cuda()
        for unit in units:
            # cpu version
            #idx = np.where(ids.cpu().data.numpy()==unit)[0]
            #times = torch.cat(d_gpu.spike_array).cpu().data.numpy()[idx]
                    
            # Third step: only deconvolve spikes where obj_function max > threshold
            idx = torch.where(ids==unit, ids*0+1, ids*0)
            idx = torch.nonzero(idx)[:,0]
            print ("spike array: ", spike_array.shape)
            times = spike_array[idx]
            
            # exclude buffer; technically should exclude both start and ends
            #idx2 = np.where(times<data.shape[1]-200)[0]
            idx2 = torch.where(times<data.shape[1], times*0+1, times*0)
            idx2 = torch.nonzero(idx2)[:,0]
            
            # grab waveforms
            wfs = data[:,times[idx2][:,None]+snipit-60].transpose(0,1).transpose(1,2).mean(0)

            # Cat: TODO: save only vis chans
            wfs_array[iter_].append(wfs)
            n_spikes_array[iter_].append(idx2.shape[0])
        
    # ******************************************************
    # ********** UPDATE TEMPLATES EVERY 60SEC **************
    # ******************************************************

    # update only every 60 seconds
    print ("time_index: ", time_index)
    # Cat: TODO read length of time to update from CONFIG
    #if (templates_in is None) or (time_index%60!=0):
    if (time_index%60!=0) or (time_index==0):
        return templates_old, wfs_array
    
    # forgetting factor ~ number of spikes
    # Cat: TODO: read from CONFIG file
    nu = 10
    
    # find max chans of templates
    max_chans = templates_in.ptp(1).argmax(0)

    # make new templates
    templates_new = np.zeros(templates_in.shape,'float32')

    # get unit ids
    ids = torch.cat(d_gpu.neuron_array)
    units = np.arange(templates_in.shape[2])
    
    # Cat: TODO: read from CONFIG; # of spikes to be used for computing template
    n_spikes_min = 5000
    
    # Cat: TODO Parallelize this step
    for unit in units:
        
        # get saved waveforms and number of spikes
        wfs_local = []
        n_spikes_local = 0
        for c in range(len(wfs_array)):
            wfs_local.append(wfs_array[c][unit])
            n_spikes_local+=n_spikes_array[c][unit]
            
        wfs_local = np.vstack(wfs_local)
        
        # limit loaded waveforms to 1000 spikes; alignment is expensive
        idx_choice = np.random.choice(np.arange(wfs_local.shape[0]),
                                        size=min(wfs_local.shape[0],n_spikes_min))
        wfs_local = wfs_local[idx_choice]
        
        # keep track of spikes in window; 
        # Cat: TODO: we are not weighing the template correctly here;
        #      # of spikes is technically larger than # of waveforms tracked...
        #n_spikes = wfs_local.shape[0]
        n_spikes = n_spikes_local
        print ("wfs local: ", wfs_local.shape)
        
        if n_spikes<2:
            templates_new[:,:,unit] = templates_in[:,:,unit]
            continue
        # Cat: TODO: implement gpu only version; 
        #            d_gpu.data should already be on GPU
        # idx = torch.where(ids==unit,
                          # ids*0+1,
                          # ids*0)
        # idx = torch.nonzero(idx)[:,0]
        # wfs = data[:,ids[idx].cpu().data.numpy()[:,None]+np.arange(61)-61]

        # align waveforms by finding best shfits
        if False:
            mc = max_chans[unit]
            best_shifts = align_get_shifts_with_ref(wfs_local[:, :, mc],ref_template)
            wfs_local = shift_chans(wfs_local, best_shifts)
        
        # compute template
        # template = wfs.mean(0).T              
        template = np.median(wfs_local, axis=0).T   
           
        # update templates; similar to Kilosort Eq (6)
        # if we're in the first block of time; just use existing spikes; 
        #     i.e. don't update based on existing tempaltes as they represent 
        #     the mean of the template over time.
        if time_index==60:
            templates_new[:,:,unit]=template
        # use KS eq (6)
        else:
            t1 = templates_in[:,:,unit]*np.exp(-n_spikes/nu)
            t2 = (1-np.exp(-n_spikes/nu))*template
            templates_new[:,:,unit] = (t1+t2)

    # save template chunk for offline analysis only
    np.save('/media/cat/1TB/liam/49channels/data1_allset/tmp/block_2/deconv/'+
            str(time_index)+'.npy', templates_new)

    return templates_new, []
