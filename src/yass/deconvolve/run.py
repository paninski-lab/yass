import os
import logging
import numpy as np
import parmap
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

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_templates = os.path.join(
        output_directory, 'templates.npy')
    fname_spike_train = os.path.join(
        output_directory, 'spike_train.npy')
    fname_templates_up = os.path.join(
        output_directory, 'templates_up.npy')
    fname_spike_train_up = os.path.join(
        output_directory, 'spike_train_up.npy')
    fname_shifts = os.path.join(
        output_directory, 'shifts.npy')
    fname_objs = os.path.join(
        output_directory, 'obj_vals.npy')
                               
    print ("Processing templates: ", fname_templates_in)

    # Cat: TODO: use Peter's conditional (below) instead of single file check
    # if (os.path.exists(fname_templates) and
        # os.path.exists(fname_spike_train) and
        # os.path.exists(fname_templates_up) and
        # os.path.exists(fname_spike_train_up)):
        # return (fname_templates, fname_spike_train,
                # fname_templates_up, fname_spike_train_up)

    if os.path.exists(fname_spike_train):
        return (fname_templates, fname_spike_train,
                fname_templates_up, fname_spike_train_up,
                fname_shifts)
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
        n_sec_chunk = CONFIG.resources.n_sec_chunk_gpu_deconv
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
                      fname_spike_train,
                      fname_spike_train_up,
                      fname_templates,
                      fname_templates_up,
                      fname_shifts,
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
                     fname_spike_train_up,
                     fname_templates,
                     fname_templates_up,
                     CONFIG)

    return (fname_templates, fname_spike_train,
            fname_templates_up, fname_spike_train_up,
            fname_shifts)



def deconv_ONgpu2(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 update_templates,
                 fname_spike_train,
                 fname_spike_train_up,
                 fname_templates,
                 fname_templates_up,
                 fname_shifts,
                 CONFIG,
                 run_chunk_sec):

    # *********** MAKE DECONV OBJECT ************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)

    #print (kfadfa)
    # Cat: TODO: gpu deconv requires own chunk_len variable
    #root_dir = '/media/cat/1TB/liam/49channels/data1_allset'
    root_dir = CONFIG.data.root_folder

    # Cat: TODO: read from CONFIG
    d_gpu.max_iter = 1000
    d_gpu.deconv_thresh=threshold

    # Cat: TODO: make sure svd recomputed for higher rank etc.
    d_gpu.svd_flag = True

    # Cat: TODO read from CONFIG file 
    d_gpu.RANK = 10
    d_gpu.vis_chan_thresh = 1.0

    # debug/printout parameters
    # Cat: TODO: read all from CONFIG
    d_gpu.save_objective = False
    d_gpu.verbose = False
    d_gpu.print_iteration_counter = 1
    d_gpu.chunk_id = 0
    
    # Turn on refactoriness
    d_gpu.refractoriness = True
    
    # Stochastic gradient descent option
    # Cat: TODO: move these and other params to CONFIG
    d_gpu.scd = True

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
    d_gpu.max_percent_update = 0.1
    if d_gpu.update_templates:
        print ("   templates being updated ...")
    else:
        print ("   templates NOT being updated ...")

    # update template time chunk; in seconds
    # Cat: TODO: read from CONFIG file
    d_gpu.template_update_time = CONFIG.deconvolution.template_update_time
    
    # set forgetting factor to 5Hz (i.e. 5 spikes per second of chunk)
    d_gpu.nu = 1 * d_gpu.template_update_time 
    
    # parameter forces deconv to do a backward step
    #   i.e. deconv is rerun on previous chunk using templates made 
    #   from that chunk
    # Cat: TODO read from CONFIG
    if d_gpu.update_templates:
        d_gpu.update_templates_backwards = 1
    else:
        d_gpu.update_templates_backwards = 0        
        

    # dummy flag that tracks the save state of deconv 
    recursion_time = 1E10 
        
    # add reader
    d_gpu.reader = reader
    
    # enforce broad buffer
    d_gpu.reader.buffer=1000


    #d_gpu.reader.n_batches = 30

    # *********************************************************
    # *********************** RUN DECONV **********************
    # *********************************************************
    begin=dt.datetime.now().timestamp()
    if d_gpu.update_templates:
        d_gpu, setup_time = run_deconv_with_templates_update(d_gpu, 
                                                            CONFIG, 
                                                            recursion_time,
                                                            output_directory)
    else:
        d_gpu, setup_time = run_deconv_no_templates_update(d_gpu, CONFIG)


    # ****************************************************************
    # *********************** GATHER SPIKE TRAINS ********************
    # ****************************************************************
    subtract_time = np.round((dt.datetime.now().timestamp()-begin),4)

    print ("-------------------------------------------")
    total_length_sec = int((d_gpu.reader.end - d_gpu.reader.start)/d_gpu.reader.sampling_rate)
    print ("Total Deconv Speed ", np.round(total_length_sec/(setup_time+subtract_time),2), " x Realtime")

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
    for chunk_id in tqdm(range(reader.n_batches)):
        #fname = os.path.join(d_gpu.seg_dir,str(chunk_id).zfill(5)+'.npz')
        time_index = (chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv
        fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
        data = np.load(fname, allow_pickle=True)

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

            # Cat: TODO: is it faster to make list and then array?
            #            or make array on the fly?
            spike_train.extend(temp)
            shifts.append(shift_list[p].cpu().data.numpy()[idx_keep])

    # Cat; TODO: sepped this up.
    print ("   vstacking spikes (TODO initalize large array and then try to fill it...): ")
    spike_train = np.vstack(spike_train)
    shifts = np.hstack(shifts)

    # add half the spike time back in to get to centre of spike
    spike_train[:,0] = spike_train[:,0]-temporal_size//2

    # sort spike train by time
    print ("   ordering spikes: ")
    idx = spike_train[:,0].argsort(0)
    spike_train = spike_train[idx]
    shifts = shifts[idx]

    np.save(fname_spike_train[:-4]+"_prededuplication.npy", spike_train)

    # remove duplicates
    # Cat: TODO: are there still duplicates in spike trains!?
    print ("removing duplicates... (TODO: remove this requirement eventually...)")
    for k in np.unique(spike_train[:,1]):
        idx = np.where(spike_train[:,1]==k)[0]
        _,idx2 = np.unique(spike_train[idx,0], return_index=True)
        idx3 = np.delete(np.arange(idx.shape[0]),idx2)
        #print ("idx: ", idx[:10], idx.shape, " idx2: ", idx2[:10], idx2.shape,
        #      " idx3: ", idx3[:10], idx2.shape,
        #      " idx[idx3]: ", idx[idx3])
        #print ("unit: ", k, "  spike train: ", spike_train[idx][:10])
     #
        if idx3.shape[0]>0:
            print ("unit: ", k, " has duplicates: ", idx3.shape[0])
            spike_train[idx[idx3],0]=-1E6
        
       #quit()
    idx = np.where(spike_train[:,0]==-1E6)[0]
    spike_train = np.delete(spike_train, idx, 0)
    shifts = np.delete(shifts, idx, 0)
        

    # save spike train
    print ("  saving spike_train: ", spike_train.shape)
    fname_spike_train = os.path.join(d_gpu.out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)
    np.save(fname_spike_train_up, spike_train)

    # save shifts
    fname_shifts = os.path.join(d_gpu.out_dir, 'shifts.npy')
    np.save(fname_shifts, shifts)

    # save templates and upsampled templates
    templates_in_original = np.load(fname_templates_in)
    np.save(fname_templates, templates_in_original)
    np.save(fname_templates_up, templates_in_original)


def run_deconv_no_templates_update(d_gpu, CONFIG):

    chunk_ids = np.arange(d_gpu.reader.n_batches)
    chunk_ids_split = np.split(chunk_ids,
                               len(CONFIG.torch_devices))
                               
    n_sec_chunk_gpu = CONFIG.resources.n_sec_chunk_gpu_deconv

    processes = []
    for ii, device in enumerate(CONFIG.torch_devices):
        p = mp.Process(target=run_deconv_no_templates_update_parallel,
                       args=(d_gpu, chunk_ids_split[ii],
                             n_sec_chunk_gpu, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    return d_gpu, 0


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
                     shift_list = d_gpu.shift_list)


def run_deconv_with_templates_update(d_gpu, CONFIG,
                                    recursion_time,
                                    output_directory):

    # ****************************************************************
    # *********************** INITIALIZE DECONV **********************
    # ****************************************************************
    begin=dt.datetime.now().timestamp()
    d_gpu.initialize()
    setup_time = np.round((dt.datetime.now().timestamp()-begin),4)
    print ("-------------------------------------------")
    print ("Total init time ", setup_time, 'sec')
    print ("-------------------------------------------")
    print ("")

    # ****************************************************************
    # ********************** ITERATE OVER CHUNKS *********************
    # ****************************************************************
    begin=dt.datetime.now().timestamp()
    # Cat: TODO: flag to run deconv just on segments of data not all data
    # Cat: this may fail if length of recording not multipole of n_sec_gpu
    #chunks = []
    #for k in range(0, CONFIG.rec_len//CONFIG.recordings.sampling_rate, 
    #                CONFIG.resources.n_sec_chunk_gpu_deconv):
    #    chunks.append([k,k+CONFIG.resources.n_sec_chunk_gpu_deconv])

    # loop over chunks and run sutraction step
    #templates_old = None
    wfs_array = []
    n_spikes_array = []    
    ptp_array = []
    ptp_time_array = []
    
    # determine if yass is recovering from crash by reading last state
    #   if state exists, then recover from there.
    # Cat: TODO: simplify this and remove metadata dependence; overwrite previous batch
    try: 
        recursion_time = np.load(os.path.join(d_gpu.seg_dir,'time.npy'))
        state = np.loadtxt(os.path.join(d_gpu.seg_dir,'state.txt'), dtype='str')
        if state=='forward':
            d_gpu.update_templates_backwards = 1
        else:
            d_gpu.update_templates_backwards = 0
    except:
        pass

    # denoiser state
    if False:
        denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                           CONFIG.neuralnetwork.denoise.filter_sizes,
                           CONFIG.spike_size_nn)
        denoiser.load(CONFIG.neuralnetwork.denoise.filename)
        denoiser = denoiser.cuda()
                          
        # 
        denoiser = denoiser.to(CONFIG.torch_devices[0])
    
    #***********************************************************
    #********************* MAIN DECONV LOOP ********************
    #***********************************************************
    chunk_id = 0
    while True:
        # keep track of chunk being deconved and time_index
        time_index = (chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv

        if d_gpu.update_templates_backwards:
            fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'_forward.npz')
        else:
            fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')

        if os.path.exists(fname)==False:
            
            # this applies to crashes mid-run
            # it's important to load the updated template file - not the starting template
            if chunk_id>0:
                out_file = os.path.join(output_directory,'template_updates',
                                'templates_'+
                                str((chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv)+
                                'sec.npy')
                d_gpu.fname_templates = out_file
    
            # if true: forward pass over data using previous chunk templates
            if (time_index>recursion_time and 
                d_gpu.update_templates_backwards==False):

                    #print ("Forward pass ON ...")
                    d_gpu.update_templates_backwards = True
                    fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'_forward.npz')

                    # save forward pass state
                    np.savetxt(os.path.join(d_gpu.seg_dir,'state.txt'),['forward'],fmt="%s")

            # printout flags
            if d_gpu.update_templates_backwards:
                print ("Forward pass - updating templates", time_index, " sec")
            else:
                print ("Backwards pass - redecon with updated templates ...", time_index, " sec")
            
            #***********************************************************
            #********************* MAIN DECONV STEP ********************
            #***********************************************************
            # run deconv
            d_gpu.run(chunk_id)
  
            # save deconv results
            np.savez(fname, 
                     spike_array = d_gpu.spike_array,
                     offset_array = d_gpu.offset_array,
                     neuron_array = d_gpu.neuron_array,
                     shift_list = d_gpu.shift_list)
            
            #***********************************************************
            #******************* UPDATE TEMPLATE STEP ******************
            #***********************************************************

            # UPDATE TEMPLATES: every batch of data (e.g. 10sec)
            # GENERATE NEW TEMPLATES every chunk (e.g. 60sec) 
            if d_gpu.update_templates_backwards:
                (d_gpu, wfs_array, n_spikes_array, ptp_array, ptp_time_array) = \
                                    update_templates_GPU_forward_backward(
                                                        d_gpu, 
                                                        CONFIG, 
                                                        time_index, 
                                                        wfs_array, 
                                                        n_spikes_array,
                                                        ptp_array,
                                                        ptp_time_array,
                                                        output_directory,
                                                        #denoiser
                                                        )

                # IF NEW TEMPLATES, re-deconvolve previous chunk
                if (time_index%d_gpu.template_update_time==0):
                    #print ("Backward pass ON ...")
                    d_gpu.update_templates_backwards = False
                    chunk_id = (chunk_id - 
                            d_gpu.template_update_time//CONFIG.resources.n_sec_chunk_gpu_deconv)
                    
                    # make a note of where the last backward step was
                    # save backward pass state
                    # Cat: TODO: unclear how important this is for save state recovery;
                    #            - may wish to just delete previous bathc of data to allow forward step recovery
                    recursion_time = time_index
                    np.savetxt(os.path.join(d_gpu.seg_dir,'state.txt'),['backward'],fmt="%s")
                    np.save(os.path.join(d_gpu.seg_dir,'time.npy'),recursion_time)

        #print ("chunk id outside OFF loop: ", chunk_id)
        chunk_id+=1       
        
        # exit when finished reading;
        if chunk_id>=d_gpu.reader.n_batches:
            break

    return d_gpu, setup_time
    
def update_templates_GPU_forward_backward(d_gpu, 
                                         CONFIG, 
                                         time_index, 
                                         wfs_array, 
                                         n_spikes_array,
                                         ptp_array,
                                         ptp_time_array,
                                         output_directory,
                                         #denoiser
                                         ):

    # ******************************************************
    # ***************** SAVE WFS FIRST *********************
    # ******************************************************

    # make new entry in array
    wfs_array.append([])
    n_spikes_array.append([])
    ptp_array.append([])
    ptp_time_array.append([])

    iter_ = len(wfs_array)-1
    
    # original templates needed for array generation
    #templates_old = d_gpu.temps
    if len(d_gpu.neuron_array)>0: 
        ids = torch.cat(d_gpu.neuron_array)
        spike_array = torch.cat(d_gpu.spike_array)
        shifts_array = torch.cat(d_gpu.shift_list)
    else:
        ids = torch.empty(0,dtype=torch.double)
        spike_array = torch.empty(0,dtype=torch.double)
        shifts_array = torch.empty(0,dtype=torch.double).cpu().data.numpy()

    units = np.arange(d_gpu.temps.shape[2])

    # Cat: TODO: move this value to CONFIG;
    # proportion of drfit allowed in each chunk of decon
    # ptp_drift_percent = 0.2
    save_flag = False
    verbose = False
    super_res_align= True
    nn_denoise = False
    d_gpu.temps_cuda = torch.from_numpy(d_gpu.temps).cuda()  #torch(d_gpu.temps).cuda
    
    # DEBUG arrays:
    denoised_wf_all_array = []
    idx_ptp_array = []
    wfs_temp_original_array = []
    wfs_temp_aligned_array = []
    #original_templates = []
    try:
        os.mkdir(d_gpu.out_dir + '/wfs/')
    except:
        pass
        
    #print ("DGPU: ", d_gpu.temps.shape)
    #print ("self.fname_templates: ", d_gpu.fname_templates)
    compressed_template_array = []
    
    # increase first 
    if time_index<d_gpu.template_update_time:
        max_percent_update = d_gpu.max_percent_update*3
    else:
        max_percent_update = d_gpu.max_percent_update
    
    unit_test = 99999
    # Cat: TODO: read debug from CONFIG
    #           - debug here saves ptp and other metadata for drift model;
    #           - important for debugging
    
    
    
    debug = True
    if True:
        # GPU version + weighted computation
        data = d_gpu.data
        snipit = torch.arange(0,d_gpu.temps.shape[1],1).cuda() 
        wfs_empty = np.zeros((d_gpu.temps.shape[1],d_gpu.temps.shape[0]),'float32')
        #total_spikes = 0
        
        for unit in units:
            # 
            idx = torch.where(ids==unit, ids*0+1, ids*0)
            idx = torch.nonzero(idx)[:,0]

            times = spike_array[idx]
            shifts = shifts_array[idx]
            
            # get indexes of spikes that are not too close to end; 
            #       note: spiketimes are relative to current chunk; i.e. start at 0
            idx2 = torch.where(times<(data.shape[1]-d_gpu.temps.shape[1]), times*0+1, times*0)
            idx2 = torch.nonzero(idx2)[:,0]
            times = times[idx2]
            shifts = shifts[idx2].cpu().data.numpy()
            
            #print (" # spikes: ", idx2.shape)
            #total_spikes+=idx2.shape[0]

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
                wf_torch = wfs_temp_original[:,0,:,d_gpu.max_chans[unit]]
                wfs_temp_original_array.append(wf_torch.cpu().data.numpy())
                
                # *********************************************************
                # **************** NN DENOISE STEP ************************
                # *********************************************************
                # select max chan only
                if nn_denoise:
                    start = dt.datetime.now().timestamp()
                    
                    # add the orignal template max channel waveform and denoise together
                    wf_torch_plus_original = torch.cat((
                                    d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit][None],
                                    wf_torch),0)
                    
                    #original_templates.append(d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit])
                    
                    denoised_wf_all = denoiser(wf_torch_plus_original)[0]
                    
                    # grab first waveform as the original compressed templated
                    template_original_denoised = denoised_wf_all[0].cpu().data.numpy()
                    
                    # all other spikes 
                    denoised_wfs = denoised_wf_all[1:].cpu().data.numpy()
                        
                    if verbose: 
                        print ("....denoise step: ", dt.datetime.now().timestamp()-start)
                
                    # 
                    denoised_wf_all_array.append(denoised_wf_all.cpu().data.numpy())

                else:
                    # these aren't really denoised
                    template_original_denoised = d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy()
                    
                    denoised_wfs = wf_torch.cpu().data.numpy()
                    denoised_wf_all_array.append(denoised_wfs)
                    template_original_denoised = d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy()

                # *********************************************************
                # ************ USE DECONV SHIFTS TO ALIGN SPIKES **********
                # *********************************************************
                # work with denoised waveforms
                #wfs_temp = denoised_wf.cpu().data.numpy()
                if super_res_align: 
                    if shifts.shape[0]==1:
                        shifts = shifts[:,None]
                    wfs_temp_aligned = shift_chans(denoised_wfs, -shifts)

                    #print ("wfs_temp: ", wfs_temp_aligned.shape)
                    #wfs_temp_aligned = shift_chans(wfs_temp, shifts)
                    wfs_temp_aligned_array.append(wfs_temp_aligned)
                else:
                    wfs_temp_aligned = denoised_wfs
                    wfs_temp_aligned_array.append(wfs_temp_aligned)

                # *********************************************************
                # ************ COMPUTE PTPS & TRIAGE STEP *****************
                # *********************************************************

                # Grab ptps from wfs using previously computed ptp_max and ptp_min values in ptp_locs
                # exclude ptps that are > or < than 20% than the current template ptp

                # OPTION 1: exclude spikes that have PTP outside of threshold
                if True:
                    start = dt.datetime.now().timestamp()
                    #if nn_denoise==True:
                    if True:
                        ptp_template_original_denoised = template_original_denoised.ptp(0)

                        # **************************************************
                        # ********** OPTION #1: PTP at fixed points ********
                        # **************************************************
                        
                        if False:
                            #ptp = (wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0], d_gpu.max_chans[unit]]-
                            #      wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1], d_gpu.max_chans[unit]])
                            #idx_ptp = np.where(np.logical_and(ptp>=d_gpu.ptps[unit]*(1-d_gpu.max_percent_update),
                            #                                  ptp<=d_gpu.ptps[unit]*(1+d_gpu.max_percent_update)))[0]
                            
                            # data is already aligned and max chan only; look for ptps that are withing triage boundary
                            ptp = (wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]]-
                                   wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]])
                            
                            #idx_ptp = np.where(np.logical_and(ptp>=d_gpu.ptps[unit]*(1-d_gpu.max_percent_update),
                            #                                  ptp<=d_gpu.ptps[unit]*(1+d_gpu.max_percent_update)))[0]
                            idx_ptp = np.where(np.logical_and(ptp>=ptp_template_original_denoised*(1-max_percent_update),
                                                              ptp<=ptp_template_original_denoised*(1+max_percent_update)))[0]
                            idx_ptp_array.append(idx_ptp)
                        
                        # *************************************************************
                        # **** OPTION #2: Maxes/Mins at fixed poitns + PTP LIMITS *****
                        # *************************************************************
                        else:
                            if False:
                                maxes = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]]
                                mins = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]]
                            else:
                                maxes = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]-1:d_gpu.ptp_locs[unit][0]+2].max(1)
                                mins = wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]-1:d_gpu.ptp_locs[unit][1]+2].min(1)
                                
                            # idx_maxes = np.where(np.logical_and(maxes>=d_gpu.temps[d_gpu.max_chans[unit],
                                                                        # d_gpu.ptp_locs[unit][0],unit]*
                                                                        # (1-d_gpu.max_percent_update),
                                                                # maxes<=d_gpu.temps[d_gpu.max_chans[unit],
                                                                # d_gpu.ptp_locs[unit][0], unit]*
                                                                # (1+d_gpu.max_percent_update)))[0]
                            idx_maxes = np.where(np.logical_and(maxes>=template_original_denoised[d_gpu.ptp_locs[unit][0]]*
                                                                        (1-max_percent_update),
                                                                maxes<=template_original_denoised[d_gpu.ptp_locs[unit][0]]*
                                                                        (1+max_percent_update))
                                                )[0]
                                                                
                            idx_mins = np.where(np.logical_and(mins<=template_original_denoised[d_gpu.ptp_locs[unit][1]]*
                                                                        (1-max_percent_update),
                                                                mins>=template_original_denoised[d_gpu.ptp_locs[unit][1]]*
                                                                        (1+max_percent_update))
                                                )[0]
                            
                            idx_ptp_max_min, idx_ptp_maxes, idx_ptp_mins = np.intersect1d(idx_maxes,idx_mins,return_indices=True)
                            

                            
                            # also add criteria that the ptp overall of the denoised aligned waveform is correct:
                            ptps_dumb = wfs_temp_aligned.ptp(1)
                            idx_ptp_dumb = np.where(np.logical_and(ptps_dumb>=ptp_template_original_denoised*(
                                                                                1-max_percent_update),
                                                              ptps_dumb<=ptp_template_original_denoised*(
                                                                                1+max_percent_update)))[0]             

                            
                            idx_ptp, idx_ptp_maxes, idx_ptp_mins = np.intersect1d(idx_ptp_max_min,
                                                            idx_ptp_dumb,return_indices=True)
                        
                            if unit==unit_test:
                                print ("UNIT: ", unit)
                                #print ("template_original_denoised: ", template_original_denoised)
                                print ("template_original_denoised[d_gpu.ptp_locs[unit][0]]: ", template_original_denoised[d_gpu.ptp_locs[unit][0]])
                                print ("template_original_denoised[d_gpu.ptp_locs[unit][1]]: ", template_original_denoised[d_gpu.ptp_locs[unit][1]])
                                print ("idx_maxes: ", idx_maxes[:10])
                                print ("idx_mins: ", idx_mins[:10])
                                
                                print ("maxes: ", maxes[:10])
                                print ("maxes average: ", maxes.mean(0))
                                print ("mins: ", mins[:10])
                                print ("mins average: ", mins.mean(0))

                                print ("maxes[idx_ptp]: ", maxes[idx_ptp][:10])
                                print ("maxes average[idx_ptp]: ", maxes[idx_ptp].mean(0))
                                print ("mins:[idx_ptp] ", mins[idx_ptp][:10])
                                print ("mins average:[idx_ptp] ", mins[idx_ptp].mean(0))
                                
                                print ("idx_ptp_max_min: ", idx_ptp_max_min[:10])
                                print ("idx_ptp_dumb: ", idx_ptp_dumb[:10])
                                print ("idx_ptp_final: ", idx_ptp[:10])
                                
                                # this uses ptps of the raw waveforms
                                ptp = (maxes-mins)
                                ptp = ptp[idx_ptp].mean(0)

                                print ("ptp value: ", ptp)
                                print ("original Template ptp without denoise: ",
                                        d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy().ptp(0))
                                        
                                print ("ptp[triaged spikes]: ", ptp)
                                print ("all spikes ptp(mean): ", wf_torch.cpu().data.numpy().mean(0).ptp(0))
                                print ("all spikes mean(ptp): ", wf_torch.cpu().data.numpy().ptp(1).mean(0))

                                print ("")
                                print ("")
                                print ("")

                            idx_ptp_array.append(idx_ptp)

                        
                    else:
                        print (" drift on raw data (without NN-denoise) NOT IMPLEMENTED (quiting) ")
                        quit()
                        # ptp = (wfs_temp.cpu().data.numpy()[:,0,d_gpu.ptp_locs[unit][0], d_gpu.max_chans[unit]]-
                              # wfs_temp.cpu().data.numpy()[:,0,d_gpu.ptp_locs[unit][1], d_gpu.max_chans[unit]])
                        # idx_ptp = np.where(np.logical_and(ptp>=d_gpu.ptps[unit]*(1-d_gpu.max_percent_update),
                                                          # ptp<=d_gpu.ptps[unit]*(1+d_gpu.max_percent_update)))[0]

                    # grab ptps at specific locations
                    if idx_ptp.shape[0]>0:
                        #print ("# of spikes survived: ", idx_ptp.shape[0], "/", ptp.shape[0])
                        
                        if True:
                            # this uses ptps of denoised waveforms
                            #ptp = (wfs_temp_aligned[:,d_gpu.ptp_locs[unit][0]]-
                            #       wfs_temp_aligned[:,d_gpu.ptp_locs[unit][1]])
                            
                            ptp = (maxes-mins)
                                
                        else:
                            
                            # this uses ptps of the raw waveforms
                            ptp = (wf_torch[:,d_gpu.ptp_locs[unit][0]]-
                                   wf_torch[:,d_gpu.ptp_locs[unit][1]])
                        
                        ptp = ptp[idx_ptp].mean(0)

                    else:
                        ptp = 0
                        
                    if verbose: 
                        print ("....ptp step: ", dt.datetime.now().timestamp() - start)

                # # OPTION 2: exclude spikes that independetly have PTP_min and PTP_max outside of threhsold
                # #           - this is a more aggressive triage method
                # else:
                    # maxes = wfs_temp.cpu().data.numpy()[:,0,d_gpu.ptp_locs[unit][0], d_gpu.max_chans[unit]]
                    # mins =  wfs_temp.cpu().data.numpy()[:,0,d_gpu.ptp_locs[unit][1], d_gpu.max_chans[unit]]
                
                    # idx_maxes = np.where(np.logical_and(maxes>=d_gpu.temps[d_gpu.max_chans[unit],
                                                                # d_gpu.ptp_locs[unit][0],unit]*
                                                                # (1-d_gpu.max_percent_update),
                                                        # maxes<=d_gpu.temps[d_gpu.max_chans[unit],
                                                        # d_gpu.ptp_locs[unit][0], unit]*
                                                        # (1+d_gpu.max_percent_update)))[0]
                    
                    # idx_mins = np.where(np.logical_and(mins<=d_gpu.temps[d_gpu.max_chans[unit],
                                                                # d_gpu.ptp_locs[unit][1],unit]*
                                                                # (1-d_gpu.max_percent_update),
                                                        # mins>=d_gpu.temps[d_gpu.max_chans[unit],
                                                        # d_gpu.ptp_locs[unit][1], unit]*
                                                        # (1+d_gpu.max_percent_update)))[0]
                    
                    # idx_ptp, idx_ptp_maxes, idx_ptp_mins = np.intersect1d(idx_maxes,idx_mins,return_indices=True)
                    # ptp = (maxes[idx_ptp]-mins[idx_ptp]).mean(0)
                    

                if idx_ptp.shape[0]>0:
                    start = dt.datetime.now().timestamp()
                    
                    # if False:
                        # wfs = torch.mean(wfs_temp[idx_ptp],0)[0]
                        # wfs_array[iter_].append(wfs.cpu().data.numpy())
                    # else:
                    
                    # Cat: TODO: THIS IS NOT CORRECT FOR THE FULL TEMPLATE MODEL!!!
                    wfs = np.mean(wfs_temp_aligned[idx_ptp],0)[0]
                    wfs_array[iter_].append(wfs)
                    
                    n_spikes_array[iter_].append(idx_ptp.shape[0])
                    ptp_array[iter_].append(ptp)
                    
                    # save this as metadata; not really required
                    ptp_time_array[iter_].append(times[idx_ptp].cpu().data.numpy())
                    if verbose: 
                        print ("....mean step: ", dt.datetime.now().timestamp()-start)
                else:
                    # default save zeros
                    wfs_array[iter_].append(d_gpu.temps[:,:,unit].transpose(1,0))
                                            #d_gpu.temps.shape[1], d_gpu.temps.shape[0])
                    n_spikes_array[iter_].append(0)
                    ptp_array[iter_].append(0)
                    ptp_time_array[iter_].append(0)
            
            # THERE ARE NO SPIKES IN TIME CHUNK
            else:
                # default save zeros
                wfs_array[iter_].append(d_gpu.temps[:,:,unit].transpose(1,0))
                                        #d_gpu.temps.shape[1], d_gpu.temps.shape[0])
                n_spikes_array[iter_].append(0)
                ptp_array[iter_].append(0)
                ptp_time_array[iter_].append(0)
                
                denoised_wf_all_array.append([])
                wfs_temp_original_array.append([])
                idx_ptp_array.append([])
                wfs_temp_aligned_array.append([])
                #original_templates.append([])
                
            # np.save('/media/cat/4TBSSD/liam/512channels/2009-04-13-5_45chan/tmp/block_2/deconv/'+
                        # str(unit)+'wfs_temp_original.npy', wfs_temp_original.cpu().data.numpy()
                                    # [:,0,:, d_gpu.max_chans[unit]])
            # np.save('/media/cat/4TBSSD/liam/512channels/2009-04-13-5_45chan/tmp/block_2/deconv/'+
                        # str(unit)+'wfs_temp_aligned.npy', wfs_temp_aligned)
            # np.save('/media/cat/4TBSSD/liam/512channels/2009-04-13-5_45chan/tmp/block_2/deconv/'+
                        # str(unit)+'wfs_temp_denoised.npy', denoised_wf.cpu().data.numpy())
    
    
    if debug:
        np.savez(d_gpu.out_dir + '/wfs/'+
            str(iter_)+'.npz',
            wfs_temp_original_array = wfs_temp_original_array,
            denoised_wf_all_array= denoised_wf_all_array,
            wfs_temp_aligned_array = wfs_temp_aligned_array,
            #original_templates = original_templates,
            idx_ptp_array = idx_ptp_array
                )
        
    # return if in middle of forward pass
    # Cat: TODO read length of time to update from CONFIG
    if (((time_index%d_gpu.template_update_time)!=0) or (time_index==0)):
                                        
        del d_gpu.temps_cuda
        torch.cuda.empty_cache()

        return (d_gpu, wfs_array, n_spikes_array, ptp_array, ptp_time_array)

    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************
    # **************************** UPDATE TEMPLATES EVERY BATCH ******************************
    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************

    # forgetting factor ~ number of spikes
    # Cat: TODO: read from CONFIG file
    #nu = d_gpu.nu
    
    # find max chans of templates
    max_chans = d_gpu.temps.ptp(1).argmax(0)

    # make new templates
    templates_new = np.zeros(d_gpu.temps.shape,'float32')
    units = np.arange(d_gpu.temps.shape[2])
    
    # *****************************************************************
    # *************** Weighted template computation *******************
    # *****************************************************************
    # n_chunks = chunk length / batch length x 2 to capture window on both sides
    n_batches_per_chunk = d_gpu.template_update_time//CONFIG.resources.n_sec_chunk_gpu_deconv
    
    # Cat: TODO: this might crash if we don't have enough spikes overall
    #           or even within a single window
    # batch_offset indicates the start of the previous chunk in order to do averaging over both prev+following chunks
    if iter_> n_batches_per_chunk:
        batch_offset = (iter_+1) - n_batches_per_chunk*2
        n_chunks = n_batches_per_chunk*2
        wfs_local = np.zeros((n_chunks, d_gpu.temps.shape[1], d_gpu.temps.shape[0]),'float32')
        n_spikes_local = np.zeros((n_chunks), 'int32')
        ptp_local = np.zeros((n_chunks), 'float32')

    # first block will use only forward data for new templates 
    #       old templates will be as usual the old templates;
    else:
        batch_offset = 0
        n_chunks = n_batches_per_chunk
        wfs_local = np.zeros((n_chunks, d_gpu.temps.shape[1], d_gpu.temps.shape[0]),'float32')
        n_spikes_local = np.zeros((n_chunks), 'int32')
        ptp_local = np.zeros((n_chunks), 'float32')

    # print ("iter: ", iter_, " batch_offset: ", batch_offset, " len wfs_array: ", len(wfs_array))
    #print ("Template updates started, n_chunks: ", n_chunks, " wfs_array: ", len(wfs_array))
    units_scaling = []
    
    ptp_flag = True
    
    # compress the templates for the 
    cuda_templates = []
    for unit in range(d_gpu.temps_cuda.shape[2]):
        cuda_templates.append(d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit][None])
        #print (" templ ate shape: ", d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].shape)

    # if using denoiser
    if False:
        #original_templates.append(d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit])
        cuda_templates = torch.cat((cuda_templates),0)
        #print ("cuda_templates: ", cuda_templates.shape)
        denoised_wf_all = denoiser(cuda_templates)[0].cpu().data.numpy()
        #print ("denoised_wf_all: ", denoised_wf_all.shape)
                        
    for unit in units:
        # get saved waveforms and number of spikes
        # Cat: TODO this is not pythonic/necessary; 
        ptp_local = []
        for c in range(n_chunks):
            wfs_local[c] = wfs_array[batch_offset+c][unit]
            n_spikes_local[c] = n_spikes_array[batch_offset+c][unit]
            ptp_local.append(ptp_array[batch_offset+c][unit])
            
        n_spikes = n_spikes_local.sum(0)
        # if there are no spikes at all matched, just use previous template 
        if n_spikes==0:
            templates_new[:,:,unit]=d_gpu.temps[:,:,unit]
            continue

        # compute scale of updated tempalte
        if ptp_flag:
            ptp_temp = np.average(np.float32(ptp_local), weights=n_spikes_local, axis=0)
            
            if False:
                scale = ptp_temp/denoised_wf_all[unit].ptp(0)
                
                # print ("unit: ", unit, "ptp_temp: ", ptp_temp, ", original template ptp: ",
                        # d_gpu.temps_cuda[d_gpu.max_chans[unit],:,unit].cpu().data.numpy().ptp(0),
                        # ", denoised_wf_all[unit].ptp(0): ", denoised_wf_all[unit].ptp(0),
                        # ", scale: ", scale)
            else:
                scale = ptp_temp/d_gpu.temps[:,:,unit].ptp(1).max(0)
            
            units_scaling.append(scale)
        else:
            # compute weighted average of the template
            template = np.average(wfs_local, weights=n_spikes_local, axis=0).T
        
        # # first chunk of data just use without weight.        
        if time_index==d_gpu.template_update_time:
            if ptp_flag:
                templates_new[:,:,unit]=d_gpu.temps[:,:,unit]*scale
            else:
                templates_new[:,:,unit]=template

        else:
            # # use KS eq (6)
            # else:
            if ptp_flag:
                #templates_new[:,:,unit] = d_gpu.temps[:,:,unit]*scale
                #templates_new[:,:,unit] = d_gpu.temps[:,:,unit]*scale
                t1 = d_gpu.temps[:,:,unit]*np.exp(-n_spikes/d_gpu.nu)
                t2 = (1-np.exp(-n_spikes/d_gpu.nu))*d_gpu.temps[:,:,unit]*scale
                templates_new[:,:,unit] = (t1+t2)
                
            else:
                #print ("temps: ", d_gpu.temps.shape, "Unit: ", unit, ", scaling factors: ", 
                #    np.exp(-n_spikes/d_gpu.nu), (1-np.exp(-n_spikes/d_gpu.nu)))
                t1 = d_gpu.temps[:,:,unit]*np.exp(-n_spikes/d_gpu.nu)
                t2 = (1-np.exp(-n_spikes/d_gpu.nu))*template
                templates_new[:,:,unit] = (t1+t2)
        
        if unit==unit_test:
            print ("  FINAL SCALE: ", scale)
            print ("  templates_new max ptp: ", templates_new[:,:,unit].ptp(1).max(0))
            print ("  ptp_local: ", ptp_local)

    #quit()
    
    # update template ptp computation:
        # compute ptps for data
    #d_gpu.ptps = templates_new.ptp(0).max(0)
                
    # # save template chunk for offline analysis only
    # np.save('/media/cat/1TB/liam/49channels/data1_allset/tmp/block_2/deconv/'+
            # str(time_index)+'.npy', templates_new)
    #np.save('/home/cat/units_scaling.npy',units_scaling)
    
    out_file = os.path.join(output_directory,'template_updates',
                    'templates_'+
                    str((d_gpu.chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv)+
                    'sec.npy')
                    
    # print (" TEMPS DONE: ", templates_in.shape, templates_new.shape)
    np.save(out_file, templates_new.transpose(2,1,0))
    np.save(out_file[:-4]+"_ptp.npy", ptp_array)
    np.save(out_file[:-4]+"_ptp_times.npy", ptp_time_array)
        
    # re-initialize d_gpu
    # change fname-templates
    # Cat: TODO: is this the best way to pass on template?  probably name is fine;
    d_gpu.fname_templates = out_file
    d_gpu.initialize()    

    # pass wfs-array back into stack
    return (d_gpu, wfs_array, n_spikes_array, ptp_array, ptp_time_array)

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
        idx2 = torch.where(times<data.shape[1], times*0+1, times*0)
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
                    str((d_gpu.chunk_id+1)*CONFIG.resources.n_sec_chunk_gpu_deconv)+
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

    # *********** MAKE DECONV OBJECT ************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)

    #print (kfadfa)
    # Cat: TODO: gpu deconv requires own chunk_len variable
    n_sec=CONFIG.resources.n_sec_chunk_gpu_deconv
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
