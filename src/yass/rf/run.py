import h5py
import parmap
import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm
import scipy
import scipy.io
import scipy.optimize as opt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import networkx as nx
from pkg_resources import resource_filename

from yass import read_config
from yass.template import upsample_resample, shift_chans

def run():
    """RF computation
    """

    CONFIG = read_config()

    stim_movie_file = os.path.join(CONFIG.data.root_folder, CONFIG.data.stimulus)
    triggers_fname = os.path.join(CONFIG.data.root_folder, CONFIG.data.triggers)
    spike_train_fname = os.path.join(CONFIG.path_to_output_directory,
                                     'spike_train.npy')
    saving_dir = os.path.join(CONFIG.path_to_output_directory, 'rf')
    
    rf = RF(stim_movie_file, triggers_fname, spike_train_fname, saving_dir)
    rf.calculate_STA()
    rf.detect_multi_rf()
    rf.classification()


class RF(object):
    def __init__(self, stim_movie_file, triggers_fname, spike_train_fname, saving_dir, matlab_bin='matlab'):
        
        # default parameter
        self.n_color_channels = 3
        self.sp_frame_rate = 20000
        self.data_sample_len = 36000000 # len of white noise data (this script doesn't look at natural scenes)
        
        self.load_spike_train(spike_train_fname)
        
        self.stim_movie_file = stim_movie_file
        self.triggers_fname = triggers_fname
        
        self.save_dir = saving_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.save_dir[-1] != '/':
            self.save_dir += '/'

        self.matlab_bin = matlab_bin

        print("spike train:\t{}".format(self.sps.shape))
        print("Number of units:\t{}".format(self.Ncells))

    def load_stimulus_trigger(self, stim_movie_file, triggers_fname):

        print('Loading Stimulus...')

        # Load stim file
        h5_temp = h5py.File(stim_movie_file, 'r')
        self.WN_stim = h5_temp['movie'][:]
        h5_temp.close()
        self.stim_size = self.WN_stim.shape[2:4]
        self.WN_stim = self.WN_stim.reshape((-1, self.n_color_channels,
                                             self.stim_size[0]*self.stim_size[1]))

        ## Load triggers
        if triggers_fname.split('.')[-1] == 'trig':
            with open(triggers_fname, 'rb'):
                 self.WN_trigger_times = np.fromfile(triggers_fname, dtype='int16')
        elif triggers_fname.split('.')[-1] == 'mat':
            self.WN_trigger_times = sio.loadmat(triggers_fname)
            self.WN_trigger_times = self.WN_trigger_times['triggers'].flatten().astype('float')

        print("stim movie:\t{}".format(self.WN_stim.shape))
        
        np.save(self.save_dir+'stim_size.npy',self.stim_size)
        
    def calculate_frame_times(self):
        
        frame_per_pulse = 100

        ## Find pulses and calculate frame times
        # Get first locations of pulses in seconds
        pulses = np.where(np.diff(self.WN_trigger_times)==-2048)[0]+1 # find where pulse starts (diff+1)
        pulses_seconds = pulses / float(self.sp_frame_rate) # divide by 20k Hz to get seconds

        self.frame_times = np.interp(
            np.arange(0,frame_per_pulse * pulses_seconds.shape[0]),
            np.arange(0,frame_per_pulse * pulses_seconds.shape[0], frame_per_pulse),
            pulses_seconds)
       
    
    def load_spike_train(self, spike_train_fname):
        
        print('Loading Spike Train...')
        
        ## Load spikes
        sps_file_ext = os.path.splitext(spike_train_fname)[1]
        if sps_file_ext == '.mat':
            #sps = sio.loadmat(spike_train_fname)['spike_train'].astype('int32')

            # for single columnd data
            sps_temp = sio.loadmat(spike_train_fname)['spike_train'].astype('int32')
            unique_ids = np.unique(sps_temp[:,1])
            unique_ids = unique_ids[unique_ids>0] - 1
            self.sps = np.zeros(sps_temp.shape, 'int32')
            self.sps[:, 0] = sps_temp[:, 0]

            for i, k in enumerate(unique_ids):
                idx = sps_temp[:, 1] == (k+1)
                self.sps[idx, 1] = i

        elif sps_file_ext == '.npy':
            self.sps = np.load(spike_train_fname)

        # Get number of cells/units
        self.Ncells = int(np.max(self.sps[:,1])+1)
    
    def calculate_STA(self):

        self.load_stimulus_trigger(self.stim_movie_file, self.triggers_fname)
        self.calculate_frame_times()

        tmp_dir = os.path.join(self.save_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        tmp_dir_sta = os.path.join(tmp_dir, 'sta')
        if not os.path.exists(tmp_dir_sta):
            os.makedirs(tmp_dir_sta)
        
        tmp_dir_rgc = os.path.join(tmp_dir, 'rgc')
        if not os.path.exists(tmp_dir_rgc):
            os.makedirs(tmp_dir_rgc)

        print('Calculating STA...')

        ############################################
        ## Get full STAs and spatial/temporal STA ##
        ############################################

        STA_temporal_length = 30 # how many bins/frames to include in STA
        Ncells = self.Ncells
        stim_size = self.stim_size
        n_color_channels = self.n_color_channels
        n_pixels = stim_size[0]*stim_size[1]

        unique_ids = np.unique(self.sps[:,1])
        
        args_in = []
        for i_cell in np.arange(Ncells):
            fname = os.path.join(tmp_dir_sta, 'unit_'+str(i_cell)+'.mat')
            if not os.path.exists(fname):

                ##################################
                ### Get spikes in stimulus bins ##
                ##################################

                # Get spike times of this cell in seconds
                these_sps = self.sps[self.sps[:,1]==i_cell]
                these_sps = these_sps[:,0]
                #spikes before 36000000 are white noise spikes, divide by frame rate to get seconds
                these_sps = these_sps[these_sps<self.data_sample_len] / float(self.sp_frame_rate)

                ## Line up spikes with frames
                binned_spikes, rr = np.histogram(these_sps,self.frame_times)
                which_spikes = np.where(binned_spikes>0)[0]
                which_spikes = which_spikes[which_spikes>STA_temporal_length]
            
                args_in.append([
                    self.WN_stim,
                    binned_spikes,
                    which_spikes,
                    STA_temporal_length,
                    stim_size,
                    fname
                ])
        
        if False:
            n_processors = 6
            parmap.map(sta_calculation_parallel,
                       args_in,
                       processes=n_processors,
                       pm_pbar=True)
        else:
            for unit in tqdm(range(len(args_in))):
                sta_calculation_parallel(args_in[unit])

        # run matlab code
        print('running matlab code')
        rf_matlab_loc = resource_filename('yass', 'rf/rf_matlab')
        command = '{} -nodisplay -r \"cd(\'{}\'); fit_sta_liam_parallel(\'{}\', \'{}\'); exit\"'.format(
            self.matlab_bin, rf_matlab_loc, tmp_dir_sta, tmp_dir_rgc)
        os.system(command)
        print('done running matlab code')

        STA_spatial = np.zeros((self.Ncells, stim_size[0], stim_size[1], n_color_channels))
        STA_temporal = np.zeros((self.Ncells, STA_temporal_length, n_color_channels))
        gaussian_fits = np.zeros((self.Ncells, 5))
        for unit in unique_ids:
            fname = os.path.join(tmp_dir_rgc, 'rgc_{}.mat'.format(unit))
            data = scipy.io.loadmat(fname)
            if 'temp_rf' in data.keys():
                STA_spatial[unit] = data['temp_rf']
                STA_temporal[unit] = data['fit_tc']
                gaussian_fits[unit] = data['temp_fit_params']['fit_params'][0][0][0][:5]
        
        np.save(os.path.join(self.save_dir, 'STA_spatial.npy'), STA_spatial)
        np.save(os.path.join(self.save_dir, 'STA_temporal.npy'), STA_temporal)
        np.save(os.path.join(self.save_dir, 'gaussian_fits.npy'), gaussian_fits)

    def detect_multi_rf(self):
        
        STA_spatial = np.load(self.save_dir+'STA_spatial.npy')

        n_rf = 1

        # loop over all units and compute 
        rfs_array = []
        units = np.arange(STA_spatial.shape[0])
        for unit in units:
            img = STA_spatial[unit,:,:,1]

            # smooth out img; seems to help overall
            img_smooth = gaussian_filter(img, sigma=1)

            # compute std over smooth image and threshold for significant pixels
            std = np.std(img_smooth)
            std_pixels = np.vstack(np.where(np.abs(img_smooth)>std*3.75))

            # exit if no significant pixels
            if std_pixels.shape[1]==0:
                rfs_array.append(0)
                continue

            # make matrix of significant pixels using euclidean pairwise distance
            std_pixels = np.array(std_pixels).T
            dists = cdist(std_pixels, std_pixels)

            # binarize distance matrix
            thresh = 1.5
            upper=1
            lower=0    
            bin_dists = np.where(dists<thresh, upper, lower)

            # enumerate over number of connected networks
            G = nx.from_numpy_array(bin_dists)
            for i, cc in enumerate(nx.connected_components(G)):
                pass

            rfs_array.append(i+1)

        # yass
        idx = np.where(np.array(rfs_array)==n_rf)[0]
        np.save(self.save_dir+'idx_single_rf.npy', idx)

        idx = np.where(np.array(rfs_array)==0)[0]
        np.save(self.save_dir+'idx_no_rf.npy', idx)

        idx = np.where(np.array(rfs_array)>n_rf)[0]
        np.save(self.save_dir+'idx_multi_rf.npy', idx)


    def classification(self):

        # wether the dataset is from 2017 or 2007
        run_2017 = False

        # load data
        STA_temporal = np.load(self.save_dir+'STA_temporal.npy')
        gaussian_fits = np.load(self.save_dir+'gaussian_fits.npy')
        idx_single_rf = np.load(self.save_dir+'idx_single_rf.npy')

        # remove double rf and think contours
        idx_keep = np.zeros(STA_temporal.shape[0], 'bool')
        areas = np.zeros(STA_temporal.shape[0])
        for unit in idx_single_rf:

            center = gaussian_fits[unit, :2]
            rad_sd = gaussian_fits[unit, 2:4]
            rot_angle = gaussian_fits[unit,4]

            # skip neurons with very thin contours
            if not ((rad_sd[0]<rad_sd[1]/5.) or (rad_sd[1]<rad_sd[0]/5.)):
                idx_keep[unit] = True
                areas[unit] = np.pi * rad_sd[0] * rad_sd[1]
        idx_keep = np.where(idx_keep)[0]
        
        # load saved feature spca and gmm params
        path_to_assets = resource_filename('yass', 'rf/classification')
        ref = np.load(os.path.join(path_to_assets, 'ref.npy'))
        if run_2017:
            temp_pca_compoents = np.load(os.path.join(path_to_assets, 'temp_pca_compoents_2017.npy'))
            temp_pca_mean = np.load(os.path.join(path_to_assets, 'temp_pca_mean_2017.npy'))
            pca_compoents = np.load(os.path.join(path_to_assets, 'pca_compoents_2017.npy'))
            pca_mean = np.load(os.path.join(path_to_assets, 'pca_mean_2017.npy'))
            gmm_covs = np.load(os.path.join(path_to_assets, 'gmm_covs_2017.npy'))
            gmm_means = np.load(os.path.join(path_to_assets, 'gmm_means_2017.npy'))
        else:
            temp_pca_compoents = np.load(os.path.join(path_to_assets, 'temp_pca_compoents.npy'))
            temp_pca_mean = np.load(os.path.join(path_to_assets, 'temp_pca_mean.npy'))
            pca_compoents = np.load(os.path.join(path_to_assets, 'pca_compoents.npy'))
            pca_mean = np.load(os.path.join(path_to_assets, 'pca_mean.npy'))
            gmm_covs = np.load(os.path.join(path_to_assets, 'gmm_covs.npy'))
            gmm_means = np.load(os.path.join(path_to_assets, 'gmm_means.npy'))


        # get features
        STA_temporal = align_tc(STA_temporal, ref)
        STA_temporal = STA_temporal.reshape(STA_temporal.shape[0], -1)

        temp_pc = np.matmul(STA_temporal - temp_pca_mean[None], temp_pca_compoents.T)
        features = np.hstack((temp_pc[:, :2], np.log(areas[:, None])))
        features = np.matmul(features - pca_mean[None], pca_compoents.T)
        features = features[idx_keep]
        
        # do classification based on mahalanobis distance
        feat_centered = (features[None] - gmm_means[:,None])
        inv_gmm_covs = np.linalg.inv(gmm_covs)
        maha_dist = np.sqrt(np.sum(np.matmul(feat_centered, inv_gmm_covs)*feat_centered, -1))
        
        labels = np.ones(STA_temporal.shape[0], 'int32')*-1
        labels_tmp = np.argmin(maha_dist, 0)
        idx_close = maha_dist.min(0) < 4
        labels[idx_keep[idx_close]] = labels_tmp[idx_close]
        
        np.save(os.path.join(self.save_dir, 'labels.npy'), labels)

    def twoD_Gaussian(self, xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        ## Define 2D Gaussian that we'll fit to spatial STAs
        (x, y) = xdata_tuple
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
        return g.ravel()

    def fit_gaussian(self):        

        if not os.path.exists(self.save_dir+'Gaussian_params.npy'):

            print('Fitting Gaussian on STA...')

            stim_size = self.stim_size

            STA_spatial = np.load(self.save_dir+'STA_spatial.npy')

            ## Fit Gaussian to STA
            use_green_only = True
            if use_green_only:
                this_STA_spatial = STA_spatial[:,1] 
            else:
                this_STA_spatial = STA_spatial_colorcat

            Gaussian_params = np.zeros((self.Ncells,7))
            Gaussian_params[:]=np.nan
            nonconverged_Gaussian_cells = np.empty(0,) # keep track of cells where fitting procedure doesn't converge

            # Loop over cells
            for i_cell in tqdm(range(self.Ncells)):

                # Get STA for this cell 
                this_STA = this_STA_spatial[i_cell].reshape((-1,)) 

                # Create x and y indices for grid for Gaussian fit
                x = np.arange(0, stim_size[1], 1)
                y = np.arange(0, stim_size[0], 1)
                x, y = np.meshgrid(x, y)

                # Get initial guess for Gaussian parameters (helps with fitting)
                init_amp = this_STA[np.argmax(np.abs(this_STA))] # get amplitude guess from most extreme (max or min) amplitude of this_STA
                init_x,init_y = np.unravel_index(np.argmax(np.abs(this_STA)),(stim_size[0],stim_size[1])) # guess center of Gaussian as indices of most extreme (max or min) amplitude
                initial_guess = (init_amp,init_y,init_x,2,2,0,0)

                # Try to fit, if it doesn't converge, log that cell
                try:
                    popt, pcov = opt.curve_fit(self.twoD_Gaussian, (x, y), this_STA, p0=initial_guess)
                    Gaussian_params[i_cell] = popt
                    Gaussian_params[i_cell,3:5] = np.abs(popt[3:5]) # sometimes sds are negative (in Gaussian def above, they're always squared)
                except:
                    nonconverged_Gaussian_cells = np.append(nonconverged_Gaussian_cells,i_cell)

            np.save(self.save_dir+'Gaussian_params.npy',Gaussian_params)
            

def get_denoiser(STA):
    max_timecourse = np.max(np.abs(STA[:,20:]), axis=1)
    max_timecourse_mean = np.mean(max_timecourse, axis=2)
    max_timecourse_std = np.std(max_timecourse, axis=2)
    good_ones = max_timecourse > (max_timecourse_mean + 4*max_timecourse_std)[:,:,np.newaxis]

    denoiser = np.zeros((3, 3, STA.shape[1]))
    for color in range(3):
        unit_id, pixel_id = np.where(good_ones[:,color])
        good_timecourse = STA[unit_id, :, color, pixel_id]
        [U,S,V] = np.linalg.svd(good_timecourse.T)
        denoiser[color] = U[:,:3].T
    
    return denoiser

def denoise_STA(STA):
    
    denoiser = get_denoiser(STA)
    
    n_units, _, _, n_pixels = STA.shape
    n_colors, n_filter, n_time = denoiser.shape
    
    STA_denoised = np.zeros(STA.shape)
    for color in range(n_colors):
        deno = np.matmul(denoiser[color].T, denoiser[color])
        STA_temp = STA[:,:,color].transpose(0,2,1).reshape(-1, n_time)
        STA_denoised[:,:,color] = np.matmul(STA_temp, deno).reshape(n_units, n_pixels, n_time).transpose(0,2,1)
    
    return STA_denoised

def get_temp_spat_filters(STA):
    [U,S,V] = np.linalg.svd(STA)

    temp_filter = U[:, 0]
    spatial_filter = V[0]
    temp_sign = np.sign(temp_filter[np.abs(temp_filter).argmax()])
    spat_sign = np.sign(spatial_filter[np.abs(spatial_filter).argmax()])
    sign = temp_sign*spat_sign
    if temp_sign != sign:
        temp_filter *= -1.0
    if spat_sign != sign:
        spatial_filter *= -1.0
    
    return temp_filter, spatial_filter


def sta_calculation_parallel(arg_in):
    
    WN_stim = arg_in[0]
    binned_spikes = arg_in[1]
    which_spikes = arg_in[2]
    STA_temporal_length = arg_in[3]
    stim_size = arg_in[4]
    fname = arg_in[5]
        
    ####################
    ### Calculate STA ##
    ####################

    ## Swap out fastest version here 
    _, n_color_channels, n_pixels = WN_stim.shape
    STA = np.zeros((STA_temporal_length, n_color_channels, n_pixels))
    for i in range(which_spikes.shape[0]):
        bin_number = which_spikes[i]
        if binned_spikes[bin_number] == 1:
            STA += WN_stim[bin_number-(STA_temporal_length-1):bin_number+1]
        else:
            STA += binned_spikes[bin_number]*WN_stim[bin_number-(STA_temporal_length-1):bin_number+1]

    # full sta
    if np.sum(binned_spikes[STA_temporal_length:])>0:
        STA = STA/np.sum(binned_spikes[STA_temporal_length:])
    STA = STA.reshape(STA_temporal_length, n_color_channels,
                      stim_size[0], stim_size[1])
    STA = STA.transpose(2,3,1,0)

    scipy.io.savemat(fname, mdict={'temp_stas': STA})

def align_tc(tc, ref):
    n_units, n_timepoints, n_channels = tc.shape

    max_channels = np.abs(tc).max(1).argmax(1)

    main_tc = np.zeros((n_units, n_timepoints))
    for j in range(n_units):
        main_tc[j] = np.abs(tc[j][:,max_channels[j]])
    
        
    best_shifts = align_get_shifts_tc(main_tc, ref, upsample_factor=1)
    shifted_tc = shift_chans(tc, best_shifts)

    return shifted_tc

def align_get_shifts_tc(wf, ref, upsample_factor = 5, nshifts = 21):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)

    wf_start = 15*upsample_factor
    wf_trunc = wf_up[:,wf_start:]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = np.roll(ref_upsampled, -s)[wf_start:]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)