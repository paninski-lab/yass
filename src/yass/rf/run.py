import h5py
import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm
import scipy.optimize as opt

from yass import read_config

def run():
    """RF computation
    """

    CONFIG = read_config()

    stim_movie_file = os.path.join(CONFIG.data.root_folder, CONFIG.data.stimulus)
    triggers_fname = os.path.join(CONFIG.data.root_folder, CONFIG.data.triggers)
    spike_train_fname = os.path.join(CONFIG.path_to_output_directory,
                                     'spike_train_post_deconv_post_merge.npy')
    saving_dir = os.path.join(CONFIG.path_to_output_directory, 'rf')
    
    rf = RF(stim_movie_file, triggers_fname, spike_train_fname, saving_dir)
    rf.calculate_STA()
    rf.fit_gaussian()


class RF(object):
    def __init__(self, stim_movie_file, triggers_fname, spike_train_fname, saving_dir):
        
        # default parameter
        self.n_color_channels = 3
        self.sp_frame_rate = 20000
        self.data_sample_len = 36000000 # len of white noise data (this script doesn't look at natural scenes)
        
        
        self.load_stimulus_trigger(stim_movie_file, triggers_fname)
        self.calculate_frame_times()
        self.load_spike_train(spike_train_fname)
        
        
        self.save_dir = saving_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if self.save_dir[-1] != '/':
            self.save_dir += '/'
             
        print("spike train:\t{}".format(self.sps.shape))
        print("stim movie:\t{}".format(self.WN_stim.shape))
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
        self.Ncells = np.unique(self.sps[:,1]).shape[0]    
    
    def calculate_STA(self):

        print('Calculating STA...')

        ############################################
        ## Get full STAs and spatial/temporal STA ##
        ############################################

        STA_temporal_length = 30 # how many bins/frames to include in STA
        Ncells = self.Ncells
        stim_size = self.stim_size
        n_color_channels = self.n_color_channels
        
        # Set up arrays
        STA_spatial = np.zeros((Ncells, n_color_channels,stim_size[0]*stim_size[1]))
        STA_temporal = np.zeros((Ncells,n_color_channels,STA_temporal_length))
        STA_spatial_colorcat = np.zeros((Ncells,stim_size[0]*stim_size[1]))
        STA_temporal_colorcat = np.zeros((Ncells,n_color_channels,STA_temporal_length))
        n_spikes = np.zeros((Ncells,))

        STA_array = []

        for i_cell in tqdm(range(Ncells)):

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
            # I don't use spikes in first 30 bins to build STA so don't have to deal with padding, probably ok anyway for ignoring transients

            ####################
            ### Calculate STA ##
            ####################

            ## Swap out fastest version here 
            STA = np.zeros((STA_temporal_length,n_color_channels,stim_size[0]*stim_size[1]))
            for i in range(which_spikes.shape[0]):
                bin_number = which_spikes[i]
                if binned_spikes[bin_number] == 1:
                    STA += self.WN_stim[bin_number-(STA_temporal_length-1):bin_number+1]
                else:
                    STA += binned_spikes[bin_number]*self.WN_stim[bin_number-(STA_temporal_length-1):bin_number+1]

            # full sta
            if np.sum(binned_spikes[STA_temporal_length:])>0:
                STA = STA/np.sum(binned_spikes[STA_temporal_length:])
            STA_array.append(STA)
            n_spikes[i_cell] = np.sum(binned_spikes[STA_temporal_length:])

            ###########################################
            ### Get separate spatial & temporal STAs ##
            ###########################################

            # Get spatial and temporal components of each color channel separately
            # second compnent taken
            for i_channel in range(n_color_channels):
                [U,S,V] = np.linalg.svd(STA[:,i_channel].reshape((STA_temporal_length,-1)))
                # I take the second component of the svd of the non-mean subtracted array because I've observed
                # that if I take the first component of the svd mean-subtracted array, there are sometimes errors 
                # (first component is nothing)

                # Align spatial and temporal components so one component always has positive max (so we can always know OFF vs ON cell from other component)
                ## UNCLEAR how to best align - I found this method lined the temporal filters up best which are clearer than the spatial
                temp_filt = U[20:,1]
                sign_mult = np.sign(np.sign(temp_filt[np.argmax(np.abs(temp_filt))]))
                STA_spatial[i_cell,i_channel] = sign_mult*V[1]
                STA_temporal[i_cell,i_channel] = sign_mult*U[:,1]

            [U,S,V] = np.linalg.svd(np.swapaxes(STA_array[i_cell],1,0).reshape((STA_temporal_length*3,-1)))
            temp_filt = U[50:60,1]
            sign_mult = np.sign(np.sign(temp_filt[np.argmax(np.abs(temp_filt))]))

            STA_temporal_colorcat[i_cell,0] = sign_mult*U[:30,1]
            STA_temporal_colorcat[i_cell,1] = sign_mult*U[30:60,1]
            STA_temporal_colorcat[i_cell,2] = sign_mult*U[60:,1]
            STA_spatial_colorcat[i_cell] = sign_mult*V[1]

        STA_array = np.asarray(STA_array)

        # Save arrays files
        np.save(self.save_dir+'STA_spatial.npy',STA_spatial)
        np.save(self.save_dir+'STA_temporal.npy',STA_temporal)
        np.save(self.save_dir+'STA.npy',STA_array)
        np.save(self.save_dir+'STA_spatial_colorcat.npy',STA_spatial_colorcat)
        np.save(self.save_dir+'STA_temporal_colorcat.npy',STA_temporal_colorcat)
        np.save(self.save_dir+'n_spikes.npy',n_spikes)

        
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