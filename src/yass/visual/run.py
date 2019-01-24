import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yaml
from tqdm import tqdm

from yass.visual.correlograms_phy import compute_correlogram
from yass.geometry import parse, find_channel_neighbors
from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans, binary_reader_waveforms
from yass.util import absolute_path_to_asset
from yass import read_config

def run():
    """Visualization Package
    """

    CONFIG = read_config()

    fname_templates = os.path.join(CONFIG.path_to_output_directory,
                                     'templates_post_deconv_post_merge.npy')
    fname_spike_train = os.path.join(CONFIG.path_to_output_directory,
                                     'spike_train_post_deconv_post_merge.npy')
    rf_dir = os.path.join(CONFIG.path_to_output_directory, 'rf')
    fname_recording = os.path.join(CONFIG.path_to_output_directory,
                                   'preprocess',
                                   'standardized.bin')
    fname_recording_yaml = os.path.join(CONFIG.path_to_output_directory,
                                        'preprocess',
                                        'standardized.yaml')
    with open(fname_recording_yaml, 'r') as stream:
        data_loaded = yaml.load(stream)
    recording_dtype = data_loaded['dtype']
    fname_geometry = os.path.join(CONFIG.data.root_folder, CONFIG.data.geometry)
    sampling_rate = CONFIG.recordings.sampling_rate
    save_dir = os.path.join(CONFIG.path_to_output_directory, 'visualize')
    
    # only for yass
    template_space_dir = absolute_path_to_asset('template_space')
    fname_residual_recording = os.path.join(CONFIG.path_to_output_directory,
                                            'deconv', 'final', 'residual.bin')
    residual_dtype = 'float32'

    vis = Visualizer(fname_templates, fname_spike_train, rf_dir,
                     fname_recording, recording_dtype, 
                     fname_geometry, sampling_rate, save_dir,
                     template_space_dir,
                     fname_residual_recording, residual_dtype)

    vis.population_level_plot()
    vis.individiual_cell_plot()


class Visualizer(object):

    def __init__(self, fname_templates, fname_spike_train, rf_dir,
                 fname_recording, recording_dtype, 
                 fname_geometry, sampling_rate, save_dir,
                 template_space_dir=None,
                 fname_residual_recording=None, residual_dtype=None):
        
        # load spike train and templates
        self.spike_train = np.load(fname_spike_train)
        self.templates = np.load(fname_templates)

        # necessary numbers
        self.n_neighbours = 3
        _, self.n_channels, self.n_units = self.templates.shape
        self.sampling_rate = sampling_rate
 
        # rf files
        self.STAs = np.load(os.path.join(rf_dir, 'STA_spatial.npy'))
        self.Gaussian_params = np.load(os.path.join(rf_dir, 'Gaussian_params.npy'))
        
        # get geometry
        self.geom = parse(fname_geometry, self.n_channels)
        
        # location of standardized recording
        self.fname_recording = fname_recording
        self.recording_dtype = recording_dtype
        
        # residual recording
        self.residual_recording = fname_residual_recording
        self.residual_dtype = residual_dtype

        # template space directory
        self.template_space_dir = template_space_dir
        
        # get colors
        self.colors = colors = [
            'black','blue','red','green','cyan','magenta','brown','pink',
            'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
            'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
            'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
            'darkmagenta','yellow','hotpink']

        # saving directory location
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
            
            
        # compute firing rates
        self.compute_firing_rates()

        # compute neighbors for each unit
        self.compute_neighbours()
        self.compute_neighbours_rf()
            
    def compute_firing_rates(self):
        
        # COMPUTE FIRING RATES
        n_chans = self.n_channels
        samplerate = self.sampling_rate
        fp_len = np.memmap(self.fname_recording, dtype=self.recording_dtype, mode='r').shape[0]
        rec_len = fp_len/n_chans/samplerate

        # compute firing rates and ptps
        unique, n_spikes = np.unique(self.spike_train[:,1], return_counts=True)
        self.f_rates = n_spikes/rec_len
        self.ptps = self.templates.ptp(0).max(0)
        
    def compute_neighbours(self):
        
        dist = self.template_dist_linear_align()
        
        nearest_units = []
        for k in range(dist.shape[0]):
            idx = np.argsort(dist[k])[1:self.n_neighbours+1]
            nearest_units.append(idx)
        self.nearest_units = np.array(nearest_units)
        
    def template_dist_linear_align(self):

        templates = self.templates.transpose(2,0,1)
        
        # find template shifts
        max_chans = templates.ptp(1).argmax(1)

        # new way using alignment only on max channel
        # maek reference template based on templates
        max_idx = templates.ptp(1).max(1).argmax(0)
        ref_template = templates[max_idx, :,max_chans[max_idx]]

        temps = []
        for k in range(max_chans.shape[0]):
            temps.append(templates[k, :, max_chans[k]])
        temps = np.vstack(temps)

        best_shifts = align_get_shifts_with_ref(temps, ref_template)    
        templates_aligned = shift_chans(templates, best_shifts)

        n_unit = templates_aligned.shape[0]
        templates_reshaped = templates_aligned.reshape([n_unit, -1])
        dist = scipy.spatial.distance.cdist(templates_reshaped, templates_reshaped)

        return dist
    
    def compute_neighbours_rf(self):
        
        th = 0.05
        STAs_th = np.copy(self.STAs)
        STAs_th[np.abs(STAs_th) < th] = 0
        STAs_th = STAs_th.reshape(self.n_units, -1)
        
        norms = np.linalg.norm(STAs_th.T, axis=0)[:, np.newaxis]
        cos = np.matmul(STAs_th, STAs_th.T)/np.matmul(norms, norms.T)
        
        nearest_units_rf = np.zeros((self.n_units, self.n_neighbours), 'int32')
        for k in range(self.n_units):
            nearest_units_rf[k] = np.argsort(cos[k])[-self.n_neighbours-1:-1][::-1]

        self.nearest_units_rf = nearest_units_rf

    def population_level_plot(self):
                
        self.make_raster_plot()
        self.make_firing_rate_plot()
        self.make_normalized_templates_plot()
        
    def individiual_cell_plot(self):
        
        # saving directory location
        save_dir_ind = os.path.join(self.save_dir,'individual')
        if not os.path.isdir(save_dir_ind):
            os.makedirs(save_dir_ind)

        for unit in tqdm(range(self.n_units)):
            
            # plotting parameters
            self.fontsize = 20
            self.figsize = [80, 40]
        
            fig=plt.figure(figsize=self.figsize)
            gs = gridspec.GridSpec(self.n_neighbours+4, 8, fig)  
            
            ## Main Unit ##
            # add template
            gs = self.add_template_plot(gs, 0, slice(2), 
                                        [unit], [self.colors[0]])
            # add rf
            gs = self.add_RF_plot(gs, 0, 2, unit)
            
            # add autocorrelogram
            gs = self.add_xcorr_plot(gs, 0, 3, unit, unit)
            
            # add example waveforms
            gs = self.add_example_waveforms(gs, 0, 4, unit)
            
            # single unit raster plot
            gs = self.add_single_unit_raster(gs, 0, slice(5,7), unit)
            
            
            ## Neighbor Units by templates ##
            neighbor_units = self.nearest_units[unit]
            for ctr, neigh in enumerate(neighbor_units):
                
                gs = self.add_template_plot(gs, ctr+1, slice(0,2), 
                                        np.hstack((unit, neigh)),
                                        [self.colors[c] for c in [0,ctr+1]])
                
                gs = self.add_RF_plot(gs, ctr+1, 2, neigh)
                
                gs = self.add_xcorr_plot(gs, ctr+1, 3, unit, neigh)

            # add contour plots
            gs = self.add_contour_plot(
                gs, self.n_neighbours+1, 2,
                np.hstack((unit, neighbor_units)),
                self.colors[:self.n_neighbours+1])
            
            ## Neighbor Units by RF ##
            neighbor_units = self.nearest_units_rf[unit]
            for ctr, neigh in enumerate(neighbor_units):
                
                gs = self.add_template_plot(gs, ctr+1, slice(4,6), 
                                        np.hstack((unit, neigh)),
                                        [self.colors[c] for c in [0,ctr+1]])
                
                gs = self.add_RF_plot(gs, ctr+1, 6, neigh)

                gs = self.add_xcorr_plot(gs, ctr+1, 7, unit, neigh)
                
            gs = self.add_contour_plot(
                gs, self.n_neighbours+1, 6,
                np.hstack((unit, neighbor_units)),
                self.colors[:self.n_neighbours+1])
            
            fname = os.path.join(save_dir_ind, 'unit_{}.png'.format(unit))
            fig.savefig(fname, bbox_inches='tight', dpi=100)
            plt.close()
            
            
    def add_example_waveforms(self, gs, x_loc, y_loc, unit):
        
        # default parameter
        n_examples = 100
        
        idx = np.where(self.spike_train[:,1]==unit)[0]
        spt = self.spike_train[idx,0]
        spt = np.random.choice(spt, n_examples, False)
        spike_size = self.templates.shape[0]
        mc = self.templates[:, :, unit].ptp(0).argmax()
        
        wf, _ = binary_reader_waveforms(self.fname_recording,
                                     self.n_channels,
                                     spike_size,
                                     spt, np.array([mc]))
        wf = wf[:, :, 0]
        
        ax = plt.subplot(gs[x_loc, y_loc])
        ax.plot(wf.T, color='k', alpha=0.1)
        title = "Unit: {}, Max Channel: {}".format(unit, mc)
        ax.set_title(title, fontsize=self.fontsize)
        
        return gs
    

    def add_template_plot(self, gs, x_loc, y_loc, units, colors):
        
        # plotting parameters
        time_scale=3.
        scale=10.
        alpha=0.4
        
        R = self.templates.shape[0]
        
        ax = plt.subplot(gs[x_loc, y_loc])
        
        for ii, unit in enumerate(units):
            ax.plot(self.geom[:,0]+np.arange(-R,0)[:,np.newaxis]/time_scale, 
             self.geom[:,1] + self.templates[:,:,unit]*scale,
             color=colors[ii], alpha=alpha)

        # add channel number
        for k in range(self.n_channels):
            ax.text(self.geom[k,0]+1, self.geom[k,1], str(k), fontsize=self.fontsize)

        # add +-1 su grey shade 
        x = np.arange(-R,0)/time_scale
        y = -np.ones(x.shape)*scale
        for k in range(self.n_channels):
            ax.fill_between(x+self.geom[k,0], 
                            y+self.geom[k,1], 
                            y+2*scale+ self.geom[k,1], color='grey', alpha=0.1)
                
        return gs
    
    
    def add_RF_plot(self, gs, x_loc, y_loc, unit):
        # COMPUTE 
        #neighbor_units = self.nearest_units[unit]
        
        ax = plt.subplot(gs[x_loc, y_loc])
        
        ax.set_title("Unit: "+str(unit)+", "+str(np.round(self.f_rates[unit],1))+
             "Hz, "+str(np.round(self.ptps[unit],1))+"SU", fontsize=self.fontsize)
        
        img = self.STAs[unit,1].reshape(64,32).T
        vmax = np.max(np.abs(img))
        vmin = -vmax
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_ylim(0,32)
        
        # also plot all in one plot
        #ax = plt.subplot(gs[self.n_neighbours+1, ax_col])
        #ax = self.plot_contours(ax, np.hstack((unit,neighbor_units)),
        #                        self.colors[:self.n_neighbours+1])

        return gs
    
    
    def add_contour_plot(self, gs, x_loc, y_loc, units, colors):
        
        ax = plt.subplot(gs[x_loc, y_loc])
        labels = []
        
        for ii, unit in enumerate(units):
            # also plot all in one plot
            plotting_data = self.get_circle_plotting_data(unit,
                                                  self.Gaussian_params)
            ax.plot(plotting_data[1],plotting_data[0],
                    color=colors[ii], linewidth=3)

            labels.append(mpatches.Patch(color = colors[ii], label = "Unit {}".format(unit)))
        
        ax.legend(handles=labels)
        ax.set_ylim(0,32)
        ax.set_xlim(0,64)
                
        return gs

    
    def add_single_unit_raster(self, gs, x_loc, y_loc, unit):
        
        ax = plt.subplot(gs[x_loc, y_loc])
        
        idx = self.spike_train[:, 1] == unit
        spt = self.spike_train[idx, 0]/self.sampling_rate
            
        plt.eventplot(spt, color='k', linewidths=0.01)
        plt.title('Single Unit Raster Plot', fontsize=self.fontsize)
        
        return gs
                        
    
    def get_circle_plotting_data(self,i_cell,Gaussian_params):
        # Adapted from Nora's matlab code, hasn't been tripled checked

        circle_samples = np.arange(0,2*np.pi,0.05)
        x_circle = np.cos(circle_samples)
        y_circle = np.sin(circle_samples)

        # Get Gaussian parameters
        angle = Gaussian_params[i_cell,5]
        sd = Gaussian_params[i_cell,3:5]
        x_shift=Gaussian_params[i_cell,1]
        y_shift = Gaussian_params[i_cell,2]

        R = np.asarray([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
        L = np.asarray([[sd[0], 0],[0, sd[1]]])
        circ = np.concatenate([x_circle.reshape((-1,1)),y_circle.reshape((-1,1))],axis=1)

        X = np.dot(R,np.dot(L,np.transpose(circ)))
        X[0] = X[0]+x_shift
        X[1] = np.abs(X[1]+y_shift)
        plotting_data = X

        return plotting_data

    
    def add_xcorr_plot(self, gs, x_loc, y_loc, unit1, unit2):
        # COMPUTE XCORRS w. neighbouring units;
        
        result = compute_correlogram(
            np.hstack((unit1, unit2)), self.spike_train)
        if result.shape[0] == 1:
            result = result[0,0]
        elif result.shape[0] > 1:
            result = result[1,0]
        
        ax = plt.subplot(gs[x_loc, y_loc])
        plt.plot(result,color='black', linewidth=2)
        plt.ylim(0, np.max(result*1.5))
        plt.plot([50,50],[0,np.max(result*1.5)],'r--')
        plt.xlim(0,101)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=6)
       
        return gs
    
    
    def add_l2features_plots(self):
        # CMOPUTE L2 PLOTS w. NEIGHBOURS
        affinity_only = True
        recompute = False
        jitter=upsample=5
        iteration = 0

        # compute affinity matrix
        merge = TemplateMerge(self.templates,self.spike_train,jitter, upsample, self.CONFIG, self.save_dir, 
                              iteration, recompute, affinity_only)
        merge.filename_residual = self.residual_dir+'/residual.bin'

        # compute l2 distance across pairs:
        for ctr, comparison_unit in enumerate(self.units[1:]):

            unit1 = self.unit
            unit2 = comparison_unit
            
            print (unit1, unit2)
            # compute l2 features
            merge.merge_templates_parallel([unit1,unit2])
            # load features from disk
            fname = os.path.join( self.save_dir + 'l2features_'+str(unit1)+'_'+str(unit2)+'_'+str(0)+'.npz')
            data= np.load(fname)
            features =data['l2_features']
            spike_ids = data['spike_ids']    
            dp_val = data['dp_val']
            # plot results
            clr_array=[]
            for i in range(spike_ids.shape[0]):
                clr_array.append(colors[0])

            idx = np.where(spike_ids==0)[0]
            for id_ in idx:
                clr_array[id_]=self.colors[ctr+1]

            ax = plt.subplot(self.gs[1:2,ctr+1:ctr+2])

            ax.scatter(features[0,:], features[1,:], c=clr_array, alpha=.5)
            ax.plot([0,np.max(features[0])], [0,np.max(features[1])],'r--')
            ax.set_title("DP: "+str(np.round(dp_val,3)),fontsize=12)
        

    def get_normalized_templates(self, templates, neigh_channels, ref_template):

        """
        plot normalized templates on their main channels and secondary channels
        templates: number of channels x temporal window x number of units
        geometry: number of channels x 2
        """

        K, R, C = templates.shape
        mc = np.argmax(templates.ptp(1), 1)

        # get main channel templates
        templates_mc = np.zeros((K, R))
        for k in range(K):
            templates_mc[k] = templates[k, :, mc[k]]

        # shift templates_mc
        best_shifts_mc = align_get_shifts_with_ref(
                        templates_mc,
                        ref_template)
        templates_mc = shift_chans(templates_mc, best_shifts_mc)
        ptp_mc = templates_mc.ptp(1)

        # normalize templates
        norm_mc = np.linalg.norm(templates_mc, axis=1, keepdims=True)
        templates_mc /= norm_mc

        # get secdonary channel templates
        templates_sec = np.zeros((0, R))
        for k in range(K):
            neighs = np.copy(neigh_channels[mc[k]])
            neighs[mc[k]] = False
            neighs = np.where(neighs)[0]
            templates_sec = np.concatenate((templates_sec, templates[k, :, neighs]), axis=0)

        # shift templates_sec
        best_shifts_sec = align_get_shifts_with_ref(
                        templates_sec,
                        ref_template)
        templates_sec = shift_chans(templates_sec, best_shifts_sec)
        ptp_sec = templates_sec.ptp(1)

        # normalize templates
        norm_sec = np.linalg.norm(templates_sec, axis=1, keepdims=True)
        templates_sec /= norm_sec

        return templates_mc, templates_sec, ptp_mc, ptp_sec

    def make_normalized_templates_plot(self):

        if self.template_space_dir is not None:
            ref_template = np.load(
                os.path.join(self.template_space_dir,
                             'ref_template.npy'))
            
            pca_main = np.load(os.path.join(
                self.template_space_dir, 'pca_main_components.npy'))
            pca_sec = np.load(os.path.join(
                self.template_space_dir, 'pca_sec_components.npy'))
            
            add_row = 1
        else:
            # template with the largest amplitude will be ref_template
            ref_template = self.templates[:, :, self.ptps.argmax()]
            mc = ref_template.ptp(0).argmax()
            ref_template = ref_template[:, mc]
            
            add_row = 0
        
        neigh_channels = find_channel_neighbors(self.geom, 70)
        
        (templates_mc, templates_sec, 
         ptp_mc, ptp_sec) = self.get_normalized_templates(
            self.templates.transpose(2, 0, 1), 
            neigh_channels, ref_template)
        
        plt.figure(figsize=(23, 5*(add_row+1)))
        plt.subplot(1+add_row,4,1)
        plt.plot(templates_mc.T, color='k', alpha=0.02)
        plt.xlabel('Main Channel', fontsize=self.fontsize//2)

        ths = [0,1,3]
        for j in range(3):
            for ii, th in enumerate(ths):
                plt.subplot(1+add_row, 4, ii+2)
                idx = ptp_sec > th
                plt.plot(templates_sec[idx].T, color='k', alpha=0.02)
                plt.xlabel('Sec. Chan., PTP > {}'.format(th), fontsize=self.fontsize//2)
                
        if add_row == 1:
            plt.subplot(2, 4, 5)
            plt.plot(pca_main.T, color='k')
            plt.xlabel('PCs for main chan. denoise', fontsize=self.fontsize//2)
            
            plt.subplot(2, 4, 6)
            plt.plot(pca_sec.T, color='k')
            plt.xlabel('PCs for sec chan. denoise', fontsize=self.fontsize//2)
            
    
        plt.suptitle('Aligned Templates on Their Main/Secondary Channels', fontsize=30)
        fname = os.path.join(self.save_dir, 'normalized_templates.png')
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def make_raster_plot(self):
        plt.figure(figsize=(30,15))
        ptps = self.ptps
        order = np.argsort(ptps)
        sorted_ptps = np.round(np.sort(ptps),2)
        for j in range(self.n_units):
            k = order[j]
            idx = self.spike_train[:,1] == k
            spt = self.spike_train[idx, 0]/self.sampling_rate
            plt.eventplot(spt, lineoffsets=j, color='k', linewidths=0.01)
        plt.yticks(np.arange(0,self.n_units,10), sorted_ptps[0:self.n_units:10])
        plt.ylabel('ptps', fontsize=self.fontsize)
        plt.xlabel('time (seconds)', fontsize=self.fontsize)
        plt.title('rater plot sorted by PTP', fontsize=self.fontsize)
        fname = os.path.join(self.save_dir, 'raster.png')
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def make_firing_rate_plot(self):
        plt.figure(figsize=(20,10))
        plt.scatter(np.log(self.ptps), self.f_rates, color='k')
        plt.xlabel('log ptps', fontsize=self.fontsize)
        plt.ylabel('firing rates', fontsize=self.fontsize)
        plt.title('Firing Rates vs. PTP', fontsize=self.fontsize)
        fname = os.path.join(self.save_dir, 'firing_rates.png')
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()
        
    def add_residual_qq_plot(self):
        
        plt.figure(figsize=(20,23))
        nrow = int(np.sqrt(self.n_channels))
        ncol = int(np.ceil(self.n_channels/nrow))
        
        sample_size = 10000

        res = np.memmap(self.residual_recording, 
                        dtype=self.residual_dtype, mode='r')
        res = np.reshape(res, [-1, self.n_channels])
        
        th = np.sort(np.random.normal(size = sample_size))
        for c in range(self.n_channels):
            qq = np.sort(np.random.choice(
                res[:, c], sample_size, False))
            
            plt.subplot(nrow, ncol, c+1)
            plt.scatter(th, qq, s=5)
            min_val = np.min((th.min(), qq.min()))
            max_val = np.max((th.max(), qq.max()))
            plt.plot([min_val, max_val], [min_val, max_val], color='r')
            plt.title('Channel: {}'.format(c))
        
        fname = os.path.join(self.save_dir, 'res_qq_plot.png')
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()