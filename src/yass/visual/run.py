import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yaml
from tqdm import tqdm
import parmap

from yass.deconvolve.correlograms_phy import compute_correlogram
from yass.deconvolve.notch import notch_finder
from yass.visual.util import compute_neighbours2, compute_neighbours_rf2, combine_two_spike_train, combine_two_rf
from yass.geometry import parse, find_channel_neighbors
from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans, binary_reader_waveforms, read_spikes
from yass.cluster.util import get_normalized_templates
from yass.cluster.util import pca_denoise
from yass.deconvolve.merge import (template_dist_linear_align, 
                                   template_spike_dist_linear_align, 
                                   test_unimodality, get_l2_features)
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
    deconv_dir = os.path.join(CONFIG.path_to_output_directory,
                              'deconv', 'final')

    vis = Visualizer(fname_templates, fname_spike_train,
                     fname_recording, recording_dtype, 
                     fname_geometry, sampling_rate, save_dir,
                     rf_dir, template_space_dir,
                     deconv_dir)

    vis.population_level_plot()
    vis.individiual_cell_plot()

        
class Visualizer(object):

    def __init__(self, fname_templates, fname_spike_train,
                 fname_recording, recording_dtype, 
                 fname_geometry, sampling_rate, save_dir,
                 rf_dir=None, template_space_dir=None,
                 deconv_dir=None, post_deconv_merge=True):
        
        # load spike train and templates
        self.spike_train = np.load(fname_spike_train)
        # load templates
        self.templates = np.load(fname_templates)

        # necessary numbers
        self.n_neighbours = 3
        _, self.n_channels, self.n_units = self.templates.shape
        self.sampling_rate = sampling_rate
 
        # rf files
        self.rf_dir = rf_dir
        if rf_dir is not None:
            self.STAs = np.load(os.path.join(rf_dir, 'STA_spatial.npy'))
            self.gaussian_fits = np.load(os.path.join(rf_dir, 'gaussian_fits.npy'))
            self.idx_single_rf = np.load(os.path.join(rf_dir, 'idx_single_rf.npy'))
            self.rf_labels = np.load(os.path.join(rf_dir, 'labels.npy'))
        
        # get geometry
        self.geom = parse(fname_geometry, self.n_channels)
        # compute neighboring channels
        self.neigh_channels = find_channel_neighbors(self.geom, 70)

        # location of standardized recording
        self.fname_recording = fname_recording
        self.recording_dtype = recording_dtype
        
        # residual recording
        self.deconv_dir = deconv_dir
        if deconv_dir is not None:
            self.residual_recording = os.path.join(deconv_dir, 'residual.bin')
            post_merge_loc = os.path.join(deconv_dir, 'results_post_deconv_post_merge_0.npz')
            if not os.path.exists(post_merge_loc) or (not post_deconv_merge):
                post_merge_loc = os.path.join(deconv_dir, 'results_post_deconv_pre_merge.npz')
            
            temp = np.load(post_merge_loc)
            self.templates_upsampled = temp['templates_upsampled']
            self.spike_train_upsampled = temp['spike_train_upsampled']

        # template space directory
        self.template_space_dir = template_space_dir
        if template_space_dir is not None:
            self.get_template_pca()
        
        # get colors
        self.colors = colors = [
            'black','blue','red','green','cyan','magenta','brown','pink',
            'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
            'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
            'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
            'darkmagenta','yellow','hotpink']

        # saving directory location
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # compute firing rates
        self.compute_firing_rates()

        # compute neighbors for each unit
        self.compute_neighbours()
        
        if rf_dir is not None:
            self.compute_neighbours_rf()

        # plotting parameters
        self.fontsize = 20
        self.figsize = [100, 40]

    def get_template_pca(self):
                    
        self.pca_main_components = np.load(os.path.join(
            self.template_space_dir, 'pca_main_components.npy'))
        self.pca_sec_components = np.load(os.path.join(
            self.template_space_dir, 'pca_sec_components.npy'))
        self.pca_main_mean = np.load(os.path.join(
            self.template_space_dir, 'pca_main_mean.npy'))
        self.pca_sec_mean = np.load(os.path.join(
            self.template_space_dir, 'pca_sec_mean.npy'))

    def compute_firing_rates(self):
        
        # COMPUTE FIRING RATES
        n_chans = self.n_channels
        samplerate = self.sampling_rate
        fp_len = np.memmap(self.fname_recording, dtype=self.recording_dtype, mode='r').shape[0]
        rec_len = fp_len/n_chans/samplerate

        # compute firing rates and ptps
        unique, n_spikes = np.unique(self.spike_train[:,1], return_counts=True)
        
        self.f_rates = np.zeros(self.templates.shape[2])
        self.f_rates[unique] = n_spikes/rec_len
        self.ptps = self.templates.ptp(0).max(0)

    def compute_neighbours(self):
        
        dist = template_dist_linear_align(self.templates.transpose(2,0,1))
        
        nearest_units = []
        for k in range(dist.shape[0]):
            idx = np.argsort(dist[k])[1:self.n_neighbours+1]
            nearest_units.append(idx)
        self.nearest_units = np.array(nearest_units)

    def compute_neighbours_rf(self):
        
        th = 0.002
        STAs_th = np.copy(self.STAs)
        STAs_th[np.abs(STAs_th) < th] = 0
        STAs_th = STAs_th.reshape(self.n_units, -1)
        
        norms = np.linalg.norm(STAs_th.T, axis=0)[:, np.newaxis]
        cos = np.matmul(STAs_th, STAs_th.T)/np.matmul(norms, norms.T)
        cos[np.isnan(cos)] = 0

        nearest_units_rf = np.zeros((self.n_units, self.n_neighbours), 'int32')
        for k in range(self.n_units):
            nearest_units_rf[k] = np.argsort(cos[k])[-self.n_neighbours-1:-1][::-1]

        self.nearest_units_rf = nearest_units_rf

    def population_level_plot(self):
                
        self.make_raster_plot()
        self.make_firing_rate_plot()
        self.make_normalized_templates_plot()
        
        if self.rf_dir is not None:
            self.make_rf_plots()
            self.cell_classification_plots()

        if self.deconv_dir is not None:
            self.add_residual_qq_plot()
            self.add_raw_deno_resid_plot()            

    def individiual_cell_plot(self, units=None):
        
        if units is None:
            units = np.arange(self.n_units)

        # saving directory location
        self.save_dir_ind = os.path.join(self.save_dir,'individual')
        if not os.path.exists(self.save_dir_ind):
            os.makedirs(self.save_dir_ind)

        #for unit in tqdm(units):
        parmap.map(self.make_individual_cell_plot, 
                   list(units),
                   processes=6,
                   pm_pbar=True)

    def make_individual_cell_plot(self, unit):

        fname = os.path.join(self.save_dir_ind, 'unit_{}.png'.format(unit))
        if os.path.exists(fname):
            return

        fig=plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(self.n_neighbours+5, 10, fig)  

        if np.sum(self.spike_train[:,1] == unit) > 0:
            ## Main Unit ##
            start_row = 0
            # add example waveforms
            gs, wf, idx = self.add_example_waveforms(gs, start_row, unit)
            start_row += 1

            # add denoised waveforms
            if self.template_space_dir is not None:
                gs = self.add_denoised_waveforms(gs, start_row, unit, wf)
                start_row += 1

            # add residual + template
            if self.deconv_dir is not None:
                gs = self.add_residual_template(gs, start_row, unit, idx)
                start_row += 1

            # add template
            gs = self.add_template_plot(gs, start_row, slice(2), 
                                        [unit], [self.colors[0]])
            # add rf
            gs = self.add_RF_plot(gs, start_row, 2, unit)

            # add autocorrelogram
            gs = self.add_xcorr_plot(gs, start_row, 3, unit, unit)

            # single unit raster plot
            gs = self.add_single_unit_raster(gs, start_row, slice(4,5), unit)

            start_row += 1

            ## Neighbor Units by templates ##
            neighbor_units = self.nearest_units[unit]
            for ctr, neigh in enumerate(neighbor_units):

                gs = self.add_template_plot(gs, ctr+start_row, slice(0,2), 
                                        np.hstack((unit, neigh)),
                                        [self.colors[c] for c in [0,ctr+1]])

                gs = self.add_RF_plot(gs, ctr+start_row, 2, neigh)

                gs = self.add_xcorr_plot(gs, ctr+start_row, 3, unit, neigh)

                if self.deconv_dir is not None:
                    gs = self.add_l2_feature_plot(gs, ctr+start_row, 4, unit, neigh,
                                                 [self.colors[c] for c in [0,ctr+1]])

            # add contour plots
            gs = self.add_contour_plot(
                gs, self.n_neighbours+start_row, 2,
                np.hstack((unit, neighbor_units)),
                self.colors[:self.n_neighbours+1])

            ## Neighbor Units by RF ##
            neighbor_units = self.nearest_units_rf[unit]
            for ctr, neigh in enumerate(neighbor_units):

                gs = self.add_template_plot(gs, ctr+start_row, slice(5,7), 
                                        np.hstack((unit, neigh)),
                                        [self.colors[c] for c in [0,ctr+1]])

                gs = self.add_RF_plot(gs, ctr+start_row, 7, neigh)

                gs = self.add_xcorr_plot(gs, ctr+start_row, 8, unit, neigh)

                if self.deconv_dir is not None:
                    gs = self.add_l2_feature_plot(gs, ctr+start_row, 9, unit, neigh,
                                                 [self.colors[c] for c in [0,ctr+1]])

            gs = self.add_contour_plot(
                gs, self.n_neighbours+start_row, 7,
                np.hstack((unit, neighbor_units)),
                self.colors[:self.n_neighbours+1])

        fig.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def pairwise_plot(self, pairs):

        # saving directory location
        save_dir_ind = os.path.join(self.save_dir,'pairs')
        if not os.path.exists(save_dir_ind):
            os.makedirs(save_dir_ind)

        max_pairs = 20
        count = -1
        n_pages = 1
        
        # plotting parameters
        self.fontsize = 20
        self.figsize = [60, 100]

        fig=plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(max_pairs, 6, fig)

        checked = np.zeros((self.n_units, self.n_units), 'bool')
        for ii in tqdm(range(len(pairs))):

            unit1 = pairs[ii][0]
            unit2 = pairs[ii][1]
            
            if not checked[unit1, unit2]:

                count += 1
                checked[unit1, unit2] = 1
                checked[unit2, unit1] = 1

                if count == max_pairs or ii == (len(pairs)-1):

                    fname = os.path.join(save_dir_ind, 'page_{}.png'.format(n_pages))
                    fig.savefig(fname, bbox_inches='tight', dpi=100)
                    plt.close()

                    if ii < len(pairs)-1:
                        fig=plt.figure(figsize=self.figsize)
                        gs = gridspec.GridSpec(max_pairs, 6, fig)

                        count = 0
                        n_pages += 1

                gs = self.add_template_plot(gs, count, slice(0,2), 
                                            np.hstack((unit1, unit2)),
                                            self.colors[:2])

                gs = self.add_RF_plot(gs, count, 2, unit1)
                gs = self.add_RF_plot(gs, count, 3, unit2)

                gs = self.add_xcorr_plot(gs, count, 4, unit1, unit2)

                if self.deconv_dir is not None:
                    gs = self.add_l2_feature_plot(gs, count, 5, unit1, unit2, self.colors[:2])
                    
        fname = os.path.join(save_dir_ind, 'page_{}.png'.format(n_pages))
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()
          

    def get_waveforms(self, unit, n_examples=500):
        
        idx = np.where(self.spike_train[:,1]==unit)[0]
        idx = np.random.choice(idx, 
                               np.min((n_examples, len(idx))),
                               False)
        spt = self.spike_train[idx,0]
        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]

        wf, _ = binary_reader_waveforms(self.fname_recording,
                                     self.n_channels,
                                     self.templates.shape[0],
                                     spt, neigh_chans)
        
        return wf, idx

    def add_example_waveforms(self, gs, x_loc, unit):
        
        wf, idx = self.get_waveforms(unit)
        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]
        
        for j, chan in enumerate(neigh_chans):
            ax = plt.subplot(gs[x_loc, j])
            ax.plot(wf[:, :, j].T, color='k', alpha=0.1)
            ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
            title = "Channel: {}".format(chan)
            if chan == mc:
                title += ', MAX CHAN'
            if j == 0:
                title = 'Raw Waveforms, Unit: {}, '.format(unit) + title
            ax.set_title(title, fontsize=self.fontsize)
        
        return gs, wf, idx

    def add_denoised_waveforms(self, gs, x_loc, unit, wf=None):
        
        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]

        if wf is None:
            wf, idx = self.get_waveforms(unit)
        
        norms = np.linalg.norm(wf, axis=1)[:,np.newaxis]
        normalized_wf = wf/norms
        denoised_wf = np.zeros(wf.shape)
        for ii in range(wf.shape[2]):
            if neigh_chans[ii] == mc:
                temp = pca_denoise(normalized_wf[:,:,ii],
                                   self.pca_main_mean,
                                   self.pca_main_components)
            else:
                temp = pca_denoise(normalized_wf[:,:,ii],
                                   self.pca_sec_mean,
                                   self.pca_sec_components)
            denoised_wf[:,:,ii] = temp*norms[:,:,ii]

        for j, chan in enumerate(neigh_chans):
            ax = plt.subplot(gs[x_loc, j])
            ax.plot(denoised_wf[:, :, j].T, color='k', alpha=0.1)
            ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
            if j == 0:
                title = 'Denoised Waveforms'
                ax.set_title(title, fontsize=self.fontsize)
        
        return gs
    
    def add_residual_template(self, gs, x_loc, unit, idx=None):

        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]

        if idx is None:
            _, idx = self.get_waveforms(unit)

        spt = self.spike_train[idx, 0]
        units = self.spike_train_upsampled[idx, 1]

        wf, _ = read_spikes(
            self.residual_recording, spt, self.n_channels,
            self.templates.shape[0], units,
            self.templates_upsampled.transpose(2,0,1),
            channels=neigh_chans,
            residual_flag=True)            

        for j, chan in enumerate(neigh_chans):
            ax = plt.subplot(gs[x_loc, j])
            ax.plot(wf[:, :, j].T, color='k', alpha=0.1)
            ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
            if j == 0:
                title = 'Residual + Template'
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
        
        img = self.STAs[unit,:,:,1].T
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
                                                  self.gaussian_fits)
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
        angle = Gaussian_params[i_cell,4]
        sd = Gaussian_params[i_cell,2:4]
        x_shift=Gaussian_params[i_cell,0]
        y_shift = Gaussian_params[i_cell,1]

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

        notch, pval1, pval2 = notch_finder(result)
        pval1 = np.round(pval1, 2)
        pval2 = np.round(pval2, 2)

        ax = plt.subplot(gs[x_loc, y_loc])
        plt.plot(result,color='black', linewidth=2)
        plt.ylim(0, np.max(result*1.5))
        plt.plot([50,50],[0,np.max(result*1.5)],'r--')
        plt.xlim(0,101)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.title('pval1: {}, pval2: {}'.format(pval1, pval2), fontsize=self.fontsize)
       
        return gs
    
    
    def add_l2_feature_plot(self, gs, x_loc, y_loc, unit1, unit2, colors):
        
        #n_samples = 5000
        l2_features, spike_ids = get_l2_features(
            self.residual_recording, self.spike_train,
            self.spike_train_upsampled,
            self.templates.transpose(2,0,1),
            self.templates_upsampled.transpose(2,0,1),
            unit1, unit2)
        try:
            dp_val, feat = test_unimodality(l2_features, spike_ids)

            #l2_1d_features = np.diff(l2_features, axis=0)[0]
            n_bins = int(len(feat)/20)
            steps = (np.max(feat) - np.min(feat))/n_bins
            bins = np.arange(np.min(feat), np.max(feat)+steps, steps)

            ax = plt.subplot(gs[x_loc, y_loc])
            plt.hist(feat, bins, color='slategrey')
            plt.hist(feat[spike_ids==0], bins, color=colors[0], alpha=0.7)
            plt.hist(feat[spike_ids==1], bins, color=colors[1], alpha=0.7)
            plt.title(
                'Dip Test: {}'.format(np.round(dp_val,4)), 
                fontsize=self.fontsize)
        except:
            print ("Diptest error for unit {} and {} with size {}".format(
                unit1, unit2, l2_features.shape[0]))

        return gs

    def make_normalized_templates_plot(self):
        
        fname = os.path.join(self.save_dir, 'normalized_templates.png')
        if os.path.exists(fname):
            return

        if self.template_space_dir is not None:
            ref_template = np.load(
                os.path.join(self.template_space_dir,
                             'ref_template.npy'))
            
            pca_main = self.pca_main_components
            pca_sec = self.pca_sec_components

            add_row = 1
        else:
            # template with the largest amplitude will be ref_template
            ref_template = self.templates[:, :, self.ptps.argmax()]
            mc = ref_template.ptp(0).argmax()
            ref_template = ref_template[:, mc]
            
            add_row = 0
        
        (templates_mc, templates_sec, 
         ptp_mc, ptp_sec, _) = get_normalized_templates(
            self.templates.transpose(2, 0, 1), 
            self.neigh_channels, ref_template)
        
        plt.figure(figsize=(14, 4*(add_row+2)))

        ths = [0,2,3]
        for ii, th in enumerate(ths):
            plt.subplot(2+add_row, 3, ii+1)

            if ii < 2:
                idx = np.logical_and(ptp_mc >= th, ptp_mc < ths[ii+1])
            else:
                idx = ptp_mc >= th
            if sum(idx) > 0:
                plt.plot(templates_mc[idx].T, color='k', alpha=0.2)

            if ii < 2:
                plt.xlabel('Main Chan., {} < PTP < {}'.format(th, ths[ii+1]), fontsize=self.fontsize//2)
            else:
                plt.xlabel('Main Chan., {} < PTP'.format(th), fontsize=self.fontsize//2)
                
        for ii, th in enumerate(ths):
            plt.subplot(2+add_row, 3, ii+4)

            if ii < 2:
                idx = np.logical_and(ptp_sec >= th, ptp_sec < ths[ii+1])
            else:
                idx = ptp_sec >= th

            if sum(idx) > 0:
                plt.plot(templates_sec[idx].T, color='k', alpha=0.05)

            if ii < 2:
                plt.xlabel('Sec. Chan., {} < PTP < {}'.format(th, ths[ii+1]), fontsize=self.fontsize//2)
            else:
                plt.xlabel('Sec. Chan., {} < PTP'.format(th), fontsize=self.fontsize//2)

        if add_row == 1:
            plt.subplot(3, 3, 7)
            plt.plot(pca_main.T, color='k')
            plt.xlabel('PCs for main chan. denoise', fontsize=self.fontsize//2)
            
            plt.subplot(3, 3, 8)
            plt.plot(pca_sec.T, color='k')
            plt.xlabel('PCs for sec chan. denoise', fontsize=self.fontsize//2)
            
    
        plt.suptitle('Aligned Templates on Their Main/Secondary Channels', fontsize=20)
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def make_raster_plot(self):
        
        fname = os.path.join(self.save_dir, 'raster.png')
        if os.path.exists(fname):
            return
        
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
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def make_firing_rate_plot(self):
        
        fname = os.path.join(self.save_dir, 'firing_rates.png')
        if os.path.exists(fname):
            return
        
        plt.figure(figsize=(20,10))
        plt.scatter(np.log(self.ptps), self.f_rates, color='k')
        plt.xlabel('log ptps', fontsize=self.fontsize)
        plt.ylabel('firing rates', fontsize=self.fontsize)
        plt.title('Firing Rates vs. PTP', fontsize=self.fontsize)
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()
        
    def add_residual_qq_plot(self):
        
        fname = os.path.join(self.save_dir, 'res_qq_plot.png')
        if os.path.exists(fname):
            return
        
        plt.figure(figsize=(20,23))
        nrow = int(np.sqrt(self.n_channels))
        ncol = int(np.ceil(self.n_channels/nrow))
        
        sample_size = 10000
        
        res = np.memmap(self.residual_recording, 
                        dtype='float32', mode='r')
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

    def add_raw_deno_resid_plot(self):
        
        fname = os.path.join(self.save_dir, 'raw_denoised_residual.png')
        #if os.path.exists(fname):
        #    return

        raw = np.memmap(self.fname_recording, 
                        dtype='float32', mode='r')
        raw = np.reshape(raw, [-1, self.n_channels])
        res = np.memmap(self.residual_recording, 
                        dtype='float32', mode='r')
        res = np.reshape(res, [-1, self.n_channels])
        
        t_max = np.abs(res[1000:-1000]).max(1).argmax() + 1000

        t_start = t_max - 500
        t_end = t_max + 500

        c_max = res[t_max].argmax()
        dist_to_max_c = np.linalg.norm(self.geom - self.geom[c_max][None], axis=1)
        channels = dist_to_max_c.argsort()[:20]

        idx_temp = np.where(np.logical_and(
            self.spike_train_upsampled[:,0] < t_end, self.spike_train_upsampled[:,0] > t_start-61))[0]
        spike_train_temp = self.spike_train_upsampled[idx_temp]

        t_tics = np.arange(t_start, t_end)/20
        summed_templates = np.zeros((t_end-t_start+61*2, len(channels)))
        for j in range(spike_train_temp.shape[0]):
            tt, ii = spike_train_temp[j]
            tt -= (t_start-61)
            summed_templates[tt:tt+61] += self.templates_upsampled[:,channels][:,:,ii]
        summed_templates = summed_templates[61:-61]
        
        raw = raw[t_start:t_end][:, channels]
        res = res[t_start:t_end][:, channels]

        spread = np.arange(len(channels))*30
        plt.figure(figsize=(40,10))
        ax = plt.subplot(111)
        ax.set_facecolor((0.1,0.1,0.1))
        ax.plot(t_tics,raw+spread[None], 'r')
        ax.plot(t_tics,summed_templates+spread[None], 'b')
        ax.plot(t_tics,res+spread[None], 'white')
        ax.set_xlabel('Time (ms)', fontsize=30)
        ax.set_xlim([min(t_tics), max(t_tics)])

        labels_legend=[]
        colors = ['r','b','white']
        names = ['raw recording', 'denoised', 'residual']
        for k in range(3):
            patch_j = mpatches.Patch(color=colors[k], label=names[k])
            labels_legend.append(patch_j)

        plt.legend(handles = labels_legend, bbox_to_anchor=[1,1], fontsize=20)
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close() 
        
    def cell_classification_plots(self):

        fname = os.path.join(self.save_dir, 'contours.png')
        if os.path.exists(fname):
            return
        
        type_names = ['off-midget','off-parasol',
               'off-sm',
               'on-midget','on-parasol',
               'unknown','multiple/no rf']

        idx_per_type = []
        for j in range(5):
            idx_per_type.append(np.where(self.rf_labels == j)[0])
        idx_per_type.append(self.idx_single_rf[self.rf_labels[self.idx_single_rf] == -1]) 
        all_units = np.arange(self.n_units)
        idx_per_type.append(all_units[~np.in1d(all_units, self.idx_single_rf)])

        plt.figure(figsize=(22,6))
        for ii, idx in enumerate(idx_per_type):
            plt.subplot(1,7,ii+1)
            for unit in idx:
                # also plot all in one plot
                plotting_data = self.get_circle_plotting_data(unit, self.gaussian_fits)
                plt.plot(plotting_data[0],plotting_data[1], 'k')
                plt.xlim([0,32])
                plt.ylim([0,64])
            plt.title(type_names[ii])
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()         
        
    def make_rf_plots(self):

        fname = os.path.join(self.save_dir, 'all_rfs.png')
        if os.path.exists(fname):
            return

        n_units = self.STAs.shape[0]
        
        type_names = ['off-midget','off-parasol',
                       'off-sm',
                       'on-midget','on-parasol',
                       'unknown','multiple/no rf']

        idx_per_type = []
        for j in range(5):
            idx_per_type.append(np.where(self.rf_labels == j)[0])
        idx_per_type.append(self.idx_single_rf[self.rf_labels[self.idx_single_rf] == -1]) 
        all_units = np.arange(n_units)
        idx_per_type.append(all_units[~np.in1d(all_units, self.idx_single_rf)])
        
        n_rows_per_type = []
        for idx in idx_per_type:
            n_rows_per_type.append(int(np.ceil(len(idx)/10.)))
        
        fig=plt.figure(figsize=(50, 100))
        gs = gridspec.GridSpec(sum(n_rows_per_type)+7, 10, fig)

        row = 0
        for ii, idx in enumerate(idx_per_type):

            col = 0

            # add label
            ax = plt.subplot(gs[row, 0])
            plt.text(0.5, 0.5, type_names[ii],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=40,
                     transform=ax.transAxes)
            ax.set_axis_off()
            
            row += 1

            idx_sort = idx[np.argsort(self.ptps[idx])[::-1]]
            for unit in idx_sort:
                gs = self.add_RF_plot(gs, row, col, unit)
                if col == 9:
                    col = 0
                    row += 1
                else:
                    col += 1
            if col != 0:
                row += 1

        fig.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()


class CompareSpikeTrains(Visualizer):
    
    def __init__(self, fname_templates, fname_spike_train, rf_dir,
                 fname_recording, recording_dtype, 
                 fname_geometry, sampling_rate, save_dir,
                 fname_templates2=None, fname_spike_train2=None, rf_dir2=None,
                 fname_set1_idx=None):
        
        if fname_set1_idx == None and (fname_templates2==None or fname_spike_train2==None or rf_dir2==None):
            raise ValueError('something is not right!!')
        
        # TODO: Finish it!!!
        if fname_set1_idx == None:
            # saving directory location
            tmp_dir = os.path.join(save_dir,'tmp')
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                
            templates1 = np.load(fname_templates)
            spike_train1 = np.load(fname_spike_train)
            templates2 = np.load(fname_templates2)
            spike_train2 = np.load(fname_spike_train2)
            
            STAs1 = np.load(os.path.join(rf_dir, 'STA_spatial.npy'))
            Gaussian_params1 = np.load(os.path.join(rf_dir, 'gaussian_fits.npy'))
            STAs2 = np.load(os.path.join(rf_dir2, 'STA_spatial.npy'))
            Gaussian_params2 = np.load(os.path.join(rf_dir2, 'gaussian_fits.npy'))
            
            idx_single_rf1 = np.load(os.path.join(rf_dir, 'idx_single_rf.npy'))
            rf_labels1 = np.load(os.path.join(rf_dir, 'labels.npy'))
            
            idx_single_rf2 = np.load(os.path.join(rf_dir2, 'idx_single_rf.npy'))
            rf_labels2 = np.load(os.path.join(rf_dir2, 'labels.npy'))
            
            templates, spike_train = combine_two_spike_train(
                templates1, templates2, spike_train1, spike_train2)
            
            STAs, Gaussian_params = combine_two_rf(
                STAs1, STAs2, Gaussian_params1, Gaussian_params2)

            K1 = templates1.shape[2]
            K2 = templates2.shape[2]
            set1_idx = np.zeros(K1+K2, 'bool')
            set1_idx[:K1] = 1
            
            idx_single_rf = np.hstack((idx_single_rf1, idx_single_rf2+K1))
            rf_labels = np.hstack((rf_labels1, rf_labels2))
            
            
            fname_templates = os.path.join(tmp_dir, 'templates_combined.npy')
            fname_spike_train = os.path.join(tmp_dir, 'spike_train_combined.npy')

            rf_dir = tmp_dir

            fname_set1_idx = os.path.join(tmp_dir, 'set1_idx.npy')
            
            np.save(fname_templates, templates)
            np.save(fname_spike_train, spike_train)
            np.save(fname_set1_idx, set1_idx)
            np.save(os.path.join(rf_dir, 'STA_spatial.npy'), STAs)
            np.save(os.path.join(rf_dir, 'gaussian_fits.npy'), Gaussian_params)    
            np.save(os.path.join(rf_dir, 'idx_single_rf.npy'), idx_single_rf)
            np.save(os.path.join(rf_dir, 'labels.npy'), rf_labels)
        
        set1_idx = np.load(fname_set1_idx)
            
        Visualizer.__init__(self, fname_templates, fname_spike_train,
                 fname_recording, recording_dtype, 
                 fname_geometry, sampling_rate, save_dir, rf_dir)
        
        self.set1_idx = set1_idx
        
        # fix nearest units!
        self.fix_nearest_units()
        

    def fix_nearest_units(self):
        
        templates1 = self.templates[:, :, self.set1_idx].transpose(2,0,1)
        templates2 = self.templates[:, :, ~self.set1_idx].transpose(2,0,1)
        
        STAs1 = self.STAs[self.set1_idx]
        STAs2 = self.STAs[~self.set1_idx]
        
        nearest_units1, nearest_units2 = compute_neighbours2(
            templates1, templates2, self.n_neighbours)
        nearest_units_rf1, nearest_units_rf2 = compute_neighbours_rf2(
            STAs1, STAs2, self.n_neighbours)
        
        set1_idx = np.where(self.set1_idx)[0]
        set2_idx = np.where(~self.set1_idx)[0]
        
        nearest_units = np.copy(self.nearest_units)
        nearest_units_rf = np.copy(self.nearest_units_rf)
        
        for k in range(self.n_units):
            
            if np.any(set1_idx==k):
                ii = np.where(set1_idx == k)[0]
                temp = nearest_units1[ii]
                nearest_units[k] = set2_idx[temp]
                
                temp = nearest_units_rf1[ii]
                nearest_units_rf[k] = set2_idx[temp]
            else:
                ii = np.where(set2_idx == k)[0]
                temp = nearest_units2[ii]
                nearest_units[k] = set1_idx[temp]
                
                temp = nearest_units_rf2[ii]
                nearest_units_rf[k] = set1_idx[temp]
        
        self.nearest_units = nearest_units
        self.nearest_units_rf = nearest_units_rf
