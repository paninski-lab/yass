import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.axes_grid1.colorbar import colorbar
import yaml
from tqdm import tqdm
import parmap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import torch

from yass.correlograms_phy import compute_correlogram
from yass.merge.notch import notch_finder
from yass.visual.util import *
from yass.geometry import parse, find_channel_neighbors
from yass.template import align_get_shifts_with_ref, shift_chans
from yass.merge.merge import (template_dist_linear_align, 
                              template_spike_dist_linear_align, 
                              test_unimodality)
from yass.util import absolute_path_to_asset
from yass import read_config
from yass.reader import READER

def run():
    """Visualization Package
    """

    CONFIG = read_config()

    fname_templates = os.path.join(CONFIG.path_to_output_directory,
                                     'templates.npy')
    fname_spike_train = os.path.join(CONFIG.path_to_output_directory,
                                     'spike_train.npy')
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
                 CONFIG, save_dir, rf_dir=None,
                 fname_residual=None, residual_dtype=None,
                 fname_soft_assignment=None):

        # saving directory location
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.tmp_folder = os.path.join(self.save_dir, 'tmp')
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
        
        # necessary numbers
        self.n_neighbours = 3
        self.sampling_rate = CONFIG.recordings.sampling_rate
        self.geom = CONFIG.geom
        self.neigh_channels = CONFIG.neigh_channels

        # load templates
        self.templates = np.load(fname_templates)
        if len(self.geom) == self.templates.shape[2]:
            self.templates = self.templates.transpose(1, 2, 0)
        self.n_times_templates, self.n_channels, self.n_units = self.templates.shape
        # compute neighbors for each unit
        self.compute_neighbours()
        self.compute_propagation()

        # load spike train and templates
        self.spike_train = np.load(fname_spike_train)
        self.unique_ids = np.unique(self.spike_train[:,1])
        if fname_soft_assignment is not None:
            self.soft_assignment = np.load(fname_soft_assignment)
        else:
            self.soft_assignment = np.ones(self.spike_train.shape[0], 'float32')

        # compute firing rates
        self.compute_firing_rates()
        self.compute_xcorrs()

        # recording readers
        self.reader = READER(fname_recording, recording_dtype, CONFIG, 1)
        # change spike size just in case
        self.reader.spike_size = self.n_times_templates

        if fname_residual is not None:
            self.reader_resid = READER(fname_residual, residual_dtype, CONFIG, 1)
            self.reader_resid.spike_size = self.n_times_templates
        else:
            self.reader_resid = None

        # rf files
        self.rf_dir = rf_dir
        if rf_dir is not None:
            self.STAs = np.load(os.path.join(rf_dir, 'STA_spatial.npy'))[:, :, :, 1]
            self.STAs = np.flip(self.STAs, axis=2)
            self.STAs_temporal = np.load(os.path.join(rf_dir, 'STA_temporal.npy'))
            self.gaussian_fits = np.load(os.path.join(rf_dir, 'gaussian_fits.npy'))

            self.cell_types = list(np.load(os.path.join(rf_dir, 'cell_types.npy')))
            self.cell_types += ['No-Rf', 'Multiple-Rf']

            if os.path.exists(os.path.join(rf_dir, 'labels_updated.npy')):
                self.rf_labels = np.load(os.path.join(rf_dir, 'labels_updated.npy'))
            else:
                self.idx_single_rf = np.load(os.path.join(rf_dir, 'idx_single_rf.npy'))
                self.idx_no_rf = np.load(os.path.join(rf_dir, 'idx_no_rf.npy'))
                self.idx_multi_rf = np.load(os.path.join(rf_dir, 'idx_multi_rf.npy'))

                self.rf_labels = np.load(os.path.join(rf_dir, 'labels.npy'))
                self.rf_labels[self.idx_multi_rf] = len(self.cell_types) - 1
                self.rf_labels[self.idx_no_rf] = len(self.cell_types) - 2

            self.stim_size = np.load(os.path.join(rf_dir, 'stim_size.npy'))

            max_t = np.argmax(np.abs(self.STAs_temporal[:,:,1]), axis=1)
            self.sta_sign = np.sign(self.STAs_temporal[
                np.arange(self.STAs_temporal.shape[0]),max_t, 1])

            # also compute rf
            self.compute_neighbours_rf()

        # get colors
        self.colors = colors = [
            'black','blue','red','green','cyan','magenta','brown','pink',
            'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
            'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
            'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
            'darkmagenta','yellow','hotpink']
        self.cmap = cm = plt.cm.get_cmap('RdYlBu')
        
    def compute_propagation(self):
        ptps = self.templates.ptp(0)
        shifts = np.zeros((self.n_units, self.n_channels))
        for k in range(self.n_units):
            temp = self.templates[:, :, k]
            mc = temp.ptp(0).argmax()
            arg_min = temp.argmin(0)
            arg_min -= arg_min[mc]

            vis_chan = np.where(temp.ptp(0) > 0.5)[0]
            shifts[k][vis_chan] = arg_min[vis_chan]
        
        #shifts[shifts < -5] = -5
        self.shifts = shifts
        self.max_shift = np.max(self.shifts)
        self.min_shift = np.min(self.shifts)
        self.min_shift = -10

    def compute_firing_rates(self):
        
        # COMPUTE FIRING RATES
        n_chans = self.n_channels
        samplerate = self.sampling_rate
        self.rec_len = np.ptp(self.spike_train[:, 0])/samplerate

        n_spikes_soft = np.zeros(self.n_units)
        for j in range(self.spike_train.shape[0]):
            n_spikes_soft[self.spike_train[j, 1]] += self.soft_assignment[j]
        n_spikes_soft = n_spikes_soft.astype('int32')

        self.f_rates = n_spikes_soft/self.rec_len
        self.ptps = self.templates.ptp(0).max(0)

    def compute_xcorrs(self):
        self.window_size = 0.04
        self.bin_width = 0.001

        fname = os.path.join(self.tmp_folder, 'xcorrs.npy')
        if os.path.exists(fname):
            self.xcorrs = np.load(fname)
        else:
            self.xcorrs = compute_correlogram(
                np.arange(self.n_units),
                self.spike_train,
                self.soft_assignment,
                sample_rate=self.sampling_rate,
                bin_width=self.bin_width,
                window_size=self.window_size)
            np.save(fname, self.xcorrs)
        
        avg_frates = (self.f_rates[self.unique_ids][:, None]+self.f_rates[self.unique_ids][None])/2
        self.xcorrs = self.xcorrs/avg_frates[:,:,None]/self.rec_len/self.bin_width

    def compute_neighbours(self):

        fname = os.path.join(self.tmp_folder, 'neighbours.npy')
        if os.path.exists(fname):
            self.nearest_units = np.load(fname)
        else:
            dist = template_dist_linear_align(self.templates.transpose(2,0,1))
            nearest_units = []
            for k in range(dist.shape[0]):
                idx = np.argsort(dist[k])[1:self.n_neighbours+1]
                nearest_units.append(idx)
            self.nearest_units = np.array(nearest_units)

            np.save(fname, self.nearest_units)

    def compute_neighbours_rf(self):

        std = np.median(np.abs(
            self.STAs - np.median(self.STAs)))/0.6745
        self.rf_std = std

        fname = os.path.join(self.tmp_folder, 'neighbours_rf.npy')
        if os.path.exists(fname):
            self.nearest_units_rf = np.load(fname)

        else:
            th = std*0.5

            STAs_th = np.copy(self.STAs)
            STAs_th[np.abs(STAs_th) < th] = 0
            STAs_th = STAs_th.reshape(self.n_units, -1)
            STAs_th = STAs_th#*self.sta_sign[:, None]

            norms = np.linalg.norm(STAs_th.T, axis=0)[:, np.newaxis]
            cos = np.matmul(STAs_th, STAs_th.T)/np.matmul(norms, norms.T)
            cos[np.isnan(cos)] = 0

            nearest_units_rf = np.zeros((self.n_units, self.n_neighbours), 'int32')
            for j in np.unique(self.rf_labels):
                units_same_class = np.where(self.rf_labels == j)[0]
                if len(units_same_class) > self.n_neighbours+1:
                    for k in units_same_class:
                        idx_ = np.argsort(cos[k][units_same_class])[::-1][1:self.n_neighbours+1]
                        nearest_units_rf[k] = units_same_class[idx_]
                else:
                    for k in units_same_class:
                        other_units = units_same_class[units_same_class != k]
                        nearest_units_rf[k][:len(other_units)] = other_units
                        nearest_units_rf[k][len(other_units):] = k


            self.nearest_units_rf = nearest_units_rf
            np.save(fname, self.nearest_units_rf)

    def compute_neighbours_xcorrs(self, unit):
        xcorrs = self.xcorrs[np.where(
            self.unique_ids == unit)[0][0]]
        xcorrs = xcorrs[self.unique_ids != unit]
        idx_others = self.unique_ids[self.unique_ids != unit]

        sig_xcorrs = np.where(xcorrs.sum(1) > 10/(self.rec_len*self.bin_width*self.f_rates[unit]))[0]
        xcorrs = xcorrs[sig_xcorrs]
        idx_others = idx_others[sig_xcorrs]

        means_ = xcorrs.mean(1)
        stds_ = np.std(xcorrs, 1)
        stds_[stds_==0] = 1
        xcorrs = (xcorrs - means_[:,None])/stds_[:,None]

        idx_max = np.argsort(xcorrs.max(1))[::-1][:self.n_neighbours]
        max_vals = xcorrs.max(1)[idx_max]
        idx_max = idx_others[idx_max]


        idx_min = np.argsort(xcorrs.min(1))[:self.n_neighbours]
        min_vals = xcorrs.min(1)[idx_min]
        idx_min = idx_others[idx_min]

        return idx_max, max_vals, idx_min, min_vals

    def population_level_plot(self):

        self.fontsize = 20
        self.make_raster_plot()
        self.make_firing_rate_plot()
        self.make_normalized_templates_plot()

        if self.rf_dir is not None:
            #self.make_rf_plots()
            self.cell_classification_plots()
        else:
            self.make_all_templates_summary_plots()

        if self.reader_resid is not None:
            #self.residual_varaince()
            self.add_residual_qq_plot()
            self.add_raw_resid_snippets()

    def individiual_level_plot(self, units_full_analysis=None, sample=False,
                               plot_all=True, plot_summary=True, divide_by_cell_types=True):

        # saving directory location
        self.save_dir_ind = os.path.join(
            self.save_dir, 'individual')
        if not os.path.exists(self.save_dir_ind):
            os.makedirs(self.save_dir_ind)

        if divide_by_cell_types:
            for cell_type in self.cell_types:
                dir_tmp = os.path.join(
                    self.save_dir_ind, cell_type)
                if not os.path.exists(dir_tmp):
                    os.makedirs(dir_tmp)

        if plot_summary:
            self.make_all_rf_templates_plots()

        # which units to do full analysis
        if units_full_analysis is None:
            units_full_analysis = np.arange(self.n_units)
        else:
            units_full_analysis = np.array(units_full_analysis)

        # random sample if requested
        n_sample = 100
        if sample and (len(units_full_analysis) > n_sample):
            fname_units = os.path.join(
                self.tmp_folder,
                'units_full_analysis_individual_plot.npy')
            if os.path.exists(fname_units):
                units_full_analysis = np.load(fname_units)
            else:
                units_full_analysis = np.random.choice(
                    units_full_analysis, n_sample, False)
                np.save(fname_units, units_full_analysis)
        full_analysis = np.zeros(self.n_units, 'bool')
        full_analysis[units_full_analysis] = True
        full_analysis = list(full_analysis)

        names = []
        if plot_all:
            units_in = np.arange(self.n_units)
        else:
            units_in = np.copy(units_full_analysis)

            
        # file names
        for unit in units_in:
            ptp = str(int(np.round(self.ptps[unit]))).zfill(3)
            name = 'ptp_{}_unit_{}'.format(ptp, unit)
            names.append(name)
        
        if False:
            parmap.map(self.make_individiual_level_plot,
                       list(units_in),
                       names,
                       full_analysis[units_in],
                       processes=3,
                       pm_pbar=True)
        else:
            for ii in tqdm(range(len(units_in))):
                self.make_individiual_level_plot(units_in[ii],
                                                 names[ii],
                                                 full_analysis[units_in[ii]],
                                                 divide_by_cell_types
                                                )

    def make_individiual_level_plot(self,
                                    unit,
                                    name,
                                    full_analysis=True,
                                    divide_by_cell_types=True
                                   ):

        # cell type
        cell_type = self.cell_types[self.rf_labels[unit]]

        if divide_by_cell_types:
            # save directory
            save_dir = os.path.join(self.save_dir_ind, cell_type)
        else:
            save_dir = self.save_dir_ind

        # template
        fname = os.path.join(save_dir, name+'_p0_template.png')
        self.make_template_plot(unit, fname)

        if full_analysis:
            # waveform plots
            fname = os.path.join(save_dir, name+'_p1_wfs.png')
            self.make_waveforms_plot(unit, fname)

            # template neighbors plots
            neighbor_units = self.nearest_units[unit]
            fname = os.path.join(save_dir, name+'_p2_temp_neigh.png')
            title = 'Unit {} ({}), Template Space Neighbors'.format(
                unit, cell_type)
            self.make_neighbors_plot(unit, neighbor_units, fname, title)

            # rf neighbors plots
            if np.max(np.abs(self.STAs[unit])) > 1.5*self.rf_std:
                neighbor_units = self.nearest_units_rf[unit]
                fname = os.path.join(save_dir, name+'_p3_rf_neigh.png')
                title = 'Unit {} ({}), RF Space Neighbors'.format(
                    unit, cell_type)
                self.make_neighbors_plot(unit, neighbor_units, fname, title)

            # xcorr neighbours
            if self.f_rates[unit] > 0.5:
                (idx_max, max_vals,
                 idx_min, min_vals) = self.compute_neighbours_xcorrs(unit)
                fname = os.path.join(save_dir, name+'_p4_high_xcorr_neigh.png')
                title = 'Unit {} ({}), Xcor Space Neighbors'.format(
                    unit, cell_type)
                self.make_neighbors_plot(unit, idx_max, fname, title)

                if np.min(min_vals) < -10:
                    fname = os.path.join(save_dir, name+'_p5_xcorr_notches.png')
                    title = 'Unit {} ({}), Xcor Notches'.format(
                        unit, cell_type)
                    self.make_neighbors_plot(unit, idx_min, fname, title)


    def make_template_plot(self, unit, fname):

        if os.path.exists(fname):
            return

        # determin channels to include
        ptp = self.templates[:, :, unit].ptp(0)
        mc = ptp.argmax()
        vis_chan = np.where(ptp > 2)[0]
        if len(vis_chan) == 0:
            vis_chan = [mc]
        geom_vis_chan = self.geom[vis_chan]
        max_x, max_y = np.max(geom_vis_chan, 0)
        min_x, min_y = np.min(geom_vis_chan, 0)
        chan_idx = np.logical_and(
            np.logical_and(self.geom[:,0] >= min_x-3, self.geom[:,0] <= max_x+3),
            np.logical_and(self.geom[:,1] >= min_y-3, self.geom[:,1] <= max_y+3))
        chan_idx = np.where(chan_idx)[0]
        chan_idx = chan_idx[ptp[chan_idx] > 1]
        # also include neighboring channels
        neigh_chans = np.where(self.neigh_channels[mc])[0]
        chan_idx = np.unique(np.hstack((chan_idx, neigh_chans)))

        # plotting parameters
        self.fontsize = 40
        fig = plt.figure(figsize=[30, 10])
        gs = gridspec.GridSpec(1, 1, fig)

        # add template summary plot
        cell_type = self.cell_types[self.rf_labels[unit]]
        fr = str(np.round(self.f_rates[unit], 1))
        ptp = str(np.round(self.ptps[unit], 1))
        title = "Template of Unit {}, {}Hz, {}SU, Max Channel: {}".format(unit, fr, ptp, mc)

        gs = self.add_template_plot(gs, 0, 0,
                        [unit], [self.colors[0]],
                        chan_idx, title)

        plt.tight_layout()
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        fig.clf()
        plt.close('all')
        gs = None


    def make_waveforms_plot(self, unit, fname):

        if os.path.exists(fname):
            return

        n_waveforms = 1000
        n_examples = 100
        fontsize = 20

        wf, wf_resid, spt, neigh_chans = self.get_waveforms(unit, n_waveforms)
        if wf.shape[0] == 0:
            return

        template = self.templates[:,:,unit][:, neigh_chans]
        spt = spt/self.sampling_rate

        spikes_ptp = self.get_spikes_ptp(wf, template)
        spikes_ptp_clean = self.get_spikes_ptp(wf_resid, template)
        template_ptp = template[:, template.ptp(0).argmax()].ptp()

        n_rows = 3
        if self.reader_resid is not None:
            n_rows += 2
        n_cols = len(neigh_chans)

        chan_order = np.argsort(template.ptp(0))[::-1]
        if n_examples < wf.shape[0]:
            idx_plot = np.random.choice(wf.shape[0], n_examples, False)
        else:
            idx_plot = np.arange(wf.shape[0])


        plt.figure(figsize=(n_cols*4, n_rows*4))
        # add raw wfs
        count = 0
        x_range = np.arange(wf.shape[1])/self.sampling_rate*1000
        for j, c in enumerate(chan_order):
            count += 1
            plt.subplot(n_rows, n_cols, count)
            plt.plot(x_range, wf[:, :, c][idx_plot].T, color='k', alpha=0.1)
            plt.plot(x_range, template[:,c], color='r', linewidth=2)
            title = "Channel: {}".format(neigh_chans[c])
            if j == 0:
                title = 'Raw Waveforms\n' + title
            plt.title(title, fontsize=fontsize)

            if j == 0:
                plt.xlabel('Time (ms)', fontsize=fontsize)
                plt.ylabel('Voltage (S.U.)', fontsize=fontsize)

        if wf_resid is None:
            count = 2
        else:
            # add clean wfs
            for j, c in enumerate(chan_order):
                count += 1
                plt.subplot(n_rows, n_cols, count)
                plt.plot(x_range, wf_resid[:, :, c][idx_plot].T, color='k', alpha=0.1)
                plt.plot(x_range, template[:,c], color='r', linewidth=2)
                if j == 0:
                    title = 'Clean Waveforms'
                    plt.title(title, fontsize=fontsize)
                    plt.xlabel('Time (ms)', fontsize=fontsize)
                    plt.ylabel('Voltage (S.U.)', fontsize=fontsize)


            # add residual variance
            residual_std = np.std(wf_resid, axis=0)
            max_std = np.max(residual_std)
            min_std = np.min(residual_std)
            for j, c in enumerate(chan_order):
                count += 1
                plt.subplot(n_rows, n_cols, count)
                plt.plot(x_range, residual_std[:, c], color='k', linewidth=2)
                plt.ylim([0.95*min_std, 1.05*max_std])
                if j == 0:
                    title = 'STD. of Residuals'
                    plt.title(title, fontsize=fontsize)
                    plt.xlabel('Time (ms)', fontsize=fontsize)
                    plt.ylabel('STD', fontsize=fontsize)
            count = 4

        # ptp vs spike times
        plt.subplot(n_rows, 1, count)
        plt.scatter(spt, spikes_ptp, c='k')
        plt.plot([np.min(spt), np.max(spt)], [template_ptp, template_ptp], 'r')
        plt.title('ptp vs spike times (red = template ptp)\n Raw Waveforms',
                  fontsize=fontsize)
        plt.xlabel('Time (seconds)', fontsize=fontsize)
        plt.ylabel('PTP (S.U.)', fontsize=fontsize)

        # ptp vs spike times (clean waveforms)
        plt.subplot(n_rows, 1, count+1)
        plt.scatter(spt, spikes_ptp_clean, c='k')
        plt.plot([np.min(spt), np.max(spt)], [template_ptp, template_ptp], 'r')
        plt.title('Clean Waveforms', fontsize=fontsize)
        plt.xlabel('Time (seconds)', fontsize=fontsize)
        plt.ylabel('PTP (S.U.)', fontsize=fontsize)

        # suptitle
        fr = np.round(float(self.f_rates[unit]), 1)
        ptp = np.round(float(self.ptps[unit]), 1)
        suptitle = 'Unit: {}, {}Hz, {}SU'.format(unit, fr, ptp)
        suptitle = suptitle + ', ' + self.cell_types[self.rf_labels[unit]]
        plt.suptitle(suptitle,
                     fontsize=int(1.5*fontsize))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
        plt.close('all')


    def make_neighbors_plot(self, unit, neighbor_units, fname, title):

        if os.path.exists(fname):
            return

        mc = self.templates[:, :, unit].ptp(0).argmax()
        chan_idx = np.where(self.neigh_channels[mc])[0]

        zoom_windows = zoom_in_window(self.STAs, unit, self.rf_std*1.5)
        if zoom_windows is not None:
            col_minus = 0
        else:
            col_minus = 1

        # plotting parameters
        self.fontsize = 30
        self.figsize = [int(6*(8-col_minus)), 25]

        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(title, fontsize=2*self.fontsize)

        gs = gridspec.GridSpec(self.n_neighbours+2, 8-col_minus, fig,
                               left=0, right=1, top=0.92, bottom=0.05,
                               hspace=0.2, wspace=0.1)

        start_row = 0

        # add template summary plot
        fr = str(np.round(self.f_rates[unit], 1))
        ptp = str(np.round(self.ptps[unit], 1))
        title = "Unit: {}, {}Hz, {}SU".format(unit, fr, ptp)
        gs = self.add_template_summary(
            gs, start_row, slice(2), unit, title)

        # add template
        title = 'Zoomed-in Templates'
        gs = self.add_template_plot(gs, start_row, 2,
                                    [unit], [self.colors[0]],
                                    chan_idx, title)

        # add rf
        title = 'Spatial RF'
        gs = self.add_RF_plot(gs, start_row, 3, unit, None, title)
        if zoom_windows is not None:
            title = 'Zoomed-in Spatial RF'
            gs = self.add_RF_plot(gs, start_row, 4,
                                  unit, zoom_windows, title)

        # add temporal sta
        title = 'Temporal RF'
        gs = self.add_temporal_sta(gs, start_row, 5-col_minus,
                                   unit, title)

        # add autocorrelogram
        title = 'Autocorrelogram'
        gs = self.add_xcorr_plot(gs, start_row, 6-col_minus, unit, unit, title)

        start_row += 1
        ## Neighbor Units by templates ##
        for ctr, neigh in enumerate(neighbor_units):
            fr = str(np.round(self.f_rates[neigh], 1))
            ptp = str(np.round(self.ptps[neigh], 1))
            title = "Unit: {}, {}Hz, {}SU".format(neigh, fr, ptp)
            gs = self.add_template_summary(
                gs, ctr+start_row, slice(2), neigh, title)

            gs = self.add_template_plot(
                gs, ctr+start_row, 2,
                np.hstack((unit, neigh)),
                [self.colors[c] for c in [0,ctr+1]],
                chan_idx
            )

            gs = self.add_RF_plot(gs, ctr+start_row, 3, neigh)
            if zoom_windows is not None:
                gs = self.add_RF_plot(gs, ctr+start_row, 4, neigh, zoom_windows)

            if ctr == len(neighbor_units)-1:
                add_label = True
            else:
                add_label = False

            gs = self.add_temporal_sta(
                gs, ctr+start_row, 5-col_minus, neigh, None, add_label)

            if ctr == 0:
                title = 'Cross-correlogram'
            else:
                title = None
            gs = self.add_xcorr_plot(gs, ctr+start_row, 6-col_minus,
                                     unit, neigh, title, add_label)

            if self.reader_resid is not None:
                if ctr == 0:
                    title = 'Histogram of\nLDA Projection of\nSpikes-to-Templates\nDistance'
                else:
                    title = None
                if ctr == len(neighbor_units)-1:
                    add_label = True
                else:
                    add_label = False
                gs = self.add_l2_feature_plot(gs, ctr+start_row, 7-col_minus, unit, neigh,
                                              [self.colors[c] for c in [0,ctr+1]],
                                              title, add_label)

        # add contour plots
        title = 'Contours of Spatial RF'
        gs = self.add_contour_plot(
            gs, self.n_neighbours+start_row, 3,
            np.hstack((unit, neighbor_units)),
            self.colors[:self.n_neighbours+1], None, title)
        if zoom_windows is not None:
            title = 'zoomed-in Contours'
            gs = self.add_contour_plot(
                gs, self.n_neighbours+start_row, 4,
                np.hstack((unit, neighbor_units)),
                self.colors[:self.n_neighbours+1],
                True, title)

        #plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        fig.clf()
        plt.close('all')
        gs = None

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

                gs = self.add_template_plot(gs, count, slice(0,2), 
                                            np.hstack((unit1, unit2)),
                                            self.colors[:2])

                gs = self.add_RF_plot(gs, count, 2, unit1)
                gs = self.add_RF_plot(gs, count, 3, unit2)

                gs = self.add_xcorr_plot(gs, count, 4, unit1, unit2)

                if self.fname_residual is not None:
                    gs = self.add_l2_feature_plot(gs, count, 5, unit1, unit2, self.colors[:2])


                if count == max_pairs or ii == (len(pairs)-1):

                    fname = os.path.join(save_dir_ind, 'page_{}.png'.format(n_pages))
                    fig.savefig(fname, bbox_inches='tight', dpi=100)
                    plt.close()

                    if ii < len(pairs)-1:
                        fig=plt.figure(figsize=self.figsize)
                        gs = gridspec.GridSpec(max_pairs, 6, fig)

                        count = 0
                        n_pages += 1
                    
        fname = os.path.join(save_dir_ind, 'page_{}.png'.format(n_pages))
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        fig.clf()
        fig.cla()
        plt.close('all')


    def get_waveforms(self, unit, n_examples=200):

        idx = np.where(self.spike_train[:,1]==unit)[0]
        spt = self.spike_train[idx, 0]
        prob = self.soft_assignment[idx]
        
        if np.sum(prob) < 1:
            return np.zeros((0,1,1)), None, None, None

        spt = np.random.choice(spt, 
                               np.min((n_examples, int(np.sum(prob)))),
                               False, prob/np.sum(prob))

        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]
        temp = self.templates[:, :, unit][:, mc]

        wf, skipped_idx = self.reader.read_waveforms(spt, self.n_times_templates, neigh_chans)
        mc_neigh = np.where(neigh_chans == mc)[0][0]
        shifts = align_get_shifts_with_ref(
                    wf[:, :, mc_neigh], temp)
        wf = shift_chans(wf, shifts)
        spt = np.delete(spt, skipped_idx)

        if self.reader_resid is not None:
            wf_resid, _ = self.reader_resid.read_waveforms(spt, self.n_times_templates, neigh_chans)
            wf_resid = wf_resid + self.templates[:, :, unit][:, neigh_chans][None]
            shifts = align_get_shifts_with_ref(
                    wf_resid[:, :, mc_neigh], temp)
            wf_resid = shift_chans(wf_resid, shifts)
        else:
            wf_resid = None

        return wf, wf_resid, spt, neigh_chans


    def get_clean_waveforms(self, unit, n_examples=200, spt=None):

        if spt is None:
            idx = np.where(self.spike_train[:,1]==unit)[0]
            idx = np.random.choice(idx,
                                   np.min((n_examples, len(idx))),
                                   False)
            spt = self.spike_train[idx,0] - self.templates.shape[0]//2
        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]

        wf_res, idx_skipped = binary_reader_waveforms(
            self.fname_residual,
            self.n_channels,
            self.templates.shape[0],
            spt, neigh_chans)
        spt = np.delete(spt, idx_skipped)

        wf = wf_res + self.templates[:, :, unit][:, neigh_chans]

        mc_neigh = np.where(neigh_chans == mc)[0][0]
        shifts = align_get_shifts_with_ref(
                    wf[:, :, mc_neigh])
        wf = shift_chans(wf, shifts)

        return wf, spt

    def add_example_waveforms(self, gs, x_loc, unit):
        
        wf, spt = self.get_waveforms(unit)

        if wf.shape[0] > 0:
            mc = self.templates[:, :, unit].ptp(0).argmax()
            neigh_chans = np.where(self.neigh_channels[mc])[0]

            order_neigh_chans = np.argsort(
                np.linalg.norm(self.geom[neigh_chans] - self.geom[[mc]], axis=1))
            for ii, j in enumerate(order_neigh_chans):
                chan = neigh_chans[j]
                ax = plt.subplot(gs[x_loc, ii])
                ax.plot(wf[:, :, j].T, color='k', alpha=0.1)
                ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
                title = "Channel: {}".format(chan)
                if ii == 0:
                    title = 'Raw Waveforms, Unit: {}, '.format(unit) + title
                ax.set_title(title, fontsize=self.fontsize)
        
        return gs, wf, spt

    def add_denoised_waveforms(self, gs, x_loc, unit, wf=None):

        if wf is None:
            wf, spt = self.get_waveforms(unit)

        if wf.shape[0] > 0:
            n_data, n_times, n_chans = wf.shape

            denoised_wf = np.zeros((n_data, n_times, n_chans))

            n_times_deno = 61
            n_times_diff = (n_times - n_times_deno)//2
            wf = wf[:, n_times_diff:-n_times_diff]

            wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times_deno)
            wf_torch = torch.FloatTensor(wf_reshaped).cuda()
            denoised_wf_short = self.denoiser(wf_torch)[0].reshape(
                n_data, n_chans, n_times_deno)
            denoised_wf_short = denoised_wf_short.cpu().data.numpy().transpose(0, 2, 1)
            denoised_wf[:, n_times_diff:-n_times_diff] = denoised_wf_short

            mc = self.templates[:, :, unit].ptp(0).argmax()
            neigh_chans = np.where(self.neigh_channels[mc])[0]

            order_neigh_chans = np.argsort(
                np.linalg.norm(self.geom[neigh_chans] - self.geom[[mc]], axis=1))
            for ii, j in enumerate(order_neigh_chans):
                chan = neigh_chans[j]
                ax = plt.subplot(gs[x_loc, ii])
                ax.plot(denoised_wf[:, :, j].T, color='k', alpha=0.1)
                ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
                if ii == 0:
                    title = 'Denoised Waveforms'
                    ax.set_title(title, fontsize=self.fontsize)

        return gs
    
    def add_residual_template(self, gs, x_loc, unit, spt=None):

        mc = self.templates[:, :, unit].ptp(0).argmax()
        neigh_chans = np.where(self.neigh_channels[mc])[0]

        wf, spt = self.get_clean_waveforms(unit, spt=spt)

        if wf.shape[0] > 0:
            order_neigh_chans = np.argsort(
                np.linalg.norm(self.geom[neigh_chans] - self.geom[[mc]], axis=1))
            for ii, j in enumerate(order_neigh_chans):
                chan = neigh_chans[j]
                ax = plt.subplot(gs[x_loc, ii])
                ax.plot(wf[:, :, j].T, color='k', alpha=0.1)
                ax.plot(self.templates[:, chan, unit].T, color='r', linewidth=2)
                if ii == 0:
                    title = 'Residual + Template'
                    ax.set_title(title, fontsize=self.fontsize)

        return gs


    def determine_channels_in(self, unit, n_max_chans = 30):

        temp = self.templates[:, :, unit]
        ptp = temp.ptp(0)
        mc = ptp.argmax()

        dist_to_mc = np.linalg.norm(
            self.geom - self.geom[mc], axis=1)

        idx_tmp = np.argsort(dist_to_mc)[:n_max_chans]
        max_dist = dist_to_mc[idx_tmp].max()
        chans_plot = np.where(dist_to_mc <= max_dist)[0]

        return chans_plot

        chans_plot = []
        n_vis_chans = 1
        while len(chans_plot) < n_max_chans:
            n_vis_chans += 1
            idx_chan = np.argsort(ptp)[::-1][:n_vis_chans]
            center = np.mean(self.geom[idx_chan], axis=0)
            max_dist = np.linalg.norm(self.geom[idx_chan] - center, axis=1).max()
            chans_plot = np.where(np.linalg.norm(self.geom - center, axis=1) <= max_dist)[0]

        n_vis_chans -= 1
        idx_chan = np.argsort(ptp)[::-1][:n_vis_chans]
        center = np.mean(self.geom[idx_chan], axis=0)
        max_dist = np.linalg.norm(self.geom[idx_chan] - center, axis=1).max()
        chans_plot = np.where(np.linalg.norm(self.geom - center, axis=1) <= max_dist)[0]

        return chans_plot

    def add_template_plot(self, gs, x_loc, y_loc, units, colors, chan_idx=None, title=None):

        if chan_idx is None:
            chan_idx = np.arange(self.n_channels)

        # plotting parameters
        time_scale=1.8
        max_ptp = np.max(self.templates[:, :, units][:, chan_idx].ptp(0))
        scale= 100/max_ptp
        alpha=0.4
        
        R = self.templates.shape[0]

        ax = plt.subplot(gs[x_loc, y_loc])
        
        for ii, unit in enumerate(units):
            ax.plot(self.geom[chan_idx, 0]+np.arange(-R, 0)[:,np.newaxis]/time_scale, 
             self.geom[chan_idx, 1] + self.templates[:, :, unit][:, chan_idx]*scale,
             color=colors[ii], linewidth=2)

        # add channel number
        #for k in chan_idx:
        #    ax.text(self.geom[k,0]+1, self.geom[k,1], str(k), fontsize=self.fontsize)

        # add +-1 su grey shade 
        x = np.arange(-R,0)/time_scale
        y = -np.ones(x.shape)*scale
        for k in chan_idx:
            ax.fill_between(x+self.geom[k,0], 
                            y+self.geom[k,1], 
                            y+2*scale+ self.geom[k,1], color='grey', alpha=0.1)
        plt.yticks([])
        plt.xticks([])

        if title is not None:
            ax.set_title(title, fontsize=self.fontsize)

        return gs


    def add_template_summary(self, gs, x_loc, y_loc, unit, title=None, add_color_bar=True, scale=50):

        temp = self.templates[:,:,unit]
        ptp = temp.ptp(0)
        min_point = self.shifts[unit]

        vis_chan = ptp > 1
        ptp = ptp[vis_chan]
        min_point = min_point[vis_chan]

        ax = plt.subplot(gs[x_loc, y_loc])

        plt.scatter(self.geom[vis_chan, 0],
                    self.geom[vis_chan, 1],
                    s=ptp*scale, c=min_point,
                    vmin=self.min_shift,
                    vmax=self.max_shift,
                    cmap=self.cmap
                   )

        np.max(self.geom[:,0]) + 30
        plt.xlim([np.min(self.geom[:,0]) - 30, np.max(self.geom[:,0]) + 30])
        plt.ylim([np.min(self.geom[:,1]) - 30, np.max(self.geom[:,1]) + 30])

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        #if add_color_bar:
        if False:
            cbar = plt.colorbar(pad=0.01, fraction=0.05)
            #ticks = cbar.get_ticks()
            ticks = np.arange(self.min_shift, self.max_shift+1, 20)
            ticklabels = np.round((
                ticks/self.sampling_rate*1000).astype('float32'), 1)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
            cbar.ax.tick_params(labelsize=self.fontsize) 

        if title is not None:
            ax.set_title(title, fontsize=self.fontsize)

        # add channel number
        #for k in np.arange(self.n_channels):
        #    plt.text(self.geom[k,0]+1, self.geom[k,1], str(k), fontsize=self.fontsize//3)

        return gs


    def add_RF_plot(self, gs, x_loc, y_loc, unit, windows=None, title=None):
        # COMPUTE 
        #neighbor_units = self.nearest_units[unit]
        
        ax = plt.subplot(gs[x_loc, y_loc])

        img = self.STAs[unit].T #*self.sta_sign[unit] 
        vmax = np.max(np.abs(img))
        vmin = -vmax
        ax.imshow(img, vmin=vmin, vmax=vmax)
        
        if windows is not None:
            ax.set_xlim([windows[0][0], windows[0][1]])
            ax.set_ylim([windows[1][0], windows[1][1]])
        else:
            ax.set_xlim([0,self.stim_size[0]])
            ax.set_ylim([0,self.stim_size[1]])

        if title is not None:
            ax.set_title(title, fontsize=self.fontsize)

        ax.set_axis_off()
        # also plot all in one plot
        #ax = plt.subplot(gs[self.n_neighbours+1, ax_col])
        #ax = self.plot_contours(ax, np.hstack((unit,neighbor_units)),
        #                        self.colors[:self.n_neighbours+1])

        return gs

    def add_contour_plot(self, gs, x_loc, y_loc, units, colors, zoom_in=False, title=None, windows=None, legend=True):
        
        ax = plt.subplot(gs[x_loc, y_loc])
        labels = []
        
        x_min = None
        for ii, unit in enumerate(units):
            # also plot all in one plot
            if np.any(self.gaussian_fits[unit] != 0):
                plotting_data = self.get_circle_plotting_data(
                    unit, self.gaussian_fits)
                ax.plot(plotting_data[1],plotting_data[0],
                        color=colors[ii], linewidth=3)

                x_min_, x_max_ = np.min(plotting_data[1]), np.max(plotting_data[1])
                y_min_, y_max_ = np.min(plotting_data[0]), np.max(plotting_data[0])
                if x_min is None:
                    x_min, x_max = x_min_, x_max_
                    y_min, y_max = y_min_, y_max_
                else:
                    x_min = np.min((x_min_, x_min))
                    x_max = np.max((x_max_, x_max))
                    y_min = np.min((y_min_, y_min))
                    y_max = np.max((y_max_, y_max))

                labels.append(mpatches.Patch(color = colors[ii], label = "Unit {}".format(unit)))

        if legend:
            ax.legend(handles=labels)
        if zoom_in and (windows is None):
            ax.set_xlim([x_min-1, x_max+1])
            ax.set_ylim([y_min-1, y_max+1])
        elif zoom_in and (windows is not None):
            ax.set_xlim([windows[0][0], windows[0][1]])
            ax.set_ylim([windows[1][0], windows[1][1]])
        else:
            ax.set_xlim([0,self.stim_size[0]])
            ax.set_ylim([0,self.stim_size[1]])

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        if title is not None:
            ax.set_title(title, fontsize=self.fontsize)

        return gs


    def add_temporal_sta(self, gs, x_loc, y_loc, unit, title=None, add_label=False):
        
        ax = plt.subplot(gs[x_loc, y_loc])
        
        lw = 3
        sta = self.STAs_temporal[unit]
        ax.plot(sta[:,0], 'r', linewidth=lw)
        ax.plot(sta[:,1], 'g', linewidth=lw)
        ax.plot(sta[:,2], 'b', linewidth=lw)

        plt.yticks([])
        #plt.xticks([])

        if title is not None:
            ax.set_title(title, fontsize=self.fontsize)

        if add_label:
            ax.set_xlabel('time (frames)', fontsize=self.fontsize)

        return gs


    def add_ptp_vs_time(self, ptp, spt, template):

        plt.scatter(spt, ptp/self.sampling_rate, c='k')
        plt.plot([np.min(spt), np.max(spt)], [temp.ptp(), temp.ptp()], 'r')

        #plt.eventplot(spt, color='k', linewidths=0.01)
        plt.title('ptp vs spike times (red = template ptp)',
                  fontsize=self.fontsize)
        
        return gs


    def get_spikes_ptp(self, wf, template):

        mc = template.ptp(0).argmax()
        temp = template[:, mc]
        min_point = temp.argmin()
        max_point = temp.argmax()
        first_point = np.min((min_point, max_point))
        second_point = np.max((min_point, max_point))

        window = np.arange(np.max((first_point-2, 0)),
                           np.min((second_point+3, len(temp))))

        ptp_spikes = wf[:, :, mc][:, window].ptp(1)

        return ptp_spikes
    
    def get_circle_plotting_data(self,i_cell,Gaussian_params):
        # Adapted from Nora's matlab code, hasn't been tripled checked

        circle_samples = np.arange(0,2*np.pi,0.05)
        x_circle = np.cos(circle_samples)
        y_circle = np.sin(circle_samples)

        # Get Gaussian parameters
        angle = -Gaussian_params[i_cell,5]
        sd = Gaussian_params[i_cell,3:5]
        x_shift = self.stim_size[1] - Gaussian_params[i_cell,1]
        y_shift = Gaussian_params[i_cell,2]

        R = np.asarray([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
        L = np.asarray([[sd[0], 0],[0, sd[1]]])
        circ = np.concatenate([x_circle.reshape((-1,1)),y_circle.reshape((-1,1))],axis=1)

        X = np.dot(R,np.dot(L,np.transpose(circ)))
        X[0] = X[0]+x_shift
        X[1] = np.abs(X[1]+y_shift)
        plotting_data = X

        return plotting_data

    
    def add_xcorr_plot(self, gs, x_loc, y_loc, unit1, unit2, title=None, add_label=False):
        # COMPUTE XCORRS w. neighbouring units;

        if (unit1 in self.unique_ids) and (unit2 in self.unique_ids):

            unit1_ = np.where(self.unique_ids == unit1)[0][0]
            unit2_ = np.where(self.unique_ids == unit2)[0][0]

            result = self.xcorrs[unit1_, unit2_]
        else:
            result = np.zeros(self.xcorrs.shape[2])

        window_size_ms = self.window_size*1000
        bin_width_ms = self.bin_width*1000
        x_range = np.arange(-(window_size_ms//2),
                            window_size_ms//2+1, bin_width_ms)

        #notch, pval1 = notch_finder(result)
        #pval1 = np.round(pval1, 2)

        ax = plt.subplot(gs[x_loc, y_loc])
        plt.plot(x_range, result,color='black', linewidth=2)
        #y_max = np.max((10, 1.5*np.max(result)))
        y_max = 1.5*np.max(result)
        plt.ylim(0, y_max)
        plt.plot([0,0],[0, y_max],'r--')
        if add_label:
            plt.xlabel('time (ms)', fontsize=self.fontsize)
        #plt.ylabel('rates (Hz)', fontsize=self.fontsize)
        #plt.ylabel('counts', fontsize=self.fontsize)
        plt.tick_params(axis='both', which='major', labelsize=self.fontsize)

        if title is not None:
            plt.title(title, fontsize=self.fontsize)
       
        return gs
    
    
    def add_l2_feature_plot(self, gs, x_loc, y_loc, unit1, unit2, colors, title=None, add_label=False):
        
        #n_samples = 5000
        l2_features, spike_ids = get_l2_features(
            self.reader_resid, self.spike_train,
            self.templates.transpose(2,0,1),
            self.soft_assignment,
            unit1, unit2)

        if l2_features is None:
            return gs

        lda = LDA(n_components = 1)
        feat = lda.fit_transform(l2_features, spike_ids).ravel()
        #try:
        #(merge,
        # lda_prob,
        # dp_val) = test_merge(l2_features, spike_ids)

        #l2_1d_features = np.diff(l2_features, axis=0)[0]
        n_bins = np.max((int(len(feat)/20), 1))
        steps = (np.max(feat) - np.min(feat))/n_bins
        bins = np.arange(np.min(feat), np.max(feat)+steps, steps)

        ax = plt.subplot(gs[x_loc, y_loc])
        plt.hist(feat, bins, color='slategrey')
        plt.hist(feat[spike_ids==0], bins, color=colors[0], alpha=0.7)
        plt.hist(feat[spike_ids==1], bins, color=colors[1], alpha=0.7)
        #ax.tick_params(labelsize=self.fontsize) 
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        
        if add_label:
            plt.xlabel('LDA Projection', fontsize=self.fontsize)
        #plt.title(
        #    'Dip Test: {}'.format(np.round(dp_val,4)),
        #    fontsize=self.fontsize)
        #except:
        #    print ("Diptest error for unit {} and {} with size {}".format(
        #        unit1, unit2, l2_features.shape[0]))

        if title is not None:
            plt.title(title, fontsize=self.fontsize)

        return gs

    def add_full_sta(self, gs, x_locs, y_locs, unit):
        fname = os.path.join(self.rf_dir, 'tmp', 'sta', 'unit_{}.mat'.format(unit))
        full_sta = loadmat(fname)['temp_stas'].transpose(1,0,2,3)[:,:,:,-len(x_locs):]
        vmax = np.max(full_sta)
        vmin = np.min(full_sta)
        for ii, (x_loc, y_loc) in enumerate(zip(x_locs, y_locs)):
            ax = plt.subplot(gs[x_loc, y_loc])
            img = full_sta[:,:,:,ii]
            ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.set_xlim([0, 64])
            ax.set_ylim([32,0])
            ax.set_title('time {}'.format(ii), fontsize=self.fontsize)
        return gs

    def make_normalized_templates_plot(self):
        
        fname = os.path.join(self.save_dir, 'normalized_templates.png')
        if os.path.exists(fname):
            return
        
        (templates_mc, templates_sec, 
         ptp_mc, ptp_sec, _) = get_normalized_templates(
            self.templates.transpose(2, 0, 1), 
            self.neigh_channels)
        
        plt.figure(figsize=(14, 8))

        x_range = np.arange(templates_mc.shape[1])/self.sampling_rate*1000

        ths = [2, 4, 6]
        for ii, th in enumerate(ths):
            plt.subplot(2, 3, ii+1)

            if ii == 0:
                idx = np.logical_and(ptp_mc >= th, ptp_mc < ths[ii+1])
                plt.title("Templates on Main Channel\n Templates with {} < PTP < {}".format(
                    th, ths[ii+1]), fontsize=self.fontsize//2)
            elif ii == 1:
                idx = np.logical_and(ptp_mc >= th, ptp_mc < ths[ii+1])
                plt.title("Templates with {} < PTP < {}".format(th, ths[ii+1]), fontsize=self.fontsize//2)
            else:
                idx = ptp_mc >= th
                plt.title("Templates with {} < PTP".format(th), fontsize=self.fontsize//2)

            if sum(idx) > 0:
                plt.plot(x_range, templates_mc[idx].T,
                         color='k', alpha=0.1)
                plt.xlabel('time (ms)')
                plt.xlim([0, np.max(x_range)])

            if ii == 0:
                plt.ylabel('Normalized Voltage (A.U.)',
                           fontsize=self.fontsize//2)

            #if ii < 2:
            #    plt.xlabel('Main Chan., {} < PTP < {}'.format(th, ths[ii+1]), fontsize=self.fontsize//2)
            #else:
            #    plt.xlabel('Main Chan., {} < PTP'.format(th), fontsize=self.fontsize//2)
                
        for ii, th in enumerate(ths):
            plt.subplot(2, 3, ii+4)

            if ii == 0:
                plt.title("Templates on Secondary Channels", fontsize=self.fontsize//2)

            if ii < 2:
                idx = np.logical_and(ptp_sec >= th, ptp_sec < ths[ii+1])
                #plt.title("Templates with {} < PTP < {}".format(th, ths[ii+1]), fontsize=self.fontsize//2)
            else:
                idx = ptp_sec >= th
                #plt.title("Templates with {} < PTP".format(th), fontsize=self.fontsize//2)

            if sum(idx) > 0:
                plt.plot(x_range, templates_sec[idx].T, color='k', alpha=0.02)
                plt.xlabel('time (ms)')
                plt.xlim([0, np.max(x_range)])

            if ii == 0:
                plt.ylabel('Normalized Voltage (A.U.)',
                           fontsize=self.fontsize//2)

            #if ii < 2:
            #    plt.xlabel('Sec. Chan., {} < PTP < {}'.format(th, ths[ii+1]), fontsize=self.fontsize//2)
            #else:
            #    plt.xlabel('Sec. Chan., {} < PTP'.format(th), fontsize=self.fontsize//2)

    
        plt.suptitle('Aligned Templates on Their Main/Secondary Channels', fontsize=20)
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        plt.savefig(fname, dpi=100)
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
            prob = self.soft_assignment[idx]
            if np.sum(prob) > 1:
                spt = np.sort(np.random.choice(
                    spt, int(np.sum(prob)), False, prob/np.sum(prob)))
                plt.eventplot(spt, lineoffsets=j, color='k', linewidths=0.01)
        plt.yticks(np.arange(0,self.n_units,10), sorted_ptps[0:self.n_units:10])
        plt.ylabel('ptps', fontsize=self.fontsize)
        plt.xlabel('time (seconds)', fontsize=self.fontsize)
        plt.title('Raster Plot Sorted by PTP', fontsize=self.fontsize)
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()

    def make_firing_rate_plot(self):
        
        fname = os.path.join(self.save_dir, 'firing_rates.png')
        if os.path.exists(fname):
            return

        unique_labels = np.unique(self.rf_labels)

        fontsize = 15
        n_figs = len(self.cell_types)
        n_cols = 3
        n_rows = int(np.ceil(n_figs/n_cols))
        max_fr = np.max(self.f_rates)
        plt.figure(figsize=(5*n_cols, n_rows*3))

        x_max = int(np.max(np.log(self.ptps))) + 1
        x_ticks = np.round(np.exp(np.arange(0, x_max+1)), 1)
        
        y_max = int(np.max(np.log(self.f_rates))) + 1
        y_ticks = np.round(np.exp(np.arange(0, y_max+1)), 1)

        for ii, label in enumerate(unique_labels):
            plt.subplot(n_rows, n_cols, ii+1)
            idx_ = self.rf_labels == label
            plt.scatter(np.log(self.ptps[idx_]),
                        np.log(self.f_rates[idx_]),
                        color=self.colors[label],
                        alpha=0.5)
            plt.xticks(np.arange(0, x_max+1), x_ticks)
            plt.yticks(np.arange(0, y_max+1), y_ticks)
            plt.xlim([-0.5, x_max+0.5])
            plt.ylim([-0.5, y_max+0.5])
            plt.title(self.cell_types[label], fontsize=fontsize)

            if ii == 0:
                plt.ylabel('firing rates', fontsize=fontsize)

            if ii == (n_rows-1)*n_cols:
                plt.xlabel('ptps (log scaled)', fontsize=fontsize)

            if ii % n_cols != 0:
                plt.yticks([])

        plt.subplots_adjust(top = 0.85, wspace = 0.001, hspace=0.3)
        plt.suptitle('Firing Rate vs. PTP', fontsize=2*fontsize)
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()
        
    def add_residual_qq_plot(self):
        
        fname = os.path.join(self.save_dir, 'residual_qq_plot.png')
        if os.path.exists(fname):
            return
        
        nrow = int(np.sqrt(self.n_channels))
        ncol = int(np.ceil(self.n_channels/nrow))

        plt.figure(figsize=(int(ncol*2.5), nrow*2))

        
        sample_size = 10000
        t_start = int(np.random.choice(
            self.reader_resid.rec_len-sample_size-1, 1)[0])
        
        res = self.reader_resid.read_data(
            t_start, t_start+sample_size)
        
        th = np.sort(np.random.normal(size = sample_size))
        for c in range(self.n_channels):
            qq = np.sort(np.random.choice(
                res[:, c], sample_size, False))
            
            plt.subplot(nrow, ncol, c+1)
            plt.subplots_adjust(top = 0.95, wspace = 0.001)
            plt.scatter(th, qq, s=5)
            min_val = np.min((th.min(), qq.min()))
            max_val = np.max((th.max(), qq.max()))
            plt.plot([min_val, max_val], [min_val, max_val], color='r')
            plt.title('Channel: {}'.format(c))
            plt.xticks([])
            plt.yticks([])

        plt.suptitle(
            'QQ plot of Residual Recording: Sample from {} to {} Timepoints'.format(
            t_start, t_start+sample_size), fontsize=int(3*ncol))
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()


    def residual_varaince(self):
        
        fname = os.path.join(self.save_dir, 'residual_variance.png')
        if os.path.exists(fname):
            return
        
        # calculate variance per unit
        n_examples = 1000
        units, counts = np.unique(self.spike_train[:, 1], return_counts=True)
        units = units[counts > 10]
        resid_var = np.zeros((len(units), self.n_times_templates))
        for ii, unit in enumerate(units):
            idx = np.where(self.spike_train[:,1]==unit)[0]
            idx = np.random.choice(idx,
                                   np.min((n_examples, len(idx))),
                                   False)
            spt = self.spike_train[idx, 0]
            mc = self.templates[:, :, unit].ptp(0).argmax()
            resid_var[ii] = np.var(self.reader_resid.read_waveforms(spt, None, [mc])[0][:,:,0], 0)

        # values for plotting
        ptps = self.ptps[units]
        max_var = resid_var.max(1)
        rf_labels = self.rf_labels[units]
        unique_labels = np.unique(rf_labels)

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.plot(np.arange(resid_var.shape[1])/self.sampling_rate*1000, resid_var.T, 'k', alpha=0.2)
        plt.ylim([0,5])
        plt.xlabel('time (ms)', fontsize=self.fontsize)
        plt.ylabel('maximum variance', fontsize=self.fontsize)
        plt.title('Time vs Residual Variance of all units')

        plt.subplot(1,2,2)
        legends = []
        for ii, label in enumerate(unique_labels):
            idx_ = rf_labels == label
            plt.scatter(np.log(ptps[idx_]), max_var[idx_], color=self.colors[label], alpha=0.5)
            legends.append(mpatches.Patch(color = self.colors[label], label = self.cell_types[ii]))
        plt.legend(handles=legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=self.fontsize)

        x_max = int(np.max(np.log(ptps))) + 1
        x_ticks = np.round(np.exp(np.arange(0, x_max+1)), 1)
        plt.xticks(np.arange(0, x_max+1), x_ticks)
        plt.xlim([0, x_max])
        plt.xlabel('ptp (log scaled)', fontsize=self.fontsize)
        plt.ylabel('Maximum Variance', fontsize=self.fontsize)
        plt.title('PTP vs Maximum Residual Variance')

        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()


    def add_raw_resid_snippets(self):

        save_dir = os.path.join(self.save_dir, 'raw_residual_snippets/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n_batches = np.minimum(self.reader.n_batches, 10)
        n_big = 5
        n_random = 5
        t_window = 50

        batch_ids = np.random.choice(self.reader.n_batches, n_batches)

        for batch_id in batch_ids:

            fname = os.path.join(save_dir, 'large_residual_chunk_{}.png'.format(batch_id))
            if os.path.exists(fname):
                continue

            offset = self.reader.idx_list[batch_id][0]

            raw = self.reader.read_data_batch(batch_id)
            res = self.reader_resid.read_data_batch(batch_id)

            max_res = np.abs(np.max(res, 1))
            max_ids = scipy.signal.argrelmax(max_res, order=2*t_window+1)[0]
            max_ids = max_ids[np.logical_and(max_ids > t_window, max_ids < res.shape[0]-t_window)]
            max_res = max_res[max_ids]

            id_keep = max_res > 4
            max_ids = max_ids[id_keep]
            max_res = max_res[id_keep]

            id_big = max_ids[np.argsort(max_res)[-n_big:]]
            id_random = np.random.choice(max_ids, n_random)

            id_big_c = res[id_big].argmax(1)
            id_random_c = res[id_random].argmax(1)

            id_check = np.hstack((id_big, id_random))
            id_check_c = np.hstack((id_big_c, id_random_c))

            x_range = np.arange(t_window*2+1)/self.sampling_rate*1000

            plt.figure(figsize=(30, 10))
            for j in range(n_big+n_random):
                chans = np.where(self.neigh_channels[id_check_c[j]])[0]
                tt = id_check[j]

                snip_raw = raw[:, chans][tt-t_window:tt+t_window+1]
                snip_res = res[:, chans][tt-t_window:tt+t_window+1]

                spread = np.arange(len(chans))*10

                plt.subplot(2, n_big, j+1)
                plt.plot(x_range, snip_raw+spread[None], 'k')
                plt.plot(x_range, snip_res+spread[None], 'r')
                plt.xlabel('Time (ms)')
                plt.yticks([])

                y = -np.ones(t_window*2+1)
                for s in spread:
                    plt.fill_between(x_range,
                                     y + s,
                                     y + s + 2,
                                     color='grey',
                                     alpha=0.1)

                plt.title('Recording Time {}, Channel {}'.format(tt + offset, id_check_c[j]))

                if j == 0:
                    legends = [mpatches.Patch(color = 'k', label = 'raw'),
                               mpatches.Patch(color = 'r', label = 'residual')
                              ]
                    plt.legend(handles=legends)
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight', dpi=100)
            plt.close('all')


    def cell_classification_plots(self):

        fname = os.path.join(self.save_dir, 'contours.png')
        if os.path.exists(fname):
            return

        idx_per_type = []
        for j in range(len(self.cell_types[:-2])):
            if self.cell_types[j] != 'Unknown':
                idx_per_type.append(np.where(self.rf_labels == j)[0])

        plt.figure(figsize=(50, 12))
        for ii, idx in enumerate(idx_per_type):
            plt.subplot(1, len(idx_per_type), ii+1)
            for unit in idx:
                # also plot all in one plot
                plotting_data = self.get_circle_plotting_data(unit, self.gaussian_fits)
                plt.plot(plotting_data[0],plotting_data[1], 'k', alpha=0.4)
                plt.xlim([0, self.stim_size[1]])
                plt.ylim([0, self.stim_size[0]])
            plt.title(self.cell_types[ii], fontsize=30)
        plt.suptitle('RF Contours by Cell Types', fontsize=50)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        plt.close()         
        
    def make_rf_plots(self):

        fname = os.path.join(self.save_dir, 'all_rfs.png')
        if os.path.exists(fname):
            return

        n_units = self.STAs.shape[0]

        idx_per_type = []
        for j in range(len(self.cell_types)):
            idx_per_type.append(np.where(self.rf_labels == j)[0])

        n_cols = 10

        n_rows_per_type = []
        for idx in idx_per_type:
            n_rows_per_type.append(int(np.ceil(len(idx)/float(n_cols))))
        n_rows = sum(n_rows_per_type)+9

        self.fontsize = 20

        fig=plt.figure(figsize=(3*n_cols, 3*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, fig,
                               left=0, right=1, top=0.95, bottom=0.05,
                               hspace=0.2, wspace=0)

        row = 0
        for ii, idx in enumerate(idx_per_type):

            col = 0

            # add label
            ax = plt.subplot(gs[row, 0])
            plt.text(0.5, 0.5, self.cell_types[ii],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=60,
                     transform=ax.transAxes)
            ax.set_axis_off()

            row += 1

            idx_sort = idx[np.argsort(self.ptps[idx])[::-1]]
            for unit in idx_sort:
                fr = str(np.round(self.f_rates[unit], 1))
                ptp = str(np.round(self.ptps[unit], 1))
                title = "Unit: {}\n{}Hz, {}SU".format(unit, fr, ptp)
                gs = self.add_RF_plot(gs, row, col, unit, None, title)
                if col == 9:
                    col = 0
                    row += 1
                else:
                    col += 1
            if col != 0:
                row += 1

        #plt.tight_layout()
        #fig.savefig(fname)
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        fig.clf()
        plt.close('all')

    def make_all_templates_summary_plots(self):

        fname = os.path.join(self.save_dir, 'all_templates.png')
        if os.path.exists(fname):
            return

        n_units = self.n_units
 
        idx_per_type = []
        for j in range(len(self.cell_types)):
            idx_per_type.append(np.where(self.rf_labels == j)[0])

        n_cols = 10

        n_rows_per_type = []
        for idx in idx_per_type:
            n_rows_per_type.append(int(np.ceil(len(idx)/float(n_cols))))
        n_rows = sum(n_rows_per_type)+9

        self.fontsize = 20

        fig=plt.figure(figsize=(3*n_cols, 3*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, fig,
                               left=0, right=1, top=0.95, bottom=0.05,
                               hspace=0.5, wspace=0)

        row = 0
        for ii, idx in enumerate(idx_per_type):

            col = 0

            # add label
            ax = plt.subplot(gs[row, 0])
            plt.text(0.5, 0.5, self.cell_types[ii],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=60,
                     transform=ax.transAxes)
            ax.set_axis_off()
            
            row += 1

            idx_sort = idx[np.argsort(self.ptps[idx])[::-1]]
            for unit in idx_sort:
                fr = str(np.round(self.f_rates[unit], 1))
                ptp = str(np.round(self.ptps[unit], 1))
                title = "Unit: {}\n{}Hz, {}SU".format(unit, fr, ptp)
                gs = self.add_template_summary(
                    gs, row, col, unit, title=title,
                    add_color_bar=False, scale=8)
                if col == 9:
                    col = 0
                    row += 1
                else:
                    col += 1
            if col != 0:
                row += 1

        #plt.tight_layout()
        fig.savefig(fname, bbox_inches='tight', dpi=100)
        fig.clf()
        plt.close('all')


    def make_all_rf_templates_plots(self):
 
        idx_per_type = []
        for j in range(len(self.cell_types)):
            idx_per_type.append(np.where(self.rf_labels == j)[0])

        n_cols = 10
        self.fontsize = 20

        for ii in range(len(self.cell_types)):
            
            type_name = self.cell_types[ii]

            fname = os.path.join(self.save_dir_ind,
                                 'all_rf_templates_{}.png'.format(type_name))
            if os.path.exists(fname):
                continue
            
            idx = idx_per_type[ii]
            idx_sort = idx[np.argsort(self.ptps[idx])[::-1]]
            n_rows = int(np.ceil(len(idx)/float(n_cols/2))) + 1

            fig=plt.figure(figsize=(3*n_cols, 3*n_rows))

            gs = gridspec.GridSpec(n_rows, n_cols, fig,
                                   left=0, right=1, top=1, bottom=0.05,
                                   hspace=0.5, wspace=0)
            
            ax = plt.subplot(gs[0, n_cols//2])
            plt.text(0.5, 0.5, type_name,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=4*self.fontsize,
                     transform=ax.transAxes)
            ax.set_axis_off()

            row = 1
            col = 0
            for unit in idx_sort:
                fr = str(np.round(self.f_rates[unit], 1))
                ptp = str(np.round(self.ptps[unit], 1))
                title = "Unit: {}, {}Hz, {}SU".format(unit, fr, ptp)
                gs = self.add_template_summary(
                    gs, row, col, unit, title=title,
                    add_color_bar=False, scale=8)
                col += 1
                gs = self.add_RF_plot(gs, row, col, unit, None, None)
                if col == n_cols-1:
                    col = 0
                    row += 1
                else:
                    col += 1

            #plt.tight_layout()
            fig.savefig(fname, bbox_inches='tight', dpi=100)
            fig.clf()
            plt.close('all')


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
            STAs_temporal1 = np.load(os.path.join(rf_dir, 'STA_temporal.npy'))
            Gaussian_params1 = np.load(os.path.join(rf_dir, 'gaussian_fits.npy'))
            STAs2 = np.load(os.path.join(rf_dir2, 'STA_spatial.npy'))
            STAs_temporal2 = np.load(os.path.join(rf_dir2, 'STA_temporal.npy'))
            Gaussian_params2 = np.load(os.path.join(rf_dir2, 'gaussian_fits.npy'))
            
            idx_single_rf1 = np.load(os.path.join(rf_dir, 'idx_single_rf.npy'))
            idx_no_rf1 = np.load(os.path.join(rf_dir, 'idx_no_rf.npy'))
            idx_multi_rf1 = np.load(os.path.join(rf_dir, 'idx_multi_rf.npy'))
            rf_labels1 = np.load(os.path.join(rf_dir, 'labels.npy'))
            
            idx_single_rf2 = np.load(os.path.join(rf_dir2, 'idx_single_rf.npy'))
            idx_no_rf2 = np.load(os.path.join(rf_dir2, 'idx_no_rf.npy'))
            idx_multi_rf2 = np.load(os.path.join(rf_dir2, 'idx_multi_rf.npy'))
            rf_labels2 = np.load(os.path.join(rf_dir2, 'labels.npy'))
            
            templates, spike_train = combine_two_spike_train(
                templates1, templates2, spike_train1, spike_train2)
            
            STAs, Gaussian_params = combine_two_rf(
                STAs1, STAs2, Gaussian_params1, Gaussian_params2)
            STAs_temporal = np.concatenate((STAs_temporal1, STAs_temporal2), axis=0)

            K1 = templates1.shape[2]
            K2 = templates2.shape[2]
            set1_idx = np.zeros(K1+K2, 'bool')
            set1_idx[:K1] = 1
            
            idx_single_rf = np.hstack((idx_single_rf1, idx_single_rf2+K1))
            idx_no_rf = np.hstack((idx_no_rf1, idx_no_rf2+K1))
            idx_multi_rf = np.hstack((idx_multi_rf1, idx_multi_rf2+K1))
            rf_labels = np.hstack((rf_labels1, rf_labels2))
            
            
            fname_templates = os.path.join(tmp_dir, 'templates_combined.npy')
            fname_spike_train = os.path.join(tmp_dir, 'spike_train_combined.npy')

            rf_dir = tmp_dir

            fname_set1_idx = os.path.join(tmp_dir, 'set1_idx.npy')
            
            np.save(fname_templates, templates)
            np.save(fname_spike_train, spike_train)
            np.save(fname_set1_idx, set1_idx)
            np.save(os.path.join(rf_dir, 'STA_spatial.npy'), STAs)
            np.save(os.path.join(rf_dir, 'STA_temporal.npy'), STAs_temporal)
            np.save(os.path.join(rf_dir, 'gaussian_fits.npy'), Gaussian_params)
            np.save(os.path.join(rf_dir, 'idx_single_rf.npy'), idx_single_rf)
            np.save(os.path.join(rf_dir, 'idx_no_rf.npy'), idx_no_rf)
            np.save(os.path.join(rf_dir, 'idx_multi_rf.npy'), idx_multi_rf)
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

def get_normalized_templates(templates, neigh_channels):

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
                    templates_mc)
    templates_mc = shift_chans(templates_mc, best_shifts_mc)
    ptp_mc = templates_mc.ptp(1)

    # normalize templates
    norm_mc = np.linalg.norm(templates_mc, axis=1, keepdims=True)
    templates_mc /= norm_mc

    # get secdonary channel templates
    templates_sec = np.zeros((0, R))
    best_shifts_sec = np.zeros(0)
    unit_ids_sec = np.zeros((0), 'int32')
    for k in range(K):
        neighs = np.copy(neigh_channels[mc[k]])
        neighs[mc[k]] = False
        neighs = np.where(neighs)[0]
        templates_sec = np.concatenate((templates_sec, templates[k, :, neighs]), axis=0)
        best_shifts_sec = np.hstack((best_shifts_sec, np.repeat(best_shifts_mc[k], len(neighs))))
        unit_ids_sec = np.hstack((unit_ids_sec, np.ones(len(neighs), 'int32')*k))

    # shift templates_sec
    best_shifts_sec = align_get_shifts_with_ref(
                    templates_sec)
    templates_sec = shift_chans(templates_sec, best_shifts_sec)
    ptp_sec = templates_sec.ptp(1)

    # normalize templates
    norm_sec = np.linalg.norm(templates_sec, axis=1, keepdims=True)
    templates_sec /= norm_sec

    return templates_mc, templates_sec, ptp_mc, ptp_sec, unit_ids_sec

def pca_denoise(data, pca_mean, pca_components):
    data_pca = np.matmul(data-pca_mean, pca_components.T)
    return np.matmul(data_pca, pca_components)+pca_mean
