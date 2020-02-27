from yass.visual.util import match_two_sorts
import os
from yass.template import run_template_computation
import numpy as np
from yass import residual, soft_assignment
from yass.merge.correlograms_phy import compute_correlogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy




class CompareTwoSorts(object):
    def __init__(self,
                 fname_spike_train2,
                 run1_name='run1',
                 run2_name='run2',
                 fname_shifts1=None,
                 fname_scales1=None,
                 fname_shifts2=None,
                 fname_scales2=None,
                 fname_templates2=None,
                 fname_softassignment1=None,
                 fname_softassignment2=None):
        
        
        
        self.fname_spike_train2 = fname_spike_train2
        self.fname_shifts1 = fname_shifts1
        self.fname_shifts2 = fname_shifts2
        self.fname_scales1 = fname_scales1
        self.fname_scales2 = fname_scales2
        self.fname_templates2 = fname_templates2
        self.fname_softassignment1 = fname_softassignment1
        self.fname_softassignment2 = fname_softassignment2
        self.run1_name = run1_name
        self.run2_name = run2_name

        if self.fname_templates is None:
            print('compute templates1')
            save_dir_template1 = os.path.join(self.save_dir, 'template1_computation')
            self.fname_templates = self.compute_templates(save_dir_template1,
                              self.fname_spike_train)

        if self.fname_templates2 is None:
            print('compute templates2')
            save_dir_template2 = os.path.join(self.save_dir, 'template2_computation')
            self.fname_templates2 = self.compute_templates(save_dir_template2,
                              self.fname_spike_train2)
            
        self.get_ptp_firing_rates()
    
        self.run_save_matching()
    
        if self.fname_softassignment1 is None:
            print('compute soft assignment 1')
            save_dir_softassignment = os.path.join(
                self.save_dir, 'softassignment1')
            self.fname_softassignment1 = self.get_soft_assignment(
                save_dir_softassignment,
                self.fname_spike_train,
                self.fname_templates,
                self.fname_shifts1,
                self.fname_scales1)
        
        if self.fname_softassignment2 is None:
            print('compute soft assignment 2')
            save_dir_softassignment = os.path.join(
                self.save_dir, 'softassignment2')
            self.fname_softassignment2 = self.get_soft_assignment(
                save_dir_softassignment,
                self.fname_spike_train2,
                self.fname_templates2,
                self.fname_shifts2,
                self.fname_scales2)
            
        (self.avg_outlier1,
         self.avg_temp_soft1) = self.compute_soft_assignment_summary(
            self.fname_softassignment1,
            self.fname_templates,
            self.fname_spike_train)
        
        (self.avg_outlier2,
         self.avg_temp_soft2) = self.compute_soft_assignment_summary(
            self.fname_softassignment2,
            self.fname_templates2,
            self.fname_spike_train2)

        fname_mad_avg1 = os.path.join(self.save_dir, 'mad_avg1.npy')
        if not os.path.exists(fname_mad_avg1):
            print('compute mad 1')
            mad1, self.mad_avg1 = self.get_mad_statistics(
                self.fname_templates, self.fname_spike_train)
            np.save(fname_mad_avg1, self.mad_avg1)
            
            fname_mad1 = os.path.join(self.save_dir, 'mad1.npy')
            np.save(fname_mad1, mad1)
            
        else:
            self.mad_avg1 = np.load(fname_mad_avg1)
            
        
            
        fname_mad_avg2 = os.path.join(self.save_dir, 'mad_avg2.npy')
        if not os.path.exists(fname_mad_avg2):
            print('compute mad 2')
            mad2, self.mad_avg2 = self.get_mad_statistics(
                self.fname_templates2, self.fname_spike_train2)
            np.save(fname_mad_avg2, self.mad_avg2)
            
            fname_mad2 = os.path.join(self.save_dir, 'mad2.npy')
            np.save(fname_mad2, mad2)

        else:
            self.mad_avg2 = np.load(fname_mad_avg2)
        
        
#         fname_cos1 = os.path.join(self.save_dir, 'cos1_only.npy')
#         fname_cos2 = os.path.join(self.save_dir, 'cos2_only.npy')
#         fname_ptp = os.path.join(self.save_dir, 'ptp_matched.npy')
#         fname_fr = os.path.join(self.save_dir, 'fr_matched.npy')
#         if not (os.path.exists(fname_cos1) and os.path.exists(fname_cos2)):
#             self.compute_cossim_summary()
#             np.save(fname_cos1, self.cos1)
#             np.save(fname_cos2, self.cos2)
#             np.save(fname_ptp, self.ptp)
#             np.save(fname_fr, self.fr)
            
#         else:
#             self.cos1 = np.load(fname_cos1)
#             self.cos2 = np.load(fname_cos2)
#             self.ptp = np.load(fname_ptp)
#             self.fr = np.load(fname_fr)
        
        
            
        
        fname_xcorr1 = os.path.join(self.save_dir, 'xcorr1.npy')
        if not os.path.exists(fname_xcorr1):
            print('compute xcorr 1')
            self.get_xcorr(self.fname_spike_train,
                           self.fname_templates,
                           fname_xcorr1)
        fname_peak_xcorr1 = os.path.join(self.save_dir, 'peak_xcorr1.npy')
        fname_notch_xcorr1 = os.path.join(self.save_dir, 'notch_xcorr1.npy')
        if (not os.path.exists(fname_peak_xcorr1)) or (not os.path.exists(fname_notch_xcorr1)):
            self.peak_xcorr1, self.notch_xcorr1 = self.get_xcorr_summary(fname_xcorr1)
            np.save(fname_peak_xcorr1, self.peak_xcorr1)
            np.save(fname_notch_xcorr1, self.notch_xcorr1)
        else:
            self.peak_xcorr1 = np.load(fname_peak_xcorr1)
            self.notch_xcorr1 = np.load(fname_notch_xcorr1)
            
        fname_xcorr2 = os.path.join(self.save_dir, 'xcorr2.npy')
        if not os.path.exists(fname_xcorr2):
            print('compute xcorr 2')
            self.get_xcorr(self.fname_spike_train2,
                           self.fname_templates2,
                           fname_xcorr2)
        fname_peak_xcorr2 = os.path.join(self.save_dir, 'peak_xcorr2.npy')
        fname_notch_xcorr2 = os.path.join(self.save_dir, 'notch_xcorr2.npy')
        if (not os.path.exists(fname_peak_xcorr2)) or (not os.path.exists(fname_notch_xcorr2)):
            self.peak_xcorr2, self.notch_xcorr2 = self.get_xcorr_summary(fname_xcorr2)
            np.save(fname_peak_xcorr2, self.peak_xcorr2)
            np.save(fname_notch_xcorr2, self.notch_xcorr2)
        else:
            self.peak_xcorr2 = np.load(fname_peak_xcorr2)
            self.notch_xcorr2 = np.load(fname_notch_xcorr2)
            

    def compute_templates(self, save_dir, fname_spike_train):
        
        fname_templates = run_template_computation(
            save_dir,
            fname_spike_train,
            self.reader,
            multi_processing=self.CONFIG.resources.multi_processing,
            n_processors=self.CONFIG.resources.n_processors)
        
        return fname_templates

    def run_save_matching(self):

        print('run matching')
        fname_results = os.path.join(self.save_dir, 'matching_results.npz')
        
        if not os.path.exists(fname_results):
            templates = np.load(self.fname_templates)
            templates2 = np.load(self.fname_templates2)
            spike_train1 = np.load(self.fname_spike_train)
            spike_train2 = np.load(self.fname_spike_train2)


            (run1_only,
             run2_only,
             run1_matched,
             run2_matched,
             matched_pairs, matched_events, run1_miss, run2_miss) = match_two_sorts(templates,
                                              templates2,
                                              spike_train1,
                                              spike_train2,
                                              overlap_threshold=0.5)
        
            np.savez(fname_results,
                     run1_only=run1_only,
                     run2_only=run2_only,
                     run1_matched=run1_matched,
                     run2_matched=run2_matched,
                     matched_pairs=matched_pairs,
                     matched_events= matched_events,
                     run1_miss = run1_miss,
                     run2_miss = run2_miss
                    )
        
        else:
            
            save_dir = '/ssd/peter/2007_5min/matching_to_ks/'
            temp = np.load(fname_results)
            run1_only = temp['run1_only']
            run2_only = temp['run2_only']
            run1_matched = temp['run1_matched']
            run2_matched = temp['run2_matched']
            matched_pairs = temp['matched_pairs']
            matched_events =temp['matched_events']
            run1_miss = temp['run1_miss']
            run2_miss = temp['run2_miss']           
            
            
            
            
        self.matched_pairs = matched_pairs
        self.matched_events = matched_events
        self.run1_only = run1_only
        self.run2_only = run2_only
        self.run1_matched = run1_matched
        self.run2_matched = run2_matched
        self.run1_miss = run1_miss
        self.run2_miss = run2_miss
        
    def get_soft_assignment(self,
                            save_dir,
                            fname_spike_train,
                            fname_templates,
                            fname_shifts=None,
                            fname_scales=None):
                    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if fname_shifts is None:
            n_spikes = np.load(fname_spike_train).shape[0]
            shifts = np.zeros(n_spikes, 'float32')
            fname_shifts = os.path.join(save_dir, 'shifts.npy')
            np.save(fname_shifts, shifts)

        if fname_scales is None:
            n_spikes = np.load(fname_spike_train).shape[0]
            scales = np.ones(n_spikes, 'float32')
            fname_scales = os.path.join(save_dir, 'scales.npy')
            np.save(fname_scales, scales)
            
        fname_residual, residual_dtype = residual.run(
            fname_shifts,
            fname_scales,
            fname_templates,
            fname_spike_train,
            os.path.join(save_dir,
                         'residual'),
            self.recording_path,
            self.recording_dtype,
            dtype_out='float32')
        
        fname_noise_soft, fname_template_soft = soft_assignment.run(
            fname_templates,
            fname_spike_train,
            fname_shifts,
            fname_scales,
            os.path.join(save_dir,
                         'soft_assignment'),
            fname_residual,
            residual_dtype,
            compute_noise_soft=False,
            compute_template_soft=True)
        
        return fname_template_soft
    
    
    def get_ptp_firing_rates(self):
        templates = np.load(self.fname_templates)
        templates2 = np.load(self.fname_templates2)
        
        self.ptps1 = templates.ptp(1).max(1)
        self.ptps2 = templates2.ptp(1).max(1)

        spike_train1 = np.load(self.fname_spike_train)
        spike_train2 = np.load(self.fname_spike_train2)
        
        n_spikes1 = np.zeros(templates.shape[0])
        a, b = np.unique(spike_train1[:, 1], return_counts=True)
        n_spikes1[a] = b

        n_spikes2 = np.zeros(templates2.shape[0])
        a, b = np.unique(spike_train2[:, 1], return_counts=True)
        n_spikes2[a] = b

        recording_length = self.reader.rec_len/self.reader.sampling_rate
        self.f_rates1 = n_spikes1/recording_length
        self.f_rates2 = n_spikes2/recording_length


    def compute_soft_assignment_summary(self,
                                        fname_soft_assignment,
                                        fname_templates,
                                        fname_spike_train):
        
        spike_train = np.load(fname_spike_train)
        templates = np.load(fname_templates)
        
        n_spikes = np.zeros(templates.shape[0])
        a, b = np.unique(spike_train[:, 1], return_counts=True)
        n_spikes[a] = b

        
        temp = np.load(fname_soft_assignment)
        outlier_scores = temp['logprobs_outliers'][:, 0]
        temp_soft = temp['probs_templates'][:,0]

        avg_outlier_scores = np.zeros(templates.shape[0])
        avg_temp_soft = np.zeros(templates.shape[0])
        for j in range(spike_train.shape[0]):
            avg_outlier_scores[spike_train[j, 1]] += outlier_scores[j]
            avg_temp_soft[spike_train[j, 1]] += temp_soft[j]

        avg_outlier = avg_outlier_scores/n_spikes
        avg_temp_soft = avg_temp_soft/n_spikes
        
        return avg_outlier, avg_temp_soft
    
    def get_mad_statistics(self, fname_templates, fname_spike_train):
        
        templates = np.load(fname_templates)
        spike_train = np.load(fname_spike_train)
        
        n_units, n_times, n_channels = templates.shape
        n_times = 61

        mad = np.zeros((n_units, n_times, n_channels))
        for k in range(n_units):
            spt = spike_train[spike_train[:, 1]==k, 0]
            if len(spt) > 500:
                spt = np.random.choice(spt, 500, False)
            wfs = self.reader.read_waveforms(spt, n_times=n_times)[0]
            mad[k] = np.median(np.abs(wfs - np.median(wfs, 0)), 0)
        
        mad_avg = np.zeros(n_units)
        for k in range(n_units):
            mad_avg[k] = np.mean(np.sort(mad[k].ravel())[-50:])

        return mad, mad_avg
    
    def get_xcorr(self, fname_spike_train, fname_templates, fname_xcorr):
        
        window_size = 0.04
        bin_width = 0.001
        
        spike_train = np.load(fname_spike_train)
        templates = np.load(fname_templates)
        n_units = templates.shape[0]
        
        xcorrs = compute_correlogram(
            np.arange(n_units),
            spike_train,
            None,
            sample_rate=self.reader.sampling_rate,
            bin_width=bin_width,
            window_size=window_size)
        
        np.save(fname_xcorr, xcorrs)
        
    def get_xcorr_summary(self, fname_xcorr):

        xcorrs = np.load(fname_xcorr)
        
        n_units = xcorrs.shape[0]
        
        means_ = xcorrs.mean(2)
        stds_ = np.std(xcorrs, 2)
        stds_[stds_==0] = 1
        xcorrs_standardized = (xcorrs - means_[:,:,None])/stds_[:,:,None]

        peak_match = np.zeros(n_units, 'int32')
        notch_match = np.zeros(n_units, 'int32')
        for k in range(n_units):
            units_ = np.where(np.sum(xcorrs[k], 1) > 10)[0]
            units_ = units_[units_ != k]

            if len(units_) == 0:
                peak_match[k] = xcorrs[k].max(1).argmax()
                notch_match[k] = xcorrs[k].min(1).argmin()
            else:
                peak_match[k] = units_[xcorrs[k][units_][:,[20,21,19]].mean(-1).argmax()]
                notch_match[k] = units_[xcorrs[k][units_][:,[20,21,19]].mean(-1).argmin()]

        peak_xcorr = xcorrs_standardized[np.arange(n_units), peak_match].max(1)
        notch_xcorr = xcorrs_standardized[np.arange(n_units), notch_match].min(1)

        return peak_xcorr, notch_xcorr
    
#     def compute_cossim_summary(self):
#         templates1 = np.load(self.fname_templates).transpose([2,0,1])
#         templates2 = np.load(self.fname_templates2)
#         units = np.unique(self.matched_pairs[:,0])
#         print(units.size, len(self.matched_events), len(self.run2_miss))
#         self.cos1 = np.zeros(units.size)
#         self.cos2 = np.zeros(units.size)
#         self.ptp  = np.zeros(units.size)
#         self.fr   = np.zeros(units.size)
#         # with tqdm_notebook(total)
#         for i, unit1 in enumerate(units):
                                                                       
#             wfs_con = self.reader.read_waveforms(self.matched_events[i][:100])[0]
#             templates_con = wfs_con.mean(0)
#             if self.run2_miss[i].shape[0] > 0.0:
#                 wfs2 = self.reader.read_waveforms(self.run2_miss[i][:100])[0]
#                 temp2 = wfs2.mean(0)
#             else:
#                 temp2 = templates_con

#             if self.run1_miss[i].shape[0] > 0.0:
#                 wfs1 = self.reader.read_waveforms(self.run1_miss[i][:100])[0]
#                 temp1 = wfs1.mean(0)
#             else:
#                 temp1 = templates_con
            
#             self.fr[i] = self.matched_events[i].size + self.run1_miss[i].size
    
    
#             self.ptp[i] = templates_con.ptp(0).max(0)
#             self.cos1[i] = 1 - scipy.spatial.distance.cosine(temp1.flatten(),templates_con.flatten())
#             self.cos2[i] = 1 - scipy.spatial.distance.cosine(temp2.flatten(),templates_con.flatten())


    
    def make_figures(self, min_fr=0.1, use_ptp=False):
        
        fs = 50
        ss = 100
        fig = plt.figure(figsize=(40, 30))

        outer = gridspec.GridSpec(2, 1, fig, height_ratios=[10, 1],
                               left=0.08, right=0.98, top=0.97, bottom=0.0,
                               hspace=0.1, wspace=0)

        gs = outer[0].subgridspec(5, 3, hspace=0.2, wspace=0.3)

        unit_keep1 = np.where(self.f_rates1 > min_fr)[0]
        unit_keep2 = np.where(self.f_rates2 > min_fr)[0]
        
        run1_only = self.run1_only[self.f_rates1[self.run1_only] > min_fr]
        run2_only = self.run2_only[self.f_rates2[self.run2_only] > min_fr]
        run1_matched = self.run1_matched[self.f_rates1[self.run1_matched] > min_fr]
        run2_matched = self.run2_matched[self.f_rates2[self.run2_matched] > min_fr]
        
        fr_max = np.max((self.f_rates1.max(), self.f_rates2.max()))
        fr_max = int(np.ceil(np.log(fr_max)))
        fr_min = np.log(min_fr)
        fr_ticks = np.round(np.exp(np.arange(fr_min, fr_max, 1)), 1)
        
        
        ptp_max = np.max((self.ptps1[unit_keep1].max(),
                          self.ptps2[unit_keep2].max())) + 5
        ptp_max = int(np.ceil(np.log(ptp_max)))

        ptp_min = np.min((self.ptps1[unit_keep1].min(),
                          self.ptps2[unit_keep2].min())) - 1
        ptp_min = np.max((ptp_min, 1))
        ptp_min = np.log(ptp_min)
        ptp_ticks = np.round(np.exp(np.arange(ptp_min, ptp_max, 1)), 1)
        if use_ptp:
            x_max = ptp_max
            x_min = ptp_min
            x_ticks = np.round(np.exp(np.arange(ptp_min, ptp_max, 2)), 1)
        else:
            x_max = fr_max
            x_min = fr_min
            x_ticks = np.round(np.exp(np.arange(fr_min, fr_max, 2)), 1)
        
        mad_max = np.max((self.mad_avg1[unit_keep1].max(),
                          self.mad_avg2[unit_keep2].max()))*1.2
        mad_min = np.min((self.mad_avg1[unit_keep1].min(),
                          self.mad_avg2[unit_keep2].min()))*0.8
        mad_min = np.max((mad_min, 0))
        
        
        temp_soft_max = np.max((self.avg_temp_soft1[unit_keep1].max(),
                          self.avg_temp_soft2[unit_keep2].max()))*1.2
        temp_soft_max = 1.1
        temp_soft_min = np.min((self.avg_temp_soft1[unit_keep1].min(),
                          self.avg_temp_soft2[unit_keep2].min()))*0.8
        temp_soft_min = np.max((temp_soft_min, 0))
        temp_soft_min = -0.1
        
        outlier_max = np.max((self.avg_outlier1[unit_keep1].max(),
                          self.avg_outlier2[unit_keep2].max()))*1.2
        outlier_min = np.min((self.avg_outlier1[unit_keep1].min(),
                          self.avg_outlier2[unit_keep2].min()))*0.8
        
        peak_max = np.max((self.peak_xcorr1[unit_keep1].max(),
                          self.peak_xcorr2[unit_keep2].max()))*1.2
        peak_min = np.min((self.peak_xcorr1[unit_keep1].min(),
                          self.peak_xcorr2[unit_keep2].min()))*0.8
        
        notch_max = np.max((self.notch_xcorr1[unit_keep1].max(),
                          self.notch_xcorr2[unit_keep2].max()))*0.8
        notch_max = 0
        notch_min = np.min((self.notch_xcorr1[unit_keep1].min(),
                          self.notch_xcorr2[unit_keep2].min()))*1.2
        
        ax = plt.subplot(gs[0, 0])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        np.log(self.f_rates1[run1_only]),
                        s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]), 
                        np.log(self.f_rates2[run2_only]),
                        s=ss, color='b')
            plt.yticks(np.arange(fr_min, fr_max, 1), fr_ticks)
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        np.log(self.ptps1[run1_only]), s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        np.log(self.ptps2[run2_only]), s=ss, color='b')
            plt.yticks(np.arange(ptp_min, ptp_max, 1), ptp_ticks)

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
        
        if use_ptp:
            plt.ylabel('Firing rates (Hz) ', fontsize=fs)
        else:
            plt.ylabel('PTP ', fontsize=fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])

        plt.xlim([x_min-0.5, x_max+0.5])
        
        
        
        ax = plt.subplot(gs[0, 1])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        self.mad_avg1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]),
                        self.mad_avg2[run2_only], s=ss, color='b')
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        self.mad_avg1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        self.mad_avg2[run2_only], s=ss, color='b')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
        plt.ylabel('MAD ', fontsize=fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([mad_min, mad_max])
        
        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[0, 2])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        self.avg_temp_soft1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]),
                        self.avg_temp_soft2[run2_only], s=ss, color='b')
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        self.avg_temp_soft1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        self.avg_temp_soft2[run2_only], s=ss, color='b')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('soft assignment', fontsize = fs)
        #plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([temp_soft_min, temp_soft_max])
        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[1, 0])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        self.avg_outlier1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]),
                        self.avg_outlier2[run2_only], s=ss, color='b')
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        self.avg_outlier1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        self.avg_outlier2[run2_only], s=ss, color='b')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('outlier scores', fontsize = fs)
        #plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([outlier_min, outlier_max])
        
        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])
        
        


        ax = plt.subplot(gs[1, 1])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        self.peak_xcorr1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]),
                        self.peak_xcorr2[run2_only], s=ss, color='b')
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        self.peak_xcorr1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        self.peak_xcorr2[run2_only], s=ss, color='b')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('xcorr peak', fontsize = fs)
#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([peak_min, peak_max])
        
        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[1, 2])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_only]),
                        self.notch_xcorr1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.ptps2[run2_only]),
                        self.notch_xcorr2[run2_only], s=ss, color='b')
        else:
            plt.scatter(np.log(self.f_rates1[run1_only]),
                        self.notch_xcorr1[run1_only], s=ss, color='r')
            plt.scatter(np.log(self.f_rates2[run2_only]),
                        self.notch_xcorr2[run2_only], s=ss, color='b')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('xcorr notch', fontsize = fs)
#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([notch_min, notch_max])

        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[2, 0])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        np.log(self.f_rates1[run1_matched]),
                        s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]), 
                        np.log(self.f_rates2[run2_matched]),
                        s=ss, color='purple')
            plt.yticks(np.arange(fr_min, fr_max, 1), fr_ticks)
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        np.log(self.ptps1[run1_matched]), s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        np.log(self.ptps2[run2_matched]), s=ss, color='purple')
            plt.yticks(np.arange(ptp_min, ptp_max, 1), ptp_ticks)

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)
        
#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
                
        if use_ptp:
            plt.ylabel('Firing rates (Hz) ', fontsize=fs)
        else:
            plt.ylabel('PTP ', fontsize=fs)

        ax.tick_params(axis='both', which='both', labelsize=fs)
        

        plt.xlim([x_min-0.5, x_max+0.5])

        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[2, 1])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        self.mad_avg1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]),
                        self.mad_avg2[run2_matched], s=ss, color='purple')
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        self.mad_avg1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        self.mad_avg2[run2_matched], s=ss, color='purple')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

#         plt.xlabel('Firing rates (Hz)', fontsize = fs)
        plt.ylabel('MAD ', fontsize=fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([mad_min, mad_max])
        
        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])


        ax = plt.subplot(gs[2, 2])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        self.avg_temp_soft1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]),
                        self.avg_temp_soft2[run2_matched], s=ss, color='purple')
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        self.avg_temp_soft1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        self.avg_temp_soft2[run2_matched], s=ss, color='purple')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('soft assignment', fontsize = fs)
        #plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([temp_soft_min, temp_soft_max])

        ax.xaxis.set_tick_params(length=15, width=2)
        ax.set_xticklabels([])

            
        ax = plt.subplot(gs[3, 0])
        
        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        self.avg_outlier1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]),
                        self.avg_outlier2[run2_matched], s=ss, color='purple')
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        self.avg_outlier1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        self.avg_outlier2[run2_matched], s=ss, color='purple')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('outlier scores', fontsize = fs)
        if use_ptp:
            plt.xlabel('PTP', fontsize = fs)
        else:
            plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([outlier_min, outlier_max])

        
        ax = plt.subplot(gs[3, 1])

        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        self.peak_xcorr1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]),
                        self.peak_xcorr2[run2_matched], s=ss, color='purple')
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        self.peak_xcorr1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        self.peak_xcorr2[run2_matched], s=ss, color='purple')
        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('xcorr peak', fontsize = fs)
        
        if use_ptp:
            plt.xlabel('PTP', fontsize = fs)
        else:
            plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)

        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([peak_min, peak_max])

        
        ax = plt.subplot(gs[3, 2])
        if use_ptp:
            plt.scatter(np.log(self.ptps1[run1_matched]),
                        self.notch_xcorr1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.ptps2[run2_matched]),
                        self.notch_xcorr2[run2_matched], s=ss, color='purple')
        else:
            plt.scatter(np.log(self.f_rates1[run1_matched]),
                        self.notch_xcorr1[run1_matched], s=ss, color='g')
            plt.scatter(np.log(self.f_rates2[run2_matched]),
                        self.notch_xcorr2[run2_matched], s=ss, color='purple')

        plt.xticks(np.arange(x_min, x_max, 2), x_ticks)

        plt.ylabel('xcorr notch', fontsize = fs)
        if use_ptp:
            plt.xlabel('PTP', fontsize = fs)
        else:
            plt.xlabel('Firing rates (Hz)', fontsize = fs)
        ax.tick_params(axis='both', which='both', labelsize=fs)
        
        plt.xlim([x_min-0.5, x_max+0.5])
        plt.ylim([notch_min, notch_max])
        
        name1_only = self.run1_name + ' only'
        name2_only = self.run2_name + ' only'
        
        name1_matched = self.run1_name + ' matched'
        name2_matched = self.run2_name + ' matched'

#         ax = plt.subplot(gs[4,0])
#         if use_ptp:
#             ax.scatter(self.ptp, self.cos1, c='g', alpha = 0.5)
#             ax.scatter(self.ptp, self.cos2, c='purple', alpha = 0.5)
#             ax.set_xlabel('PTP')
#         else:
#             ax.scatter(self.fr, self.cos1, c = 'g', alpha = 0.5)
#             ax.scatter(self.fr, self.cos2, c = 'purple', alpha= 0.5)
#             ax.set_xlabel('Firing_rate (Hz)')
        
#         ax.xticks(np.arange(x_min, x_max, 2), x_ticks)
#         ax.tick_params(axis='both', which='both', labelsize=fs)
#         plt.xlim([x_min-0.5, x_max+0.5])
#         plt.ylim([outlier_min, outlier_max])
            
        
        

        gs = outer[1].subgridspec(1, 1, hspace=0, wspace=0)
        ax = plt.subplot(gs[0,0])
        
        custom_lines = [Line2D([0], [0], marker='o', color='w', label=name1_only,
                          markerfacecolor='r', markersize=fs),
                        Line2D([0], [0], marker='o', color='w', label=name2_only,
                          markerfacecolor='b', markersize=fs),
                        Line2D([0], [0], marker='o', color='w', label=name1_matched,
                          markerfacecolor='g', markersize=fs),
                        Line2D([0], [0], marker='o', color='w', label=name2_matched,
                          markerfacecolor='purple', markersize=fs)
                       ]
        ax.set_axis_off()
        ax.legend(handles=custom_lines, loc='upper left',
                  bbox_to_anchor=(-0.05, 1), ncol=4, fontsize=fs)



        #plt.tight_layout()
        #plt.subplots_adjust(wspace=0.2)

        if  use_ptp:
            plt.savefig(os.path.join(self.save_dir, 'figure_ptp.png'))
        else:
            plt.savefig(os.path.join(self.save_dir, 'figure_fr.png'))

        
        plt.show()

