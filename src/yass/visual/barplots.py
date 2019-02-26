import numpy as np
import os
import parmap
import scipy.spatial
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

def match_units(selected_unit, fname_templates, fname_spike_train, 
                fname_templates_ground_truth, fname_spike_train_ground_truth, 
                purity_threshold, cos_sim_thresh, save_dir, rerun=True):
    
    templates_ground_truth = np.load(fname_templates_ground_truth)#[10:-10]
    spike_train_ground_truth = np.load(fname_spike_train_ground_truth)
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)
        
    fname_out = os.path.join(save_dir, 'match_unit_{}.npz'.format(selected_unit))
    if (os.path.exists(fname_out)==False) or rerun:
        #print('matching gt unit {}'.format(selected_unit))
        ''' **** MATCH USING TEMPLATES ****
        '''
        
        ctr_clr=1
        match_ids = []
        shifts = []
        units = np.arange(templates.shape[2])
        
        data1 = templates_ground_truth[:,:,selected_unit].T
        for unit in units:
            data2 = templates[:,:,unit].T.ravel()
            best_result = 0
            shift = 0
            for k in range(-10,10,1):
                data_test = np.roll(data1,k,axis=1).ravel()
                result = 1 - scipy.spatial.distance.cosine(data_test,data2)
                if result>best_result:
                    best_result = result
                    shift = -k
                
            # if poor match go to next unit
            if best_result < cos_sim_thresh: 
                continue
            #print (" GT Unit: ", selected_unit, " cos sim: ", best_result, "matching sorted unit: ", unit)
            match_ids.append(unit)
            shifts.append(shift)

        ''' **** COUNT MATCHING SPIKES FOR THE MATCHING UNITS ****
        '''
        # count spike time matches within window
        max_dist = 5
        spt_gt = spike_train_ground_truth[spike_train_ground_truth[:,1]==selected_unit,0]
        
        spike_match_array=np.zeros((len(match_ids), len(spt_gt)), 'bool')
        n_spikes_found = np.zeros(len(match_ids), 'int32')
        # keep only units with high enough purity
        idx_keep = np.zeros(len(match_ids), 'bool')
        for ctr, match_id in enumerate(match_ids):
            
            shift = shifts[ctr]
            # need to offset spiketimes for kilosort

            spt = spike_train[spike_train[:,1]==match_id, 0] + shift
            #print ("GT unit: ",selected_unit, " sorted match: ",match_id, " matching spikes found: ", n_spikes_found[ctr])
            
            spike_match = np.zeros(len(spt_gt), 'bool')
            for ii, s in enumerate(spt_gt):
                if np.min(np.abs(spt - s)) < max_dist:
                    spike_match[ii] = True
            
            if np.sum(spike_match) > purity_threshold*len(spt):
                spike_match_array[ctr] = spike_match
                n_spikes_found[ctr] = len(spt)
                idx_keep[ctr] = True
        
        match_ids = np.array(match_ids)[idx_keep]
        spike_match_array = spike_match_array[idx_keep]
        n_spikes_found = n_spikes_found[idx_keep]
        
        n_matched = np.sum(spike_match_array, axis=1)
        idx = np.argsort(n_matched)[::-1]
        
        matched_spikes = []
        unmatched_spikes = []
        already_matched = np.zeros(len(spt_gt), 'bool')
        for ii in idx:
            spike_match = spike_match_array[ii]
            spike_match[already_matched] = False
            
            matched_spikes.append(np.sum(spike_match))
            unmatched_spikes.append(n_spikes_found[ii]-matched_spikes[-1])
            already_matched[spike_match] = True
        
        match_ids = np.array(match_ids)[idx]
            
        np.savez(fname_out, match_ids = match_ids,
                 matched_spikes = matched_spikes,
                 unmatched_spikes = unmatched_spikes)
        

def make_purity_complete_plots(ax, idx_sorted, all_matched_spikes, all_unmatched_spikes, n_spikes_gt, title):
    
    colors = ['black','blue','red','green','cyan','magenta','brown','pink',
    'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
    'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
    'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
    'darkmagenta','yellow','hotpink']
    
    x_ = 0
    for ctr in idx_sorted:

        matched_spikes = all_matched_spikes[ctr]
        unmatched_spikes = all_unmatched_spikes[ctr]
        idx = np.argsort(matched_spikes)[::-1]
        matched_spikes = matched_spikes[idx]
        unmatched_spikes = unmatched_spikes[idx]        

        clr_ctr=1
        cum_sum = 0

        for k in range(len(matched_spikes)):
            temp_val = matched_spikes[k]/float(n_spikes_gt[ctr])*1E2
            plt.bar(x_, temp_val, bottom=cum_sum, width=.95, color=colors[clr_ctr]) #, bottom=np.sum(times[:k]), color=clrs[k])
            cum_sum+= temp_val

            temp_val = unmatched_spikes[k]/float(n_spikes_gt[ctr])*1E2
            plt.bar(x_, temp_val, bottom=cum_sum, width=.95, color=colors[clr_ctr],  hatch='//') #, bottom=np.sum(times[:k]), color=clrs[k])
            cum_sum+= temp_val

            clr_ctr+=1
        x_+=1

    #idx_sorted=np.delete(idx_sorted)
    plt.xticks(np.arange(len(idx_sorted)),idx_sorted)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot([-0.5,len(idx_sorted)-0.5],[100,100],'r--', color='black')
    plt.xlim(-0.5, len(idx_sorted)-0.5)    
    plt.ylim(0,200)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel("Ground truth unit ID", fontsize=20)
    plt.ylabel("Percent of spikes found \n(colors represent different units;\nhatches are unmatched spikes)",
               fontsize=20)
    plt.title(title, fontsize=20)
    
    return ax


def make_ptp_fr_plot(ax, idx_sorted, ptps, n_spikes_gt):

    units = np.arange(len(idx_sorted))
    
    # **************** PTP SCATTER PLOT *****************
    plt.scatter(units,ptps[idx_sorted])
    plt.xticks(np.arange(len(idx_sorted)),idx_sorted)
    plt.tick_params(axis='both', which='both', labelsize=10)
    plt.ylabel("PTP (SU) (blue)",fontsize=20)
    plt.xlim(-0.5, units.shape[0]-0.5)
    plt.ylim(0,25)

    x = units -0.5 
    y = np.zeros(x.shape)
    plt.fill_between(x, y+5, y+10, color='grey', alpha=0.1)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel("Unit ID (ptp order)",fontsize=20)

    # ***************** FIRING RATE BAR PLOT *************
    ax2 = ax.twinx()
    rec_len = 1200.
    for x_, ctr in enumerate(idx_sorted):
        plt.bar(x_, n_spikes_gt[ctr]/rec_len, width=.95, color='black',alpha=.5)

    plt.plot([-0.5,len(idx_sorted)],[1,1],'r--',c='black')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel("Firing rate (Hz) (grey bars)", fontsize=20)
    plt.xlabel("Unit ID (ptp order)",fontsize=20)
    plt.xlim(-0.5,units.shape[0]-0.5)
    plt.xticks(np.arange(len(idx_sorted)),idx_sorted)
    plt.yticks([1,10,20])
    
    return ax

def compare_bar_plot(fname_templates_ground_truth, fname_spike_train_ground_truth, 
                  dir1, dir2, title1, title2, fname_fig):

    templates_ground_truth=np.load(fname_templates_ground_truth)
    spike_train_ground_truth = np.load(fname_spike_train_ground_truth)

    ptps = templates_ground_truth.ptp(0).max(0)
    idx_sorted = np.argsort(ptps)[::-1]

    n_units_gt = templates_ground_truth.shape[2]
    n_spikes_gt = np.zeros(n_units_gt, 'int32')
    unique_ids, unique_n_spikes_gt = np.unique(spike_train_ground_truth[:,1], return_counts=True)
    n_spikes_gt[unique_ids] = unique_n_spikes_gt
 
    units = np.arange(n_units_gt)
 
    # Load all the match data
    all_matched_spikes1 = []
    all_unmatched_spikes1 = []
    for k in units:
        fname_out = os.path.join(dir1, 'match_unit_{}.npz'.format(k))
        data = np.load(fname_out)
        match_ids = data['match_ids']
        matched_spikes = data['matched_spikes']
        unmatched_spikes = data['unmatched_spikes']

        all_matched_spikes1.append(matched_spikes)
        all_unmatched_spikes1.append(unmatched_spikes)
        
    # Load all the match data
    all_matched_spikes2 = []
    all_unmatched_spikes2 = []
    for k in units:
        fname_out = os.path.join(dir2, 'match_unit_{}.npz'.format(k))
        data = np.load(fname_out)
        match_ids = data['match_ids']
        matched_spikes = data['matched_spikes']
        unmatched_spikes = data['unmatched_spikes']

        all_matched_spikes2.append(matched_spikes)
        all_unmatched_spikes2.append(unmatched_spikes)

    ##############
    # MAKE PLOTS #
    ##############
    
    plt.figure(figsize=(15,20))

    # **************** BAR PLOTS *******************
    ax=plt.subplot(3,1,1)
    _, make_purity_complete_plots(
        ax, idx_sorted, all_matched_spikes1,
        all_unmatched_spikes1, n_spikes_gt, title1)
    
    ax=plt.subplot(3,1,2)
    _, make_purity_complete_plots(
        ax, idx_sorted, all_matched_spikes2,
        all_unmatched_spikes2, n_spikes_gt, title2)
    
    # ************ PTP and Firing rates ************
    ax=plt.subplot(3,1,3)
    _, make_ptp_fr_plot(ax, idx_sorted, ptps, n_spikes_gt)
 

    plt.savefig(fname_fig)
    plt.show()