import numpy as np
import os

from yass.correlograms_phy import compute_correlogram

def remove_high_xcorr_peaks(save_dir,
                            fname_spike_train,
                            fname_templates,
                            sampling_rate,
                            threshold=5,
                            units_in=None):
    
    if not os.path.exists(save_dir):
        os.makedir(save_dir)

    window_size = 0.04
    bin_width = 0.001
        
    spike_train = np.load(fname_spike_train)
    templates = np.load(fname_templates)
    n_units = templates.shape[0]
    
    if units_in is None:
        units_in = np.arange(n_units)

    units_in_orig = np.copy(units_in)
    
    # only consider units with more than 0 spikes
    unique_units = np.unique(spike_train[:, 1])
    units_in = units_in[np.in1d(units_in, unique_units)]
    
    ptp_sum = templates.ptp(1).sum(1)
    ptp_sum = ptp_sum[units_in]

    xcorrs = compute_correlogram(
        units_in,
        spike_train,
        None,
        sample_rate=sampling_rate,
        bin_width=bin_width,
        window_size=window_size)
    np.save(os.path.join(save_dir, 'xcorrs.npy'), xcorrs)

    means_ = xcorrs.mean(2)
    stds_ = np.std(xcorrs, 2)
    stds_[stds_==0] = 1
    xcorrs_standardized = (xcorrs - means_[:,:,None])/stds_[:,:,None]

    a, b = np.where(xcorrs_standardized.max(2) > threshold)
    kill = []
    for j in range(len(a)):
        k1 = a[j]
        k2 = b[j]

        if xcorrs[k1,k2].max() > 10:
            if ptp_sum[k1] > ptp_sum[k2]:
                kill.append(k2)
            else:
                kill.append(k1)

    units_killed = units_in[np.unique(np.array(kill))]
    units_out = units_in[~np.in1d(units_in, units_killed)]

    return units_out
