from yass.reader import READER
import numpy as np
import os
import parmap
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_cov(templates):

    dat = templates.transpose(0, 2, 1).reshape(-1, templates.shape[1])
    dat = dat[:, 30:-30]
    dat = dat[dat.ptp(1) > 0.5]
    dat = dat - np.mean(dat, 1, keepdims=True)
    dat = dat/np.std(dat, 1, keepdims=True)

    acov = np.ones(3)
    for ii in range(1,3):
        acov[ii] = np.mean(dat[:,ii:]*dat[:,:-ii])
    cov = np.eye(3)
    cov[0,1] = acov[1]
    cov[1,0] = acov[1]
    cov[1,2] = acov[1]
    cov[2,1] = acov[1]
    cov[0,2] = acov[2]
    cov[2,0] = acov[2]

    w, v = np.linalg.eig(cov)
    w[w<=0] = 1e-9
    cov_half = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    return cov_half


def get_noise_samples(cov_half, n_samples):
    return np.matmul(np.random.randn(n_samples, 3), cov_half)


def split_templates(templates_dir, ptp_threshold, n_updates, update_time):
    
    # load initial templates
    init_templates = np.load(
        os.path.join(templates_dir, 'templates_{}sec.npy').format(0))
    n_units, n_times, n_channels = init_templates.shape

    vis_chans = [None]*n_units
    templates_sparse = [None]*n_units
    for k in range(n_units):
        ptp_ = init_templates[k].ptp(0)
        vis_chan_ = np.where(ptp_ > ptp_threshold)[0]
        vis_chans[k] = vis_chan_[np.argsort(ptp_[vis_chan_])[::-1]]
        templates_sparse[k] = np.zeros((n_updates, n_times, len(vis_chans[k])), 'float32')

    for j in range(n_updates):
        templates = np.load(os.path.join(templates_dir, 'templates_{}sec.npy').format(
            update_time*j)).astype('float32')
        for k in range(n_units):
            templates_sparse[k][j] = templates[k, :, vis_chans[k]].T

    return templates_sparse, vis_chans

def get_data(k, vis_chans_all, templates_sparse, init_templates,
             spike_train, scales, shifts, reader_resid,
             n_updates, update_time, meta_data_dir, cov_half):
    
    vis_chan = vis_chans_all[k]
    templates_unit = templates_sparse[k]
    ptps_unit = templates_unit.ptp(1).astype('float32')
    
    n_units, n_times, n_channels = init_templates.shape

    # spike times, shifts and scales
    idx_ = np.where(spike_train[:, 1]==k)[0]
    if len(idx_) > 20000:
        idx_ = np.random.choice(idx_, 20000, False)
    spt_ = spike_train[idx_, 0]
    scale_ = scales[idx_]
    shift_ = shifts[idx_]

     # get residual
    residual, idx_skipped = reader_resid.read_waveforms(spt_, n_times=n_times, channels=vis_chan)
    spt_ = np.delete(spt_, idx_skipped)
    scale_ = np.delete(scale_, idx_skipped)
    shift_ = np.delete(shift_, idx_skipped)
    spt_sec = spt_/reader_resid.sampling_rate

    # get residual variance and residual ptp
    residual_variance = np.zeros((n_updates, len(vis_chan)), 'float32')
    ptps_clean = np.zeros((len(residual), len(vis_chan)), 'float32')
    ptps_unit = np.zeros((n_updates, len(vis_chan)), 'float32')
    
    for j in range(n_updates):
        t_start = update_time*j
        t_end = update_time*(j+1)
        idx_ = np.where(np.logical_and(spt_sec >= t_start, spt_sec < t_end))[0]

        residual_variance[j] = np.var(residual[idx_], axis=0).mean(0)

        min_loc = templates_unit[j].argmin(0)
        max_loc = templates_unit[j].argmax(0)
        min_loc[min_loc < 1] = 1
        max_loc[max_loc < 1] = 1
        min_loc[min_loc > 99] = 99
        max_loc[max_loc > 99] = 99
        #min_max_loc = np.stack((min_loc, max_loc), 1)

        # get residual ptp
        t_range = np.arange(-1, 2)
        for ii in range(len(vis_chan)):

            f = interp1d(np.arange(n_times), templates_unit[j, :, ii],
                         'cubic', fill_value='extrapolate')
            mins_temp = f((min_loc[ii] + t_range)[:, None] - shift_[idx_]).T
            maxs_temp = f((max_loc[ii] + t_range)[:, None] - shift_[idx_]).T
            
            mins_temp *= scale_[idx_][:, None]
            maxs_temp *= scale_[idx_][:, None]

            mins_ = mins_temp + residual[idx_][:, min_loc[ii] + t_range, ii]
            maxs_ = maxs_temp + residual[idx_][:, max_loc[ii] + t_range, ii]

            ptps_clean[idx_, ii] = (maxs_.max(1) - mins_.min(1))#*scale_[idx_]
            
            noise_sample = get_noise_samples(cov_half, 2*len(idx_))
            ptps_unit[j, ii] = np.mean(np.max(maxs_temp + noise_sample[:len(idx_)], 1) - 
                                       np.min(mins_temp + noise_sample[len(idx_):], 1), 0)

    # save meta data
    fname_meta_data = os.path.join(meta_data_dir, 'unit_{}.npz'.format(k))
    np.savez(fname_meta_data,
             ptps_clean=ptps_clean,
             ptps_unit=ptps_unit,
             residual_variance=residual_variance,
             vis_chan=vis_chan,
             update_time=update_time,
             spt_sec=spt_sec
            )

def get_plot_ptps(save_dir, fname_raw, fname_residual,
                  fname_spike_train,  fname_scales,
                  fname_shifts, templates_dir,
                  ptp_threshold, n_col, CONFIG,
                  units_in=None,
                  fname_drifts_gt=None,
                  n_nearby_units=3
                 ):

    reader_raw = READER(fname_raw, 'float32', CONFIG)
    reader_resid = READER(fname_residual, 'float32', CONFIG)
    update_time = CONFIG.deconvolution.template_update_time

    # load initial templates
    init_templates = np.load(
        os.path.join(templates_dir, 'templates_{}sec.npy').format(0))
    n_units = init_templates.shape[0]
    
    
    meta_data_dir = os.path.join(save_dir, 'meta_data')
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)

    figs_dir = os.path.join(save_dir, 'figs')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    if units_in is None:
        units_in = np.arange(n_units)
    units_in = units_in[units_in < n_units]

    get_plot_ptps_parallel(
        units_in,
        reader_raw,
        reader_resid,
        fname_spike_train,
        fname_scales,
        fname_shifts, 
        templates_dir,
        meta_data_dir,
        figs_dir,
        update_time, 
        ptp_threshold,
        n_col,
        fname_drifts_gt,
        n_nearby_units
    )

def get_plot_ptps_parallel(
    units_in, reader_raw, reader_resid,
    fname_spike_train, fname_scales,
    fname_shifts, templates_dir,
    meta_data_dir, figs_dir, update_time, 
    ptp_threshold, n_col, fname_drifts_gt=None, n_nearby_units=3):

    # basic info about templates update
    n_updates = int(np.ceil(reader_resid.rec_len/reader_raw.sampling_rate/update_time))

    # load initial templates
    init_templates = np.load(
        os.path.join(templates_dir, 'templates_{}sec.npy').format(0))
    n_units, n_times, n_channels = init_templates.shape
    
    cov_half = get_cov(init_templates)

    # spike train
    spike_train = np.load(fname_spike_train)
    
    f_rates = np.zeros(n_units)
    a, b = np.unique(spike_train[:, 1], return_counts=True)
    f_rates[a] = b
    f_rates = f_rates/(reader_raw.rec_len/20000)
    
    # scale and shifts
    scales = np.load(fname_scales)
    shifts = np.load(fname_shifts)

    fname_templates_sprase = os.path.join(meta_data_dir, 'templates_sparse.npy')
    fname_vis_chans = os.path.join(meta_data_dir, 'vis_chans.npy')
    if not os.path.exists(fname_templates_sprase):
        templates_sparse, vis_chans_all = split_templates(
            templates_dir, ptp_threshold, n_updates, update_time)
        
        #np.save(fname_templates_sprase, templates_sparse, allow_pickle=True)
        #np.save(fname_vis_chansi, vis_chans_all, allow_pickle=True)

    else:
        templates_sprase = np.load(fname_templates_sprase, allow_pickle=True)
        vis_chans_all = np.load(fname_vis_chans, allow_pickle=True)
        
    ptps_all = init_templates.ptp(1)
    nearby_units = np.zeros((len(units_in), n_nearby_units), 'int32')
    for ii, unit in enumerate(units_in):
        nearby_units[ii] = np.argsort(
            np.square(ptps_all - ptps_all[unit]).sum(1))[:n_nearby_units]

    # order units in
    #units_in_ordered = units_in[np.argsort(init_templates[units_in].ptp(1).max(1))[::-1]]
    #units_in_ordered = units_in
    run_order = np.argsort(init_templates[units_in].ptp(1).max(1))[::-1]

    if fname_drifts_gt is not None:
        drifts_gt = np.load(fname_drifts_gt)
    else:
        drifts_gt = None
        
    for idx_ in run_order:
        unit = units_in[idx_]
        print(unit)
        nearby_units_ = nearby_units[idx_]
        for k in nearby_units_:
            # save meta data
            fname_meta_data = os.path.join(meta_data_dir, 'unit_{}.npz'.format(k))
            if not os.path.exists(fname_meta_data):
                get_data(k, vis_chans_all, templates_sparse, init_templates,
                 spike_train, scales, shifts, reader_resid,
                 n_updates, update_time, meta_data_dir, cov_half)


        vis_chan = vis_chans_all[unit]
        templates_unit = templates_sparse[unit]
        ptp_unit_init = init_templates[unit, :, vis_chan].ptp(1)

        colors = ['k', 'red', 'blue', 'yellow']

        # make figs
        n_row = int(np.ceil(len(vis_chan)/n_col))
        ptp_max_ = np.round(float(ptp_unit_init.max()), 1)
        fname_out = 'ptp_{}_unit{}_clean_ptp.png'.format(int(ptp_max_), unit)
        fname_out = os.path.join(figs_dir, fname_out)
        if not os.path.exists(fname_out):
            
            fig = plt.figure(figsize=(3*n_col, 3*n_row))

            for jj, k in enumerate(nearby_units_):
                
                if jj == 0:
                    fname_meta_data = os.path.join(meta_data_dir, 'unit_{}.npz'.format(k))            
                    temp = np.load(fname_meta_data, allow_pickle=True)
                    ptps_clean = temp['ptps_clean']
                    ptps_unit = temp['ptps_unit']
                    vis_chan = temp['vis_chan']
                    update_time = temp['update_time']
                    spt_sec = temp['spt_sec']
                    residual_variance = temp['residual_variance']

                    if len(spt_sec) == 0:
                        continue

                    alpha = np.min((0.05*20000/len(spt_sec), 0.5))
                    for ii in range(len(vis_chan)):
                        plt.subplot(n_row, n_col, ii+1)
                        plt.scatter(spt_sec, ptps_clean[:, ii], c='k', s=10, alpha=0.05)
                        
                        for j in range(n_updates):
                            t_start = update_time*j
                            t_end = update_time*(j+1)
                            ptp_ = ptps_unit[j, ii]
                            plt.plot([t_start, t_end], [ptp_, ptp_], 'orange', linewidth=5)

                        if drifts_gt is not None:
                            c = vis_chan[ii]
                            ptp_start = init_templates[k, :, c].ptp()
                            ptp_end = ptp_start*drifts_gt[k, c]
                            plt.plot([0, reader_raw.rec_len/reader_raw.sampling_rate],
                                     [ptp_start, ptp_end], 'b', linewidth=3)

                        plt.title('Channel {}'.format(vis_chan[ii]))
                
                else:
                    fname_meta_data = os.path.join(meta_data_dir, 'unit_{}.npz'.format(k))            
                    temp = np.load(fname_meta_data, allow_pickle=True)
                    ptps_clean_k = temp['ptps_clean']
                    vis_chan_k = temp['vis_chan']
                    spt_sec_k = temp['spt_sec']
                    
                    if len(spt_sec_k) == 0:
                        continue

                    alpha = np.min((0.005*20000/len(spt_sec_k), 0.5))

                    for ii in range(len(vis_chan)):

                        if np.any(vis_chan_k == vis_chan[ii]):
                            ii2 = np.where(vis_chan_k==vis_chan[ii])[0][0]
                            plt.subplot(n_row, n_col, ii+1)
                            plt.scatter(spt_sec_k, ptps_clean_k[:, ii2],
                                        c=colors[jj], s=3, alpha=alpha)

            f_rate_print = np.round(float(f_rates[unit]), 1)
            suptitle = 'Clean PTP, Unit {}, PTP {}, {}Hz'.format(unit, ptp_max_, f_rate_print)
            for jj in range(1, n_nearby_units):
                ptp_print = np.round(float(ptps_all[nearby_units_[jj]].max()),1)
                f_rate_print = np.round(float(f_rates[nearby_units_[jj]]), 1)
                suptitle += '\n{}: '.format(colors[jj])
                suptitle += ' ID {}, PTP {}, {}Hz '.format(nearby_units_[jj], ptp_print, f_rate_print)
            plt.suptitle(suptitle, fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])

            plt.savefig(fname_out)
            plt.close()