"""
Filtering functions
"""
import logging
import os
import numpy as np
import math
from scipy.interpolate import griddata, interp2d
from scipy.signal import butter, filtfilt
import torch
from fast_histogram import histogram2d


def _butterworth(ts, low_frequency, high_factor, order, sampling_frequency):
    """Butterworth filter
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    low = float(low_frequency) / sampling_frequency * 2
    high = float(high_factor) * 2
    b, a = butter(order, low, btype='high', analog=False)
    
    if ts.ndim == 1:
        return filtfilt(b, a, ts)
    else:
        T, C = ts.shape
        output = np.zeros((T, C), 'float32')
        for c in range(C):
            output[:, c] = filtfilt(b, a, ts[:, c])

        return output


def _mean_standard_deviation(rec, centered=False):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
    centered : bool
        if not standardized, center it
    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # find standard deviation using robust method
    if not centered:
        centers = np.mean(rec, axis=0)
        rec = rec - centers[None]
    else:
        centers = np.zeros(rec.shape[1], 'float32')

    return np.median(np.abs(rec), 0)/0.6745, centers


def _standardize(rec, sd=None, centers=None):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
        recording
    sd : vector [number of chnanels,]
        standard deviation
    centered : bool
        if not standardized, center it
    Returns
    -------
    matrix [length of recording, number of channels]
        standardized recording
    """

    # find standard deviation using robust method
    if (sd is None) or (centers is None):
        sd, centers = _mean_standard_deviation(rec, centered=False)

    # standardize all channels with SD> 0.1 (Voltage?) units
    # Cat: TODO: ensure that this is actually correct for all types of channels
    idx1 = np.where(sd>=0.1)[0]
    rec[:,idx1] = np.divide(rec[:,idx1] - centers[idx1][None], sd[idx1])
    
    # zero out bad channels
    idx2 = np.where(sd<0.1)[0]
    rec[:,idx2]=0.
    
    return rec
    #return np.divide(rec, sd)

def whiten(ts):
    
    wrot = np.load('/ssd/nishchal/temp_/new/neuropixels-data-sep-2020/scripts/recordings/wrot.npy')
    return np.matmul(wrot, ts.T).T

def filter_standardize_batch(batch_id, reader, fname_mean_sd,
                             apply_filter, out_dtype, output_directory,
                             low_frequency=None, high_factor=None,
                             order=None, sampling_frequency=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """
    logger = logging.getLogger(__name__)
    
    if os.path.exists(os.path.join(
        output_directory,
        "filtered_{}.npy".format(
            str(batch_id).zfill(6)))):
        return
    
        
    # filter
    if apply_filter:
        # read a batch
        ts = reader.read_data_batch(batch_id, add_buffer=True)
        ts = _butterworth(ts, low_frequency, high_factor,
                              order, sampling_frequency)
        ts = ts[reader.buffer:-reader.buffer]
    else:
        ts = reader.read_data_batch(batch_id, add_buffer=False)
    

    # standardize
    temp = np.load(fname_mean_sd)
    sd = temp['sd']
    centers = temp['centers']
    ts = _standardize(ts, sd, centers)
    
    ts = whiten(ts)
    
    # save
    fname = os.path.join(
        output_directory,
        "filtered_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, ts.astype(out_dtype))




def get_std(ts,
            sampling_frequency,
            fname,
            apply_filter=False, 
            low_frequency=None,
            high_factor=None,
            order=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    # filter
    if apply_filter:
        ts = _butterworth(ts, low_frequency, high_factor,
                          order, sampling_frequency)

    # standardize
    sd, centers = _mean_standard_deviation(ts)
    
    # save
    np.savez(fname,
             centers=centers,
             sd=sd)

def make_histograms(batch_id, reader, output_directory, output_directory_spikes, num_chan, sample_rate, template_time, voltage_threshold, 
    length_um, num_bins, num_y_pos, quantile_comp, br_quantile, iter_quantile, neighboring_chan, M):
    """Create histograms for each batches, before registration
    Parameters
    ----------
    template_time : length of a template 
    voltage_threshold : threshold for spike detection 
    length_um : length of the electrode 
    num_bins : number of bins for x axis of the histograms (log ptps)
    num_y_pos : number of bins for y axis of the histograms 
    quantile_comp : number of shifts for the sliding window for computing quantiles 
    br_quantile : background removal quantile to consider 
    neighboring_chan : number of neighboring channels in spatial radius 
    M : electrode2space mapping
    """
    logger = logging.getLogger(__name__)

    ts = reader.read_data_batch(batch_id, add_buffer=False)
    ts = ts.T

    ### Detect spikes by simple thresholding ###
    num_timesteps = ts.shape[1]    
    ### Make histograms ###

    hist_arrays = np.zeros((num_bins, num_y_pos))


    #### Get sliding window ptp ####
    timepoints_bin = int((num_timesteps-template_time)/quantile_comp)
    ptp_sliding = np.zeros((num_chan, timepoints_bin)) #Sampling Rate 
    for j in range(timepoints_bin):
        ptp_sliding[:, j] = ts[:, int(j*quantile_comp):int(j*quantile_comp + template_time)].ptp(1)

    ##### Sinkhorn denoising #####
    for k in range(iter_quantile): 
        quantile_s = np.quantile(ptp_sliding, br_quantile, axis = 0) #size num_timepoints
        ptp_sliding = np.maximum(ptp_sliding - quantile_s, 0)
        quantile_t = np.quantile(ptp_sliding, br_quantile, axis = 1) #size num_channels
        ptp_sliding = np.maximum((ptp_sliding.T - quantile_t).T, 0)
    
    #### Make histograms ####
    for spike_time in np.where(ptp_sliding.max(0)>=voltage_threshold)[0]:
        electrode_ptp_int = np.log1p(np.matmul(ptp, M))
        hist_plot = np.histogram2d(electrode_ptp_int, np.arange(0, num_y_pos), bins=(20, num_y_pos), range = [[np.log1p(voltage_threshold), np.log1p(5*voltage_threshold)], [0, num_y_pos]])[0]
        hist_arrays += hist_plot

    log_hist_arrays = np.log1p(hist_arrays) #Take log counts 
    ####### Save histogram arrays #######

    fname = os.path.join(
        output_directory,
        "histogram_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, log_hist_arrays)

    # print("HISTOGRAMS CREATED !!!!!!!!!")

def register_data(batch_id, filtered_directory, output_directory, estimated_displacement, geomarray, out_dtype):
    """Create histograms for each batches, before registration
    Parameters
    ----------
    estimated_displacement : np array containing displacements 
    """
    logger = logging.getLogger(__name__)
#     ts = reader.read_data_batch(batch_id, add_buffer=False)
    fname = os.path.join(
        filtered_directory,
        "filtered_{}.npy".format(
            str(batch_id).zfill(6)))
    ts = np.load(fname)
    ts = ts.T

    
    new_data = griddata(geomarray, ts, geomarray + [0, estimated_displacement[batch_id]], fill_value = 0)
    # Griddata

#     mat = np.zeros((geomarray.shape[0], geomarray.shape[0])) #displacement = 0
#     mat_bis = np.zeros((geomarray.shape[0], geomarray.shape[0]))
#     sigma = 20*np.sqrt(2)
#     sigma = 10
#     # if batch_id == 10:
#     #     print("GPR")
#     for i in range(geomarray.shape[0]):
#         mat_bis[:, i] = np.exp(-np.sum((geomarray-geomarray[i])**2, axis = 1)/(2*sigma**2))
#         mat[:, i] = np.exp(-np.sum((geomarray-geomarray[i]+[0, estimated_displacement[batch_id]])**2, axis=1)/(2*sigma**2))
#     mat += 0.01 * np.eye(mat.shape[0])
#     mat_bis += 0.01 * np.eye(mat.shape[0])
# #     mat[np.where(np.isnan(mat))]=0
# #     mat_bis /= mat_bis.sum(0)
# #     mat_bis[np.where(np.isnan(mat_bis))]=0
# # 
#     inv_mat = np.linalg.inv(mat_bis)
#     mat = np.matmul(mat, inv_mat)
# #     mat[np.where(np.isnan(mat))]=0
#     new_data = np.matmul(mat, ts)
    # save
    fname = os.path.join(
        output_directory,
        "registered_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, new_data.T.astype(out_dtype))

    if (batch_id == 100):
        fname = 'Matrix_Before_Registration.npy'
        np.save(fname, ts.T.astype(out_dtype))
        fname = 'Matrix_After_Registration.npy'
        np.save(fname, new_data.T.astype(out_dtype))

def make_histograms_gpu(batch_id, filtered_directory, output_directory, output_directory_spikes, num_chan, sample_rate, template_time, voltage_threshold, 
    length_um, num_bins, num_y_pos, quantile_comp, br_quantile, iter_quantile, neighboring_chan, M):
    """Create histograms for each batches, before registration
    Parameters
    ----------
    template_time : length of a template 
    voltage_threshold : threshold for spike detection 
    length_um : length of the electrode 
    num_bins : number of bins for x axis of the histograms (log ptps)
    num_y_pos : number of bins for y axis of the histograms 
    quantile_comp : number of shifts for the sliding window for computing quantiles 
    br_quantile : background removal quantile to consider 
    neighboring_chan : number of neighboring channels in spatial radius 
    M : electrode2space mapping
    """
    logger = logging.getLogger(__name__)
#     print("GPU")
#     ts = torch.from_numpy(reader.read_data_batch(batch_id, add_buffer=False)).float().cuda()
    
    fname = os.path.join(
        output_directory,
        "histogram_{}.npy".format(
            str(batch_id).zfill(6)))
    if os.path.exists(fname):
        return np.load(fname)[None]
    ts = torch.from_numpy(np.load(os.path.join(
        filtered_directory,
        "filtered_{}.npy".format(
            str(batch_id).zfill(6))))).float().cuda()

    ### Detect spikes by simple thresholding ###
    num_timesteps = ts.shape[1]

    ### Make histograms ###

    mp2d= torch.nn.MaxPool2d(kernel_size = [template_time, 1], stride = [quantile_comp,1])
    ptp_sliding = mp2d(ts[None])[0] + mp2d(-ts[None])[0]
    
#     print("what the fuck")
    
    for k in range(iter_quantile): 
        quantile_s = torch.kthvalue(ptp_sliding, int(br_quantile * ptp_sliding.shape[1]), dim = 1, keepdims = True)[0] #size num_timepoints
        ptp_sliding = torch.nn.functional.relu(ptp_sliding - quantile_s)
        quantile_t = torch.kthvalue(ptp_sliding, int(br_quantile * ptp_sliding.shape[0]), dim = 0, keepdims = True)[0] #size num_channels
        ptp_sliding = torch.nn.functional.relu(ptp_sliding - quantile_t)
        
    if batch_id == 5:
        print(ptp_sliding.sum())
    ptp_sliding = ptp_sliding.cpu().numpy()
    electrode_ptp_int = np.matmul(ptp_sliding,M)
    if batch_id == 5:
        print(electrode_ptp_int.sum(), np.arange(num_y_pos).max(), electrode_ptp_int.shape[0])
    
    bins = np.tile(np.arange(num_y_pos), electrode_ptp_int.shape[0])
    
    hist_arrays = histogram2d(electrode_ptp_int.ravel(), bins, bins= (num_bins, num_y_pos), range = [[voltage_threshold, 4*voltage_threshold], [0, num_y_pos]])
    if batch_id == 5:
        print(hist_arrays.ptp(), voltage_threshold, 2.5*voltage_threshold, num_bins, num_y_pos)
    
    hist_arrays /= (hist_arrays.sum(0, keepdims = True)+1e-6)
    if batch_id == 5:
        print(hist_arrays.ptp())
    log_hist_arrays = np.log1p(hist_arrays) #Take log counts
    ####### Save histogram arrays #######
    np.save(fname, log_hist_arrays)
    
    return(log_hist_arrays[None])

    # print("HISTOGRAMS CREATED !!!!!!!!!")
def calc_displacement(displacement, niter = 50):
    p = np.zeros(displacement.shape[0])
    pprev = p
    for i in range(niter):

        p = (displacement.sum(1) - (p-p.sum()))/(displacement.shape[0]-1)
        if np.allclose(p - p[0], pprev - pprev[0]):
            break
        else:
            pprev = p
    return p

def merge_filtered_files(filtered_location, output_directory, delete=True, op = 'standardize'):

    logger = logging.getLogger(__name__)

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    if op == 'standardize':
        f_out = os.path.join(output_directory, "standardized.bin")
        logger.info('...saving standardized file: %s', f_out)
    elif op == 'register':
        f_out = os.path.join(output_directory, "registered.bin")
        logger.info('...saving registered file: %s', f_out)

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        if '.ipynb' in fname:
            continue
        res = np.load(os.path.join(filtered_location, fname)).astype('float32')
        res.tofile(f)
        if delete==True:
            os.remove(os.path.join(filtered_location, fname))
