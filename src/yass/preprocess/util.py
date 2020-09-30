"""
Filtering functions
"""
import logging
import os
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt


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
    
    # save
    fname = os.path.join(
        output_directory,
        "standardized_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, ts.astype(out_dtype))

    #fname = os.path.join(
    #    output_directory,
    #    "standardized_{}.bin".format(
    #        str(batch_id).zfill(6)))
    #f = open(fname, 'wb')
    #f.write(ts.astype(out_dtype))



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

def make_histograms(batch_id, reader, output_directory, num_chan, sample_rate, template_time, voltage_threshold, 
    length_um, num_bins, num_y_pos, quantile_comp, br_quantile, neighboring_chan, M):
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

    print("ts SHAPE :")
    print(ts.shape)

    ### Detect spikes by simple thresholding ###
    num_timesteps = ts.shape[1]
    threshold_spikes = []
    num_sliding_windows = int(num_timesteps/(template_time/2))
    for i in range(num_sliding_windows):
        ptp = ts[:, i*int(template_time/2):(i+1)*(int(template_time/2))].ptp(1)
        top_chan = ptp.argmax()
        if ptp[top_chan] > voltage_threshold:
            threshold_spikes.append(ts[top_chan, i*int(template_time/2):(i+1)*(int(template_time/2))].argmin() + i*int(template_time/2))

    spike_times = np.array(threshold_spikes)
    
    ### Make histograms ###

    hist_arrays = np.zeros((num_bins, num_y_pos))

    #### Get 0.9 quantile per channel ####
    timepoints_bin = int((num_timesteps-template_time)/quantile_comp)
    ptp_sliding = np.zeros((num_chan, timepoints_bin)) #Sampling Rate 
    for j in range(timepoints_bin):
        ptp_sliding[:, j] = ts[:, int(j*quantile_comp):int(j*quantile_comp + template_time)].ptp(1)
        quantile = np.quantile(ptp_sliding, br_quantile, axis = 1)
    
    #### Make histograms ####
    for spike_t in spike_times:
        ptp = np.maximum(ts[:, max(int(spike_t-template_time/2), 0):min(int(spike_t + template_time/2), num_timesteps)].ptp(1) - quantile, 0) 
        top_chan = ptp.argmax()
        if (top_chan+neighboring_chan/2 < num_chan):
            ptp[(top_chan+int(neighboring_chan/2)):] = 0
        if (top_chan-neighboring_chan/2 > 0):
            ptp[:(top_chan-int(neighboring_chan/2))] = 0
        electrode_ptp_int = np.log1p(np.matmul(ptp, M))
        hist_plot = np.histogram2d(electrode_ptp_int, np.arange(0, length_um), bins=(num_bins, num_y_pos), range = [[np.log(voltage_threshold), 6.0], [0, length_um]])[0]
        hist_arrays += hist_plot

    log_hist_arrays = np.log1p(hist_arrays) #Take log counts 

    ####### Save histogram arrays #######

    fname = os.path.join(
        output_directory,
        "histogram_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, log_hist_arrays)

    print("HISTOGRAMS CREATED !!!!!!!!!")

def register_data(batch_id, reader, output_directory, estimated_displacement, out_dtype):
    """Create histograms for each batches, before registration
    Parameters
    ----------
    estimated_displacement : np array containing displacements 
    """
    logger = logging.getLogger(__name__)
    ts = reader.read_data_batch(batch_id, add_buffer=False)
    ts = ts.T

    new_data = np.zeros(ts.shape)
    for i in range(num_hist):
        new_data = griddata(geomarray, ts, geomarray - [0, -estimated_displacement[batch_id]])

    # save
    fname = os.path.join(
        output_directory,
        "registered_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(new_data, new_data.T.astype(out_dtype))



def merge_filtered_files(filtered_location, output_directory, delete=True):

    logger = logging.getLogger(__name__)

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standardized.bin")
    logger.info('...saving standardized file: %s', f_out)

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        res = np.load(os.path.join(filtered_location, fname))
        res.tofile(f)
        if delete==True:
            os.remove(os.path.join(filtered_location, fname))
