import numpy as np
import sklearn.decomposition as decomp
import scipy.optimize as opt


def get_fit_on_sta(sta_array):
    """fit sta to get spaital and temporal sta and also gaussian fit
    
    Input:
        sta_array: array of shape (# units, stim size 0, stim size 1, # color channels, # frames)

    Output:
        spatial_sta: array of shape (# units, stim size 0, stim size 1)
        temporal_sta: array of shape (# units, # color channels, # frames)
        gaussian_fit: array of shape (# units, 6)
                      column 1: amplitude of gaussian fit
                      column 2,3: x, y location
                      column 4,5: x, y sd size
                      column 6: angle
    """
    n_units, stim_size0, stim_size1, n_channels, n_frames = sta_array.shape
    
    spatial_sta = np.zeros((n_units, stim_size0, stim_size1))
    temporal_sta = np.zeros((n_units, n_channels, n_frames))
    gaussian_fit = np.zeros((n_units, 6))

    max_per_frame = np.max(np.abs(sta_array[:,:,:,1]-0.5), (1,2))
    max_frames = max_per_frame.argmax(1)
    max_val = max_per_frame.max(1)
    peak_frame = int(np.median(max_frames))
    frames_in = np.arange(peak_frame-1, peak_frame+2)

    for j in range(sta_array.shape[0]):
        spatial_sta[j], temp_, gau_ =fit_sta(sta_array[j], frames_in)
        if temp_ is not None:
            temporal_sta[j] = temp_
            gaussian_fit[j] = gau_

    temporal_sta = denoise_sta(temporal_sta, frames_in)
    
    return spatial_sta, temporal_sta, gaussian_fit


def fit_sta(sta, frames_in):

    stim_size0, stim_size1, n_channel, n_frames = sta.shape
    
    # center it
    sta = sta - 0.5
    
    # get spatial sta
    spatial_sta = np.mean(sta[:, :, 1, frames_in], 2)

    # Create x and y indices for grid for Gaussian fit
    x = np.arange(0, stim_size1, 1)
    y = np.arange(0, stim_size0, 1)
    x, y = np.meshgrid(x, y)

    # Get initial guess for Gaussian parameters (helps with fitting)
    this_STA = spatial_sta.reshape(-1)
    init_amp = this_STA[np.argmax(np.abs(this_STA))] # get amplitude guess from most extreme (max or min) amplitude of this_STA
    init_x,init_y = np.unravel_index(np.argmax(np.abs(this_STA)),(stim_size0, stim_size1)) # guess center of Gaussian as indices of most extreme (max or min) amplitude
    initial_guess = (init_amp,init_y,init_x,2,2,0)

    # Try to fit, if it doesn't converge, log that cell
    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), this_STA, initial_guess)
        gaussian_param = np.copy(popt)
        gaussian_param[3:5] = np.abs(popt[3:5]) # sometimes sds are negative (in Gaussian def above, they're always squared)
    
        # sign of fit
        sign = np.sign(gaussian_param[0])
        
        # get temporal sta
        gaussian_image = twoD_Gaussian((x,y), *gaussian_param).reshape(stim_size0, stim_size1)
        temporal_sta = sign*np.sum(sta*gaussian_image[:,:,None,None], (0, 1))

    except:
        temporal_sta = None
        gaussian_param = None
        
    return spatial_sta, temporal_sta, gaussian_param


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):
    ## Define 2D Gaussian that we'll fit to spatial STAs
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g =  amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
    return g.ravel()


def denoise_sta(temporal_sta, frames_in):
    temporal_sta_reshaped = temporal_sta.reshape(-1, temporal_sta.shape[2])
    temp_sta_mean = np.mean(temporal_sta_reshaped, 1, keepdims=True)
    temp_sta_std = np.std(temporal_sta_reshaped, 1, keepdims=True)
    temp_sta_std[temp_sta_std==0] = 1
    temporal_sta_std = (temporal_sta_reshaped - temp_sta_mean)/temp_sta_std
    good_sta = temporal_sta_reshaped[np.abs(temporal_sta_std[:, frames_in]).max(1) > 3]
    
    pca = decomp.PCA(n_components = 3)
    pca.fit(good_sta)

    temporal_sta_denoised = np.zeros_like(temporal_sta)
    for j in range(temporal_sta.shape[1]):
        temp_ = temporal_sta[:,j]
        temporal_sta_denoised[:,j] = pca.inverse_transform(pca.transform(temp_))
        
    return temporal_sta_denoised
