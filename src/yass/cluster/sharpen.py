import numpy as np 
from scipy.interpolate import interp1d

def sharpen_templates(fname_templates):

    templates = np.load(fname_templates)

    n_units, n_times, n_channels = templates.shape
    mcs = templates.ptp(1).argmax(1)

    t_range = np.arange(n_times)

    templates_new = np.zeros_like(templates)
    for unit in range(n_units):

        temp_ = templates[unit, :, mcs[unit]]
        f = interp1d(t_range, -temp_, 'cubic')
        shift = find_peak_binary_search(-temp_, f) - np.argmin(temp_)

        vis_chan = np.where(templates[unit].ptp(0) > 1)[0]
        temp_new = np.copy(templates[unit])
        t_range_new = t_range + shift
        for c in vis_chan:
            f = interp1d(t_range, templates[unit,:,c], 'cubic', fill_value='extrapolate')
            temp_new[:, c] = f(t_range_new)
        templates_new[unit] = temp_new    

    np.save(fname_templates, templates_new)
    
    return fname_templates

def find_peak_binary_search(obj, func, threshold= 0.0001):
    small, big = np.argsort(obj)[-2:]
    big_val = obj[big]
    small = small
    big = big
    mid = (small+big)/2
    mid_val = func(mid)
    
    while np.abs(big_val - mid_val) > threshold:
        if mid_val > big_val:
            small = big
            big = mid
            big_val = mid_val
            
        else:
            small = mid
        mid = (small+big)/2
        mid_val = func(mid)
    
    return big
