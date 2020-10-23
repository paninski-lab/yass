import numpy as np
import os

def remove_small_units(templates, threshold=2.5, units_in=None):

    print (" .....PTP threshold: ", threshold)
    if units_in is None:
        n_units = templates.shape[0]
        units_in = np.arange(n_units)

    # compute ptp
    ptps = templates[units_in].ptp(1).max(1)
    
    # units_keep 
    units_keep = units_in[ptps>=threshold]

    return units_keep
