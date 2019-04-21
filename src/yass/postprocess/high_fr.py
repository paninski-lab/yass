import numpy as np
import os

def remove_high_fr_units(n_spikes, rec_len_sec,
                        threshold=70, units_in=None):

    if units_in is None:
        units_in = np.arange(len(n_spikes))

    # units_keep 
    units_keep = units_in[
        n_spikes[units_in]<=threshold*rec_len_sec]

    return units_keep