import numpy as np
import os

def remove_off_centered_units(templates, threshold=5, units_in=None):

    if units_in is None:
        n_units = templates.shape[0]
        units_in = np.arange(n_units)

    # compute min points
    min_points = templates[units_in].min(2).argmin(1)
    idx_keep = np.abs(min_points - np.mean(min_points)) <= threshold

    # units_keep 
    units_keep = units_in[idx_keep]

    return units_keep