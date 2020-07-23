import numpy as np
import os

def remove_off_centered_units(templates, threshold=5, units_in=None):

    if units_in is None:
        n_units = templates.shape[0]
        units_in = np.arange(n_units)

    # compute min points
    mc = templates[units_in].ptp(1).argmax(1)
    min_points = np.zeros(len(units_in))
    for j in range(len(units_in)):
        min_points[j] = templates[units_in[j]][:, mc[j]].argmin()
    idx_keep = np.abs(min_points - np.median(min_points)) <= threshold

    # units_keep 
    units_keep = units_in[idx_keep]

    return units_keep
