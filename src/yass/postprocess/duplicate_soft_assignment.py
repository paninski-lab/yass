import numpy as np

def duplicate_soft_assignment(fname_template_soft_assignment,
                              threshold=0.7, units_in=None):

    # load soft assignment
    tmp = np.load(fname_template_soft_assignment)
    probs_all = tmp['probs_templates']
    units_all = tmp['units_assignment']

    # number of units
    n_units = int(np.unique(units_all[:,0]).max() + 1)
    if units_in is None:
        units_in = np.arange(n_units)

    # compute avg template soft assignment per unit
    n_neigh = units_all.shape[1]
    probs = np.zeros((n_units, n_neigh), 'float32')
    units_neigh = np.zeros((n_units, n_neigh), 'int32')
    for k in range(n_units):
        idx_ = np.where(units_all[:, 0] == k)[0]

        if len(idx_) > 0:
            probs[k] = probs_all[idx_].mean(0)
            units_neigh[k] = units_all[idx_[0]]

    # do the comparison
    kill = np.zeros(n_units, 'bool')
    for k in units_in:

        # if the avg soft assignment is less than the threshold, do the comparison
        if probs[k,0] < threshold:

            # find the closest unit by soft assignment
            #k2 = units_neigh[k, probs[k,1:].argmax() + 1]

            for j2, k2 in enumerate(units_neigh[k, 1:]):
                # if the closest unit is stable (soft assignment > threshold),
                # then kill the unit
                #if probs[k2, 0] >= threshold:
                if probs[k, j2+1] > 0.1 and probs[k2, 0] >= probs[k,0]:
                    kill[k] = True

            # if the closest unit is also not stable, but
            # the closest unit of the closted unit is also the unit of interest, (i.e. paired)
            # then kill the less stable unit
            #else:
            #    k3 = units_neigh[k2, probs[k2, 1:].argmax() + 1]
            #    if k == k3 and probs[k2,0] > probs[k,0]:
            #        kill[k] = True
            
    # units not killed
    kill = np.where(kill)[0]
    units_out = units_in[~np.in1d(units_in, kill)]

    return units_out



