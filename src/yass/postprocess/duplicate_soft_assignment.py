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
    no_spikes_units = []
    for k in range(n_units):
        idx_ = np.where(units_all[:, 0] == k)[0]

        if len(idx_) > 0:
            probs[k] = probs_all[idx_].mean(0)
            units_neigh[k] = units_all[idx_[0]]
        else:
            probs[k, 0] = 1
            no_spikes_units.append(k)
            units_neigh[k] = k

    # paired soft assignment 
    paired_probs = probs[:,[0]]/(probs[:,1:]+probs[:,[0]])
    paired_probs[no_spikes_units] = 0

    # for comparing which unit is more stable
    min_paired_probs = np.min(paired_probs, 1)

    # do the comparison
    pairs = []
    kill = np.zeros(n_units, 'bool')
    for k in units_in:
        # if the avg soft assignment is less than the threshold, do the comparison
        if np.any(paired_probs[k] < threshold):
            candidate_pairs = units_neigh[k, 1:][paired_probs[k] < threshold]
            for k2 in candidate_pairs:
                if min_paired_probs[k] < min_paired_probs[k2]:
                    pairs.append([k ,k2])
                    kill[k] = True

    # units not killed
    kill = np.where(kill)[0]
    units_out = units_in[~np.in1d(units_in, kill)]

    return units_out
