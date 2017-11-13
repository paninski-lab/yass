import numpy as np
from scipy.spatial import cKDTree


# FIXME: dotriage should not be here
def triage(score, channel_index, triage_k, triage_percent, do_triage):
    """

    Note
    ----
    This object mutates score and clr_idx
    """

    K = triage_k + 1
    th = (1 - triage_percent)*100
    nmax = 10000

    n_data, n_features, n_channels = score.shape
    index_keep_bool = np.zeros(n_data,'bool')
    if n_data > nmax:
        index_keep = np.random.permutation(n_data)[:nmax]
        score = score[idx_keep]
    else:
        index_keep = np.arange(n_data)

    if do_triage and (n_data > K):
        score_temp = score[:, :, channel_index]
        tree = cKDTree(score_temp)
        dist, ind = tree.query(score_temp, k=K)
        dist = np.sum(dist, 1)
        index_keep = index_keep[dist < np.percentile(dist, th)]

    index_keep_bool[index_keep] = 1

    return index_keep_bool

def triage_depreciated(score, clr_idx, n_channels, triage_k, triage_percent,
           neighbors, doTriage):
    """

    Note
    ----
    This object mutates score and clr_idx
    """
    C = n_channels
    K = triage_k + 1
    th = (1 - triage_percent)*100
    nmax = 10000

    score_clean = list()
    clr_idx_clean = list()

    if doTriage:
        for c in range(C):
            nc = score[c].shape[0]
            if nc > K:
                if nc > nmax:
                    idx_keep2 = np.random.permutation(nc)
                    score[c] = score[c][idx_keep2[:nmax]]
                    clr_idx[c] = clr_idx[c][idx_keep2[:nmax]]

                c_idx = int(np.where(np.where(neighbors[c])[0] == c)[0])
                score_temp = score[c][:, :, c_idx]
                tree = cKDTree(score_temp)
                dist, ind = tree.query(score_temp, k=K)
                dist = np.sum(dist, 1)
                idx_keep = dist < np.percentile(dist, th)
                n_keep = sum(idx_keep)
                score_clean.append(score[c][idx_keep])
                clr_idx_clean.append(clr_idx[c][idx_keep])

            else:
                score_clean.append(np.copy(score[c]))
                clr_idx_clean.append(np.copy(clr_idx[c]))

    else:
        for c in range(C):
            if score[c].shape[0] > nmax:
                idx_keep2 = np.random.permutation(score[c].shape[0])
                score_clean.append(score[c][idx_keep2[:nmax]])
                clr_idx_clean.append(clr_idx[c][idx_keep2[:nmax]])
            else:
                score_clean.append(np.copy(score[c]))
                clr_idx_clean.append(np.copy(clr_idx[c]))

    return score_clean, clr_idx_clean