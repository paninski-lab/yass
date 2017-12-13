import progressbar
import numpy as np

from yass.mfm import spikesort


# TODO: documentation
# TODO: comment code, it's not clear what it does
def runSorter(score_all, mask_all, clr_idx_all, group_all,
              channel_groups, neighbors, n_features, config):
    """Run sorting algorithm for every channel group

    Parameters
    ----------

    Returns
    -------
    spike_train:
        ?
    """
    # FIXME: mutating parameter
    # this function is passing a config object and mutating it,
    # this is not a good idea as having a mutable object lying around the code
    # can break things and make it hard to debug
    # (09/27/17) Eduardo

    nG = len(channel_groups)
    nmax = 10000

    K = 0
    spike_train = 0

    bar = progressbar.ProgressBar(maxval=nG)

    # iterate over every channel group (this is computed in config.py)
    for g in range(nG):

        # get the channels that conform this group
        ch_idx = channel_groups[g]

        neigh_chan = np.sum(neighbors[ch_idx], axis=0) > 0

        score = np.zeros(
            (nmax*ch_idx.shape[0], n_features, np.sum(neigh_chan)))
        index = np.zeros((nmax*ch_idx.shape[0], 2), 'int32')
        mask = np.zeros((nmax*ch_idx.shape[0], np.sum(neigh_chan)))
        group = np.zeros(nmax*ch_idx.shape[0], 'int16')

        count = 0
        Ngroup = 0

        for j in range(ch_idx.shape[0]):
            c = ch_idx[j]
            if score_all[c].shape[0] > 0:

                ndataTemp = score_all[c].shape[0]

                score[count:(count+ndataTemp), :, neighbors[c]
                      [neigh_chan]] = score_all[c]

                clr_idx_temp = clr_idx_all[c]
                index[count:(count+ndataTemp)] = np.concatenate(
                    (np.ones((ndataTemp, 1))*c,
                        clr_idx_temp[:, np.newaxis]), axis=1)

                mask[count:(count+ndataTemp), neighbors[c]
                     [neigh_chan]] = mask_all[c]

                group[count:(count+ndataTemp)] = group_all[c] + Ngroup + 1

                Ngroup += np.amax(group_all[c]) + 1
                count += ndataTemp

        score = score[:count]
        index = index[:count]
        mask = mask[:count]
        group = group[:count] - 1

        if score.shape[0] > 0:
            L = spikesort(score, mask, group, config)
            idx_triage = L == -1
            L = L[~idx_triage]
            index = index[~idx_triage]

            spikeTrain_temp = np.concatenate(
                (L[:, np.newaxis]+K, index), axis=1)
            K += np.amax(L)+1
            if g == 0:
                spike_train = spikeTrain_temp
            else:
                spike_train = np.concatenate(
                    (spike_train, spikeTrain_temp))

        bar.update(g+1)

    bar.finish()

    return spike_train
