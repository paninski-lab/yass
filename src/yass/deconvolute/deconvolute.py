"""
[Description of file content]
"""

import logging
import datetime as dt
import numpy as np

from ..geometry import n_steps_neigh_channels


# TODO: documentation
# TODO: comment code, it's not clear what it does
class Deconvolution(object):

    def __init__(self, config, templates, spike_index, wrec):

        self.config = config
        self.templates = templates
        self.spike_index = spike_index
        self.logger = logging.getLogger(__name__)
        self.wrec = wrec

    def fullMPMU(self):

        start_time = dt.datetime.now()

        neighchan = n_steps_neigh_channels(self.config.neighChannels, steps=3)
        C = self.config.recordings.n_channels
        R = self.config.spikeSize
        shift = 3  # int(R/2)
        K = self.templates.shape[2]
        nrank = self.config.deconvolution.rank
        lam = self.config.deconvolution.lam
        Th = self.config.deconvolution.threshold

        # used to have 3
        iter_max = 1

        amps = np.max(np.abs(self.templates), axis=0)
        amps_max = np.max(amps, axis=0)

        templatesMask = np.zeros((K, C), 'bool')

        for k in range(K):
            templatesMask[k] = amps[:, k] > amps_max[k]*0.5

        W_all, U_all, mu_all = decompose_dWU(self.templates, nrank)

        spiketime_all = np.zeros(0, 'int32')
        assignment_all = np.zeros(0, 'int32')

        for c in range(C):
            nmax = 1000000
            ids = np.zeros(nmax, 'int32')
            sts = np.zeros(nmax, 'int32')
            ns = np.zeros(nmax, 'int32')

            idx_c = self.spike_index[:, 1] == c
            nc = np.sum(idx_c)

            if nc > 0:
                spt_c = self.spike_index[idx_c, 0]
                ch_idx = np.where(neighchan[c])[0]

                # this line is in the old pipeline
                ch_idx = np.arange(C)

                k_idx = np.where(templatesMask[:, c])[0]
                tt = self.templates[:, ch_idx][:, :, k_idx]
                Kc = k_idx.shape[0]

                if nc > 0 and Kc > 0:
                    mu = np.reshape(mu_all[k_idx], [1, 1, Kc])

                    # commented out since it is unused
                    # lam1 = lam/np.square(mu)

                    wf = np.zeros((nc, 2*(R+shift)+1, ch_idx.shape[0]))

                    for j in range(nc):
                        wf[j] = self.wrec[
                            spt_c[j]+np.arange(-(R+shift), R+shift+1)][:,
                                                                       ch_idx]

                    n = np.arange(nc)
                    i0 = 0
                    it = 0

                    while it < iter_max:
                        nc = n.shape[0]
                        wf_projs = np.zeros((nc, 2*(R+shift)+1, nrank, Kc))

                        for k in range(Kc):
                            wf_projs[:, :, :, k] = np.reshape(np.matmul(
                                np.reshape(wf[n], [-1,
                                                   ch_idx.shape[0]]),
                                U_all[ch_idx, k_idx[k]]),
                                [nc, -1, nrank])

                        obj = np.zeros((nc, 2*shift+1, Kc))

                        for j in range(2*shift+1):
                            obj[:, j, :] = np.sum((wf_projs[:, j:(
                                j+2*R+1)]*np.transpose(W_all[:, k_idx],
                                                       [0, 2, 1])[np.newaxis,
                                                                  :]),
                                axis=(1, 2))

                        # this block is commented out in the old pipeline
                        # Ci = obj+(mu*lam1)
                        # Ci = np.square(Ci)/(1+lam1)
                        # Ci = Ci - lam1*np.square(mu)

                        # this block is in the old pipeline
                        scale = np.abs((obj-mu)/np.sqrt(mu/lam)) - 3
                        scale = np.minimum(np.maximum(scale, 0), 1)
                        scale[scale < 0] = 0
                        scale[scale > 1] = 1
                        Ci = np.multiply(np.square(obj), (1-scale))

                        mX = np.max(Ci, axis=1)
                        st = np.argmax(Ci, axis=1)
                        idd = np.argmax(mX, axis=1)
                        st = st[np.arange(st.shape[0]), idd]

                        idx_keep = np.max(mX, axis=1) > Th*Th
                        st = st[idx_keep]
                        idd = idd[idx_keep]
                        n = n[idx_keep]

                        n_detected = np.sum(idx_keep)

                        if it > 0:
                            idx_keep2 = np.zeros(n_detected, 'bool')

                            for j in range(n_detected):

                                if (np.sum(ids[:i0][ns[:i0] == n[j]] ==
                                           idd[j]) == 0):
                                    idx_keep2[j] = 1

                            st = st[idx_keep2]
                            idd = idd[idx_keep2]
                            n = n[idx_keep2]
                            n_detected = np.sum(idx_keep2)

                        if not st.any():
                            it = iter_max
                        else:
                            sts[i0:(i0+n_detected)] = st
                            ids[i0:(i0+n_detected)] = idd
                            ns[i0:(i0+n_detected)] = n
                            i0 = i0 + n_detected
                            it += 1

                            if it < iter_max:
                                for j in range(st.shape[0]):
                                    wf[n[j], st[j]:(
                                        st[j]+2*R+1)] -= tt[:, :, idd[j]]

                    ids = k_idx[ids[:i0]]
                    sts = sts[:i0] - shift - 1 + spt_c[ns[:i0]]

                    spiketime_all = np.concatenate((spiketime_all, sts))
                    assignment_all = np.concatenate((assignment_all, ids))

        current_time = dt.datetime.now()
        self.logger.info("Deconvolution done in {0} seconds.".format(
                         (current_time-start_time).seconds))

        return np.concatenate((spiketime_all[:, np.newaxis],
                               assignment_all[:, np.newaxis]), axis=1)


def decompose_dWU(templates, nrank):
    R, C, K = templates.shape
    W = np.zeros((R, nrank, K), 'float32')
    U = np.zeros((C, nrank, K), 'float32')
    mu = np.zeros((K, 1), 'float32')

    templates[np.isnan(templates)] = 0
    for k in range(K):
        W[:, :, k], U[:, :, k], mu[k] = get_svds(templates[:, :, k], nrank)

    U = np.transpose(U, [0, 2, 1])
    W = np.transpose(W, [0, 2, 1])

    U[np.isnan(U)] = 0

    return W, U, mu


def get_svds(template, nrank):
    Wall, S_temp, Uall = np.linalg.svd(template)
    imax = np.argmax(np.abs(Wall[:, 0]))
    ss = np.sign(Wall[imax, 1])
    Uall[0, :] = -Uall[0, :]*ss
    Wall[:, 0] = -Wall[:, 0]*ss

    Sv = np.zeros((Wall.shape[0], Uall.shape[0]))
    nn = np.min((Wall.shape[0], Uall.shape[0]))
    Sv[:nn, :nn] = np.diag(S_temp)

    Wall = np.matmul(Wall, Sv)

    mu = np.sqrt(np.sum(np.square(np.diag(Sv)[:nrank])))
    Wall = Wall/mu

    W = Wall[:, :nrank]
    U = (Uall.T)[:, :nrank]

    return W, U, mu
