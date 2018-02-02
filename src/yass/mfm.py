# FIXME: this file needs refactoring
import logging

import numpy as np
import scipy.special as specsci
import math
from numpy.random import dirichlet
import scipy.spatial as ss

logger = logging.getLogger(__name__)


class maskData:
    """
        Class for creating masked virtual data

        Attributes:
        -----------

        sumY: np.array
            Ngroup x nfeature x nchannel numpy array which contain the sum of
            the expectation of the
             masked virtual data for each group. Here Ngroup is number of
             unique groups given by the
             coresetting. nfeature is the number of features and nchannel is
             the number of channels.
        sumYSq: np.array
            Ngroup x nfeature x nchannel numpy array which contain the sum of
            the expectation of yy.T
            where y = M * score for each group.
        sumEta: np.array
            Ngroup x nfeature x nchannel numpy array sum of the variance for
            all points in a group.
        weight: np.array
            Ngroup x 1 numpy arraay which contains in the number of points in
            each group
        meanY: np.array
            Ngroup x nfeature x nchannel numpy array which is sumY/weight or
            the empirical mean of sumY
        meanYSq: np.array
            Ngroup x nfeature x nfeature x nchannel numpy array which is
            sumYSq/weight or the empirical
            mean of sumYSq
        meanEta: np.array
            Ngroup x nfeature x nfeature x nchannel numpy array whihc is
            sumEta/weight or the empirical
            mean of sumEta
        groupmask: np.array
            Ngroup x nchannel emprical average of the mask for the datapoints
            for a given group
    """

    def __init__(self, *args):
        """
            Initialization of class attributes. Class method
            calc_maskedData_mfm() is called to actually
            calculate the attributes.

            Parameters
            ----------
            score: np.array
                N x nfeature x nchannel numpy array, where N is the number of
                spikes, nfeature is the
                number of features and nchannel is the number of channels.
                Contains multichannel spike
                data in a low dimensional space.
            mask:  np.array
                N x nchannel numpy array, where N is the number of spikes,
                nchannel is the number of
                channels. Mask for the data.
            group: np.array
                N x 1 numpy array, where N is the number of spikes.
                Coresetting group assignments for
                each spike

        """
        if len(args) > 0:
            self.calc_maskedData_mfm(args[0], args[1], args[2])

    def calc_maskedData_mfm(self, score, mask, group):
        """
            Calculation of class attributes happen here.


            Parameters
            ----------
            score: np.array
                N x nfeature x nchannel numpy array, where N is the number of
                spikes, nfeature is the
                number of features and nchannel is the number of channels.
                Contains multichannel spike
                data in a low dimensional space.
            mask:  np.array
                N x nchannel numpy array, where N is the number of spikes,
                nchannel is the number of
                channels. Mask for the data.
            group: np.array
                N x 1 numpy array, where N is the number of spikes.
                Coresetting group assignments for
                each spike
        """
        N, nfeature, nchannel = score.shape
        uniqueGroup = np.unique(group)
        Ngroup = uniqueGroup.size
        y = mask[:, np.newaxis, :] * score

        y_temp = y[:, :, np.newaxis, :]
        ySq = np.matmul(
            np.transpose(y_temp, [0, 3, 1, 2]),
            np.transpose(y_temp, [0, 3, 2, 1])).transpose((0, 2, 3, 1))

        score_temp = score[:, :, np.newaxis, :]
        scoreSq = np.matmul(
            np.transpose(score_temp, [0, 3, 1, 2]),
            np.transpose(score_temp, [0, 3, 2, 1])).transpose((0, 2, 3, 1))
        z = mask[:, np.newaxis, np.newaxis, :] * scoreSq + \
            (1 - mask)[:, np.newaxis, np.newaxis, :] * \
            (np.eye(nfeature)[np.newaxis, :, :, np.newaxis])
        eta = z - ySq

        if Ngroup == N:
            sumY = y
            sumYSq = ySq
            sumEta = eta
            groupMask = mask
            weight = np.ones(N)

        elif Ngroup < N:

            sumY = np.zeros((Ngroup, nfeature, nchannel))
            sumYSq = np.zeros((Ngroup, nfeature, nfeature, nchannel))
            sumEta = np.zeros((Ngroup, nfeature, nfeature, nchannel))
            groupMask = np.zeros((Ngroup, nchannel))
            weight = np.zeros(Ngroup)
            for n in range(N):
                idx = group[n]
                sumY[idx] += y[n]
                sumYSq[idx] += ySq[n]
                sumEta[idx] += eta[n]
                groupMask[idx] += mask[n]
                weight[idx] += 1

        else:
            raise ValueError(
                "Number of groups is larger than the size of the data")
        # self.y = y
        self.sumY = sumY
        self.sumYSq = sumYSq
        self.sumEta = sumEta
        self.weight = weight
        self.groupMask = groupMask / self.weight[:, np.newaxis]
        self.meanY = self.sumY / self.weight[:, np.newaxis, np.newaxis]
        self.meanYSq = self.sumYSq / \
                       self.weight[:, np.newaxis, np.newaxis, np.newaxis]
        self.meanEta = self.sumEta / \
                       self.weight[:, np.newaxis, np.newaxis, np.newaxis]


class vbPar:
    """
        Class for all the parameters for the VB inference

        Attributes:
        -----------

        rhat: np.array
            Ngroup x K numpy array containing the probability of each
            representative point of being assigned
            to cluster  0<=k<K. Here K is the number of clusters

        ahat: np.array
            K x 1 numpy array. Posterior dirichlet parameters
        lambdahat, nuhat: np.array
            K x 1 numpy array. Posterior Normal wishart parameters
        muhat, Vhat, invVhat: np.array
            nfeaature x K x nchannel, nfeature x nfeature x K x nchannel,
            nfeature x nfeature x K x nchannel
            respectively. Posterior parameters for the normal wishart
            distribution
    """

    def __init__(self, rhat):
        """
            Iniitalizes rhat defined above

            Parameters:
            -----------

            rhat: np.array
                Ngroup x K numpy array defined above

        """

        self.rhat = rhat

    def update_local(self, maskedData):
        """
            Updates the local parameter rhat for VB inference

            Parameters:
            -----------

            maskedData: maskData object
        """

        pik = dirichlet(self.ahat.ravel())
        Khat = self.ahat.size
        Ngroup = maskedData.meanY.shape[0]
        # nchannel = maskedData.meanY.shape[2]
        log_rho = np.zeros([Ngroup, Khat])
        for k in range(Khat):
            mvn = multivariate_normal_logpdf(
                maskedData.meanY, self.muhat[:, k, :],
                self.Vhat[:, :, k, :] * self.nuhat[k])
            log_rho[:, k] = log_rho[:, k] + mvn
            log_rho[:, k] = log_rho[:, k] + np.log(pik[k])
        log_rho = log_rho - np.max(log_rho, axis=1)[:, np.newaxis]
        rho = np.exp(log_rho)
        self.rhat = rho / np.sum(rho, axis=1, keepdims=True)

    def update_global(self, suffStat, param):
        """
            Updates the global variables muhat, invVhat, Vhat, lambdahat,
            nuhat, ahat for VB inference

            Parameters:
            ----------

            suffStat: suffStatistics object

            param: Config object (See config.py for details)
        """
        prior = param.cluster_prior
        nfeature, Khat, nchannel = suffStat.sumY.shape
        self.ahat = prior.a + suffStat.Nhat
        self.lambdahat = prior.lambda0 + suffStat.Nhat
        self.muhat = suffStat.sumY / self.lambdahat[:, np.newaxis]
        invV = np.eye(nfeature) / prior.V
        self.Vhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        self.invVhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        for n in range(nchannel):
            for k in range(Khat):
                self.invVhat[:, :, k, n] = self.invVhat[:, :, k, n] + invV
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k,
                     n] + self.lambdahat[k] * np.dot(
                    self.muhat[:, np.newaxis, k,
                    n],
                    self.muhat[:, np.newaxis, k,
                    n].T)
                temp = np.dot(self.muhat[:, np.newaxis, k, n],
                              suffStat.sumY[:, np.newaxis, k, n].T)
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k, n] - temp - temp.T
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k,
                     n] + suffStat.sumYSq[:, :, k, n]
                self.Vhat[:, :, k, n] = np.linalg.solve(
                    np.squeeze(self.invVhat[:, :, k, n]), np.eye(nfeature))
        self.nuhat = prior.nu + suffStat.Nhat

    def update_global_selected(self, suffStat, param):
        """
            Updates the global variables muhat, invVhat, Vhat, lambdahat,
            nuhat, ahat for VB inference for
            a given cluster (Unused; needs work)

            Parameters:
            ----------

            suffStat: suffStatistics object

            param: Config object (See config.py for details)
        """
        prior = param.cluster_prior
        nfeature, Khat, nchannel = suffStat.sumY.shape
        self.ahat = prior.a + suffStat.Nhat
        self.lambdahat = prior.lambda0 + suffStat.Nhat
        self.muhat = suffStat.sumY / self.lambdahat[:, np.newaxis]
        invV = np.eye(nfeature) / prior.V
        self.Vhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        self.invVhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        for n in range(nchannel):
            for k in range(Khat):
                self.invVhat[:, :, k, n] = self.invVhat[:, :, k, n] + invV
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k,
                     n] + self.lambdahat[k] * np.dot(
                    self.muhat[:, np.newaxis, k,
                    n],
                    self.muhat[:, np.newaxis, k,
                    n].T)
                temp = np.dot(self.muhat[:, np.newaxis, k, n],
                              suffStat.sumY[:, np.newaxis, k, n].T)
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k, n] - temp - temp.T
                self.invVhat[:, :, k,
                n] = self.invVhat[:, :, k,
                     n] + suffStat.sumYSq[:, :, k, n]
                self.Vhat[:, :, k, n] = np.linalg.solve(
                    np.squeeze(self.invVhat[:, :, k, n]), np.eye(nfeature))
        self.nuhat = prior.nu + suffStat.Nhat


class suffStatistics:
    """
        Class to calculate precompute sufficient statistics for increased
        efficiency

        Attributes:
        -----------

        Nhat : np.array
            K x 1 numpy array which stores pseudocounts for the number of
            elements in each cluster. K
            is the number of clusters
        sumY : np.array
            nfeature x K x nchannel which stores weighted sum of the
            maskData.sumY weighted by the cluster
            probabilities. (See maskData for more details)
        sumYSq1: np.array
            nfeaature x nfeature x K x nchannel which stores weighted sum of
            the maskData.sumYSq weighted
            by the cluster probabilities. (See maskData for more details)
        sumYSq2: np.array
            nfeaature x nfeature x K x nchannel which stores weighted sum of
            the maskData.sumEta weighted
            by the cluster probabilities. (See maskData for more details)
        sumYSq: np.array
            nfeature x nfeature x K x nchannel. Stores sumYSq1 + sumYSq2.
    """

    def __init__(self, *args):
        """
            Initializes the above attributes and calls calc_suffstat().

            Parameters:
            -----------
            maskedData: maskData object

            vbParam: vbPar object

                or

            suffStat: suffStatistics object
        """

        if len(args) == 2:
            maskedData, vbParam = args
            Khat = vbParam.rhat.shape[1]
            Ngroup, nfeature, nchannel = maskedData.sumY.shape
            self.Nhat = np.sum(
                vbParam.rhat * maskedData.weight[:, np.newaxis], axis=0)
            self.sumY = np.zeros([nfeature, Khat, nchannel])
            self.sumYSq = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.sumYSq1BS = np.zeros(
                [Ngroup, nfeature, nfeature, Khat, nchannel])
            self.sumYSq1 = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.sumYSq2 = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.calc_suffstat(maskedData, vbParam, Ngroup, Khat, nfeature,
                               nchannel)
        elif len(args) == 1:
            self.Nhat = args[0].Nhat.copy()
            self.sumY = args[0].sumY.copy()
            self.sumYSq = args[0].sumYSq.copy()
            self.sumYSq1 = args[0].sumYSq1.copy()
            self.sumYSq2 = args[0].sumYSq2.copy()

    def calc_suffstat(self, maskedData, vbParam, Ngroup, Khat, nfeature,
                      nchannel):
        """
            Calcation of the above attributes happens here. Called by
            __init__().

            Parameters:
            -----------
            maskedData: maskData object

            vbParam: vbPar object

            Ngroup: int
                Number of groups defined by coresetting

            Khat: int
                Number of clusters

            nfeature: int
                Number of features

            nchannel: int
                Number of channels
        """

        for n in range(nchannel):
            noMask = maskedData.groupMask[:, n] > 0
            nnoMask = np.sum(noMask)
            if nnoMask == 0:
                for k in range(Khat):
                    self.sumYSq[:, :, k, n] = np.eye(nfeature) * self.Nhat[k]

            elif nnoMask < Ngroup:
                maskedEta = np.eye(nfeature)
                unmaskedY = maskedData.sumY[noMask, :, n]
                unmaskedsumYSq = maskedData.sumYSq[noMask, :, :, n]
                unmaskedEta = maskedData.sumEta[noMask, :, :, n]
                unmaskedWeight = maskedData.weight[noMask]

                rhat = vbParam.rhat[noMask, :]
                self.sumY[:, :, n] = np.dot(unmaskedY.T, rhat)

                visibleCluster = np.sum(rhat, axis=0) > 1e-10
                sumMaskedRhat = self.Nhat - \
                                np.sum(rhat * unmaskedWeight[:, np.newaxis],
                                       axis=0)
                self.sumYSq2[:, :, :, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, :]
                    * unmaskedEta[:, :, :, np.newaxis], axis=0) + \
                                           sumMaskedRhat * maskedEta[:, :,
                                                           np.newaxis]
                self.sumYSq1[:, :, visibleCluster, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, visibleCluster] *
                    unmaskedsumYSq[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq[:, :, :,
                n] = self.sumYSq1[:, :, :,
                     n] + self.sumYSq2[:, :, :, n]

            elif nnoMask == Ngroup:
                unmaskedY = maskedData.sumY[:, :, n]
                unmaskedsumYSq = maskedData.sumYSq[:, :, :, n]
                unmaskedEta = maskedData.sumEta[:, :, :, n]

                rhat = vbParam.rhat[noMask, :]
                self.sumY[:, :, n] = np.dot(unmaskedY.T, rhat)

                visibleCluster = np.sum(rhat, axis=0) > 1e-10
                sumMaskedRhat = self.Nhat - \
                                np.sum(rhat * maskedData.weight[:, np.newaxis],
                                       axis=0)
                self.sumYSq2[:, :, :, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, :] *
                    unmaskedEta[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq1[:, :, visibleCluster, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, visibleCluster] *
                    unmaskedsumYSq[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq[:, :, :,
                n] = self.sumYSq1[:, :, :,
                     n] + self.sumYSq2[:, :, :, n]


class ELBO_Class:
    """
        Class for calculating the ELBO for VB inference

        Attributes:
        -----------

        percluster: np.array
            K x 1 numpy array containing part of ELBO value that depends on
            each cluster. Can be used to calculate
            value for only a given cluster

        rest_term: float
            Part of ELBO value that has to be calculated regardless of cluster

        total: float
            Total ELBO total = sum(percluster) + rest_term
    """

    def __init__(self, *args):
        """
            Initializes attributes. Calls cal_ELBO_opti()

            Parameters:
            -----------
            maskedData: maskData object

            suffStat: suffStatistics object

            vbParam: vbPar object

            param: Config object (see Config.py)

            K_ind (optional): list
                Cluster indices for which the partial ELBO is being
                calculated. Defaults to all clusters
        """
        if len(args) == 1:
            self.total = args[0].total
            self.percluster = args[0].percluster
            self.rest_term = args[0].rest_term
        else:
            self.calc_ELBO_Opti(args)

    def calc_ELBO_Opti(self, *args):
        """
            Calculates partial (or total) ELBO for the given cluster indices

            Parameters:
            -----------
            maskedData: maskData object

            suffStat: suffStatistics object

            vbParam: vbPar object

            param: Config object (see Config.py)

            K_ind (optional): list
                Cluster indices for which the partial ELBO is being
                calculated. Defaults to all clusters
        """

        if len(args[0]) < 5:
            maskedData, suffStat, vbParam, param = args[0]
            nfeature, Khat, nchannel = vbParam.muhat.shape
            P = nfeature * nchannel
            k_ind = np.arange(Khat)
        else:
            maskedData, suffStat, vbParam, param, k_ind = args[0]
            nfeature, Khat, nchannel = vbParam.muhat.shape
            P = nfeature * nchannel

        prior = param.cluster_prior
        fit_term = np.zeros(Khat)
        bmterm = np.zeros(Khat)
        entropy_term = np.zeros(Khat)
        rhatp = vbParam.rhat[:, k_ind]
        muhat = np.transpose(vbParam.muhat, [1, 2, 0])
        Vhat = np.transpose(vbParam.Vhat, [2, 3, 0, 1])
        sumY = np.transpose(suffStat.sumY, [1, 2, 0])
        logdetVhat = np.sum(np.linalg.slogdet(Vhat)[1], axis=1, keepdims=False)

        # fit term

        fterm1temp = np.squeeze(
            np.sum(
                np.matmul(
                    np.matmul(muhat[k_ind, :, np.newaxis, :],
                              Vhat[k_ind, :, :, :]),
                    muhat[k_ind, :, :, np.newaxis]),
                axis=1,
                keepdims=False),
            axis=(1, 2))
        fterm1 = -fterm1temp * \
                 vbParam.nuhat[k_ind] * suffStat.Nhat[k_ind] / 2.0

        fterm2 = -2.0 * np.squeeze(
            np.sum(
                np.matmul(
                    np.matmul(sumY[k_ind, :, np.newaxis, :], Vhat[
                                                             k_ind, :, :, :]),
                    muhat[k_ind, :, :, np.newaxis]),
                axis=(1),
                keepdims=False))
        fterm2 *= -vbParam.nuhat[k_ind] / 2.0

        fterm3 = np.sum(
            suffStat.sumYSq1[:, :, k_ind, :] * vbParam.Vhat[:, :, k_ind, :],
            axis=(0, 1, 3),
            keepdims=False)
        fterm3 *= -vbParam.nuhat[k_ind] / 2.0

        fterm4 = -nchannel * suffStat.Nhat[k_ind] / 2.0 * (
            nfeature / vbParam.lambdahat[k_ind] - nfeature * np.log(2.0) -
            mult_psi(vbParam.nuhat[k_ind, np.newaxis] / 2.0,
                     nfeature).ravel() + nfeature * np.log(2 * np.pi))

        fterm5 = suffStat.Nhat[k_ind] * logdetVhat[k_ind] / 2.0

        fterm6 = -np.sum(
            np.trace(
                np.matmul(
                    np.transpose(suffStat.sumYSq2[:, :, k_ind, :],
                                 [2, 3, 0, 1]), Vhat[k_ind, :, :, :]),
                axis1=2,
                axis2=3),
            axis=1,
            keepdims=0) * vbParam.nuhat[k_ind] / 2.0

        fit_term[k_ind] = fterm1 + fterm2 + fterm3 + fterm4 + fterm5 + fterm6

        # BM Term

        bmterm1 = 0.5 * prior.nu * \
                  np.sum(np.linalg.slogdet(
                      Vhat[k_ind, :, :, :] / prior.V)[1], axis=1)

        bmterm2 = -0.5 * vbParam.nuhat[k_ind] * (np.sum(
            np.trace(Vhat[k_ind, :, :, :] / prior.V, axis1=2, axis2=3),
            axis=1))

        bmterm3 = 0.5 * (vbParam.nuhat[k_ind] * P + P -
                         P * prior.lambda0 / vbParam.lambdahat[k_ind] +
                         P * np.log(prior.lambda0 / vbParam.lambdahat[k_ind]))

        bmterm4 = -0.5 * vbParam.nuhat[k_ind] * prior.lambda0 * fterm1temp

        bmterm5 = nchannel * (
            specsci.multigammaln(vbParam.nuhat[k_ind] / 2.0, nfeature) -
            specsci.multigammaln(prior.nu / 2.0, nfeature) + 0.5 *
            (prior.nu - vbParam.nuhat[k_ind]) * mult_psi(
                vbParam.nuhat[k_ind, np.newaxis] / 2.0, nfeature).ravel())

        bmterm[k_ind] = bmterm1 + bmterm2 + bmterm3 + bmterm4 + bmterm5

        # Entropy term
        entropy_term1 = np.sum(
            vbParam.rhat * maskedData.weight[:, np.newaxis] *
            (specsci.digamma(vbParam.ahat) -
             specsci.digamma(np.sum(vbParam.ahat))),
            axis=0)

        entropy_term2 = np.zeros(Khat)
        entropy_term2[k_ind] = -np.sum(
            maskedData.weight[:, np.newaxis] * rhatp * np.log(rhatp + 1e-200),
            axis=0).ravel()

        entropy_term = entropy_term1 + entropy_term2

        # Dirichlet terms

        dc_term = - specsci.gammaln(np.sum(vbParam.ahat)) + np.sum(
            specsci.gammaln(vbParam.ahat)) \
                  + specsci.gammaln(
            Khat * param.cluster_prior.a) - Khat * specsci.gammaln(
            param.cluster_prior.a) \
                  + np.sum(
            (param.cluster_prior.a - vbParam.ahat) * (
            specsci.digamma(vbParam.ahat) - specsci.digamma(
                np.sum(vbParam.ahat))))

        # prior term

        pterm = np.log(
            prior.beta ** Khat * np.exp(-prior.beta) / math.factorial(Khat))

        self.percluster = fit_term + bmterm + entropy_term2
        self.rest_term = np.sum(entropy_term1) + dc_term + pterm
        self.total = np.sum(self.percluster) + self.rest_term


def multivariate_normal_logpdf(x, mu, Lam):
    """
        Calculates the gaussian density of the given point(s). returns N x 1
        array which is the density for
        the given cluster (see vbPar.update_local())

        Parameters:
        -----------
        x: np.array
            N x nfeature x nchannel where N is the number of datapoints
            nfeature is the number of features
            and nchannel is the number of channels

        mu: np.array
            nfeature x nchannel numpy array. channelwise mean of the gaussians

        cov: np.array
            nfeature x nfeauter x nchannel numpy array. Channelwise covariance
            of the gaussians

    """

    p, C = mu.shape

    xMinusMu = np.transpose((x - mu), [2, 0, 1])
    maha = -0.5 * np.sum(
        np.matmul(xMinusMu, np.transpose(Lam, [2, 0, 1])) * xMinusMu,
        axis=(0, 2))

    const = -0.5 * p * C * np.log(2 * math.pi)

    logpart = 0

    for c in range(C):
        logpart = logpart + logdet(Lam[:, :, c])
    logpart = logpart * 0.5

    return maha + const + logpart


def logdet(X):
    """
        Calculates log of the determinant of the given symmetric positive
        definite matrix. returns float

        parameters:
        -----------
        X: np.array
            M x M. Symmetric positive definite matrix

    """

    L = np.linalg.cholesky(X)
    return 2 * np.sum(np.log(np.diagonal(L)))


def mult_psi(x, d):
    """
        Calculates the multivariate digamma function. Returns N x 1 array of
        the multivariaate digamma values

        parameters:
        -----------
        x: np.array
            M x 1 array containing the
    """
    v = x - np.asarray([range(d)]) / 2.0
    u = np.sum(specsci.digamma(v), axis=1)
    return u[:, np.newaxis]


def init_param(maskedData, K, param):
    """
        Initializes vbPar object using weighted kmeans++ for initial cluster
        assignment. Calculates sufficient
        statistics. Updates global parameters for the created vbPar object.

        Parameters:
        -----------
        maskedData: maskData object

        K: int
            Number of clusters for weighted kmeans++

        param: Config object (see config.py)

    """
    N, nfeature, nchannel = maskedData.sumY.shape
    allocation = weightedKmeansplusplus(
        maskedData.meanY.reshape([N, nfeature * nchannel], order='F').T,
        maskedData.weight, K)

    if N < K:
        rhat = np.zeros([N, N])
    else:
        rhat = np.zeros([N, K])
    rhat[np.arange(N), allocation] = 1
    vbParam = vbPar(rhat)
    suffStat = suffStatistics(maskedData, vbParam)
    vbParam.update_global(suffStat, param)
    # vbParam.update_local(maskedData)
    # suffStat = suffStatistics(maskedData, vbParam)
    return vbParam, suffStat


def weightedKmeansplusplus(X, w, k):
    """


    """
    L = np.asarray([])
    L1 = 0
    p = w ** 2 / np.sum(w ** 2)
    n = X.shape[1]
    while np.unique(L).size != k:
        ii = np.random.choice(np.arange(n), size=1, replace=True, p=p)
        C = X[:, ii]
        L = np.zeros([1, n]).astype(int)
        for i in range(1, k):
            D = X - C[:, L.ravel()]
            D = np.sum(D * D, axis=0)
            if np.max(D) == 0:
                # C[:, i:k] = X[:, np.ones([1, k - i + 1]).astype(int)]
                return L
            D = D / np.sum(D)
            ii = np.random.choice(np.arange(n), size=1, replace=True, p=D)
            C = np.concatenate((C, X[:, ii]), axis=1)
            L = np.argmax(
                2 * np.dot(C.T, X) - np.sum(C * C, axis=0)[:, np.newaxis],
                axis=0)
        while np.any(L != L1):
            L1 = L
            for i in range(k):
                l = L == i
                if np.sum(l) > 0:
                    C[:, i] = np.dot(X[:, l], w[l] / np.sum(w[l]))
            L = np.argmax(
                2 * np.dot(C.T, X) - np.sum(C * C, axis=0)[:, np.newaxis],
                axis=0)
    return L


def birth_move(maskedData, vbParam, suffStat, param, L):
    Khat = suffStat.sumY.shape[1]
    collectionThreshold = 0.1
    extraK = param.clustering.n_split
    weight = (suffStat.Nhat + 0.001) * L ** 2
    weight = weight / np.sum(weight)
    idx = np.zeros(1).astype(int)
    while np.sum(idx) == 0:
        kpicked = np.random.choice(np.arange(Khat).astype(int), p=weight)
        idx = vbParam.rhat[:, kpicked] > collectionThreshold

    idx = np.where(idx)[0]
    if idx.size > 10000:
        idx = idx[:10000]
    L = L * 2
    L[kpicked] = 1

    # Creation

    maskedDataPrime = maskData()
    if idx.shape[0] > 0:
        maskedDataPrime.sumY = maskedData.sumY[idx, :, :]
        maskedDataPrime.sumYSq = maskedData.sumYSq[idx, :, :, :]
        maskedDataPrime.sumEta = maskedData.sumEta[idx, :, :, :]
        maskedDataPrime.groupMask = maskedData.groupMask[idx, :]
        maskedDataPrime.weight = maskedData.weight[idx]
        maskedDataPrime.meanY = maskedData.meanY[idx, :, :]
        maskedDataPrime.meanEta = maskedData.meanEta[idx, :, :, :]
    elif idx.shape[0] == 1:
        maskedDataPrime.sumY = maskedData.sumY[idx:(idx + 1), :, :]
        maskedDataPrime.sumYSq = maskedData.sumYSq[idx:(idx + 1), :, :, :]
        maskedDataPrime.sumEta = maskedData.sumEta[idx:(idx + 1), :, :, :]
        maskedDataPrime.groupMask = maskedData.groupMask[idx:(idx + 1), :]
        maskedDataPrime.weight = maskedData.weight[idx:(idx + 1)]
        maskedDataPrime.meanY = maskedData.meanY[idx:(idx + 1), :, :]
        maskedDataPrime.meanEta = maskedData.meanEta[idx:(idx + 1), :, :, :]
    vbParamPrime, suffStatPrime = init_param(maskedDataPrime, extraK, param)

    for iter_creation in range(3):
        vbParamPrime.update_local(maskedDataPrime)
        suffStatPrime = suffStatistics(maskedDataPrime, vbParamPrime)
        vbParamPrime.update_global(suffStatPrime, param)

    temp = vbParamPrime.rhat * maskedDataPrime.weight[:, np.newaxis]
    goodK = ((np.sum(temp, axis=0) / np.sum(maskedDataPrime.weight)) >
             (1.0 / (extraK * 2)))
    Nbirth = np.sum(goodK)
    if Nbirth >= 1:
        vbParam.ahat = np.concatenate(
            (vbParam.ahat, vbParamPrime.ahat[goodK]), axis=0)
        vbParam.lambdahat = np.concatenate(
            (vbParam.lambdahat, vbParamPrime.lambdahat[goodK]), axis=0)
        vbParam.muhat = np.concatenate(
            (vbParam.muhat, vbParamPrime.muhat[:, goodK, :]), axis=1)
        vbParam.Vhat = np.concatenate(
            (vbParam.Vhat, vbParamPrime.Vhat[:, :, goodK, :]), axis=2)
        vbParam.invVhat = np.concatenate(
            (vbParam.invVhat, vbParamPrime.invVhat[:, :, goodK, :]), axis=2)
        vbParam.nuhat = np.concatenate(
            (vbParam.nuhat, vbParamPrime.nuhat[goodK]), axis=0)
        L = np.concatenate((L, np.ones(Nbirth)), axis=0)

    vbParam.update_local(maskedData)
    suffStat = suffStatistics(maskedData, vbParam)
    vbParam.update_global(suffStat, param)

    return vbParam, suffStat, L


def merge_move(maskedData, vbParam, suffStat, param, L, check_full):
    n_merged = 0
    ELBO = ELBO_Class(maskedData, suffStat, vbParam, param)
    nfeature, K, nchannel = vbParam.muhat.shape
    if K > 1:
        all_checked = 0
    else:
        all_checked = 1

    while (not all_checked) and (K > 1):
        m = np.transpose(vbParam.muhat, [1, 0, 2]).reshape(
            [K, nfeature * nchannel], order="F")
        kdist = ss.distance_matrix(m, m)
        kdist[np.tril_indices(K)] = np.inf
        merged = 0
        k_untested = np.ones(K)
        while np.sum(k_untested) > 0 and merged == 0:
            untested_k = np.argwhere(k_untested)
            ka = untested_k[np.random.choice(len(untested_k), 1)].ravel()[0]
            kb = np.argmin(kdist[ka, :]).ravel()[0]
            k_untested[ka] = 0
            if np.argmin(kdist[kb, :]).ravel()[0] == ka:
                k_untested[kb] = 0
            kdist[min(ka, kb), max(ka, kb)] = np.inf

            vbParam, suffStat, merged, L, ELBO = check_merge(
                maskedData, vbParam, suffStat, ka, kb, param, L, ELBO)
            if merged:
                n_merged += 1
                K -= 1
        if not merged:
            all_checked = 1
    return vbParam, suffStat, L


def check_merge(maskedData, vbParam, suffStat, ka, kb, param, L, ELBO):
    K = vbParam.rhat.shape[1]
    no_kab = np.ones(K).astype(bool)
    no_kab[[ka, kb]] = False
    ELBO_bmerge = np.sum(ELBO.percluster[[ka, kb]]) + ELBO.rest_term

    vbParamTemp = vbPar(
        np.concatenate(
            (vbParam.rhat[:, no_kab],
             np.sum(vbParam.rhat[:, [ka, kb]], axis=1, keepdims=True)),
            axis=1))
    suffStatTemp = suffStatistics()
    suffStatTemp.Nhat = np.append(suffStat.Nhat[no_kab],
                                  np.sum(suffStat.Nhat[[ka, kb]]))
    suffStatTemp.sumY = np.concatenate(
        (suffStat.sumY[:, no_kab, :],
         np.sum(suffStat.sumY[:, (ka, kb), :], axis=1, keepdims=True)),
        axis=1)
    suffStatTemp.sumYSq = np.concatenate(
        (suffStat.sumYSq[:, :, no_kab, :],
         np.sum(suffStat.sumYSq[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)
    suffStatTemp.sumYSq1 = np.concatenate(
        (suffStat.sumYSq1[:, :, no_kab, :],
         np.sum(suffStat.sumYSq1[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)
    suffStatTemp.sumYSq2 = np.concatenate(
        (suffStat.sumYSq2[:, :, no_kab, :],
         np.sum(suffStat.sumYSq2[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)

    vbParamTemp.ahat = param.cluster_prior.a + suffStatTemp.Nhat
    vbParamTemp.lambdahat = param.cluster_prior.lambda0 + suffStatTemp.Nhat
    vbParamTemp.nuhat = param.cluster_prior.nu + suffStatTemp.Nhat
    vbParamTemp.muhat = np.concatenate(
        (vbParam.muhat[:, no_kab, :],
         suffStatTemp.sumY[:, [K - 2], :] / vbParamTemp.lambdahat[K - 2]),
        axis=1)
    nfeature, Khat, nchannel = suffStat.sumY.shape
    invV = np.eye(nfeature) / param.cluster_prior.V
    invVhatTemp = np.zeros((nfeature, nfeature, 1, nchannel))
    VhatTemp = np.zeros((nfeature, nfeature, 1, nchannel))
    for n in range(nchannel):
        muhatC = vbParamTemp.muhat[:, np.newaxis, K - 2, n]
        temp = np.dot(muhatC, suffStatTemp.sumY[:, np.newaxis, K - 2, n].T)
        invVhatTemp[:, :, 0, n] = invV + vbParamTemp.lambdahat[K - 2] * \
                                         np.dot(muhatC,
                                                muhatC.T) - temp - temp.T + \
                                  suffStatTemp.sumYSq[:, :, K - 2, n]
        VhatTemp[:, :, 0, n] = np.linalg.solve(invVhatTemp[:, :, 0, n],
                                               np.eye(nfeature))
    vbParamTemp.Vhat = np.concatenate(
        (vbParam.Vhat[:, :, no_kab, :], VhatTemp), axis=2)
    vbParamTemp.invVhat = np.concatenate(
        (vbParam.invVhat[:, :, no_kab, :], invVhatTemp), axis=2)

    ELBO_amerge = ELBO_Class(maskedData, suffStatTemp, vbParamTemp, param,
                             [K - 2])
    if ELBO_amerge.total < ELBO_bmerge:
        merged = 0
        return vbParam, suffStat, merged, L, ELBO
    else:
        merged = 1
        d = np.asarray([np.min(L[[ka, kb]])])
        L = np.concatenate((L[no_kab], d), axis=0)

        ELBO_amerge.percluster[0:-1] += ELBO.percluster[no_kab]
        ELBO_amerge.total = np.sum(
            ELBO_amerge.percluster) + ELBO_amerge.rest_term

        if L.size == 1:
            L = np.asarray([1])
        return vbParamTemp, suffStatTemp, merged, L, ELBO_amerge


def spikesort(score, mask, group, param):
    usedchan = np.asarray(np.where(np.sum(mask, axis=0) >= 1)).ravel()
    score = score[:, :, usedchan]
    mask = mask[:, usedchan]
    # FIXME: seems like this is never used
    # param.n_chan = np.sum(usedchan)

    maskedData = maskData(score, mask, group)

    vbParam = split_merge(maskedData, param)

    if param.clustering.clustering_method == '2+3':
        return vbParam, maskedData
    else:
        assignmentTemp = np.argmax(vbParam.rhat, axis=1)

        assignment = np.zeros(score.shape[0], 'int16')
        for j in range(score.shape[0]):
            assignment[j] = assignmentTemp[group[j]]

        idx_triage = cluster_triage(vbParam, score, 3)
        assignment[idx_triage] = -1

        return assignment


def split_merge(maskedData, param):
    vbParam, suffStat = init_param(maskedData, 1, param)
    iter = 0
    L = np.ones([1])
    n_iter = 1
    extra_iter = 5
    k_max = 1
    while iter < n_iter:
        iter += 1
        vbParam, suffStat, L = birth_move(maskedData, vbParam, suffStat, param,
                                          L)
        # print('birth',vbParam.rhat.shape[1])
        vbParam, suffStat, L = merge_move(maskedData, vbParam, suffStat, param,
                                          L, 0)
        # print('merge',vbParam.rhat.shape[1])

        k_now = vbParam.rhat.shape[1]
        if (k_now > k_max) and (iter + extra_iter > n_iter):
            n_iter = iter + extra_iter
            k_max = k_now

    vbParam, suffStat, L = merge_move(maskedData, vbParam, suffStat, param, L,
                                      1)

    return vbParam


def cluster_triage(vbParam, score, threshold):
    prec = np.transpose(
        vbParam.Vhat * vbParam.nuhat[np.newaxis, np.newaxis, :, np.newaxis],
        axes=[2, 3, 0, 1])
    scoremhat = np.transpose(
        score[:, :, np.newaxis, :] - vbParam.muhat, axes=[0, 2, 3, 1])
    maha = np.sqrt(
        np.sum(
            np.matmul(
                np.matmul(scoremhat[:, :, :, np.newaxis, :], prec),
                scoremhat[:, :, :, :, np.newaxis]),
            axis=(3, 4),
            keepdims=False))
    idx = np.any(np.all(maha >= threshold, axis=1), axis=1)
    return idx
