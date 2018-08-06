import copy
import numpy as np
import scipy

#from deconv_exp_utils import snr_main_chans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm

class MatchPursuitAnalyze(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, data, spike_train, temps, n_channels, n_features):
        """Sets up the deconvolution object.
        Parameters:
        -----------
        data: numpy.ndarray of shape (T, C)
            Where T is number of time samples and C number of channels.
        spike_train: numpy.ndarray of shape(T, 2)
            First column represents spike times and second column
            represents cluster ids.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        """
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps
        self.spike_train = spike_train
        self.data = data
        self.data_len = data.shape[0]
        #
        self.n_main_chan = n_channels
        self.n_feat = n_features
        #
        self.features = None
        self.cid = None
        self.means = None
        self.snrs = snr_main_chans(temps)
        self.n_clusters = 0
        #
        self.residual = None
        self.get_residual()

    def get_residual(self):
        """Returns the residual or computes it the first time."""
        if self.residual is None:         
            self.residual = copy.copy(self.data)
            for i in tqdm(range(self.n_unit), 'Computing Residual'):
                unit_sp = self.spike_train[self.spike_train[:, 1] == i, :]
                self.residual[np.arange(0, self.n_time) + unit_sp[:, :1], :] -= self.temps[:, :, i]

        return self.residual

    def get_unit_spikes(self, unit):
        """Gets clean spikes for a given unit."""
        unit_sp = self.spike_train[self.spike_train[:, 1] == unit, :]
        # Add the spikes of the current unit back to the residual
        return self.residual[np.arange(0, self.n_time) + unit_sp[:, :1], :] + self.temps[:, :, unit]

    def snr_main_chans(temps):
        """Computes peak to peak SNR for given templates."""
        chan_peaks = np.max(temps, axis=0)
        chan_lows = np.min(temps, axis=0)
        peak_to_peak = chan_peaks - chan_lows
        return np.argsort(peak_to_peak, axis=0)


    def featurize_spikes(self, spikes, channels):
        """Given a set of spikes computes low dim representations of them."""
        features = np.zeros([spikes.shape[0], self.n_main_chan, self.n_feat])
        for i, chan in enumerate(channels):
            pca = PCA(n_components=self.n_feat)
            features[:, i, :] = pca.fit_transform(spikes[:, :, chan])
        return features

    def get_features(self):
        """Returns the features or computes it the first time."""
        if self.features is None:
            self.features = []
            for i in tqdm(range(self.n_unit), 'Computing Features'):
                unit_spikes = self.get_unit_spikes(unit=i)
                self.features.append(self.featurize_spikes(
                    unit_spikes, channels=self.snrs[-self.n_main_chan:, i]))

        return self.features

    def split_units(self, n_clusters):
        """Splits recovered spikes per units into clusters."""
        self.n_clusters = n_clusters
        if self.cid is None:
            self.cid = []
            for i in tqdm(range(self.n_unit), 'Splitting Units'):
                f = self.features[i]
                f = f.reshape([f.shape[0], self.n_feat * self.n_main_chan])
                clustering = BayesianGaussianMixture(
                    n_components=n_clusters, max_iter=500)
                clustering.fit(f)
                self.cid.append(clustering.predict(f))
        return self.cid

    def split_units_means(self, n_clusters):
        """Splits recovered spikes per units into clusters."""
        if not(self.n_clusters == n_clusters) or self.means is None:
            res = self.residual
            self.means = np.zeros([self.n_unit, self.n_clusters, self.n_time, self.n_chan])
            for i in tqdm(range(self.n_unit), 'Computing Wave Forms'):
                unit_sp = self.spike_train[self.spike_train[:, 1] == i, :]
                unit_spikes = res[np.arange(0, self.n_time) + unit_sp[:, :1], :] + self.temps[:, :, i]
                for cluster_id in range(self.n_clusters):
                    idx = self.cid[i] == cluster_id
                    self.means[i, cluster_id, :, :] = np.mean(unit_spikes[idx, :, :], axis=0)
        return self.means
