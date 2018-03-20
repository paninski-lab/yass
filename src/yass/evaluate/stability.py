"""
Stability evaluation

Original source:
https://github.com/hooshmandshr/yass_visualization/blob/master/src/stability/stability_evaluation.py
"""

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm

from yass.geometry import find_channel_neighbors, parse
from yass.evaluate.stability_filters import butterworth, whitening


def clean_spike_train(spt):
    units = np.unique(spt[:, 1])
    spt[:, 1] += len(units)
    units += len(units)
    for i, u in enumerate(units):
        u_idx = spt[:, 1] == u
        spt[u_idx, 1] = i
    return spt


# TODO: remove this and use new yass Reader
class RecordingBatchIterator(object):

    def __init__(self, rec_file, geom_file, sample_rate,
                 n_batches, batch_time_samples, n_chan,
                 radius, scale=1e2, filter_std=True, whiten=True):
        """Sets up the object for reading from a binary file.

        Parameters
        ----------
        rec_file: str
            Path to binary file that contains the raw recording file.
        geom_file: str
            Path to text file containing the geometry file. The file should
            contain n_chan lines and each line should contain two numbers that
            are separated by ' '.
        sample_rate: int
            Recording sample rate in Hz
        n_batches: int
            Processes the recording in n_batches number of consecuitive
            segments that start from the beginning.
        batch_time_samples: int
            Number of time samples per each batch to be used.
        filter_std: bool
            The iterator both filters and standardizes the recording (dividing
            by standard deviation.
        whiten: bool
            Spatially whiten the recording.
        scale: float
            In case filter and whitening is not needed and the binary data is
            scaled up.
        """
        self.s_rate = sample_rate
        self.batch_time_samples = batch_time_samples
        self.n_batches = n_batches
        self.n_chan = n_chan
        self.radius = radius
        self.geometry = parse(geom_file, n_chan)
        self.neighbs = find_channel_neighbors(
            self.geometry, self.radius)
        self.filter_std = filter_std
        self.whiten = whiten
        self.scale = scale
        self.file = open(rec_file, 'r')

    def next_batch(self):
        """Gets the next temporal batch of recording."""
        ts = np.fromfile(
            self.file,
            count=self.n_chan * self.batch_time_samples,
            dtype=np.int16)
        ts = np.reshape(ts, [self.batch_time_samples, self.n_chan])
        if not self.filter_std:
            return ts / self.scale
        ts = butterworth(ts, 300, 0.1, 3, self.s_rate)
        ts = ts / np.std(ts)
        if not self.whiten:
            return ts
        ts = whitening(ts, self.neighbs, 40)
        return ts

    def reset_cursor(self):
        """Resets the cursor of the open file to the beginning."""
        self.file.seek(0)

    def close_iterator(self):
        self.file.close()


class MeanWaveCalculator(object):

    def __init__(self, batch_reader, spike_train, window=range(-10, 30)):
        """Sets up the object for mean wave computation.

        Parameters
        ----------
        spt: numpy.ndarray
            Shape [N, 2] where N is the total number of events. First column
            indicates the spike times in time sample and second is cluster
            identity of the spike times.
        window: list
            List of consecuitive integers. Indicating the window around spike
            times that indicate an event.

        Returns
        -------
        int
            The number of boundary violations in batch processing part of the
            mean wave calculation.
        """
        self.batch_reader = batch_reader
        self.spike_train = spike_train
        self.window = window
        self.spike_train = clean_spike_train(
            self.spike_train)
        self.n_units = max(self.spike_train[:, 1] + 1)
        self.templates = np.zeros(
            [len(self.window), batch_reader.n_chan, self.n_units])

    def compute_templates(self, n_batches):
        """Computes the templates from a given number of batches."""
        self.batch_reader.reset_cursor()
        counts = np.zeros(self.n_units)
        boundary_violation = 0
        n_samples = self.batch_reader.batch_time_samples
        for i in tqdm(range(n_batches)):
            batch_idx = np.logical_and(
                self.spike_train[:, 0] > i * n_samples,
                self.spike_train[:, 0] < (i + 1) * n_samples)
            spt = self.spike_train[batch_idx, :]
            spt[:, 0] -= n_samples * i
            ts = self.batch_reader.next_batch()
            for j in range(spt.shape[0]):
                try:
                    self.templates[:, :, spt[j, 1]] += (ts[spt[j, 0] +
                                                        self.window, :])
                    counts[spt[j, 1]] += 1
                except Exception:
                    boundary_violation += 1
        for u in range(self.n_units):
            if counts[u]:
                self.templates[:, :, u] /= counts[u]
        return boundary_violation

    def close_reader(self):
        self.batch_reader.close()


class RecordingAugmentation(object):

    def __init__(self, mean_wave_calculator, move_rate,
                 augment_rate, dist_factor=0.5, refractory_period=2.0):
        """Sets up the object for stability metric computations.

        Parameters
        ----------
        mean_wave_calculator: MeanWaveCalculator
            mean_wave_calculator: MeanWaveCalculator object.
            move_rate: float [0, 1]. The rate at which original clusters
            will be moved spatially around.
            dist_factor: float [0, 1]. How far the should the template
            move spatially. 0 represents no movement and 1 furtherst.
            refractory_period: float
            The minimum time between spikes of the same unit/cluster in
            milli-seconds.
        """
        self.template_comp = mean_wave_calculator
        self.geometry = mean_wave_calculator.batch_reader.geometry
        self.n_chan = self.geometry.shape[0]
        self.template_calculator = mean_wave_calculator
        # Number of samples per batches.
        n_samp = self.template_calculator.batch_reader.batch_time_samples
        self.batch_num_samples = n_samp
        self.construct_channel_map()
        self.compute_stat_summary()
        self.move_rate = move_rate
        self.augment_rate = augment_rate
        self.dist_factor = dist_factor
        # Convert refractory period to time samples.
        sampling_rate = mean_wave_calculator.batch_reader.s_rate
        self.refrac_period = refractory_period * 1e-3 * sampling_rate

    def construct_channel_map(self):
        """Constucts a map of coordinate to channel index."""
        self.geom_map = {}
        for i in range(self.n_chan):
            self.geom_map[(self.geometry[i, 0], self.geometry[i, 1])] = i
        pair_dist = squareform(pdist(self.geometry))
        self.closest_channels = np.argsort(pair_dist, axis=1)

    def correct_spike_time(self, spike_times, aug_spike_times):
        """Corrects any violation of refractory period for spike times.

        Parameters
        ----------
        spike_times: numpy.array
            Sorted numpy.array of base spike times.
        aug_spike_times: numpy.array
            Sorted numpy.array of spike times to be added to the base. These
            should not violate refractory period among themselves.

        Returns
        -------
        numpy.array
            New augmented spike times where there is no violation of refractory
            period time with respect to the combined spike train.
        """
        if len(spike_times) == 0 or len(aug_spike_times) == 0:
            return aug_spike_times
        # Number of spikes that violate refractory period.
        num_violation = 0
        # Silent periods that more spikes can be added.
        silent_period = []
        # Last spike time that was added, combined between the two.
        last_spike_time = 0
        # The spike that was added the current iteration, combined between
        # the two spike trains.
        current_spike_time = 0
        valid_spike_times = []

        remove_idx = []
        for i in range(1, len(aug_spike_times)):
            diff = aug_spike_times[i] - aug_spike_times[i - 1]
            if diff < self.refrac_period:
                remove_idx.append(i - 1)
                num_violation += 1
        aug_spike_times = np.delete(aug_spike_times, remove_idx)
        # Cursor on the base spike times.
        i = 0
        # Cursor on the augmented spike_times.
        j = 0
        while i < len(spike_times) or j < len(aug_spike_times):
            diff = 0
            if i >= len(spike_times):
                # We saw all base spike times.
                diff = self.refrac_period + 1
            elif j >= len(aug_spike_times):
                # We saw all augmented spikes.
                diff = - self.refrac_period - 1
            else:
                diff = spike_times[i] - aug_spike_times[j]

            if diff > self.refrac_period:
                current_spike_time = aug_spike_times[j]
                valid_spike_times.append(current_spike_time)
                j += 1

            elif diff > - self.refrac_period and diff < self.refrac_period:
                # Violating refrac period with respect to base spike_times
                j += 1
                current_spike_time = last_spike_time
                num_violation += 1
            else:
                current_spike_time = spike_times[i]
                i += 1
            # Check whether there is a silent period.
            silence = current_spike_time - last_spike_time
            if silence > 2 * self.refrac_period:
                silent_period.append((last_spike_time, current_spike_time))
            last_spike_time = current_spike_time.astype('int')

        # Add as many unvalid augmented spike times as possible back.
        i = 0
        while num_violation > 0 and i < len(silent_period):
            valid_spike_times.append(
                    silent_period[i][0] + self.refrac_period)
            i += 1
            num_violation -= 1
        return np.sort(np.array(valid_spike_times)).astype('int')

    def move_spatial_trace(self, template, spatial_size=10, mode='amp'):
        """Moves the waveform spatially around the probe.

        Parameters
        ----------
        template: numpy.ndarray
            Shape [T, C]
        spatial_size: int
            How many channels comprise the spatial trace of the given template.
        mode: str
            Main channels are detected using amplitude if 'amp' and energy
            otherwise.
        """
        new_temp = np.zeros(template.shape)
        if mode == 'amp':
            location = np.argsort(
                np.max(np.abs(template), axis=0))[-spatial_size:]
        main_channel = location[-1]
        # Move the main channel to another channel which is sampled
        # according to a binomial distribution with n_channel trial
        # on a sorted channel list based on distance from the original
        # channel.
        rand_int = np.random.binomial(n=self.n_chan, p=self.dist_factor)
        new_main_channel = self.closest_channels[main_channel, rand_int]
        prior_coordinates = self.geometry[main_channel, :]
        new_coordinates = self.geometry[new_main_channel, :]
        translation = new_coordinates - prior_coordinates
        x_move = translation[0]
        y_move = translation[1]
        # the vector of translation from original location to new one.
        trans = np.zeros([len(location), 2]).astype('int') - 1
        trans[:, 0] = location
        for i, l in enumerate(location):
            new_x_coord = self.geometry[l, 0] + x_move
            new_y_coord = self.geometry[l, 1] + y_move
            candidate = (new_x_coord, new_y_coord)
            if candidate in self.geom_map:
                trans[i, 1] = self.geom_map[candidate]
            else:
                continue
        idx_origin = trans[trans[:, 1] >= 0, 0]
        idx_moved = trans[trans[:, 1] >= 0, 1]
        new_temp[:, idx_moved] = template[:, idx_origin]
        return new_temp

    def compute_stat_summary(self):
        """Sets up statistic summary of given spike train.

        This function models the difference in time sample
        between consecutive firings of a particular unit
        as a log-normal distribution.

        Returns
        -------
        np.ndarray
            Shape [U, 3] where U is the number of units in the spike train.
            The columns of the summary respectively correspond to mean,
            standard devation of the log-normal and the total count of spikes
            for units.
        """
        self.stat_summary = np.zeros(
            [self.template_comp.n_units, 3])
        spt = self.template_comp.spike_train
        for u in range(self.template_comp.n_units):
            # spike train of unit u
            spt_u = np.sort(spt[spt[:, 1] == u, 0])
            if len(spt > 2):
                # We estimate the difference between
                # consecutive firing times of the same unit
                u_firing_diff = spt_u[1:] - spt_u[:-1]
                # Getting rid of duplicates.
                # TODO: do this more sensibly.
                u_firing_diff[u_firing_diff == 0] = 1
                u_firing_diff = np.log(u_firing_diff)
                u_mean = np.mean(u_firing_diff)
                u_std = np.std(u_firing_diff)
                self.stat_summary[u, :] = u_mean, u_std, len(spt_u)
        return self.stat_summary

    def make_fake_spike_train(self, augment_rate):
        """Augments the data and saves the result to binary.

        Parameters
        ----------
        augment_rate: float
            Between 0 and 1. Augmented spikes per unit (percentage of total
            spikes per unit).
        """
        spt = self.template_comp.spike_train
        # We sample a new set of spike times per cluster.
        times = []
        cid = []
        for u in range(self.template_comp.n_units):

            if (np.isnan(self.stat_summary[u, 0]) or
               np.isnan(self.stat_summary[u, 1])):
                continue

            spt_u = np.sort(spt[spt[:, 1] == u, 0])
            new_spike_count = int(
                self.stat_summary[u, 2] * augment_rate)
            diffs = np.exp(np.random.normal(
                self.stat_summary[u, 0],
                self.stat_summary[u, 1],
                new_spike_count)).astype('int')
            # Offsets for adding new spikes based on the
            # sampled differential times.
            offsets = np.sort(
                np.random.choice(spt_u, new_spike_count, replace=False))
            # Enforce refractory period.
            diffs[diffs < self.refrac_period] = self.refrac_period
            new_spikes = offsets + diffs
            new_spikes = self.correct_spike_time(spt_u, new_spikes)
            times += list(new_spikes)
            cid += [u] * new_spike_count
        return np.array([times, cid]).T

    def save_augment_recording(self, out_file_name, length, scale=1e2):
        """Augments recording and saves it to file.

        Parameters
        ----------
        out_file_name: str
            Name of output file where the augmented recording is writen to.
        length: int
            Length of augmented recording in batch size of the originial batch
            iterator object which is in the mean wave calculatro object.
        move_rate: float
            Between 0 and 1. Percentage of units whose augmented spike wave
            form is spatially moved.

        Returns
        -------
        tuple
            Tuple with two members. First is a numpy.ndarray which
            is the new ground truth spike train. Second is the status
            which is a list of string, each is an error regarding
            boundary violation for batch processing.
        """
        status = []
        reader = self.template_comp.batch_reader
        reader.reset_cursor()
        # Determine which clusters are spatially moved.
        orig_templates = self.template_comp.templates
        n_units = self.template_comp.n_units
        # list of unit numbers which we move spatially.
        moved_units = np.sort(
            np.random.choice(range(n_units),
                             int(self.move_rate * n_units),
                             replace=False))
        temp_shape = self.template_comp.templates.shape
        moved_templates = np.zeros(
            [temp_shape[0], temp_shape[1], len(moved_units)])
        # An array size of n_units where 0 indicates no movement
        # otherwise the index of the moved template in the move_templates
        # np.ndarray.
        moved = np.zeros(n_units)
        for i, u in enumerate(moved_units):
            moved[u] = i
            # Spatial distance is drawn from a poisson distribution.
            moved_templates[:, :, i] = self.move_spatial_trace(
                orig_templates[:, :, u])
        # Create augmented spike train.
        aug_spt = self.make_fake_spike_train(self.move_rate)
        reader = self.template_comp.batch_reader
        boundary_violation = 0
        n_samples = reader.batch_time_samples
        f = open(out_file_name, 'w')
        # TODO: for debugging, remove later.
        moved = moved.astype('int')
        for i in tqdm(range(length)):
            batch_idx = np.logical_and(
                aug_spt[:, 0] > i * n_samples,
                aug_spt[:, 0] < (i + 1) * n_samples)
            spt = aug_spt[batch_idx, :]
            spt[:, 0] -= n_samples * i
            ts = reader.next_batch()
            for j in range(spt.shape[0]):
                cid = spt[j, 1]
                try:
                    # Time window around spike
                    spike_win = spt[j, 0] + self.template_comp.window
                    if moved[cid]:
                        sup_signal = moved_templates[:, :, moved[cid]]
                        ts[spike_win, :] += sup_signal
                    else:
                        sup_signal = orig_templates[:, :, cid]
                        ts[spike_win, :] += sup_signal
                except Exception as e:
                    status.append('warning:{}'.format(str(e)))
                    boundary_violation += 1
            ts *= scale
            ts = ts.astype('int16')
            ts.tofile(f)
        # Reassign spikes from moved clusters to new units.
        new_unit_id = self.template_comp.n_units
        for u in range(self.template_comp.n_units):
            if moved[u]:
                aug_spt[aug_spt[:, 1] == u, 1] = new_unit_id
                new_unit_id += 1
        f.close()
        orig_count = self.template_comp.spike_train.shape[0]
        aug_count = aug_spt.shape[0]
        # Appends the new synthetic spike train to the base spike train.
        new_aug_spike_train = np.append(
            np.append(self.template_comp.spike_train,
                      np.zeros([orig_count, 1], dtype='int'),
                      axis=1),
            np.append(aug_spt, np.ones([aug_count, 1], dtype='int'), axis=1),
            axis=0)
        # Gets rid of any spike times beyond the length of the augmented
        # Data set.
        aug_rec_len = length * self.batch_num_samples
        valid_idx = new_aug_spike_train[:, 0] < aug_rec_len
        new_aug_spike_train = new_aug_spike_train[valid_idx, :]
        return new_aug_spike_train, status


class SpikeSortingEvaluation(object):

    def __init__(self, spt_base, spt, tmp_base=None, tmp=None,
                 method='hungarian'):
        """Sets up the evaluation object with two spike trains.

        Parameters
        ----------
        spt_base: numpy.ndarray
            Shape [N, 2]. base line spike train. First column is spike times
            and second the cluster identities.
        spt: numpy.ndarray
            Shape [M, 2].
        tmp_base: numpy.ndarray or None
            Shape [T1, C, N]. Ground truth unit mean waveforms. If None,
            hungarian algorithm is used for matching.
        tmp_base: numpy.ndarray or None
            Shape [T2, C, M]. Clustering units mean waveforms. If None,
            the hungarian algorithm is used for matching.
        method: str, 'greedy' or 'hungarian'
            Method for matching clusters/units.
        """
        if tmp_base is None or tmp is None:
            method = 'hungarian'
        # clean the spike train before calling this function.
        self.tmp_base = tmp_base
        self.tmp = tmp
        spt_base = clean_spike_train(spt_base)
        spt = clean_spike_train(spt)
        self.n_units = np.max(spt_base[:, 1]) + 1
        self.n_clusters = np.max(spt[:, 1]) + 1
        self.spt_base = spt_base
        self.spt = spt
        # Spike counts per unit and cluster
        self.spike_count_base = self.count_spikes(spt_base)
        self.spike_count_cluster = self.count_spikes(spt)
        # Compute matching and accuracies.
        self.confusion_matrix = None
        self.compute_confusion_matrix()
        self.true_positive = np.zeros(self.n_units)
        self.false_positive = np.zeros(self.n_units)
        self.unit_cluster_map = np.zeros(self.n_units, dtype='int')
        self.compute_accuracies(method)

    def count_spikes(self, spt):
        """Counts spike events per cluster/units.

        Parameters
        ----------
        spt: numpy.ndarray
            Shape [N, 2]. Clean spike train where cluster ids are 0, ..., N-1.
        """
        n_cluster = np.max(spt[:, 1]) + 1
        counts = np.zeros(n_cluster)
        for u in range(n_cluster):
            counts[u] = np.sum(spt[:, 1] == u)
        return counts

    def compute_confusion_matrix(self):
        """Calculates the confusion matrix of two spike trains.

        The first spike train is the instances original spike train.
        The second one is given as an argument.
        """
        confusion_matrix = np.zeros(
            [self.n_units, self.n_clusters])
        for unit in tqdm(range(self.n_units)):
            idx = self.spt_base[:, 1] == unit
            spike_times_base = np.sort(self.spt_base[idx, 0])
            for cluster in range(self.n_clusters):
                idx = self.spt[:, 1] == cluster
                spike_times_cluster = np.sort(self.spt[idx, 0])
                confusion_matrix[unit, cluster] = self.count_matches(
                    spike_times_base, spike_times_cluster)
        self.confusion_matrix = confusion_matrix

    def count_matches(self, array1, array2):
        """Finds the matches between two count process.

        Returns
        -------
        int
            Number of temporal collisions of spikes in array1 vs spikes in
            array2.
        """
        # In time samples
        self.admissible_proximity = 60
        m, n = len(array1), len(array2)
        i, j = 0, 0
        count = 0
        while i < m and j < n:
            if abs(array1[i] - array2[j]) < self.admissible_proximity:
                i += 1
                j += 1
                count += 1
            elif array1[i] < array2[j]:
                i += 1
            else:
                j += 1
        return count

    def compute_accuracies(self, method):
        """Computes the TP/FP accuracies for the given spike trains.

        Parameters:
        -----------
        method: str. 'hungarian', 'greedy'
            Method of matching base units to clusters.
        """
        # Maps ground truth unit to matched cluster unit.
        # -1 indicates no matching if n_units > n_clusters.
        unmatched_clusters = list(range(self.n_clusters))
        self.unit_cluster_map = np.zeros(self.n_units, dtype='int') - 1

        if method == 'hungarian':
            # Compute the accuracy confusion matrix.
            percent_matrix = self.confusion_matrix / np.reshape(
                self.spike_count_base, [self.n_units, 1])
            units, clusters = linear_sum_assignment(
                -percent_matrix)
            self.unit_cluster_map[units] = clusters

        elif method == 'greedy':
            # Calculate and match energy of templates.
            # The energy is based on amplitude (l-inf norm).
            energy_base = np.max(self.tmp_base, axis=0)
            energy = np.max(self.tmp, axis=0)
            energy_dist = cdist(energy_base.T, energy.T)
            ordered_units = reversed(
                np.argsort(np.linalg.norm(energy_base, axis=0)))
            # First match the largest energy ground truth templates.
            for unit in ordered_units:
                if len(unmatched_clusters) < 1:
                    break
                # TODO(hooshmand): Find a fix for template comparison.
                # If the closest template is not very similar skip it.
                # if (np.min(energy_dist[unit, unmatched_clusters]) >
                # 1/4 * np.linalg.norm(energy_base[:, unit])):
                # continue
                # Also, something like selecting the template with
                # the closest shape, e.g. the following zombie code line.
                matched_cluster_id = unmatched_clusters[np.argmin(
                    energy_dist[unit, unmatched_clusters])]
                matched_cluster_id = unmatched_clusters[np.argmax(
                    self.confusion_matrix[unit, unmatched_clusters])]
                unmatched_clusters.remove(matched_cluster_id)
                self.unit_cluster_map[unit] = matched_cluster_id
        # Units which have a match in the clusters.
        rec_units = np.where(self.unit_cluster_map > -1)[0]
        recovered = np.zeros(self.n_units)

        for unit in rec_units:
            recovered[unit] = (self.confusion_matrix[unit,
                               self.unit_cluster_map[unit]])

        self.true_positive = recovered / self.spike_count_base
        match_count = self.spike_count_cluster[self.unit_cluster_map]
        self.false_positive = (match_count - recovered) / match_count
