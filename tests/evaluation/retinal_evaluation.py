import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import yaml

from numpy import genfromtxt
from subprocess import call
from yass.evaluate.stability import (MeanWaveCalculator,
                                     RecordingAugmentation,
                                     RecordingBatchIterator,
                                     SpikeSortingEvaluation)


def main_channels(template):
    """Computes the main channel of a list of templates.

    Parameters
    ----------
    template: numpy.ndarray
        The shape of the array should be (T, C, K) where T indicates
        time samples, C number of channels and K total number of
        units/clusters.
    """
    return np.argsort(np.max(
        np.abs(template), axis=0), axis=0).T


def temp_snr(templates):
    """Computes the PNR of a list of templates.

    Parameters
    ----------
    template: numpy.ndarray
        The shape of the array should be (T, C, K) where T indicates
        time samples, C number of channels and K total number of
        units/clusters.
    """
    tot = templates.shape[2]
    res = np.zeros(tot)
    for unit, c in enumerate(main_channels(templates)[:, -1]):
        res[unit] = np.linalg.norm(templates[:, c, unit], np.inf)
    return res


class EvaluationPlot(object):
    """Standard figure for evaluation comparison."""

    def __init__(self, data_set_title, n_dataset, methods=['Method'],
                 logit_y=True, eval_type='Accuracy'):
        """Setup pyplot figures.

        Parameters
        ----------
        data_set_title: str
            Title of the data set that will be displayed in the plots.
        n_dataset: int
            Total umber of data sets that evaluations are performed on.
        methods: list of str
            The spike sorting methods that evaluations are done for.
        logit_y: bool
            Logit transform the y-axis (metric axis) to emphasize near 1
            and near 0 values.
        eval_type: str
            Type of metric (for display purposes only) which appears in the
            plots.
        """
        self.new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                           '#bcbd22', '#17becf']
        self.n_dataset = n_dataset
        self.methods = methods
        self.data_set_title = data_set_title
        self.eval_type = eval_type
        self.logit_y = logit_y
        self.data_set_title = data_set_title
        # Contains evaluation metrics for methods and datasets.
        self.metric_matrix = {}
        for method in self.methods:
            self.metric_matrix[method] = []
            for i in range(self.n_dataset):
                self.metric_matrix[method].append(None)

    def logit(self, x, inverse=False):
        """Logit transfors the array x.

        Parameters
        ----------
        x: numpy.ndarray
            List of values [0-1] only to be logit transformed.
        inverse: bool
            Inverse-logit transforms if True.
        """
        # Add apsilon to avoid boundary conditions for transform.
        x[x == 0] += 0.0001
        x[x == 1] -= 0.0001
        if inverse:
            return 1 / (1 + np.exp(-x))
        return np.log(x / (1 - x))

    def set_logit_labels(
            self, labs=np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])):
        """Logit transforms the y axis.

        Parameters
        ----------
        labs: numpy.ndarray
            List of values ([0-1] only) to be displayed as ticks on
            y axis.
        """
        for i in range(self.n_dataset):
            self.ax[i].set_yticks(self.logit(labs))
            self.ax[i].set_yticklabels(labs)

    def add_metric(self, snr_list, percent_list, dataset_number,
                   method_name='Method'):
        """Adds accuracy percentages for clusters/units of a method.

        Parameters
        ----------
        snr_list: numpy.ndarray of shape(N,)
            List of SNR/PNR values for clusters/units of the corresponding
            dataset number and spike sorting method.
        percent_list: numpy.ndarray of shape(N,)
            List of SNR/PNR values for clusters/units of the corresponding
            dataset number and spike sorting method.
        dataset_number: int
            Value should be between 0 and self.n_dataset - 1. Indicates
            which dataset are the evaluations for.
        method: str
            Should be a member of self.methods. Indicates which of the
            spike sorting methods the evaluations correspond to.
        """
        if method_name not in self.methods:
            raise KeyError('Method name does not exist in methods list.')
        if np.any(percent_list < 0) or np.any(percent_list > 1):
            raise TypeError(
                    'Percent accuracy list should contain only [0-1] values.')
        eval_tup = (snr_list, percent_list)
        self.metric_matrix[method_name][dataset_number] = eval_tup

    def generate_snr_metric_plot(self):
        """Generate pdf plots of evaluations for the datasets and methods."""
        self.fig, self.ax = plt.subplots(self.n_dataset, 1)
        for i in range(self.n_dataset):
            self.ax[i].set_title(
                    self.data_set_title + 'Dataset {}'.format(i + 1))
            self.ax[i].set_ylabel('Percent {}'.format(self.eval_type))
            self.ax[i].legend(self.methods)
        self.ax[i].set_xlabel('Log PNR')
        if self.logit_y:
            self.set_logit_labels()
        for method_idx, method in enumerate(self.methods):
            for i in range(self.n_dataset):
                try:
                    metric_tuple = self.metric_matrix[method][i]
                    metrics = metric_tuple[1]
                    if self.logit_y:
                        metrics = self.logit(metrics)
                    self.ax[i].scatter(
                        metric_tuple[0], metrics,
                        color=self.new_colors[method_idx])
                except Exception:
                    print("No metric found for {} for dataset {}".format(
                        method, i + 1))
        self.fig.set_size_inches(12, 4 * self.n_dataset)
        plt.savefig('{}_{}.pdf'.format(self.data_set_title, self.eval_type))


def main(n_batches=6):
    """Runs the procedure for evaluating yass on retinal data."""

    config_file = open('config_template.yaml', 'r')
    config = yaml.load(config_file)
    config_file.close()
    # Extracting window around spikes.
    sampling_rate = config['recordings']['sampling_rate']
    n_chan = config['recordings']['n_channels']
    dtype = config['recordings']['dtype']
    spike_length = config['recordings']['spike_size_ms']
    window_radius = int(spike_length * sampling_rate / 1e3)
    window = range(-window_radius, window_radius)

    k_tot_data = 4

    # Set up the pyplot figures
    stb_plot = EvaluationPlot('EJ Retinal', k_tot_data, eval_type='Stability')
    acc_plot = EvaluationPlot('EJ Retinal', k_tot_data)

    for data_idx, data_number in enumerate(range(1, k_tot_data + 1)):
        # Setting up config file for yass.
        bin_file = 'ej49_data{}.bin'.format(data_number)
        geom_file = 'ej49_geometry{}.txt'.format(data_number)
        config['data']['recordings'] = bin_file
        config['data']['geometry'] = geom_file
        with open('config_ej49.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        # TODO(hooshmand): Find the internal call for yass.
        call(['yass', 'config_ej49.yaml'])
        yass_spike_train_file = 'yass_spike_train_{}.csv'.format(data_number)
        call(['cp', 'spike_train.csv', yass_spike_train_file])
        spike_train = genfromtxt(
            yass_spike_train_file, delimiter=',').astype('int32')
        # Data augmentation setup.
        os.path.getsize(bin_file)
        file_size_bytes = os.path.getsize(bin_file)
        tot_samples = file_size_bytes / (np.dtype(dtype).itemsize * n_chan)
        radius = 70
        n_batch_samples = int(tot_samples / n_batches)
        batch_reader = RecordingBatchIterator(
            bin_file, geom_file, sample_rate=sampling_rate,
            batch_time_samples=n_batch_samples, n_batches=n_batches,
            n_chan=n_chan, radius=radius, whiten=False)
        mean_wave = MeanWaveCalculator(
                batch_reader, spike_train, window=window)
        mean_wave.compute_templates(n_batches=n_batches)
        # Augment with new spikes.
        stab = RecordingAugmentation(
                mean_wave, augment_rate=0.25, move_rate=0.2)
        aug_bin_file = 'ej49_data{}.aug.bin'.format(data_number)
        aug_gold_spt, status = stab.save_augment_recording(
                aug_bin_file, n_batches)
        np.save('ej49_data{}.aug.npy'.format(data_number), aug_gold_spt)
        # Setting up config file for yass to run on augmented data.
        config['data']['recordings'] = aug_bin_file
        config['data']['geometry'] = geom_file
        with open('config_ej49.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        # TODO(hooshmand): Find the internal call for yass.
        call(['yass', 'config_ej49.yaml'])
        yass_aug_spike_train_file = 'yass_aug_spike_train_{}.csv'.format(
                data_number)
        call(['cp', 'spike_train.csv', yass_aug_spike_train_file])

        # Evaluate accuracy of yass.
        gold_std_spike_train_file = 'groundtruth_ej49_data{}.mat'.format(
                data_number)
        gold_std_map = scipy.io.loadmat(gold_std_spike_train_file)
        gold_std_spike_train = np.append(
            gold_std_map['spt_gt'], gold_std_map['L_gt'], axis=1)
        gold_standard_mean_wave = MeanWaveCalculator(
            batch_reader, gold_std_spike_train, window=window)
        gold_standard_mean_wave.compute_templates(n_batches=n_batches)
        accuracy_eval = SpikeSortingEvaluation(
            gold_std_spike_train, spike_train,
            gold_standard_mean_wave.templates, mean_wave.templates)
        acc_tp = accuracy_eval.true_positive
        acc_plot.add_metric(
                np.log(temp_snr(gold_standard_mean_wave.templates)),
                acc_tp, data_idx)
        batch_reader.close_iterator()

        # Evaluate stability of yass.
        yass_aug_spike_train = genfromtxt(
            yass_aug_spike_train_file, delimiter=',').astype('int32')
        batch_reader = RecordingBatchIterator(
            aug_bin_file, geom_file, sample_rate=sampling_rate,
            batch_time_samples=n_batch_samples, n_batches=n_batches,
            n_chan=n_chan, radius=radius, whiten=False)
        aug_gold_standard_mean_wave = MeanWaveCalculator(
            batch_reader, aug_gold_spt, window=window)
        aug_gold_standard_mean_wave.compute_templates(n_batches=n_batches)
        aug_yass_mean_wave = MeanWaveCalculator(
            batch_reader, yass_aug_spike_train, window=window)
        aug_yass_mean_wave.compute_templates(n_batches=n_batches)
        stability_eval = SpikeSortingEvaluation(
            aug_gold_spt, yass_aug_spike_train,
            aug_gold_standard_mean_wave.templates,
            aug_yass_mean_wave.templates)
        stb_tp = stability_eval.true_positive
        stb_plot.add_metric(
                np.log(temp_snr(aug_gold_standard_mean_wave.templates)),
                stb_tp, data_idx)
        batch_reader.close_iterator()

    # Render the plots and save them.
    acc_plot.generate_snr_metric_plot()
    stb_plot.generate_snr_metric_plot()


if __name__ == '__main__':
    main()
