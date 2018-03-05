import numpy as np
import os
import yaml

from yass.evaluate.stability import (MeanWaveCalculator,
                                     RecordingAugmentation,
                                     RecordingBatchIterator,
                                     SpikeSortingEvaluation)
from yass.evaluate.visualization import ChristmasPlot
from yass.evaluate.util import temp_snr
from yass.pipeline import run


class Analyzer(object):
    """Class that analyzes the performance of yass on a certain dataset"""

    def __init__(self, config, gold_std_spike_train=None):
        """Sets up the analyzer object with configurations.

        Parameters:
            config: str or map
            Absolute path to the yass config file for the dataset.
            Alternatively, map of configuration options.
            gold_std_spike_train: numpy.ndarray of shape (N, 2) or None
            Gold standard spike train where first column corresponds to
            spike times and second column corresponds to cluster identities.
            If None, then no accuracy evaluation will be done.
        """
        self.config_file = None
        self.config = config
        if isinstance(self.config, str):
            self.config_file = config
            self.load_config()
        elif not isinstance(self.config, map):
            raise ValueError("config should either of type map or str.")
        self.gold_std_spike_train = gold_std_spike_train
        # Directories that contain necessary metrics, intermediate files.
        self.root_dir = self.config['data']['root_folder']
        self.tmp_dir = os.path.join(self.root_dir, 'tmp', 'eval')
        if not os.path.isdir(os.path.join(self.root_dir, 'tmp')):
            os.mkdir(os.path.join(self.root_dir, 'tmp'))
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        # Files that will contain necessary metrics, intermediate computations.
        self.gold_templates_file = os.path.join(
                self.tmp_dir, 'gold_templates.npy')
        self.yass_templates_file = os.path.join(
                self.tmp_dir, 'yass_templates.npy')
        self.gold_aug_templates_file = os.path.join(
                self.tmp_dir, 'aug_templates.npy')
        self.yass_aug_templates_file = os.path.join(
                self.tmp_dir, 'yass_aug_templates.npy')
        # Spike train files.
        self.aug_spike_train_file = os.path.join(
                self.tmp_dir, 'augmented_spike_train.npy')
        self.yass_spike_train_file = os.path.join(
                self.root_dir, 'tmp', 'spike_train.npy')
        self.yass_aug_spike_train_file = os.path.join(
                self.tmp_dir, 'tmp', 'spike_train.npy')
        # Metric files.
        self.accuracy_file = os.path.join(self.tmp_dir, 'accuracy.npy')
        self.stability_file = os.path.join(self.tmp_dir, 'stability.npy')

    def load_config(self):
        """Loads config .yaml file indicated in constructor."""
        conf_file = open(self.config_file, 'r')
        self.config = yaml.load(conf_file)
        conf_file.close()

    def run_stability(self, n_batches=6):
        """Runs stability metric computation for the given config file.

        Parameters:
            n_batchs: int
            Break down the processing of the dataset in these many batches.
        """
        # Check whether this analysis is not done already.
        if os.path.isfile(self.stability_file):
            return
        sampling_rate = self.config['recordings']['sampling_rate']
        n_chan = self.config['recordings']['n_channels']
        dtype = self.config['recordings']['dtype']
        spike_length = self.config['recordings']['spike_size_ms']
        # Extracting window around spikes.
        window_radius = int(spike_length * sampling_rate / 1e3)
        window = range(-window_radius, window_radius)

        bin_file = os.path.join(
                self.root_dir, self.config['data']['recordings'])
        geom_file = os.path.join(
                self.root_dir, self.config['data']['geometry'])
        # Check whether spike sorting has already been completed.
        if not os.path.isfile(self.yass_spike_train_file):
            spike_train = run(config=self.config)
        else:
            spike_train = np.load(self.yass_spike_train_file)
        # Data augmentation setup.
        os.path.getsize(bin_file)
        file_size_bytes = os.path.getsize(bin_file)
        tot_samples = file_size_bytes / (np.dtype(dtype).itemsize * n_chan)
        radius = 70
        n_batch_samples = int(tot_samples / n_batches)

        bin_extension = os.path.splitext(bin_file)[1]
        aug_file_name = 'augmented_recording{}'.format(bin_extension)
        aug_bin_file = os.path.join(self.tmp_dir, aug_file_name)

        # Check whether data augmentation has been done before or not.
        is_file_aug_bin = os.path.isfile(aug_bin_file)
        is_file_aug_spt = os.path.isfile(self.aug_spike_train_file)
        is_file_yass_temp = os.path.isfile(self.yass_templates_file)
        if is_file_aug_bin and is_file_aug_spt and is_file_yass_temp:
            aug_gold_spt = np.load(self.aug_spike_train_file)
        else:
            batch_reader = RecordingBatchIterator(
                bin_file, geom_file, sample_rate=sampling_rate,
                batch_time_samples=n_batch_samples, n_batches=n_batches,
                n_chan=n_chan, radius=radius, whiten=False)
            mean_wave = MeanWaveCalculator(
                    batch_reader, spike_train, window=window)
            mean_wave.compute_templates(n_batches=n_batches)
            np.save(self.yass_templates_file, mean_wave.templates)
            # Compute gold standard mean waveforms too.
            gold_standard_mean_wave = MeanWaveCalculator(
                    batch_reader, self.gold_std_spike_train, window=window)
            gold_standard_mean_wave.compute_templates(n_batches=n_batches)
            np.save(self.gold_templates_file,
                    gold_standard_mean_wave.templates)
            # Augment with new spikes.
            stab = RecordingAugmentation(
                    mean_wave, augment_rate=0.25, move_rate=0.2)
            aug_gold_spt, status = stab.save_augment_recording(
                    aug_bin_file, n_batches)
            np.save(self.aug_spike_train_file, aug_gold_spt)
            np.save(os.path.join(self.tmp_dir, 'geom.npy'),
                    batch_reader.geometry)

        # Setting up config file for yass to run on augmented data.
        self.config['data']['root_folder'] = self.tmp_dir
        self.config['data']['recordings'] = aug_file_name
        self.config['data']['geometry'] = 'geom.npy'
        # Check whether spike sorting has already been completed.
        if not os.path.isfile(self.yass_aug_spike_train_file):
            yass_aug_spike_train = run(config=self.config)
        else:
            yass_aug_spike_train = np.load(self.yass_aug_spike_train_file)
        # Evaluate stability of yass.
        # Check whether the mean wave of the yass spike train on the augmented
        # data has been computed before or not.
        is_file_temp_yass = os.path.isfile(self.yass_aug_templates_file)
        is_file_temp_gold = os.path.isfile(self.gold_aug_templates_file)
        if is_file_temp_gold and is_file_temp_yass:
            gold_aug_templates = np.load(self.gold_aug_templates_file)
            yass_aug_templates = np.load(self.yass_aug_templates_file)
        else:
            batch_reader = RecordingBatchIterator(
                aug_bin_file, geom_file, sample_rate=sampling_rate,
                batch_time_samples=n_batch_samples, n_batches=n_batches,
                n_chan=n_chan, radius=radius, filter_std=False, whiten=False)
            aug_gold_standard_mean_wave = MeanWaveCalculator(
                batch_reader, aug_gold_spt, window=window)
            aug_gold_standard_mean_wave.compute_templates(n_batches=n_batches)
            gold_aug_templates = aug_gold_standard_mean_wave.templates
            aug_yass_mean_wave = MeanWaveCalculator(
                batch_reader, yass_aug_spike_train, window=window)
            aug_yass_mean_wave.compute_templates(n_batches=n_batches)
            yass_aug_templates = aug_yass_mean_wave.templates
            batch_reader.close_iterator()
            np.save(self.gold_aug_templates_file, gold_aug_templates)
            np.save(self.yass_aug_templates_file, yass_aug_templates)
        # Finally, instantiate a spike train evaluation object for comparisons.
        stability_eval = SpikeSortingEvaluation(
            aug_gold_spt, yass_aug_spike_train, gold_aug_templates,
            yass_aug_templates)
        # Saving results of evaluation for stability.
        np.save(self.stability_file, stability_eval.true_positive)

    def run_accuracy(self):
        """Runs accuracy evaluation for the config file.

        Note: This can be performed only if there is a gold standard
        spike train and run_stability has been called before.
        """
        # Check whether this analysis is not done already.
        if self.gold_std_spike_train is None:
            print("Can not run accuracy if there is no gold standard.")
        if os.path.isfile(self.accuracy_file):
            return
        # Evaluate accuracy of yass.
        spike_train = np.load(self.yass_spike_train_file)
        gold_templates = np.load(self.gold_templates_file)
        templates = np.load(self.yass_templates_file)
        accuracy_eval = SpikeSortingEvaluation(
            self.gold_std_spike_train, spike_train, gold_templates, templates)
        # Saving results of evaluation for accuracy.
        np.save(self.accuracy_file, accuracy_eval.true_positive)

    def run_analyses(self, n_batches=6):
        """Runs the analyses on the dataset indicated in config.

        Note: This step has to be called only when the stability has been
        evaluated.

        Parameters:
            n_batchs: int
            Break down the processing of the dataset in these many batches.
        """
        self.run_stability(n_batches=n_batches)
        self.run_accuracy()

    def visualize(self, metric):
        """Visualizes the metric of interest.

        Parameters:
        -----------
        metric: str, 'stability', 'accuracy'
            The type of metric that should be visualized.
        """
        if metric == 'stability':
            data_title = self.config['data']['recordings']
            plot = ChristmasPlot(data_title, eval_type=metric)
            templates = np.load(self.gold_aug_templates_file)
            stability = np.load(self.stability_file)
            # Add the log PNR of templates and stability.
            plot.add_metric(np.log(temp_snr(templates)), stability)
            plot.generate_snr_metric_plot(show_id=True)
            plot.generate_curve_plots()

        if metric == 'accuracy':
            data_title = self.config['data']['recordings']
            plot = ChristmasPlot(data_title, eval_type=metric)
            templates = np.load(self.gold_templates_file)
            accuracy = np.load(self.accuracy_file)
            # Add the log PNR of templates and stability.
            plot.add_metric(np.log(temp_snr(templates)), accuracy)
            plot.generate_snr_metric_plot(show_id=True)
            plot.generate_curve_plots()
