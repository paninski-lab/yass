import numpy as np
import os
import yaml

from yass.evaluate.stability import (MeanWaveCalculator,
                                     RecordingAugmentation,
                                     RecordingBatchIterator,
                                     SpikeSortingEvaluation)
from yass.pipeline import run


class Analyzer(object):
    """Class that analyzes the performance of yass on a certain dataset"""

    def __init__(self, config, gold_std_spike_train=None):
        """Sets up the analyzer object with configurations.

        Parameters:
            config: str
            Absolute path to the yass config file for the dataset.
            gold_std_spike_train: numpy.ndarray of shape (N, 2)
            Gold standard spike train where first column corresponds to
            spike times and second column corresponds to cluster identities.
            If None, then no accuracy evaluation will be done.
        """
        self.config = config
        self.gold_std_spike_train = gold_std_spike_train

    def run(self, n_batches=6):
        """Runs the analyses on the dataset indicated in config.

        Parameters:
            n_batchs: int
            Break down the processing of the dataset in these many batches.
        """
        config_file = open(self.config, 'r')
        config = yaml.load(config_file)
        config_file.close()
        # Extracting window around spikes.
        sampling_rate = config['recordings']['sampling_rate']
        n_chan = config['recordings']['n_channels']
        dtype = config['recordings']['dtype']
        spike_length = config['recordings']['spike_size_ms']
        window_radius = int(spike_length * sampling_rate / 1e3)
        window = range(-window_radius, window_radius)

        # Passing the config to pipeline to be run.
        root_dir = config['data']['root_folder']
        # Temp directory where all intermediate files are stored.
        tmp_dir = os.path.join(root_dir, 'tmp', 'eval')
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        bin_file = os.path.join(root_dir, config['data']['recordings'])
        geom_file = os.path.join(root_dir, config['data']['geometry'])
        spike_train = run(config=config)
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
        # Extension of the binary file.
        bin_extension = os.path.splitext(bin_file)[1]
        aug_file_name = 'augmented_recording{}'.format(bin_extension)
        aug_bin_file = os.path.join(tmp_dir, aug_file_name)
        aug_spt_file = os.path.join(tmp_dir, 'augmented_spike_train.npy')
        if not os.path.isfile(aug_bin_file):
            aug_gold_spt, status = stab.save_augment_recording(
                    aug_bin_file, n_batches)
            np.save(aug_spt_file, aug_gold_spt)
        # Setting up config file for yass to run on augmented data.
        np.save(os.path.join(tmp_dir, 'geom.npy'), batch_reader.geometry)
        config['data']['root_folder'] = tmp_dir
        config['data']['recordings'] = aug_file_name
        config['data']['geometry'] = 'geom.npy'
        yass_aug_spike_train = run(config=config)

        # Evaluate accuracy of yass.
        gold_standard_mean_wave = MeanWaveCalculator(
            batch_reader, self.gold_std_spike_train, window=window)
        gold_standard_mean_wave.compute_templates(n_batches=n_batches)
        accuracy_eval = SpikeSortingEvaluation(
            self.gold_std_spike_train, spike_train,
            gold_standard_mean_wave.templates, mean_wave.templates)
        # Saving results of evaluation for accuracy.
        self.gold_templates_file = os.path.join(tmp_dir, 'gold_templates.npy')
        self.accuracy_file = os.path.join(tmp_dir, 'accuracy.npy')
        np.save(self.gold_templates_file, gold_standard_mean_wave.templates)
        np.save(self.accuracy_file, accuracy_eval.true_positive)
        batch_reader.close_iterator()

        # Evaluate stability of yass.
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
        # Saving results of evaluation for stability.
        self.stability_file = os.path.join(tmp_dir, 'accuracy.npy')
        np.save(self.stability_file, stability_eval.true_positive)
        batch_reader.close_iterator()

    def visualize(self):
        """Renders results of the evaluation in plots."""
        # Set up the pyplot figures
        # TODO(hooshmand): This is not functioning now.
        pass
