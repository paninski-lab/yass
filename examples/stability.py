"""
Stability metric example
"""
import os.path as path

import numpy as np
import matplotlib.pyplot as plt

from stability_evaluation import (RecordingBatchIterator, MeanWaveCalculator,
                                  RecordingAugmentation,
                                  SpikeSortingEvaluation)


ROOT = path.join(path.expanduser('~'), 'data/yass')
path_to_spike_train = path.join(ROOT, 'ej49_spikeTrain1_1.csv')
path_to_data = path.join(ROOT, 'ej49_data1_set1.bin')
path_to_geom = path.join(ROOT, 'ej49_geometry1.txt')

path_to_augmented = path.join(ROOT, 'augmented.bin')

spike_train = np.loadtxt(path_to_spike_train, dtype='int32', delimiter=',')
spike_train

br = RecordingBatchIterator(path_to_data, path_to_geom, sample_rate=30000,
                            batch_time_samples=1000000, n_batches=5,
                            n_chan=200, radius=100, whiten=False)

mwc = MeanWaveCalculator(br, spike_train)


# plot some of the recovered templates
for i in range(2):
    plt.plot(mwc.templates[:, :, i])
    plt.show()

# here we indicate what is the length of the augmented data in terms of
# batches (with respect to the batch iterator object.)
stab = RecordingAugmentation(mwc, augment_rate=0.25, move_rate=0.2)

# New ground truth spike train
new_gt_spt, status = stab.save_augment_recording(path_to_augmented, 5)

# Creating evaluation object for matching, TP, and FP
spt_ = spike_train[spike_train[:, 0] < 1e6, :]
tmp_ = mwc.templates[:, :, np.unique(spt_[:, 1])]

# Let's create a fake new spike train with only 100
# first units of the ground truth as clusters
spt_2 = spt_[spt_[:, 1] < 100, :]
tmp_2 = tmp_[:, :, :100]

# Here we just demonstrate with the sampe spike train
# The second argument should be a different spike train
ev = SpikeSortingEvaluation(spt_, spt_2, tmp_, tmp_2)

print(ev.true_positive)
print(ev.false_positive)
print(ev.unit_cluster_map)
