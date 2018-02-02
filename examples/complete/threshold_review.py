"""
Checking results from threshold detector
"""
import matplotlib.pyplot as plt

from yass.explore import SpikeTrainExplorer, RecordingExplorer

path_to_data = '/Users/Edu/data/yass/tmp/standarized.bin'
path_to_spike_train = '/Users/Edu/data/yass/tmp/spike_train.npy'

exp = RecordingExplorer(path_to_data, spike_size=15)
spe = SpikeTrainExplorer(path_to_spike_train, exp)

spe.plot_templates(group_ids=range(10))
plt.show()
