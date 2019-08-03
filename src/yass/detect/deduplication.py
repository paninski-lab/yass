import numpy as np
import torch

def deduplicate_gpu(spike_index_torch, energy_torch,
                    recording_shape, channel_index,
                    max_window=5):

    # find device
    device = spike_index_torch.device
    # initialize energy train
    energy_train = torch.zeros(recording_shape).to(device)
    energy_train[spike_index_torch[:,0], spike_index_torch[:, 1]] = energy_torch

    # get temporal max
    maxpool = torch.nn.MaxPool2d(kernel_size=[max_window*2+1, 1],
                                 stride=1, padding=[max_window, 0])
    max_energy = maxpool(energy_train[None, None])[0, 0]
    # get spatial max
    max_energy = torch.cat(
        (max_energy,
         torch.zeros([max_energy.shape[0], 1]).to(device)),
        1)
    max_energy = torch.max(max_energy[:,channel_index], 2)[0] - 1e-8
    
    # deduplicated spikes: temporal and spatial locally it has the maximum energy
    spike_index_dedup = torch.nonzero((energy_train >= max_energy) & (energy_train > 2))

    return spike_index_dedup

def deduplicate(spike_index, energy,
                recording_shape, channel_index,
                max_window=5):

    # initialize energy train
    energy_train = np.zeros(recording_shape)
    energy_train[spike_index[:,0], spike_index[:, 1]] = energy
    energy_train = torch.from_numpy(energy_train).float()

    maxpool = torch.nn.MaxPool2d(kernel_size=[max_window*2+1, 1],
                                 stride=1, padding=[max_window, 0])
    max_energy = maxpool(energy_train[None, None])[0, 0]
    max_energy = torch.cat(
        (max_energy,
         torch.zeros([max_energy.shape[0], 1]).float()),
        1)
    max_energy = torch.max(max_energy[:, channel_index], 2)[0] - 1e-8
    spike_train_dedup = torch.nonzero(
        (energy_train >= max_energy) & (energy_train > 2)).data.numpy()

    return spike_index_dedup
