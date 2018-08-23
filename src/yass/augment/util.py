"""Utility functions for augmenting data
"""
import random
import numpy as np
import logging

import yass.array as yarr
from yass import util


logger = logging.getLogger(__name__)


# TODO: remove
def _make_noisy(x, the_noise):
    """Make a noisy version of x
    """
    noise_sample = the_noise[np.random.choice(the_noise.shape[0],
                                              x.shape[0],
                                              replace=False)]
    return x + noise_sample


def sample_from_zero_axis(x):
    """Sample from a certain axis
    """
    idx = np.random.choice(x.shape[0], 1, replace=True)[0]
    return x[idx], idx


def amplitudes(x):
    """Compute amplitudes
    """
    return np.max(np.abs(x), axis=(1, 2))


def draw_with_group_probabilities(elements, probabilities):
    """
    Group elements in a 1D array and draw them depending on the probabilities
    passed
    """
    n_groups = len(probabilities)
    groups = np.array_split(elements, n_groups)

    def draw_one():
        group_idx = (np.random
                       .choice(np
                               .arange(n_groups), size=1, p=probabilities)[0])

        element = np.random.choice(groups[group_idx], size=1)[0]
        return element

    elements_new = np.empty(elements.shape)

    for i in range(len(elements)):
        elements_new[i] = draw_one()

    return elements_new


def make_from_templates(templates, min_amplitude, max_amplitude,
                        n_per_template, probabilities=None):
    """Make spikes with varying amplitudes from templates

    Parameters
    ----------
    templates: numpy.ndarray, (n_templates, waveform_length, n_channels)
        Templates

    min_amplitude: float
        Minimum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    max_amplitude: float
        Maximum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    n_per_template: int
        How many spikes to generate per template

    probabilities: tuple
        Tuple of probabilities for the amplitude range. When the linear
        amplitude range is generated, equal number of spikes are generated
        along the range, by passing probabolities, you can choose how this
        distribution looks like

    Returns
    -------
    numpy.ndarray (n_templates * n_per_template, waveform_length, n_channels)
        Clean spikes
    """
    logger = logging.getLogger(__name__)

    logger.debug('templates shape: %s, min amplitude: %s, '
                 'max_amplitude: %s', templates.shape, min_amplitude,
                 max_amplitude)

    n_templates, waveform_length, n_neighbors = templates.shape

    x = np.zeros((n_per_template * n_templates,
                  waveform_length, n_neighbors))

    d = max_amplitude - min_amplitude
    amps_range = (min_amplitude + np.arange(n_per_template) * d/n_per_template)

    if probabilities is not None:
        amps_range = draw_with_group_probabilities(amps_range, probabilities)

    amps_range = amps_range[:, np.newaxis, np.newaxis]

    # go over every template
    for k in range(n_templates):

        # get current template and scale it
        current = templates[k]
        amp = np.max(np.abs(current))
        scaled = (current/amp)[np.newaxis, :, :]

        # create n clean spikes by scaling the template along the range
        x[k * n_per_template: (k + 1) * n_per_template] = (scaled
                                                           * amps_range)

    return x


def make_collided(x, n_per_spike, multi_channel, min_shift,
                  amp_tolerance=0.2, max_shift='auto', return_metadata=False):
    """Make collided spikes

    Parameters
    ----------
    x
    n_per_spike
    multi_channel
    amp_tolerance: float, optional
        Maximum relative difference in amplitude between the collided spikes,
        defaults to 0.2
    max_shift: int or string, optional
        Maximum amount of shift for the collided spike. If 'auto', it sets
        to half the waveform length in x
    min_shift: float, int
        Minimum shift
    return_metadata: bool, optional
        Return data used to generate the collisions
    """
    # NOTE: i think this is generated collided spikes where one of them
    # is always centered, this may not be the desired behavior sometimes
    # FIXME: maybe it's better to take x_misaligned as parameter, there is
    # redundant shifting logic here

    logger = logging.getLogger(__name__)

    n_clean, wf_length, n_neighbors = x.shape

    if max_shift == 'auto':
        max_shift = int((wf_length - 1) / 2)

    logger.debug('Making collided spikes with n_per_spike: %s max shift: '
                 '%s, multi_channel: %s, amp_tolerance: %s, clean spikes with'
                 ' shape: %s', n_per_spike, max_shift, multi_channel,
                 amp_tolerance, x.shape)

    x_first = np.zeros((n_clean*int(n_per_spike),
                        wf_length,
                        n_neighbors))

    n_collided, _, _ = x_first.shape

    amps = amplitudes(x)

    spikes_first = []
    spikes_second = []
    shifts = []

    for j in range(n_collided):

        # random shifting
        shift = random.choice([-1, 1]) * random.randint(min_shift, max_shift)

        # sample a clean spike - first spike
        x_first[j], i = sample_from_zero_axis(x)

        # get amplitude for sampled x and compute bounds
        amp = amps[i]
        lower = amp * (1.0 - amp_tolerance)
        upper = amp * (1.0 + amp_tolerance)

        # generate 100 possible scaling values and select one of them
        scale_factor = np.random.uniform(lower, upper)

        # draw another clean spike and scale it to be within bounds
        scale_factor = np.linspace(lower, upper, num=50)[random.randint(0, 49)]
        x_second, i = sample_from_zero_axis(x)
        x_second = scale_factor * x_second / amps[i]

        # FIXME: remove this
        x_second = x_second[0, :, :]

        x_second = shift_waveform(x_second, shift)

        # if multi_channel, shuffle neighbors
        if multi_channel:
            shuffled_neighs = np.random.choice(n_neighbors, n_neighbors,
                                               replace=False)
            x_second = x_second[:, shuffled_neighs]

        # if on debug mode, add the spikes and shift to the lists
        if return_metadata:
            spikes_first.append(np.copy(x_first[j]))
            spikes_second.append(x_second)
            shifts.append(shift)

        # add the two spikes
        x_first[j] += x_second

    if return_metadata:
        spikes_first = np.stack(spikes_first)
        spikes_second = np.stack(spikes_second)

        params = dict(n_per_spike=n_per_spike,
                      multi_channel=multi_channel,
                      min_shift=min_shift,
                      max_shift=max_shift,
                      amp_tolerance=amp_tolerance,
                      yass_version=util.get_version())

        metadata = dict(first=spikes_first,
                        second=spikes_second,
                        shift=shifts,
                        params=params)

        return yarr.ArrayWithMetadata(x_first, metadata)
    else:
        return x_first


def shift_waveform(x, shift):

    wf_length, _, = x.shape
    zeros = np.zeros(x.shape)

    if shift > 0:
        zeros[:(wf_length-shift)] += x[shift:]
        return zeros
    elif shift < 0:
        zeros[(-shift):] += x[:(wf_length+shift)]
        return zeros
    else:
        return x


# TODO: remove this function and use separate functions instead
def make_misaligned(x, max_shift, misalign_ratio, misalign_ratio2,
                    multi_channel):
    """Make temporally and spatially misaligned from spikes

    Parameters
    ----------
    multi_channel: bool
        Whether to return multi channel or single channel spikes
    """
    ################################
    # temporally misaligned spikes #
    ################################

    x_temporally = make_temporally_misaligned(x, misalign_ratio,
                                              multi_channel, max_shift)

    ###############################
    # spatially misaligned spikes #
    ###############################

    if multi_channel:
        x_spatially = make_spatially_misaligned(x, misalign_ratio2)

        return x_temporally, x_spatially

    else:
        return x_temporally


def make_spatially_misaligned(x, n_per_spike):
    """Make spatially misaligned spikes (main channel is not the first channel)
    """

    n_spikes, waveform_length, n_neigh = x.shape
    n_out = int(n_spikes * n_per_spike)

    x_spatially = np.zeros((n_out, waveform_length, n_neigh))

    for j in range(n_out):
        x_spatially[j] = np.copy(x[np.random.choice(
            n_spikes, 1, replace=True)][:, :, np.random.choice(
                n_neigh, n_neigh, replace=False)])

    return x_spatially


def make_temporally_misaligned(x, n_per_spike, multi_channel,
                               max_shift='auto'):
    """Make temporally shifted spikes from clean spikes
    """
    n_spikes, waveform_length, n_neigh = x.shape
    n_out = int(n_spikes * n_per_spike)

    if max_shift == 'auto':
        max_shift = int(waveform_length / 2)

    x_temporally = np.zeros((n_out, waveform_length, n_neigh))

    logger.debug('Making spikes with max_shift: %i, output shape: %s',
                 max_shift, x_temporally.shape)

    temporal_shifts = np.random.randint(-max_shift, max_shift, size=n_out)

    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    for j in range(n_out):
        shift = temporal_shifts[j]
        if multi_channel:
            x2 = np.copy(x[np.random.choice(
                x.shape[0], 1, replace=True)][:, :, np.random.choice(
                    n_neigh, n_neigh, replace=False)])
            x2 = np.squeeze(x2)
        else:
            x2 = np.copy(x[np.random.choice(
                x.shape[0], 1, replace=True)])
            x2 = np.squeeze(x2)

        if shift > 0:
            x_temporally[j, :(x_temporally.shape[1]-shift)] += x2[shift:]

        elif shift < 0:
            x_temporally[
                j, (-shift):] += x2[:(x_temporally.shape[1]+shift)]
        else:
            x_temporally[j] += x2

    return x_temporally


def make_noise(shape, spatial_SIG, temporal_SIG):
    """Make noise

    Returns
    ------
    numpy.ndarray
        Noise array with the desired shape
    """
    n_out, waveform_length, n_neigh = shape

    # get noise
    noise = np.random.normal(size=(n_out, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n_out, waveform_length, n_neigh))

    return the_noise


def add_noise(x, spatial_SIG, temporal_SIG):
    """Returns a noisy version of x
    """
    noise = make_noise(x.shape, spatial_SIG, temporal_SIG)
    return x + noise


class ArrayWithMetadata:
    """Wrapper to store metadata in numpy.ndarray, see the metadata attribute
    """

    def __init__(self, array, metadata):
        self.array = array
        self._metadata = metadata

    @property
    def metadata(self):
        return self._metadata

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, key):
        return self.array[key]
