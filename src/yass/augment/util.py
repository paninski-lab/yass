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
                        n_per_template, probabilities=None,
                        return_metadata=False):
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

    amps_range = (min_amplitude + np.arange(n_per_template)
                  * d / (n_per_template - 1))

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

    if return_metadata:
        ids = [[k]*n_per_template for k in range(n_templates)]
        ids = np.array([item for sublist in ids for item in sublist])
        metadata = dict(ids=ids)
        return yarr.ArrayWithMetadata(x, metadata)
    else:
        return x


def make_collided(x1, x2, n_per_spike, min_shift='auto',
                  amp_tolerance=0.2, max_shift='auto', return_metadata=False):
    """Make collided spikes

    Parameters
    ----------
    x1
    x2
    n_per_spike
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
    logger = logging.getLogger(__name__)

    n_clean, wf_length, n_neighbors = x1.shape

    if max_shift == 'auto':
        max_shift = int(0.5 * wf_length)

    if min_shift == 'auto':
        min_shift = int(0.1 * wf_length)

    logger.debug('Making collided spikes with n_per_spike: %s max shift: '
                 '%s, amp_tolerance: %s, clean spikes with'
                 ' shape: %s', n_per_spike, max_shift,
                 amp_tolerance, x1.shape)

    x_first = np.zeros((n_clean*int(n_per_spike),
                        wf_length,
                        n_neighbors))

    n_collided, _, _ = x_first.shape

    amps1 = amplitudes(x1)
    amps2 = amplitudes(x2)

    spikes_first = []
    spikes_second = []
    shifts = []

    for j in range(n_collided):

        # random shifting
        shift = random.choice([-1, 1]) * random.randint(min_shift, max_shift)

        # sample a clean spike - first spike
        x_first[j], i = sample_from_zero_axis(x1)

        # get amplitude for sampled x and compute bounds
        amp = amps1[i]
        lower = amp * (1.0 - amp_tolerance)
        upper = amp * (1.0 + amp_tolerance)

        # generate 100 possible scaling values and select one of them
        scale_factor = np.random.uniform(lower, upper)

        # draw another clean spike and scale it to be within bounds
        scale_factor = np.linspace(lower, upper, num=50)[random.randint(0, 49)]
        x_second, i = sample_from_zero_axis(x2)
        x_second = scale_factor * x_second / amps2[i]

        x_second = shift_waveform(x_second, shift)

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


def make_spatially_misaligned(x, n_per_spike,
                              force_first_channel_shuffle=True):
    """Make spatially misaligned spikes (shuffles channels)
    """

    n_spikes, waveform_length, n_neigh = x.shape
    n_out = int(n_spikes * n_per_spike)

    x_spatially = np.zeros((n_out, waveform_length, n_neigh))

    def shuffle():
        if force_first_channel_shuffle:
            success = False

            while not success:
                shuffled = np.random.choice(n_neigh, n_neigh, replace=False)
                success = shuffled[0] != 0
        else:
            shuffled = np.random.choice(n_neigh, n_neigh, replace=False)

        return shuffled

    for j in range(n_out):
        shuffled = shuffle()

        to_shuffle = x[np.random.choice(n_spikes, 1, replace=True)]
        x_spatially[j] = to_shuffle[:, :, shuffled]

    return x_spatially


def make_temporally_misaligned(x, n_per_spike=1, min_shift='auto',
                               max_shift='auto'):
    """Make temporally shifted spikes from clean spikes
    """
    n_spikes, waveform_length, n_neigh = x.shape
    n_out = int(n_spikes * n_per_spike)

    if max_shift == 'auto':
        max_shift = int(0.5 * waveform_length)

    if min_shift == 'auto':
        min_shift = int(0.1 * waveform_length)

    x_temporally = np.zeros((n_out, waveform_length, n_neigh))

    logger.debug('Making spikes with max_shift: %i, output shape: %s',
                 max_shift, x_temporally.shape)

    for j in range(n_out):

        shift = random.choice([-1, 1]) * random.randint(min_shift, max_shift)

        idx = np.random.choice(x.shape[0], 1, replace=True)[0]
        spike = x[idx]

        if shift > 0:
            x_temporally[j, :(x_temporally.shape[1]-shift)] += spike[shift:]

        elif shift < 0:
            x_temporally[
                j, (-shift):] += spike[:(x_temporally.shape[1]+shift)]
        else:
            x_temporally[j] += spike

    return x_temporally


def make_noise(n, spatial_SIG, temporal_SIG):
    """Make noise

    Parameters
    ----------
    n: int
        Number of noise events to generate

    Returns
    ------
    numpy.ndarray
        Noise
    """
    n_neigh, _ = spatial_SIG.shape
    waveform_length, _ = temporal_SIG.shape

    # get noise
    noise = np.random.normal(size=(n, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n, waveform_length, n_neigh))

    return the_noise


def add_noise(x, spatial_SIG, temporal_SIG):
    """Returns a noisy version of x
    """
    noise = make_noise(x.shape[0], spatial_SIG, temporal_SIG)
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
