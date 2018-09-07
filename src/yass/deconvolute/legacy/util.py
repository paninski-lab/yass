from scipy.interpolate import interp1d
import numpy as np


def upsample_templates(template, n_shifts=5):
    """
    get n_shifts number of shifted templates.
    The amount of shift is evenly distributed from
    -0.5 to 0.5 timebin including -0.5 and 0.5.

    Parameters
    ----------

    templates: numpy.ndarray (n_channels, waveform_size)
       A 2D array of a template

    n_shifts: int
       number of shifted templates to make

    Returns
    -------
    shifted_templates: numpy.ndarray (n_shifts, n_channels,
                                      waveform_size)
        A 3D array with shifted templates
    """

    # get shapes
    n_channels, waveform_size = template.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size-1, num=waveform_size, endpoint=True)
    ff = interp1d(x, template, kind='cubic')

    # get shifted templates
    shifts = np.linspace(-0.5, 0.5, n_shifts, endpoint=False)
    shifted_templates = np.zeros((n_shifts, n_channels, waveform_size))
    for j in range(n_shifts):
        xnew = x - shifts[j]
        idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size-1)
        shifted_templates[j][:, idx_good] = ff(xnew[idx_good])

    return shifted_templates


def make_spt_list(spike_index, n_channels):
    """
    Change a data structure of spike_index from an array of two
    columns to a list

    Parameters
    ----------

    spike_index: numpy.ndarray (n_spikes, 2)
       A 2D array containing spikes information with two columns,
       where the first column is spike time and the second is channel.

    n_channels: int
       the number of channels in recording (or length of output list)

    Returns
    -------
    spike_index_list: list (n_channels)
        A list such that spike_index_list[c] cointains all spike times
        whose channel is c
    """

    spike_index_list = [None]*n_channels

    for c in range(n_channels):
        spike_index_list[c] = spike_index[spike_index[:, 1] == c, 0]

    return spike_index_list


def get_longer_spt_list(spt, n_explore):
    """
    Given a spike time, -n_explore to n_explore time points
    around the spike time is also included as spike times

    Parameters
    ----------

    spt: numpy.ndarray (n_spikes)
       A list of spike times

    n_explore: int
       2*n_explore additional points will be included into spt

    Returns
    -------
    spt_long: numpy.ndarray
        A new list containing additions spike times
    """

    # sort spike time
    spt = np.sort(spt)

    # add -n_explore to n_explore points around each spike time
    all_spikes = np.reshape(np.add(spt[:, np.newaxis],
                                   np.arange(-n_explore, n_explore+1)
                                   [np.newaxis, :]), -1)

    # if there are any duplicate remove it
    spt_long = np.sort(np.unique(all_spikes))

    return spt_long
