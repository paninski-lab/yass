from scipy.interpolate import interp1d
import numpy as np


def clean_up(spike_train, templates, max_spikes):
    ''' Function removes units with with spikes < max_spikes
    '''

    n_templates, n_temporal_big, n_channels = templates.shape

    units_keep = np.zeros(n_templates, 'bool')
    for k in range(n_templates):
        if np.sum(spike_train[:, 1] == k) >= max_spikes:
            units_keep[k] = 1

    Ks = np.where(units_keep)[0]
    templates_clean = np.zeros((Ks.shape[0], n_temporal_big, n_channels))
    spike_train_clean = np.zeros((0, 2), 'int32')
    for j, k in enumerate(Ks):
        templates_clean[j] = templates[k]
        spike_train_temp = np.copy(
            spike_train[spike_train[:, 1] == k])
        spike_train_temp[:, 1] = j
        spike_train_clean = np.vstack((spike_train_clean, spike_train_temp))

    idx_sort = np.argsort(spike_train_clean[:, 0])

    return spike_train_clean[idx_sort], templates_clean


def calculate_temp_temp(temporal_features, spatial_features):
    ''' Function computes all pair-wise template convolutions and saves
    '''

    import datetime
    n_templates, n_shifts, waveform_size, n_features = temporal_features.shape

    temp_temp = np.zeros(
        (n_templates, n_templates, n_shifts, n_shifts, 2 * waveform_size - 1),
        'float32')

    for j1 in range(n_features):
        for j2 in range(n_features):
            for s1 in range(n_shifts):
                for s2 in range(n_shifts):
                    print('temp_temp: ', j1, '/', n_features, '  ', j2, '/',
                          n_features, '   ', s1, '/', n_shifts,
                          '   ', s2, '/', n_shifts, '   ',
                          datetime.datetime.now().strftime('%H:%M:%S'))
                    temp1 = temporal_features[:, s1, :, j1]
                    temp2 = np.flip(temporal_features[:, s2, :, j2], 1)
                    spat1 = spatial_features[:, s1, j1]
                    spat2 = spatial_features[:, s2, j2]
                    temporal_conv = np.zeros((2 * waveform_size - 1,
                                              n_templates, n_templates))
                    for k in range(n_templates):
                        for k2 in range(n_templates):
                            temporal_conv[:, k, k2] = np.convolve(
                                temp2[k2], temp1[k])
                    temp_temp[:, :, s1, s2] += np.transpose(
                        temporal_conv * np.matmul(spat1, spat2.T)[np.newaxis],
                        (1, 2, 0))

    return temp_temp


def calculate_temp_temp_parallel(templates_selected, temporal_features,
                                 spatial_features):
    ''' Function computes all pair-wise template convolutions and saves
    '''

    n_templates, n_shifts, waveform_size, n_features = temporal_features.shape

    temp_temp = np.zeros((n_templates, len(templates_selected), n_shifts,
                          n_shifts, 2 * waveform_size - 1), 'float32')
    temporal_conv = np.zeros(
        (2 * waveform_size - 1, n_templates, len(templates_selected)),
        dtype='float32')
    for j1 in range(n_features):
        for j2 in range(n_features):
            for s1 in range(n_shifts):
                for s2 in range(n_shifts):
                    temp1 = temporal_features[:, s1, :, j1]
                    temp2 = np.flip(
                        temporal_features[templates_selected][:, s2, :, j2], 1)
                    spat1 = spatial_features[:, s1, j1]
                    spat2 = spatial_features[templates_selected][:, s2, j2]

                    temporal_conv *= 0
                    for k in range(n_templates):
                        for k2 in range(len(templates_selected)):
                            temporal_conv[:, k, k2] = np.convolve(
                                temp2[k2], temp1[k]).T
                    temp_temp[:, :, s1, s2] += np.transpose(
                        temporal_conv * np.matmul(spat1, spat2.T)[np.newaxis],
                        (1, 2, 0))

    return temp_temp


def upsample_templates(templates, upsample_factor):
    ''' Function upsamples temporal resolution (usually 3 x) to better
        compute alignment
    '''

    # get shapes
    n_channels, waveform_size, n_templates = templates.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size - 1, num=waveform_size, endpoint=True)
    shifts = np.linspace(-0.5, 0.5, upsample_factor, endpoint=False)
    xnew = np.sort(np.reshape(x[:, np.newaxis] + shifts, -1))

    upsampled_templates = np.zeros(
        (n_channels, waveform_size * upsample_factor, n_templates))
    for j in range(n_templates):
        ff = interp1d(x, templates[:, :, j], kind='cubic')
        idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size - 1)
        upsampled_templates[:, idx_good, j] = ff(xnew[idx_good])

    return upsampled_templates


def make_spike_index_per_template(spike_index, templates, n_explore):
    '''
    '''

    n_channels, n_temporal_big, n_templates = templates.shape

    principal_channels = np.argmax(np.max(np.abs(templates), 1), 0)

    spt_list = make_spt_list(spike_index, n_channels)

    for c in range(n_channels):
        spt_list[c] = get_longer_spt_list(spt_list[c], n_explore)

    spike_index_template = np.zeros((0, 2), 'int32')
    template_id = np.zeros(0, 'int32')
    for k in range(n_templates):

        mainc = principal_channels[k]
        spt = spt_list[mainc]

        spike_index_template = np.concatenate(
            (spike_index_template,
             np.concatenate(
                 (spt[:, np.newaxis], np.ones(
                     (spt.shape[0], 1), 'int32') * mainc), 1)), 0)
        template_id = np.hstack((template_id, np.ones(
            (spt.shape[0]), 'int32') * k))

    idx_sort = np.argsort(spike_index_template[:, 0])

    return spike_index_template[idx_sort], template_id[idx_sort]


def get_smaller_shifted_templates(shifted_templates, channel_index,
                                  principal_channels, spike_size):
    '''
    '''

    n_shifts, n_channels, waveform_size, n_templates = shifted_templates.shape
    n_neigh = channel_index.shape[1]

    shifted_templates = np.transpose(shifted_templates, (3, 2, 1, 0))
    shifted_templates = np.concatenate(
        (shifted_templates, np.zeros(
            (n_templates, waveform_size, 1, n_shifts))), 2)

    mid_t = int((waveform_size - 1) / 2)
    templates_small = np.zeros((n_templates, 2 * spike_size + 1, n_neigh,
                                n_shifts))
    for k in range(n_templates):
        mainc = principal_channels[k]
        temp = shifted_templates[k, mid_t - spike_size:mid_t + spike_size + 1]
        templates_small[k] = temp[:, channel_index[mainc]]

    return templates_small


def small_shift_templates(templates, n_shifts=5):
    """
    get n_shifts number of shifted templates.
    The amount of shift is evenly distributed from
    -0.5 to 0.5 timebin including -0.5 and 0.5.

    Parameters
    ----------

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
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
    n_templates, waveform_size, n_channels = templates.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size - 1, num=waveform_size, endpoint=True)
    shifts = np.linspace(-0.5, 0.5, n_shifts, endpoint=False)

    shifted_templates = np.zeros((n_templates, n_shifts, waveform_size,
                                  n_channels))
    for k in range(n_templates):
        ff = interp1d(x, templates[k], kind='cubic', axis=0)

        # get shifted templates
        for j in range(n_shifts):
            xnew = x - shifts[j]
            idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size - 1)
            shifted_templates[k, j][idx_good] = ff(xnew[idx_good])

    return shifted_templates


def svd_shifted_templates(shifted_templates, n_features):
    ''' Compute SVD matrices for shifted_templates (usually upsampled x3)
        and save right/left singular matrices to n_features resolution
        These are used in deconvolution
    '''

    # n_templates, n_shifts, waveform_size, n_channels =
    # shifted_templates.shape
    # R = int((waveform_size-1)/4)
    # a,b,c = np.linalg.svd(shifted_templates[:, :, R:3*R+1],[0,3,1,2])

    a, b, c = np.linalg.svd(shifted_templates, [0, 3, 1, 2])
    temporal_features = a[:, :, :, :n_features]
    spatial_features = c[:, :, :n_features, :] * b[:, :, :
                                                   n_features][:, :, :,
                                                               np.newaxis]

    return temporal_features, spatial_features


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

    spike_index_list = [None] * n_channels

    for c in range(n_channels):
        spike_index_list[c] = spike_index[spike_index[:, 1] == c, 0]

    return spike_index_list


def make_spt_list_parallel(spike_index, n_channels):
    ''' Convert 2 column spike_index to a channel based index
    '''

    spike_index_list = [None] * n_channels

    for c in range(n_channels):
        #if c % 50 == 0:
        #    print("making spt list channel: ", c)
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
    all_spikes = np.reshape(
        np.add(spt[:, np.newaxis],
               np.arange(-n_explore, n_explore + 1)[np.newaxis, :]), -1)

    # if there are any duplicate remove it
    spt_long = np.sort(np.unique(all_spikes))

    return spt_long
