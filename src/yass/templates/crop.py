import logging

import numpy as np

from yass.geometry import order_channels_by_distance


def main_channels(templates):
    """Find main channel for every template
    """
    abs_templates = np.abs(templates)

    # get the maximum along the waveform to get the highest point in every
    # channel
    max_along_time = np.amax(abs_templates, axis=1)

    # return the index maximum  value along the channels - this is the main
    # channel
    return np.argmax(max_along_time, axis=1)


def amplitudes(templates):
    """Find amplitudes
    """
    return np.amax(np.abs(templates), axis=(1, 2))


def on_main_channel(templates):
    """Get templates on its main and neighboring channels
    """
    pass


def align(templates, crop=True):
    """Align templates spatially
    """
    pass


def crop_and_align_templates(big_templates, R, neighbors, geom):
    """Crop (spatially) and align (temporally) templates

    Parameters
    ----------

    Returns
    -------
    """
    logger = logging.getLogger(__name__)

    # copy templates to avoid modifying the original ones
    big_templates = np.copy(big_templates)

    n_templates, _, _ = big_templates.shape

    # main channels ad amplitudes for each template
    main_ch = main_channels(big_templates)
    amps = amplitudes(big_templates)

    # get a template on a main channel and align them
    K_big = np.argmax(amps)
    templates_mainc = np.zeros((n_templates, big_templates.shape[1]))
    t_rec = big_templates[K_big, :, main_ch[K_big]]
    t_rec = t_rec/np.sqrt(np.sum(np.square(t_rec)))

    for k in range(n_templates):
        t1 = big_templates[k, :, main_ch[k]]
        t1 = t1/np.sqrt(np.sum(np.square(t1)))
        shift = align_templates(t1, t_rec)

        logger.debug('Template %i will be shifted by %i', k, shift)

        if shift > 0:
            templates_mainc[k, :(big_templates.shape[1]-shift)] = t1[shift:]
            big_templates[k, :(big_templates.shape[1]-shift)
                          ] = big_templates[k, shift:]

        elif shift < 0:
            templates_mainc[k, (-shift):] = t1[:(big_templates.shape[1]+shift)]
            big_templates[k,
                          (-shift):] = big_templates[k,
                                                     :(big_templates.shape[1]
                                                       + shift)]

        else:
            templates_mainc[k] = t1

    # determin temporal center of templates and crop around it
    R2 = int(R/2)
    center = np.argmax(np.convolve(
        np.sum(np.square(templates_mainc), 0), np.ones(2*R2+1), 'valid')) + R2
    big_templates = big_templates[:, (center-3*R):(center+3*R+1)]

    # spatially crop
    nneigh = np.max(np.sum(neighbors, 0))

    small_templates = np.zeros((n_templates, big_templates.shape[1], nneigh))

    for k in range(n_templates):
        ch_idx = np.where(neighbors[main_ch[k]])[0]
        ch_idx, temp = order_channels_by_distance(main_ch[k], ch_idx, geom)
        small_templates[k, :, :ch_idx.shape[0]] = big_templates[k][:, ch_idx]

    return small_templates


def align_templates(t1, t2):
    """Align templates

    Parameters
    ----------

    Returns
    -------
    """
    temp = np.convolve(t1, np.flip(t2, 0), 'full')
    shift = np.argmax(temp)
    return shift - t1.shape[0] + 1
