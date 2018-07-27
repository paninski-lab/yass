import logging
import numpy as np
from yass.geometry import order_channels_by_distance
from yass.templates.util import main_channels, amplitudes

# FIXME: keeping this just because make_training data is still using it
# use template processor instead, remove as soon as make_training_data
# is refactored


def crop_and_align_templates(big_templates, R, neighbors, geom,
                             crop_spatially=True):
    """Crop (spatially) and align (temporally) templates

    Parameters
    ----------

    Returns
    -------
    """
    logger = logging.getLogger(__name__)

    logger.debug('crop and align input shape %s', big_templates.shape)

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

    # crop templates, now they are from 4*R to 3*R
    logger.debug('6*R+1 %s', 6*R+1)
    big_templates = big_templates[:, (center-3*R):(center+3*R+1)]

    logger.debug('crop and align output shape %s', big_templates.shape)

    if not crop_spatially:

        return big_templates

    else:
        # spatially crop (only keep neighbors)
        n_neigh_to_keep = np.max(np.sum(neighbors, 0))
        small = np.zeros((n_templates, big_templates.shape[1],
                          n_neigh_to_keep))

        for k in range(n_templates):

            # get neighbors for the main channel in the kth template
            ch_idx = np.where(neighbors[main_ch[k]])[0]

            # order channels
            ch_idx, _ = order_channels_by_distance(main_ch[k], ch_idx, geom)

            # new kth template is the old kth template by keeping only
            # ordered neighboring channels
            small[k, :, :ch_idx.shape[0]] = big_templates[k][:, ch_idx]

        return small


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
