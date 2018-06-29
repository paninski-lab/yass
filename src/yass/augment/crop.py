import numpy as np

from yass.geometry import order_channels_by_distance


def crop_and_align_templates(big_templates, R, neighbors, geom):
    """Crop (spatially) and align (temporally) templates

    Parameters
    ----------

    Returns
    -------
    """
    K, wf_length, _ = big_templates.shape

    # main channel for each template and amplitudes
    mainC = np.argmax(np.amax(np.abs(big_templates), axis=1), axis=1)
    amps = np.amax(np.abs(big_templates), axis=(1, 2))

    # get a template on a main channel and align them
    K_big = np.argmax(amps)
    templates_mainc = np.zeros((K, wf_length))
    t_rec = big_templates[K_big, :, mainC[K_big]]
    t_rec = t_rec/np.sqrt(np.sum(np.square(t_rec)))

    for k in range(K):
        t1 = big_templates[k, :, mainC[k]]
        t1 = t1/np.sqrt(np.sum(np.square(t1)))
        shift = align_templates(t1, t_rec)

        if shift > 0:
            templates_mainc[k, :(wf_length-shift)] = t1[shift:]
            big_templates[k, :(wf_length-shift)
                          ] = big_templates[k, shift:]

        elif shift < 0:
            templates_mainc[k, (-shift):] = t1[:(wf_length+shift)]
            big_templates[k, (-shift):] = big_templates[k,
                                                        :(wf_length
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

    small_templates = np.zeros((K, wf_length, nneigh))

    for k in range(K):
        ch_idx = np.where(neighbors[mainC[k]])[0]
        ch_idx, temp = order_channels_by_distance(mainC[k], ch_idx, geom)
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
