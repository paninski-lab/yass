import numpy as np

from ..geometry import order_channels_by_distance


# TODO: documentation
# TODO: comment code, it's not clear what it does
def crop_templates(templatesBig, R, neighbors, geom):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    # number of templates
    K = templatesBig.shape[0]

    # main channel for each template and amplitudes
    mainC = np.argmax(np.amax(np.abs(templatesBig), axis=1), axis=1)
    amps = np.amax(np.abs(templatesBig), axis=(1, 2))

    # get a template on a main channel and align them
    K_big = np.argmax(amps)
    templates_mainc = np.zeros((K, templatesBig.shape[1]))
    t_rec = templatesBig[K_big, :, mainC[K_big]]
    t_rec = t_rec/np.sqrt(np.sum(np.square(t_rec)))
    for k in range(K):
        t1 = templatesBig[k, :, mainC[k]]
        t1 = t1/np.sqrt(np.sum(np.square(t1)))
        shift = align_templates(t1, t_rec)
        if shift > 0:
            templates_mainc[k, :(templatesBig.shape[1]-shift)] = t1[shift:]
            templatesBig[k, :(templatesBig.shape[1]-shift)
                         ] = templatesBig[k, shift:]

        elif shift < 0:
            templates_mainc[k, (-shift):] = t1[:(templatesBig.shape[1]+shift)]
            templatesBig[k, (-shift):] = templatesBig[k,
                                                      :(templatesBig.shape[1]
                                                        + shift)]

        else:
            templates_mainc[k] = t1

    # determin temporal center of templates and crop around it
    center = np.argmax(np.convolve(
        np.sum(np.square(templates_mainc), 0), np.ones(2*R+1), 'valid')) + R
    templatesBig = templatesBig[:, (center-3*R):(center+3*R+1)]

    # spatially crop
    nneigh = np.max(np.sum(neighbors, 0))
    templatesBig2 = np.zeros(
        (templatesBig.shape[0], templatesBig.shape[1], nneigh))
    for k in range(K):
        ch_idx = np.where(neighbors[mainC[k]])[0]
        ch_idx, temp = order_channels_by_distance(mainC[k], ch_idx, geom)
        templatesBig2[k, :, :ch_idx.shape[0]] = templatesBig[k][:, ch_idx]

    return templatesBig2


# TODO: documentation
# TODO: comment code, it's not clear what it does
def align_templates(t1, t2):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    temp = np.convolve(t1, np.flip(t2, 0), 'full')
    shift = np.argmax(temp)
    return shift - t1.shape[0] + 1
