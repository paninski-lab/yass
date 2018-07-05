import numpy as np


def choose_templates(templates, chosen_templates_indexes,
                     minimum_amplitude=4):
    """
    Keep only selected templates and from those, only the ones above certain
    value

    Returns
    -------
    """
    try:
        chosen_templates = templates[chosen_templates_indexes]
    except IndexError:
        raise IndexError('Error getting chosen_templates, make sure the ids'
                         'exist')

    amplitudes = np.max(np.abs(chosen_templates), axis=(1, 2))
    big_templates_indexes = amplitudes > minimum_amplitude
    big_templates = chosen_templates[big_templates_indexes]

    return big_templates
