import numpy as np


# TODO: documentation
# TODO: comment code, it's not clear what it does
def choose_templates(templates, chosen_templates):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    try:
        templates = templates[chosen_templates]
    except IndexError:
        raise IndexError('Error getting chosen_templates, make sure the ids'
                         'exist')

    templates_small = np.max(templates, axis=(1, 2)) > 4
    templates = templates[templates_small]

    return templates
