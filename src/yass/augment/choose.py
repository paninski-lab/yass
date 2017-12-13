import numpy as np


# TODO: documentation
# TODO: comment code, it's not clear what it does
def choose_templates(templates, template_choice):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    chosen_templates = np.arange(49)
    templates = templates[chosen_templates]

    templates_small = np.max(templates, axis=(1, 2)) > 4
    templates = templates[templates_small]

    return templates
