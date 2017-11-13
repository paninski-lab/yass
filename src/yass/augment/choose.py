import numpy as np

def choose_templates(templates, template_choice):
    """

    Returns
    -------
    
    """
    chosen_templates = np.arange(49)
    templates = templates[chosen_templates]

    templates_small = np.max(templates,axis=(1,2)) > 4
    templates = templates[templates_small]

    return templates