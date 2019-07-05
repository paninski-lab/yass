"""
Custom rules to validate specific files
"""
import pkg_resources
import yaml
import os
from cerberus import Validator
from pkg_resources import resource_filename

def expand_asset_model(mapping, section, subsection, field):
    """Expand filenames
    """
    value = mapping[section][subsection][field]

    # if absolute path, just return the value
    if value.startswith('/'):
        new_value = value

    # else, look into assets
    else:
        path = 'assets/nn_models/{}'.format(value)
        new_value = pkg_resources.resource_filename('yass', path)

    mapping[section][subsection][field] = new_value

def expand_to_root(mapping, section, subsection, field):
    """Expand filenames
    """
    value = mapping[section][subsection][field]

    # if root_folder, expand and return
    if value.startswith('/'):
        new_value = value

    else:
        root = mapping['data']['root_folder']
        new_value = os.path.join(root, value)

    mapping[section][subsection][field] = new_value


def validate(mapping, silent=True):
    """Validate values in the input dictionary using a reference schema
    """
    path_to_validator = resource_filename('yass',
                                          'assets/config/schema.yaml')
    with open(path_to_validator) as file:
        schema = yaml.load(file)

    validator = Validator(schema)
    is_valid = validator.validate(mapping)

    if not is_valid and not silent:
        raise ValueError('Errors occurred while validating the '
                         'configuration file: {}'
                         .format(validator.errors))

    document = validator.document

    # expand paths to filenames
    expand_asset_model(document, 'neuralnetwork', 'detect',
                       'filename')
    expand_asset_model(document, 'neuralnetwork', 'denoise',
                       'filename')

    if document['neuralnetwork']['training']['input_spike_train_filname'] is not None:
        expand_to_root(document, 'neuralnetwork', 'training',
                       'input_spike_train_filname')

    return document
