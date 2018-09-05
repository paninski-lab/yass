"""
Custom rules to validate specific files
"""
import pkg_resources
import yaml
from cerberus import Validator
from pkg_resources import resource_filename


def expand_asset_model(mapping, section, subsection, field):
    """Expand filenames
    """
    value = mapping[section][subsection][field]

    # if root_folder, expand and return
    if '$ROOT_FOLDER' in value:
        root = mapping['data']['root_folder']
        new_value = value.replace('$ROOT_FOLDER', root)

    # if absolute path, just return the value
    elif value.startswith('/'):
        new_value = value

    # else, look into assets
    else:
        path = 'assets/models/{}'.format(value)
        new_value = pkg_resources.resource_filename('yass', path)

    mapping[section][subsection][field] = new_value


def validate(mapping):
    """Validate values in the input dictionary using a reference schema
    """
    path_to_validator = resource_filename('yass',
                                          'assets/config/schema.yaml')
    with open(path_to_validator) as file:
        schema = yaml.load(file)

    validator = Validator(schema)
    is_valid = validator.validate(mapping)

    if not is_valid:
        raise ValueError('Errors occurred while validating the '
                         'configuration file: {}'
                         .format(validator.errors))

    document = validator.document

    # expand paths to filenames
    expand_asset_model(document, 'detect', 'neural_network_detector',
                       'filename')
    expand_asset_model(document, 'detect', 'neural_network_triage',
                       'filename')
    expand_asset_model(document, 'detect', 'neural_network_autoencoder',
                       'filename')

    return document
