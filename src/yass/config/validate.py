"""
Custom rules to validate specific files
"""
import yaml
from cerberus import Validator
from pkg_resources import resource_filename


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

    return validator.document
