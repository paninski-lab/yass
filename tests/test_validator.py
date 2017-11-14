import pkg_resources
import pytest

from yass.config.validator import Validator


def test_rejects_dict_with_missing_required_sections():
    d = dict(a=1, b=2)
    required_sections = ['a', 'b', 'c']
    optional_sections = dict(d=4)
    validator = Validator(d, required_sections, optional_sections)

    message = 'The following sections are required c'

    with pytest.raises(ValueError, message=message):
        validator.validate()


def test_rejects_dict_with_invalid_sections():
    d = dict(a=1, b=2, c=3, e=5, f=6)
    required_sections = ['a', 'b', 'c']
    optional_sections = dict(d=4)
    validator = Validator(d, required_sections, optional_sections,
                          allow_extras=False)
    message = 'The following sections are invalid: e, f'

    with pytest.raises(ValueError) as exception:
        validator.validate()

    assert str(exception.value) == message


def test_fills_optional_missing_sections():
    d = dict(a=1, b=2, c=3)
    required_sections = ['a', 'b', 'c']
    optional_sections = dict(d=4)
    validator = Validator(d, required_sections, optional_sections)

    validated = validator.validate()

    assert validated['d'] == 4


def test_validates_fields_type():
    d = dict(a=dict(a_num='not a num', a_cat='val', a_str='a str'), b=2, c=3)
    required_sections = ['a', 'b', 'c']
    optional_sections = dict(d=4)
    fields_validator = dict(a=dict(a_num=dict(type='int'),
                                   a_cat=dict(type='str', values='val'),
                                   a_str=dict(tyle='str')))

    validator = Validator(d, required_sections, optional_sections,
                          fields_validator)

    message = 'Value in "a.a_num" must be "int" but it is "str"'

    with pytest.raises(ValueError) as exception:
        validator.validate()

    assert str(exception.value) == message


def test_validates_fields_values():
    d = dict(a=dict(a_num=1, a_cat='not val', a_str='a str'), b=2, c=3)
    required_sections = ['a', 'b', 'c']
    optional_sections = dict(d=4)
    fields_validator = dict(a=dict(a_num=dict(type='int'),
                                   a_cat=dict(type='str', values=['val',
                                                                  'val2']),
                                   a_str=dict(tyle='str')))

    validator = Validator(d, required_sections, optional_sections,
                          fields_validator)

    message = 'Value in "a.a_cat" is invalid, valid values are "val, val2"'

    with pytest.raises(ValueError) as exception:
        validator.validate()

    assert str(exception.value) == message


def test_can_validate_with_custom_rules():
    d = dict(a=dict(a_path='file_in_assets_model.txt'))
    required_sections = ['a']
    optional_sections = dict(b=1)
    fields_validator = dict(a=dict(a_path=dict(function='expand_asset_model')))

    validator = Validator(d, required_sections, optional_sections,
                          fields_validator)
    validated = validator.validate()

    path = 'assets/models/file_in_assets_model.txt'
    path_absolute = pkg_resources.resource_filename('yass', path)

    assert path_absolute == validated['a']['a_path']
