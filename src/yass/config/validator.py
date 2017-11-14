from . import custom_rules

from functools import reduce


class Validator(object):
    """
    Validate a mapping object, it checks that all required sections are
    present, that no invalid sections exist, fills optional sections with
    provided defaults and validates types and values for fields

    Parameters
    ----------
    d: dict
        The mapping to validate

    required_sections: set, optional
        The set of top-level valid sections

    optional_sections: dict, optional
        The set of top-level optional sections, each key represents the name
        of an optional section and the content the default value if the section
        is not present in d

    fields_validator: dict, optional
        A dict used to validate values inside sections, each key must contain
        a valid section name, within each section, there must be another
        dictionary whose keys represent valid keys inside the section. Each
        of these dictionaries can have any of the optional keys: values
        (list of permitted valyes) and type (Python data type)

    allow_extras: bool, optional
        Ignore extra sections, defaults to True
    """
    def __init__(self, d, required_sections=None, optional_sections=None,
                 fields_validator=None, allow_extras=True):
        self.d = d
        self.required_sections = (set(required_sections) if required_sections
                                  else set())
        self.optional_sections = (set(optional_sections.keys())
                                  if optional_sections else set())
        self.optional_sections_defaults = optional_sections
        self.fields_validator = fields_validator
        self.sections = set(self.d.keys())
        self.allow_extras = allow_extras

    def _validate_required_sections(self):
        missing = self.required_sections - self.sections

        if missing:
            raise ValueError('The following sections are required: {}'
                             .format(_pretty_iter(missing)))

    def _validate_extra_sections(self):
        extra = self.sections - self.required_sections
        invalid = extra - self.optional_sections

        if invalid:
            raise ValueError('The following sections are invalid: {}'
                             .format(_pretty_iter(invalid)))

    def _fill_default_values(self):
        total = self.required_sections | self.optional_sections
        missing_optional = total - self.sections

        for missing in missing_optional:
            self.d[missing] = self.optional_sections_defaults[missing]

    def _validate_fields(self):
        for section, section_validator in self.fields_validator.items():
            for subsection, subsection_validator in section_validator.items():
                required_type = subsection_validator.get('type')
                function = subsection_validator.get('function')
                permitted_values = subsection_validator.get('values')
                value = self.d[section][subsection]
                actual_type = type(value).__name__

                if required_type and actual_type != required_type:
                    raise ValueError('Value in "{}.{}" must be "{}" but it '
                                     'is "{}"'
                                     .format(section, subsection,
                                             required_type, actual_type))

                if permitted_values and value not in permitted_values:
                    raise ValueError('Value in "{}.{}" is invalid, valid '
                                     'values are "{}"'
                                     .format(section, subsection,
                                             _pretty_iter(permitted_values)))

                if function:
                    fn = getattr(custom_rules, function)
                    self.d[section][subsection] = fn(self.d, section,
                                                     subsection)

    def validate(self):
        if self.required_sections:
            self._validate_required_sections()

        if not self.allow_extras:
            self._validate_extra_sections()

        if self.optional_sections:
            self._fill_default_values()

        if self.fields_validator:
            self._validate_fields()

        return self.d


def _pretty_iter(iterator):
    return reduce(lambda x, y: x+', '+y, sorted(iterator))
