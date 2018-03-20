"""
Custom rules to validate specific files
"""
import pkg_resources


def expand_asset_model(d, section, subsection):
    value = d[section][subsection]

    # if root_folder, expand and return
    if '$ROOT_FOLDER' in value:
        root = d['data']['root_folder']
        new_value = value.replace('$ROOT_FOLDER', root)

    # if absolute path, just return the value
    elif value.startswith('/'):
        new_value = value

    # else, look into assets
    else:
        path = 'assets/models/{}'.format(value)
        new_value = pkg_resources.resource_filename('yass', path)

    d[section][subsection] = new_value
