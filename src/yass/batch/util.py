from pathlib import Path
import yaml
import numbers


def make_chunk_path(output_path, i):
    name, ext = output_path.parts[-1].split('.')
    filename = name+str(i)+'.'+ext
    chunk_path = Path(*output_path.parts[:-1], filename)

    return chunk_path


def make_metadata(channels, n_channels, dtype, output_path):
    """Make and save metadata for a binary file

    Parameters
    ----------
    chahnels: str or int
        The value of the channels parameter
    n_channels: int
        Number of channels in the whole dataset (not necessarily match the
        number of channels in the subset)
    dtype: str
        dtype
    output_path: str
        Where to save the file
    """
    if channels == 'all':
        _n_channels = n_channels
    elif isinstance(channels, numbers.Integral):
        _n_channels = 1
    else:
        _n_channels = len(channels)

    # save yaml file with params
    path_to_yaml = output_path.replace('.bin', '.yaml')

    params = dict(dtype=dtype, n_channels=_n_channels, data_order='samples')

    with open(path_to_yaml, 'w') as f:
        yaml.dump(params, f)

    return params
