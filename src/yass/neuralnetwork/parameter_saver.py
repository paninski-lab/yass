import yaml
from yaml import Dumper


class map_data(dict):
    pass


class CustomYAMLDumper(Dumper):
    pass


def map_rep(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data,
                                    flow_style=False)


class seq_data(list):
    pass


def seq_rep(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data,
                                     flow_style=True)


# CustomYAMLDumper.add_representer(map_data, map_data)
# CustomYAMLDumper.add_representer(seq_data, map_rep)


def save_detect_network_params(filters_size, waveform_length, n_neighbors,
                               output_path):
    """Generate yaml file with parameters for a detect network

    Parameters
    ----------
    filters_size: list
        List with number of filters in each layer

    waveform_length: int
        Temporal filter size

    n_neighbors: int
        Number of neighboring channels

    output_path: str
        Where to save the file
    """
    d = dict(filters_size=filters_size, waveform_length=waveform_length,
             n_neighbors=n_neighbors)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)


def save_triage_network_params(filters_size, waveform_length, n_neighbors,
                               output_path):
    """Generate yaml file with parameters for a triage network

    Parameters
    ----------
    filters_size: list
        List filters size

    waveform_length: int
        Temporal filter size

    n_neighbors: int
        Number of neighboring channels

    output_path: str
        Where to save the file
    """
    d = dict(filters_size=filters_size, waveform_length=waveform_length,
             n_neighbors=n_neighbors)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)


def save_ae_network_params(waveform_length, n_features, output_path):
    """Generate yaml file with parameters for a ae network

    Parameters
    ----------
    waveform_length: int
        Dimension of input

    n_features: int
        Number of features

    output_path: str
        Where to save the file
    """
    d = dict(waveform_length=waveform_length, n_features=n_features)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)


def save_params(output_path, **kwargs):
    with open(output_path, 'w') as f:
        yaml.dump(kwargs, f, CustomYAMLDumper)
