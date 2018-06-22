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


CustomYAMLDumper.add_representer(map_data, map_data)
CustomYAMLDumper.add_representer(seq_data, map_rep)


def save_detect_network_params(filters, size, n_neighbors, output_path):
    """Generate yaml file with parameters for a detect network

    Parameters
    ----------
    filters: list
        List with number of filters in each layer

    size: int
        Temporal filter size

    n_neighbors: int
        Number of neighboring channels

    output_path: str
        Where to save the file
    """
    d = dict(filters=filters, size=size, n_neighbors=n_neighbors)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)


def save_triage_network_params(filters, size, n_neighbors, output_path):
    """Generate yaml file with parameters for a triage network

    Parameters
    ----------
    filters: list
        List filters size

    size: int
        Temporal filter size

    n_neighbors: int
        Number of neighboring channels

    output_path: str
        Where to save the file
    """
    d = dict(filters=filters, size=size, n_neighbors=n_neighbors)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)


def save_ae_network_params(n_input, n_features, output_path):
    """Generate yaml file with parameters for a ae network

    Parameters
    ----------
    n_input: int
        Dimension of input

    n_features: int
        Number of features

    output_path: str
        Where to save the file
    """
    d = dict(n_input=n_input, n_features=n_features)

    with open(output_path, 'w') as f:
        yaml.dump(d, f, CustomYAMLDumper)
