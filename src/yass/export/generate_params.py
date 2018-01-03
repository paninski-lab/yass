from datetime import datetime
import os.path as path

from ..util import load_asset, load_yaml


def generate_params(path_to_config):
    """
    Generate phy's params.py from YASS' config.yaml
    """
    template = load_asset('phy/params.py')
    config = load_yaml(path_to_config)

    timestamp = datetime.now().strftime('%B %-d, %Y at %H:%M')
    dat_path = path.join(config['data']['root_folder'],
                         config['data']['recordings'])
    n_channels_dat = config['recordings']['n_channels']
    dtype = config['recordings']['dtype']
    sample_rate = config['recordings']['sampling_rate']

    params = template.format(timestamp=timestamp,
                             dat_path=dat_path,
                             n_channels_dat=n_channels_dat,
                             dtype=dtype,
                             offset=0,
                             sample_rate=sample_rate,
                             hp_filtered='True')

    return params
