import os
import yaml
from ml_collections import ConfigDict




def save_config_yaml(config, file_name):
    file_name = os.path.join('..', 'saved_configs', file_name)
    with open(file_name, 'w') as f:
        yaml.dump(config.to_dict(), f)




def load_config_yaml(file_name):
    file_name = os.path.join('..', 'saved_configs', file_name)
    with open(file_name, 'r') as f:
        config = ConfigDict(yaml.safe_load(f))
    return config