import os

import torch

from models.cspnet.initialization import init_cspnet
from models.cspnet_light.initialization import init_cspnet_light

from saving_utils.get_repo_root import get_repo_root




def init_model(config):
    switch = {
        'CSPNet'      : init_cspnet,
        'CSPNetLight' : init_cspnet_light
    }
    if 'model_name' in config:
        model = switch[config['model_name']](config)
        model.model_config = config

    elif 'saved_model' in config:
        path = os.path.join(get_repo_root(), 'saved_models', config['saved_model'])
        model = load_model_from_path(path)

    else:
        raise NotImplementedError

    model.device = 'cpu'
    return model




def save_model_to_path(model, path):
    state_dict = model.state_dict()
    state_dict['model_config'] = model.model_config
    torch.save(state_dict, path)




def load_model_from_path(path):
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    config = state_dict['model_config']
    model = init_model(config)
    del state_dict['model_config']
    state_dict = { k.removeprefix('module.') : v for k, v in state_dict.items() }
    model.load_state_dict(state_dict)
    model.model_config = config
    return model