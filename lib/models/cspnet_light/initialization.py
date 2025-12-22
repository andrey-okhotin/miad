import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.cspnet_light.cspnet_light import CSPNetLight


# gen

def default_cspnet_light_gen_config():
    cspnet_gen_config = ml_collections.ConfigDict({
        "hidden_dim"    : 512,
        "num_layers"    : 6,
        "time_dim"      : 256,
        "num_freqs"     : 128,
        "max_atoms"     : 100
    })
    return cspnet_gen_config

def cspnet_s_gen_config():
    cspnet_gen_config = default_cspnet_light_gen_config()
    scale_factor = 1.0
    cspnet_gen_config.hidden_dim = int(cspnet_gen_config.hidden_dim * scale_factor)
    cspnet_gen_config.num_layers = int(cspnet_gen_config.num_layers * scale_factor)
    return cspnet_gen_config

def cspnet_m_gen_config():
    cspnet_gen_config = default_cspnet_light_gen_config()
    scale_factor = 1.5
    cspnet_gen_config.hidden_dim = int(cspnet_gen_config.hidden_dim * scale_factor)
    cspnet_gen_config.num_layers = int(cspnet_gen_config.num_layers * scale_factor)
    return cspnet_gen_config






def init_cspnet_light(model_config):
    switch_default_configs = {
        # ab-initio generation
        'cspnet-light-gen-default' : default_cspnet_light_gen_config,
        'cspnet-s-gen'             : cspnet_s_gen_config,
        'cspnet-m-gen'             : cspnet_m_gen_config
    }

    if 'default' in model_config.keys():
        cspnet_config = switch_default_configs[model_config['default']]()
        return CSPNetLight(**cspnet_config)

    elif 'saved_model' in model_config.keys():
        model_domain = model_config['saved_model'].split('_')[0]
        model = CSPNetLight(**switch_default_configs[model_domain]())
        path = os.path.join(get_repo_root(), 'saved_models', model_config['saved_model'])
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = { k.removeprefix('module.') : v for k, v in state_dict.items() }
        model.load_state_dict(state_dict)
        print(f'Initialized CSPNetLight from {path}')
        return model
    
    raise NotImplementedError
