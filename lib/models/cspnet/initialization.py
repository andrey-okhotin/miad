import os
import ml_collections
import torch

from saving_utils.get_repo_root import get_repo_root
from models.cspnet.cspnet import CSPNet




def default_cspnet_config():
    cspnet_config = ml_collections.ConfigDict({
        "hidden_dim"    : 512,
        "num_layers"    : 6,
        "num_freqs"     : 128,
        
        "latent_dim"    : 256,
        "max_atoms"     : 100,
        "act_fn"        : "silu",
        "dis_emb"       : "sin",
        "edge_style"    : "fc",
        "max_neighbors" : 20,
        "cutoff"        : 7.0,
        "ln"            : True,
        "ip"            : True
    })
    return cspnet_config




def small_cspnet_config():
    cspnet_config = ml_collections.ConfigDict({
        "hidden_dim"    : 256,
        "num_layers"    : 4,
        "num_freqs"     : 10,
        
        "latent_dim"    : 256,
        "max_atoms"     : 100,
        "act_fn"        : "silu",
        "dis_emb"       : "sin",
        "edge_style"    : "fc",
        "max_neighbors" : 20,
        "cutoff"        : 7.0,
        "ln"            : True,
        "ip"            : True
    })
    return cspnet_config


# gen


def default_cspnet_gen_config():
    cspnet_gen_config = default_cspnet_config()
    cspnet_gen_config.pred_type = True
    cspnet_gen_config.smooth = True
    return cspnet_gen_config




def small_cspnet_gen_config():
    cspnet_gen_config = small_cspnet_config()
    cspnet_gen_config.pred_type = True
    cspnet_gen_config.smooth = True
    return cspnet_gen_config




def init_cspnet(config):
    switch_default_configs = {
        # ab-initio generation
        'cspnet-gen-default'       : default_cspnet_gen_config,
        'cspnet-gen-small'         : small_cspnet_gen_config
    }

    if 'default_config' in config.keys():
        default_config = switch_default_configs[config['default_config']]()
        return CSPNet(**default_config)

    else:
        return CSPNet(**config)
    
    raise NotImplementedError
