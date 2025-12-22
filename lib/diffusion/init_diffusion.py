from diffusion.default_diffusion_configs import set_default_diffusion

from diffusion.crystal_diffusions.general_crystal_diffusion import CrystalGen
from diffusion.crystal_diffusions.special_diffcsp import DiffCSP




def init_diffusion(diffusion_config, logger):
    if 'default_config' in diffusion_config:
        diffusion_config = set_default_diffusion(default_config)
    switch = {
        'Default' : CrystalGen,
        'DiffCSP' : DiffCSP,
    }
    if diffusion_config['method'] in switch:
        diffusion = switch[diffusion_config['method']](diffusion_config, logger)
    else:
        raise NotImplementedError
    return diffusion
