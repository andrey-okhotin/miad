from ml_collections import ConfigDict

# base diffusion configs

# lattice

switch_lat = {

    'diffcsp_ddpm_config' : ConfigDict({
        'method'    : 'ddpm',
        'scheduler' : 'diffcsp_cosine',
        'cond_coef' : 1.
    }),

    'ddpm_config' : ConfigDict({
        'method'    : 'ddpm',
        'scheduler' : 'cosine',
        'cond_coef' : 1.
    }),

    'fm_config' : ConfigDict({
        'method'    : 'fm',
        'cond_coef' : 1.,
        'parameterization' : 'eps'
    }),

    'fm_v_config' : ConfigDict({
        'method'    : 'fm',
        'cond_coef' : 1.,
        'parameterization' : 'v'
    }),

    'fm_lenang_config' : ConfigDict({
        'method'    : 'fm_lenang',
        'cond_coef' : 1.
    })

}

# coords

switch_coord = {

    'wrapped_normal_config' : ConfigDict({
        'method'    : 'wrapped_normal',
        'scheduler' : 'default_wrapped_normal',
        'cond_coef' : 1.
    }),

    'pfm_config' : ConfigDict({
        'method'    : 'pfm',
        'cond_coef' : 1.
    })

}

# types

switch_type = {

    'd3pm_config' : ConfigDict({
        'method'    : 'd3pm',
        'scheduler' : 'default_d3pm',
        'cond_coef' : 1.
    }),

    'ddpm_onehot_config' : ConfigDict({
        'method'    : 'ddpm_onehot',
        'scheduler' : 'diffcsp_cosine',
        'cond_coef' : 1.
    })

}


def set_default_diffusion(diffusion_name):
    # EXAMPLE: DiffCSP:diffcsp_ddpm_config:wrapped_normal_config:d3pm_config
    combination_method, lat_m, coord_m, type_m = diffusion_name.split(':')
    config = ConfigDict({
        'method'         : combination_method,
        'cont_time'      : False,
        'num_steps'      : 1000,
        'task'           : '',
        'lat_diffusion'  : switch_lat[lat_m],
        'frac_diffusion' : switch_coord[coord_m],
        'type_diffusion' : switch_type[type_m] if type_m in switch_type else None
    })
    return config