import os
import warnings
import pickle

import torch

from saving_utils.get_repo_root import get_repo_root




def energy_above_hull(
    process,
    config,
    structures,
    save_report=False
):
    if not ('name' in config) and save_report:
        raise KeyError(
            "Key: \'name\' is not defined in the config, while function is run w"+
            "ith flag \'save_report\'. In order to save the results of function, \'outp"+
            "ut_path\' need to be specified in the config."
        )

    ppd_mp_path = os.path.join(get_repo_root(), 'datasets', 'energy_hulls', f'{config.energy_hull}')
    with open(ppd_mp_path, "rb") as f:
        ppd_mp = pickle.load(f)

    list_energy_above_hull_per_atom = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(structures['structure'])):
            try:
                assert structures['valid'][i]
                energy_hull_per_atom = ppd_mp.get_hull_energy_per_atom(structures['structure'][i].composition)
                energy_above_hull_per_atom = structures['energy_per_atom'][i] - energy_hull_per_atom
            except:
                energy_above_hull_per_atom = float('nan')
            list_energy_above_hull_per_atom.append(energy_above_hull_per_atom)
    structures['energy_above_hull_per_atom'] = list_energy_above_hull_per_atom

    if save_report:
        path = os.path.join(get_repo_root(), 'saved_results', config.name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(structures, os.path.join(path, 'report.pt'))
    return structures