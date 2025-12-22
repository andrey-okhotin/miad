import io
import os
import time
import pickle
import warnings
from contextlib import redirect_stdout

trap = io.StringIO()

import glob
from natsort import natsorted

import pandas as pd

from chgnet.model import CHGNet
from pymatgen.core import Structure

import torch

from metrics.compress_results import compress_data
from saving_utils.get_repo_root import get_repo_root

from prerelaxations.prerelax_chgnet import prerelaxation_chgnet
from prerelaxations.prerelax_eqv2 import prerelaxation_eqv2




def metric_energy_above_hull(config, skip_prerelaxation=False):
    """
    Example:

        config = ConfigDict({
            'input_path' : ... ,
            'output_path' : ... ,
            'prerelaxation_config' : ConfigDict({
                'method' : 'CHGNet',
                 ...
            })
            'energy_estimator_config' : ConfigDict({
                'method' : 'CHGNet',
                 ...
            })
        })
    """
    t1 = time.time()

    path = os.path.join(get_repo_root(), 'saved_results', config.output_path)
    if not os.path.exists(path):
        os.mkdir(path)

    config.prerelaxation_config.world_size = config.world_size
    config.prerelaxation_config.rank = config.rank
    config.prerelaxation_config.output_path = os.path.join(path, "prerelaxed_crystals")

    # data operations
    if not skip_prerelaxation:
        fname = compress_data(config.input_path, False)

        # prerelax
        switch_energy_prerelaxation = {
            'CHGNet': prerelax_chgnet,
            'EqV2' :  prerelax_eqv2
        }
        config.prerelaxation_config.name = config.output_path
        config.prerelaxation_config.input_path = fname
        config.prerelaxation_config.num_structures = config.num_structures
        config.prerelaxation_config.num_workers = config.num_workers
        config.prerelaxation_config.device = config.device
        config.prerelaxation_config.gpu = config.gpu

        # Perform actual computations. We assume that it will be performed on a [multi]-GPU
        _ = switch_energy_prerelaxation[config.prerelaxation_config.method](config.prerelaxation_config)

    # Dumb way to collect results. Just read them and then concat
    # Perform synchronization in multi-GPU setup
    if config.prerelaxation_config.world_size > 1:
        torch.distributed.barrier()
    # Assume that all necessary files in the folder will match
    if config.prerelaxation_config.rank == 0:
        pattern = f'{config.prerelaxation_config.output_path}_[0-9]*.json'
        if os.path.exists(f'{config.prerelaxation_config.output_path}.json'):
            print(
                '{0}.json already exists. Do not overwrite from {1}'.format(
                    config.prerelaxation_config.output_path, pattern
                )
            )
        else:
            df = pd.concat([
                pd.read_json(path) for path in natsorted(glob.glob(pattern))
            ])
            df.to_json(f'{config.prerelaxation_config.output_path}.json')
    # Perform synchronization in multi-GPU setup
    if config.prerelaxation_config.world_size > 1:
        torch.distributed.barrier()
    prerelaxation_results = pd.read_json(f'{config.prerelaxation_config.output_path}.json')

    t2 = time.time()

    # energy above hull
    config.energy_estimator_config.input_path = config.prerelaxation_config.output_path
    config.energy_estimator_config.output_path = os.path.join(path, "energy_above_hull")
    config.energy_estimator_config.num_workers = config.num_workers
    ehull_results = energy_above_hull(config.energy_estimator_config)

    t3 = time.time()

    # time estimations
    prerelax_time = torch.tensor(t2 - t1)
    ehull_time = torch.tensor(t3 - t2)
    torch.save(prerelax_time, f"{config.prerelaxation_config.output_path}-time.pt")
    torch.save(ehull_time, f"{config.energy_estimator_config.output_path}-time.pt")

    return prerelaxation_results, ehull_results, prerelax_time, ehull_time




def load_metric_energy_above_hull(config):
    path = os.path.join(get_repo_root(), 'saved_results', config.output_path)
    prerelaxation_path = os.path.join(path, "prerelaxed_crystals")
    ehull_path = os.path.join(path, "energy_above_hull")
    return (
        pd.read_json(f'{prerelaxation_path}.json'),
        pd.read_json(f'{ehull_path}.json'),
        torch.load(f"{prerelaxation_path}-time.pt"),
        torch.load(f"{ehull_path}-time.pt")
    )




def energy_above_hull(energy_estimator_config):
    # split structures into parts
    df = pd.read_json(f'{energy_estimator_config.input_path}.json')
    # DataGrame with columns: structure, e_gen, e_relax, num_sites

    # ehull
    switch_ehull = {
        'flowmm': '2023-02-07-ppd-mp.pkl',
        'alex-mp20' : ''
    }
    ppd_mp_path = os.path.join(
        get_repo_root(), 'datasets', 'energy_hulls',
        switch_ehull[energy_estimator_config.ehull]
    )
    with open(ppd_mp_path, "rb") as f:
        ppd_mp = pickle.load(f)
    e_hulls = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, structure in enumerate(df["structure"]):
            try:
                structure = to_structure(structure)
                e_hull = ppd_mp.get_hull_energy_per_atom(structure.composition)
            except (ValueError, AttributeError, ZeroDivisionError, KeyError):
                e_hull = float("nan")
            e_hulls.append(e_hull)

    out = pd.DataFrame(data={"e_hull_per_atom": e_hulls})
    out.index = df.index  # this works because we filtered out exceptions above!
    default_e_above_hull = (df["e_gen"] / df["num_sites"]) - out["e_hull_per_atom"]
    prerelaxed_e_above_hull = (df["e_relax"] / df["num_sites"]) - out["e_hull_per_atom"]

    # save
    out["e_above_hull_per_atom_default"] = default_e_above_hull
    out["e_above_hull_per_atom_prerelaxed"] = prerelaxed_e_above_hull
    out.to_json(f'{energy_estimator_config.output_path}.json')
    return out


def to_structure(structure: Structure | dict | str) -> Structure:
    with redirect_stdout(trap):
        if isinstance(structure, dict):
            return Structure.from_dict(structure)
        elif isinstance(structure, str):
            return Structure.from_str(structure, fmt="cif")
        else:
            return structure