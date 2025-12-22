import os
import time
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist

from ase.optimize import LBFGS
import fairchem.core.common.distutils
from fairchem.core import OCPCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from saving_utils.get_repo_root import get_repo_root




def prerelax_eqv2(
    process, 
    config,
    structures
):
    # split structures into parts
    if process.distributed:
        torch.distributed.barrier()
    time.sleep(2 * process.rank)

    num_structures = len(structures['structure'])
    if hasattr(config, 'num_structures'):
        num_structures = config.num_structures
    index = np.arange(num_structures)
    indexes = np.array_split(index, process.world_size)
    indexes = [i for i in indexes if i.size > 0]  # filter out blanks
    
    results = prerelax_multiple(
        indexes[process.rank],
        {
            'cif_name'  : [ structures['cif_name'][j]  for j in indexes[process.rank] ],
            'valid'     : [ structures['valid'][j]     for j in indexes[process.rank] ],
            'structure' : [ structures['structure'][j] for j in indexes[process.rank] ]
        }, 
        config, 
        process.rank,
        process
    )

    # collect results
    if process.distributed:
        dist.barrier()
        if process.is_root_process:
            gathered_results = [ None for _ in range(process.world_size) ]
        else:
            gathered_results = None
        dist.gather_object(results, gathered_results, dst=0)
        dist.barrier()
        if process.is_root_process:
            detailed_report = pd.concat(gathered_results)
    else:
        detailed_report = results

    # save main results to the report
    if process.is_root_process:
        none_to_nan = lambda l: [ float('nan') if v is None else v for v in l ]
        structures['energy'] = none_to_nan(list(detailed_report['e_relax']))
        structures['energy_per_atom'] = none_to_nan(list(detailed_report['e_relax_per_atom']))
        empty_dict_to_none = lambda l: [ Structure.from_dict(v) if v != {} else None for v in l ]
        structures['prerelaxed_structure'] = empty_dict_to_none(list(detailed_report['structure']))
        return structures, detailed_report
    else:
        return None, None




def prerelax_multiple(
    index_array, 
    structures,
    config, 
    rank,
    process
):
    if 'name' in config:
        logfiles_folder = os.path.join(get_repo_root(), 'logs', 'progress_logs', 'per_process_eqv2_prerelax_logs')
        if not os.path.exists(logfiles_folder):
            os.mkdir(logfiles_folder)
        logfile_path = os.path.join(logfiles_folder, f'{config.name}_{rank}.log')
        logfile = open(logfile_path, 'w')
    else:
        logfile = None

    with redirect_stdout(logfile):
        with redirect_stderr(logfile):
            checkpoint_path = os.path.join(get_repo_root(), 'saved_models', 'eqV2_153M_omat_mp_salex.pt')
            # hack to overcome multiprocessing inside fairchem
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
            checkpoint_config = checkpoint['config']
            del checkpoint
            checkpoint_config['local_rank'] = rank
            checkpoint_config['dataset']['train'] = checkpoint_config['dataset']
            _is_master = fairchem.core.common.distutils.is_master
            _initialized = fairchem.core.common.distutils.initialized
            fairchem.core.common.distutils.is_master = lambda: True
            fairchem.core.common.distutils.initialized = lambda: False

            calc = OCPCalculator(
                config_yml=checkpoint_config,
                checkpoint_path=checkpoint_path,
                cpu=(process.gpu == 'cpu')
            )
            rd = {
                'cif_name'          : [],
                'valid_for_reading' : [],
                'index'             : [],
                'structure'         : [],
                'e_gen'             : [],
                'e_relax'           : [],
                'e_relax_per_atom'  : [],
                'e_delta'           : [],
                'num_sites'         : [],
                'exception'         : []
            }
            for i, index in enumerate(index_array):
                rd['index'].append(index)
                rd['cif_name'].append(structures['cif_name'][i])
                rd['valid_for_reading'].append(structures['valid'][i])
                structure = structures['structure'][i]

                try:
                    ase_obj = AseAtomsAdaptor.get_atoms(structure)
                    ase_obj.calc = calc
                    # noinspection PyUnresolvedReferences
                    e_gen = ase_obj.get_potential_energy()
                    dyn = LBFGS(ase_obj, logfile=logfile)

                    dyn.run(0.05, config.steps)
                    # noinspection PyUnresolvedReferences
                    e_relax = ase_obj.get_potential_energy()
                    prerelaxed_structure = AseAtomsAdaptor.get_structure(ase_obj)

                    structure = prerelaxed_structure.as_dict()
                    num_sites = prerelaxed_structure.num_sites
                    e_delta = np.abs(e_relax - e_gen)

                    rd['structure'].append(structure)
                    rd['num_sites'].append(num_sites)
                    rd['e_gen'].append(e_gen)
                    rd['e_relax'].append(e_relax)
                    rd['e_relax_per_atom'].append(e_relax / num_sites)
                    rd['e_delta'].append(e_delta)
                    rd['exception'].append(False)
                
                except Exception as exp:
                    cif_name = structures['cif_name'][i]
                    print(f'Exception. Index: {index} CIF-name: {cif_name}', exp)
                    rd['structure'].append({})
                    rd['num_sites'].append(pd.NA)
                    rd['e_gen'].append(None)
                    rd['e_relax'].append(None)
                    rd['e_relax_per_atom'].append(None)
                    rd['e_delta'].append(None)
                    rd['exception'].append(True)

    if not (logfile is None):
        logfile.close()
    df = pd.DataFrame.from_dict(rd).set_index('index')
    return df
