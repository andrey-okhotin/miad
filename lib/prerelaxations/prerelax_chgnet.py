import os
import copy
import time
import psutil
import multiprocessing
from typing import Any
from contextlib import redirect_stdout, redirect_stderr

import torch

import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from dataclasses import dataclass
from functools import cached_property

from chgnet.model import StructOptimizer
from chgnet.model.dynamics import TrajectoryObserver
from chgnet.model.model import CHGNet
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms, Atoms

from prerelaxations.error_safety_utils import limit_memory
from saving_utils.get_repo_root import get_repo_root




def prerelax_chgnet(
    process, 
    config,
    structures
):
    # num workers
    available_cpu = len(psutil.Process().cpu_affinity()) - 4
    if config.num_workers == 'auto':
        num_threads = 4
        num_workers = available_cpu // num_threads
    else:
        num_workers = config.num_workers
        num_threads = available_cpu // num_workers

    # split structures into parts
    num_structures = len(structures['structure'])
    index = np.arange(num_structures)
    indexes = np.array_split(index, num_workers)
    indexes = [i for i in indexes if i.size > 0]  # filter out blanks

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    return_dict = manager.dict()

    limit_memory(percent_of_max_memory=10)
    processes = []
    for i in range(len(indexes)):
        process = ctx.Process(
            target=prerelax_multiple_crystals,
            args=(
                copy.deepcopy(indexes[i]),
                {
                    'cif_name'  : [ structures['cif_name'][j]  for j in indexes[i] ],
                    'valid'     : [ structures['valid'][j]     for j in indexes[i] ],
                    'structure' : [ structures['structure'][j] for j in indexes[i] ]
                },
                copy.deepcopy(config),
                i, 
                return_dict, 
                num_threads
            )
        )
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    results = [ return_dict[i] for i in range(len(indexes)) ]
    detailed_report = pd.concat(results)

    # save main results to the report
    none_to_nan = lambda l: [ float('nan') if v is None else v for v in l ]
    structures['energy'] = none_to_nan(list(detailed_report['e_relax']))
    structures['energy_per_atom'] = none_to_nan(list(detailed_report['e_relax_per_atom']))
    empty_dict_to_none = lambda l: [ Structure.from_dict(v) if v != {} else None for v in l ]
    structures['prerelaxed_structure'] = empty_dict_to_none(list(detailed_report['structure']))
    return structures, detailed_report




def prerelax_multiple_crystals(
    index_array, 
    structures,
    config, 
    rank, 
    return_dict, 
    num_treads
):
    torch.set_num_threads(num_treads)

    # for per process progress printing
    if 'name' in config:
        logfiles_folder = os.path.join(get_repo_root(), 'logs', 'progress_logs', 'per_process_chgnet_prerelax_logs')
        if not os.path.exists(logfiles_folder):
            os.mkdir(logfiles_folder)
        logfile_path = os.path.join(logfiles_folder, f'{config.name}_{rank}.log')
        logfile = open(logfile_path, 'w')
    else:
        logfile = None

    with redirect_stdout(logfile):
        print('Run prerelaxation with rank {0} on {1} threads'.format(rank, num_treads))
        chgnet = CHGNet.load(verbose=False, use_device=config.device)
        relaxer = StructOptimizer(model=chgnet, use_device=config.device)

    rd = {
        'cif_name'          : [],
        'valid_for_reading' : [], 
        'index'             : [],
        'e_gen'             : [],
        'e_relax'           : [],
        'e_relax_per_atom'  : [],
        'n_to_relax'        : [],
        'rms_dist'          : [],
        'matched'           : [],
        'converged'         : [],
        'exception'         : [],
        'num_sites'         : [],
        'structure'         : [],
        'e_delta'           : []
    }
    st_time = time.time()

    for i, index in enumerate(index_array):
        if i % 1 == 0 and not (logfile is None):
            elapsed_time = (time.time() - st_time) / 60
            print('Processed: {0}/{1}. Elapsed: {2:.2f}/{3:.2f}'.format(
                i, len(index_array),
                elapsed_time, elapsed_time * len(index_array) / (i + 1)
            ), file=logfile, flush=True)

        # prerelax
        rd['cif_name'].append(structures['cif_name'][i])
        rd['valid_for_reading'].append(structures['valid'][i])
        rd['index'].append(index)
        try:
            assert structures['valid'][i]
            structure = structures['structure'][i]

            results = prerelax_crystal(
                structure=structure,
                chgnet=chgnet,
                relaxer=relaxer,
                steps=config.steps,
                logfile=logfile
            )

            e_gen = results.energies[0]
            e_relax = results.energies[1]
            e_delta = np.abs(e_relax - e_gen)

            rd['e_gen'].append(e_gen)
            rd['e_relax'].append(e_relax)
            rd['e_relax_per_atom'].append(e_relax / structure.num_sites)
            rd['n_to_relax'].append(results.n_steps_to_relax)
            rd['rms_dist'].append(results.rms_dist)
            rd['matched'].append(results.match)
            rd['converged'].append(True if results.n_steps_to_relax < config.steps else False)
            rd['exception'].append(False)
            rd['num_sites'].append(structure.num_sites)
            rd['structure'].append(results.structure_dicts[-1])
            rd['e_delta'].append(e_delta)

        except Exception as exp:
            cif_name = structures['cif_name'][i]
            print(f'Exception. Index: {index} CIF-name: {cif_name}', exp)
            rd['e_gen'].append(None)
            rd['e_relax'].append(None)
            rd['e_relax_per_atom'].append(None)
            rd['n_to_relax'].append(pd.NA)
            rd['rms_dist'].append(None)
            rd['matched'].append(False)
            rd['converged'].append(False)
            rd['exception'].append(True)
            rd['num_sites'].append(pd.NA)
            rd['structure'].append({})
            rd['e_delta'].append(None)

    if not (logfile is None):
        logfile.close()
    df = pd.DataFrame.from_dict(rd).set_index("index")
    return_dict[rank] = df




def prerelax_crystal(structure, relaxer, chgnet, steps, logfile=None):
    with redirect_stdout(logfile):
        with redirect_stderr(logfile):
            prediction = chgnet.predict_structure(structure)
            relaxation = relaxer.relax(structure, steps=steps, verbose=True)
    return PrerelaxationResultsCollector.from_chgnet(structure, prediction, relaxation)




@dataclass
class PrerelaxationResultsCollector:

    structure_dicts: tuple[dict, dict]
    energies: tuple[float, float]
    n_steps_to_relax: int
    stol: float = 0.5
    angle_tol: int = 10
    ltol: float = 0.3

    def __post_init__(self):
        self.matcher = StructureMatcher(
            stol=self.stol,
            angle_tol=self.angle_tol,
            ltol=self.ltol,
        )

    @cached_property
    def structures(self) -> tuple[Structure, ...]:
        return tuple(Structure.from_dict(sd) for sd in self.structure_dicts)

    @cached_property
    def atoms(self) -> tuple[MSONAtoms | Atoms, ...]:
        return tuple(
            AseAtomsAdaptor.get_atoms(structure) for structure in self.structures
        )

    @cached_property
    def match(self) -> bool:
        return False if self.rms_dist is None else True

    @cached_property
    def rms_dist(self) -> Any | None:
        out = self.matcher.get_rms_dist(self.structures[0], self.structures[1])
        if out is None:
            return out
        elif isinstance(out, tuple):
            return out[0]
        else:
            raise ValueError()

    @classmethod
    def from_chgnet(
            cls,
            initial_structure: Structure,
            prediction: dict[str, torch.Tensor],
            relaxation: dict[str, Structure | TrajectoryObserver],
    ):
        initial_structure.add_site_property("magmom", prediction["m"])
        final_structure = relaxation["final_structure"]
        trajectory = relaxation["trajectory"]
        return cls(
            structure_dicts=(initial_structure.as_dict(), final_structure.as_dict()),
            energies=(
                prediction["e"] * initial_structure.num_sites,
                trajectory.energies[-1],
            ),
            n_steps_to_relax=len(trajectory.energies),
        )