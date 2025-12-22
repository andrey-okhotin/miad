import os

import pandas as pd

import torch

from metrics.data_preprocess_utils import preprocess_folder_with_cif
from prerelaxations.run_prerelax import run_prerelax
from metrics.energy_above_hull import energy_above_hull
from metrics.uniqueness import uniqueness
from metrics.novelty import novelty

from saving_utils.get_repo_root import get_repo_root




def run(process, config, logger):
    """
    config = ConfigDict({
        'name'         : str,
        'input'        : str,
        'prerelaxation'     : ConfigDict({}),
        'energy_above_hull' : ConfigDict({}),
        'uniqueness'        : ConfigDict({}),
        'novelty'           : ConfigDict({})
    })
    """
    input_path = os.path.join(get_repo_root(), 'saved_results', config.input)

    # main scripts
    logger.root_fprint(f"<> stage: preprocessing .cif files from folder {input_path}")
    structures = preprocess_folder_with_cif(input_path, num_crystals=config.num_crystals)
    
    logger.root_fprint(f"<> stage: prerelaxation via {config.prerelaxation.method}")
    config.prerelaxation.name = config.name
    structures = run_prerelax(process, config.prerelaxation, structures, save_prerelaxed_structures_separatly=True)
    
    # all other parts don't need speed up, so they won't be parallelized
    # therefore, all processes except root one stop their pipelines here
    if not process.is_root_process:
        return None

    logger.root_fprint(f"<> stage: energy above hull {config.energy_above_hull.energy_hull}")
    structures = energy_above_hull(process, config.energy_above_hull, structures)
    
    logger.root_fprint(f"<> stage: uniqueness")
    structures = uniqueness(process, config.uniqueness, structures)
    
    logger.root_fprint(f"<> stage: novelty relatively to {config.novelty.reference} train set")
    structures = novelty(process, config.novelty, structures)

    # sun components
    logger.root_fprint(f"<> stage: sun calculation")
    structures = pd.DataFrame(structures)
    structures['is_non_trivial'] = [ len(set(s.composition)) > 1 for s in structures['structure'] ]
    compare_with_bound = lambda energy_bound, energy_above_hull_per_atom: [ 
        eah < energy_bound if eah != float('nan') else False for eah in energy_above_hull_per_atom
    ]
    structures['metastability'] = compare_with_bound(0.1, structures['energy_above_hull_per_atom'])
    structures['stability'] = compare_with_bound(0.0, structures['energy_above_hull_per_atom'])
    structures['unique&novel_among_metastable'] = (
        structures['uniqueness'] & structures['novelty'] & structures['is_non_trivial']
    )[structures['metastability']]      # rate among metastable
    structures['unique&novel_among_stable'] = (
        structures['uniqueness'] & structures['novelty'] & structures['is_non_trivial']
    )[structures['stability']]          # rate among stable
    structures['msun'] = (
        structures['metastability'] & structures['uniqueness'] & structures['novelty'] 
        & structures['is_non_trivial']  # in original definition we don't take into account crystals of 1 atom type
    )
    structures['sun'] = (
        structures['stability'] & structures['uniqueness'] & structures['novelty']
        & structures['is_non_trivial']  # in original definition we don't take into account crystals of 1 atom type
    )
    
    # save results per crystal
    logger.root_fprint(f"<> stage: saving results")    
    path = os.path.join(get_repo_root(), 'saved_results', config.name)
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(structures.to_dict(), os.path.join(path, 'report.pt'))
    # save statistic
    rate = lambda bool_column: round(100 * bool_column.mean(), 2)
    str_results = (
        f"|---------------------------------------------|\n"+
        f"|  Number of crystals:               | {len(structures):6d} |\n"+
        f"|  Succesfully read:                 | {sum(structures['valid']):6d} |\n"+
        f"|  Succesfully compute energy:       | {sum(~structures['energy_above_hull_per_atom'].isna()):6d} |\n"+
        f"|  More than 1 atom type:            | {sum(structures['is_non_trivial']):6d} |\n"+
        f"|---------------------------------------------|\n"+
        f"|  Metastable (E_hull < 0.1), %      | {rate(structures['metastability']):6.2f} |\n"+
        f"|  Unique&Novel among Metastable %   | {rate(structures['unique&novel_among_metastable']):6.2f} |\n"+
        f"|  M.S.U.N., %                       | {rate(structures['msun']):6.2f} |\n"+
        f"|---------------------------------------------|\n"+
        f"|  Stable (E_hull < 0), %            | {rate(structures['stability']):6.2f} |\n"+
        f"|  Unique&Novel among Stable, %      | {rate(structures['unique&novel_among_stable']):6.2f} |\n"+
        f"|  S.U.N., %                         | {rate(structures['sun']):6.2f} |\n"+
        f"|---------------------------------------------|\n"
    )
    logger.root_fprint(f"<> results:\n{str_results}", end="")
    with open(os.path.join(path, 'metric_values.txt'), 'w') as f:
        f.write(
            f"|---------------------------------------------|\n"+
            f"|           S.U.N. pipeline results           |\n"+
            f"|---------------------------------------------|\n"+
            str_results
        )