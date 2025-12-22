import os
from collections import defaultdict

import pandas as pd

import torch

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from saving_utils.get_repo_root import get_repo_root




def novelty(
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
    
    # preprocess reference
    path_to_reference_folder = os.path.join(get_repo_root(), 'datasets', config.reference)
    path_to_reference = os.path.join(path_to_reference_folder, 'novelty_reference.pt')
    if not os.path.exists(path_to_reference):
        path_to_train_part = os.path.join(path_to_reference_folder, 'train.csv')
        dataset = pd.read_csv(path_to_train_part, index_col=0)
        train_structures = dataset['cif'].apply(Structure.from_str, fmt="cif")
        reference = defaultdict(list)
        for structure in train_structures:
            chem_system = str(sorted(list(structure.atomic_numbers)))
            reference[chem_system].append(structure)
        torch.save(reference, path_to_reference)
    else:
        reference = torch.load(path_to_reference)
        
    novelty = []
    for i in range(len(structures['structure'])):
        if not structures['valid']:
            uniqueness.append(False)
            continue
        structure = structures['structure'][i]

        chem_system = frozenset(structure.composition)
        if chem_system not in reference:
            novelty.append(True)
        else:
            for reference_structure in reference[chem_system]:
                if StructureMatcher(attempt_supercell=config.attempt_supercell).fit(
                    structure, reference_structure, symmetric=config.symmetric):                
                    novelty.append(False)
                    break
            else:
                novelty.append(True)
    structures['novelty'] = novelty

    if save_report:
        path = os.path.join(get_repo_root(), 'saved_results', config.name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(structures, os.path.join(path, 'report.pt'))
    return structures