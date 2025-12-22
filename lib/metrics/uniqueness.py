import os
from collections import defaultdict

import torch

from pymatgen.analysis.structure_matcher import StructureMatcher

from saving_utils.get_repo_root import get_repo_root




def uniqueness(
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

    present = defaultdict(list)
    uniqueness = []
    for i in range(len(structures['structure'])):
        if not structures['valid']:
            uniqueness.append(False)
            continue
        structure = structures['structure'][i]

        chem_system = frozenset(structure.composition)
        if chem_system not in present:
            uniqueness.append(True)
        else:
            for present_structure in present[chem_system]:
                if StructureMatcher(attempt_supercell=config.attempt_supercell).fit(
                    structure, present_structure, symmetric=config.symmetric):
                    uniqueness.append(False)
                    break
            else:
                uniqueness.append(True)
        present[chem_system].append(structure)
    structures['uniqueness'] = uniqueness

    if save_report:
        path = os.path.join(get_repo_root(), 'saved_results', config.name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(structures, os.path.join(path, 'report.pt'))
    return structures