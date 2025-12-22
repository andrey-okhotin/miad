import os
import shutil

import torch

from prerelaxations.prerelax_chgnet import prerelax_chgnet
from prerelaxations.prerelax_eqv2 import prerelax_eqv2

from saving_utils.get_repo_root import get_repo_root




def run_prerelax(
    process, 
    config,
    structures,
    save_report=False,
    save_detailed_report=False,
    save_prerelaxed_structures_separatly=False
):
    """
    Example:

        config = ConfigDict({
            'name' : ... ,
            'method': ..., 
            'num_structures': args.num_samples,
            'num_workers': num_workers_prerelaxation,
            'device': f'cuda:{process.gpu}' if process.gpu != 'cpu' else process.gpu
        })
    """
    if not ('name' in config) and (save_report or save_detailed_report or save_prerelaxed_structures_separatly):
        raise KeyError(
            "Key: \'name\' is not defined in config, while function is run with "+
            "one of saving flags \'save_report\' or \'save_detailed_report\'. In order "+
            "to save the results of function, \'name\' need to be specified in t"+
            "he config."
        )

    config.device = f'cuda:{process.gpu}' if process.gpu != 'cpu' else process.gpu
    switch_energy_prerelaxation = {
        'CHGNet' : prerelax_chgnet,
        'eq-V2'  : prerelax_eqv2
    }
    # Perform actual computations. We assume that it will be performed on a [multi]-GPU
    structures, detailed_report = switch_energy_prerelaxation[config.method](
        process,
        config,
        structures
    )
    if process.is_root_process and (save_report or save_detailed_report or save_prerelaxed_structures_separatly):
        path = os.path.join(get_repo_root(), 'saved_results', config.name)
        if not os.path.exists(path):
            os.mkdir(path)
        if save_report:
            torch.save(structures, os.path.join(path, 'report.pt'))
        if save_detailed_report and not (detailed_report is None):
            detailed_report.to_json(os.path.join(path, 'detailed-report_prerelaxation.json'))
        if save_prerelaxed_structures_separatly:
            folder = os.path.join(path, 'prerelaxed_crystals_cif')
            if process.is_root_process and not os.path.exists(folder):
                os.mkdir(folder)
            for name, prerelaxed_structure in zip(structures['cif_name'], structures['prerelaxed_structure']):
                if not (prerelaxed_structure is None):
                    prerelaxed_structure.to(os.path.join(folder, f'prerelaxed_{name}.cif'))
            if process.is_root_process:
                shutil.make_archive(folder, 'zip', folder)
    return structures
    