import os
from collections import defaultdict
from natsort import natsorted

import torch
from torch_geometric.data import Batch

from data.crystal_datasets.cryst_utils import lattices_to_params_shape
from saving_utils.get_repo_root import get_repo_root




def compress_data(dataset_name, multi_eval):
    path = os.path.join(get_repo_root(), 'results', dataset_name)
    if not os.path.exists(path) and os.path.exists(path+'.pt'):
        return path+'.pt'
    all_results = defaultdict(list)
    results_dir = os.path.join(path, 'generated_samples')
    files = natsorted(os.listdir(results_dir))
    for f in files:
        if 'tag' in f:
            tag = int(f.split('tag')[1].split('.')[0])
            rank = int(f.split('_')[1].split('_')[0])
            all_results[1000000 * rank + tag].append(
                torch.load(os.path.join(results_dir, f), map_location='cpu', weights_only=False)
            )

    frac_coords, num_atoms, atom_types, lattices = [], [], [], []
    crystal_indexes = []
    input_data_list = []
    for tag, list_of_batches in all_results.items():
        frac_coords_4tag, num_atoms_4tag, atom_types_4tag, lattices_4tag = [], [], [], []
        for batch in list_of_batches:
            frac_coords_4tag.append(batch['frac_coords'])
            num_atoms_4tag.append(batch['num_atoms'])
            atom_types_4tag.append(batch['atom_types'])
            lattices_4tag.append(batch['lattices'])
            if not multi_eval:
                break
        
        frac_coords.append(torch.stack(frac_coords_4tag, dim=0))
        atom_types.append(torch.stack(atom_types_4tag, dim=0))
        num_atoms.append(torch.stack(num_atoms_4tag, dim=0))
        lattices.append(torch.stack(lattices_4tag, dim=0))
        if hasattr(batch['input_data_batch'], "_object_index_in_dataset_"):
            crystal_indexes += batch['input_data_batch']._object_index_in_dataset_
        input_data_list += batch['input_data_batch'].to_data_list()       

    frac_coords = torch.cat(frac_coords, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    if multi_eval:
        fname = 'eval_results_best20.pt'
    else:
        fname = 'eval_results.pt'
    output_pack = {
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles
    }
    if hasattr(batch['input_data_batch'], "_object_index_in_dataset_"):
        output_pack['crystal_indexes'] = crystal_indexes
    torch.save(output_pack, os.path.join(results_dir, fname))
    return os.path.join(results_dir, fname)




