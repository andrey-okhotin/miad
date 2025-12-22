import os
import shutil

import pandas as pd
from p_tqdm import p_umap

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Data, Batch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from saving_utils.get_repo_root import get_repo_root
from data.crystal_utils import (
    lattice_to_lengths_and_angles,
    lengths_and_angles_to_lattice,
    frac_and_lattice_to_cart_coords
)
from data.scalers import init_scaler






class CrystalDataStorage:

    """
    config = {
        'type' : 'CrystalDataStorage'
        'name' : str,
        'parts' : [ 'part_1', 'part_2', 'part_3' ],
        'main_part' : 'part_i',
        'properties' : [ 
            ('column_in_csv_prop_1', 'discrete', 'NoScaler'),
            ('column_in_csv_prop_2', 'continious', 'StandardContiniousScaler')
         ],
        'crystal_preprocess' : ConfigDict({}),
        'batch_size' : N
    }
    """

    def __init__(self, config, process):
        self.config = config
        self.process = process
        # preprocess dataset
        if process.distributed:
            # running for the first time, the following code preprocess dataset in the root
            # process, and then use its results in other processes
            if process.is_root_process:
                for part in self.config.parts:
                    setattr(self, f"dataset_{part}", CrystalDataset(self.config, part))
                dist.barrier()
            else:
                dist.barrier()
                for part in self.config.parts:
                    setattr(self, f"dataset_{part}", CrystalDataset(self.config, part))
        else:
            for part in self.config.parts:
                setattr(self, f"dataset_{part}", CrystalDataset(self.config, part))
            
        # in order to have the same scalers in all parts of the dataset as in the main dataset part
        self.param_scalers = getattr(getattr(self, f"dataset_{self.config.main_part}"), "param_scalers")
        self.prop_scalers = getattr(getattr(self, f"dataset_{self.config.main_part}"), "prop_scalers")
        for part in self.config.parts:
            getattr(self, f"dataset_{part}").set_crystal_parameters_scalers(self.param_scalers)
            getattr(self, f"dataset_{part}").set_crystal_properties_scalers(self.prop_scalers)
        self.set_dataloaders()




    def set_dataloaders(self):
        for part in self.config.parts:
            setattr(
                self, 
                f"dataloader_{part}",
                DataLoader(
                    getattr(self, f"dataset_{part}"),
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                    collate_fn=lambda data_list: Batch.from_data_list(data_list)
                )
            )




    def rescale_crystals_in_datasets(self):
        for part in self.config.parts:
            getattr(self, f"dataset_{part}").rescale_crystals_accordingly_to_current_scalers()
        self.set_dataloaders()




    def to_distributed(self, part):
        self.distributed_sampler = DistributedSampler(
            getattr(self, f"dataset_{part}"),
            rank=self.process.rank,
            seed=self.process.seed,
            shuffle=True
        )
        setattr(
            self, 
            f"distributed_dataloader_{part}",
            DataLoader(
                getattr(self, f"dataset_{part}"),
                sampler=self.distributed_sampler,
                batch_size=self.config.batch_size // self.process.world_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda data_list: Batch.from_data_list(data_list)
            )
        )



    
    def ab_init_gen_training_batch(self, crystal_batch, device='cpu'):
        num_crystals = len(crystal_batch.to_data_list())
        x0 = [ 
            crystal_batch.lattice,          # lattice
            crystal_batch.frac_coords,      # fractional
            crystal_batch.atom_types        # atom types
        ]
        if device == 'cuda':
            x0[0] = x0[0].cuda(non_blocking=True)
            x0[1] = x0[1].cuda(non_blocking=True)
            x0[2] = x0[2].cuda(non_blocking=True)
            device = torch.cuda.current_device()
        return {
            'x0'         : x0,
            'batch_size' : num_crystals,
            'num_atoms'  : crystal_batch.num_nodes,
            'device'     : device,
            'batch'      : crystal_batch.to(device)
        }




    def ab_init_gen_sampling_batch(self, batch_size, device='cpu'):
        num_atoms = self.dataset_train.num_atoms_distribution.multinomial(num_samples=batch_size, replacement=True)
        
        # mirage infusion code
        if "miad:add_mirage_atoms_upto" in os.environ['MODIFICATIONS_FIELD']:
            N_m = int(os.environ['MODIFICATIONS_FIELD'].split('miad:add_mirage_atoms_upto')[1].split('+')[0])
            num_atoms = torch.tensor([ N_m ] * batch_size)
        
        x0 = [ None, None, None ]
        crystal_batch = Batch.from_data_list([ Data(num_atoms=torch.LongTensor([n]), num_nodes=n) for n in num_atoms ])
        if device == 'cuda':
            device = torch.cuda.current_device()
        return {
            'x0'         : x0,
            'batch_size' : batch_size,
            'num_atoms'  : crystal_batch.num_nodes,
            'device'     : device,
            'batch'      : crystal_batch.to(device)
        }




    def save_batch(self, batch, folder, rank, tag=''):
        already_sampled = -1
        for f in os.listdir(folder):
            if int(f.split('_')[-3].split('_')[0]) == rank:
                already_sampled = max(already_sampled, int(f.split('_')[-2].split('_tag')[0]))
        
        lengths, angles = lattice_to_lengths_and_angles(batch['x0_prediction'][0])
        lengths = lengths.cpu()
        angles = angles.cpu()
        num_atoms = batch['batch'].num_atoms
        frac_coords = batch['x0_prediction'][1].cpu()
        atom_types = batch['x0_prediction'][2].cpu()

        # mirage infusion code
        if "miad:add_mirage_atoms_upto" in os.environ['MODIFICATIONS_FIELD']:
            mirage_type = 0
            mask = (atom_types != mirage_type)
            num_atoms = mask.reshape(batch['batch_size'], -1).to(torch.long).sum(dim=-1)
            frac_coords = frac_coords[mask]
            atom_types = atom_types[mask]

        num_atoms_offset = 0
        for i in range(batch['batch_size']):
            structure = Structure(
                lattice=Lattice.from_parameters(
                    *(lengths[i].tolist() + angles[i].tolist())),
                species=atom_types[num_atoms_offset:num_atoms_offset+num_atoms[i]],
                coords=frac_coords[num_atoms_offset:num_atoms_offset+num_atoms[i]],
                coords_are_cartesian=False
            )
            num_atoms_offset += num_atoms[i]
            structure.to(filename=os.path.join(folder, f"crystal_{rank}_{already_sampled+1+i}_tag{tag}.cif"))
        pass






class CrystalDataset(Dataset):

    cont_dtype = torch.float32
    disc_dtype = torch.long

    def __init__(self, config, part):
        self.name = config.name
        self.dataset_part = part
        self.properties = config.properties
        self.preprocess_config = config.crystal_preprocess

        path_to_dataset = os.path.join(get_repo_root(), 'datasets', self.name)
        if not os.path.exists(path_to_dataset):
            raise Exception(
                f"No such dataset: {self.name}"
            )
        path_to_dataset_part = os.path.join(path_to_dataset, self.dataset_part+'.csv')
        path_to_preprocessed_dataset_part = os.path.join(path_to_dataset, self.dataset_part+'.pt')
        if not (os.path.exists(path_to_dataset_part) or os.path.exists(path_to_preprocessed_dataset_part)):
            raise Exception(
                f"No such part {self.dataset_part}.csv or its already prepreprocessed version" + 
                f" {self.dataset_part}.pt in dataset: {self.name}"
            )
        self.preprocessed_data, self.param_scalers, self.prop_scalers = self.preprocess_dataset_part(
            path_to_dataset_part,
            path_to_preprocessed_dataset_part
        )
        self.num_atoms_distribution = torch.zeros(100, dtype=CrystalDataset.cont_dtype)
        for crystal_dict in self.preprocessed_data:
            self.num_atoms_distribution[crystal_dict['parameters']['num_atoms']] += 1
        self.num_atoms_distribution = self.num_atoms_distribution / self.num_atoms_distribution.sum()


    def preprocess_dataset_part(
        self, 
        path_to_dataset_part,
        path_to_preprocessed_dataset_part
    ):
        if os.path.exists(path_to_preprocessed_dataset_part):
            preprocessed_data = torch.load(path_to_preprocessed_dataset_part)
        
        else:
            df = pd.read_csv(path_to_dataset_part)

            # check data validity

            if not ('cif' in df.columns):
                raise Exception(f"No \'cif\' column in {path_to_dataset_part}")
            if not ('material_id' in df.columns):
                print(f"No \'material_id\' column in {path_to_dataset_part}. IDs will be created")
                df['material_id'] = [ f'index-{i}' for i in range(len(df)) ]
            for prop_info in self.properties:
                prop_name, _, _ = prop_info
                if not (prop_name in df.columns):
                    raise Exception(f"No \'{prop}\' column in {path_to_dataset_part}")

            # preprocess crystals

            def preprocess_crystal(row):
                try:
                    # crystal parameters
                    crystal = Structure.from_str(row['cif'], fmt='cif')
                    if ('primitive' in self.preprocess_config) and self.preprocess_config.primitive:
                        crystal = crystal.get_primitive_structure()
                    if ('niggli' in self.preprocess_config) and self.preprocess_config.niggli:
                        crystal = crystal.get_reduced_structure()
                    canonical_crystal = Structure(
                        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
                        species=crystal.species,
                        coords=crystal.frac_coords,
                        coords_are_cartesian=False,
                    )
                    # crystal properties
                    properties = {}
                    for prop_info in self.properties:
                        prop_name, prop_dtype, _ = prop_info
                        if prop_dtype == 'discrete':
                            dtype = CrystalDataset.disc_dtype
                        if prop_dtype == 'continious':
                            dtype = CrystalDataset.cont_dtype
                        if torch.is_tensor(row[prop_name]):
                            if row[prop_name].dtype != dtype:
                                raise TypeError(
                                    f"Type of property {prop_name} value is not the same as selected property dtype.")
                            value = row[prop_name]
                        else:
                            value = torch.tensor(row[prop_name], dtype=dtype)
                        properties[prop_name] = value
                
                except Exception as e:
                    print(f"While processing cif of crystal {row['material_id']} catch:\n    {e}", flush=True)
                    return None
                
                return {
                    'id' : row['material_id'],
                    'cif' : row['cif'],
                    'parameters' : {
                        'lattice'     : torch.tensor(
                            canonical_crystal.lattice.matrix, dtype=CrystalDataset.cont_dtype
                        )[None,:],
                        'lengths'     : torch.tensor(
                            canonical_crystal.lattice.parameters[:3], dtype=CrystalDataset.cont_dtype
                        )[None,:],
                        'angles'      : torch.tensor(
                            canonical_crystal.lattice.parameters[3:], dtype=CrystalDataset.cont_dtype
                        )[None,:],
                        'frac_coords' : torch.tensor(
                            canonical_crystal.frac_coords, dtype=CrystalDataset.cont_dtype
                        ),
                        'cart_coords' : torch.tensor(
                            canonical_crystal.cart_coords, dtype=CrystalDataset.cont_dtype
                        ),
                        'atom_types'  : torch.tensor(
                            canonical_crystal.atomic_numbers, dtype=CrystalDataset.disc_dtype
                        ),
                        'num_atoms'   : torch.tensor(
                            len(canonical_crystal.atomic_numbers), dtype=CrystalDataset.disc_dtype
                        )
                    },
                    'properties' : { prop_name : prop_value for prop_name, prop_value in properties.items() }
                }

            preprocessed_data = p_umap(
                preprocess_crystal,
                [ df.iloc[idx] for idx in range(len(df)) ],
                num_cpus=None
            )
            only_succesfully_preprocessed_data = []
            for crystal_dict in preprocessed_data:
                if not (crystal_dict is None):
                    only_succesfully_preprocessed_data.append(crystal_dict)
            torch.save(only_succesfully_preprocessed_data, path_to_preprocessed_dataset_part)
            preprocessed_data = only_succesfully_preprocessed_data

        # set scalers
        self.rescaled_dataset = False

        param_scalers = {}
        for param_name in [ 'lattice', 'lengths', 'angles', 'cart_coords' ]:
            param_arr = [ crystal_dict['parameters'][param_name] for crystal_dict in preprocessed_data ]
            param_scalers[param_name] = init_scaler(
                scaler_name="StandardContiniousScaler", 
                data_sample=param_arr
            )

        prop_scalers = {}
        for prop_info in self.properties:
            prop_name, prop_dtype, scaler_name = prop_info
            try:
                prop_arr = [ crystal_dict['properties'][prop_name] for crystal_dict in preprocessed_data ]
                prop_scalers[prop_name] = init_scaler(
                    scaler_name=scaler_name,
                    data_sample=prop_arr
                )
            except Exception as e:
                raise Exception(f"While processing prop {prop_name} catch:\n    {e}", flush=True)

        return preprocessed_data, param_scalers, prop_scalers


    def set_crystal_parameters_scalers(self, param_scalers):
        self.param_scalers = param_scalers


    def set_crystal_properties_scalers(self, prop_scalers):
        self.prop_scalers = prop_scalers


    def rescale_crystals_accordingly_to_current_scalers(self):
        # remove scale from data
        self.rescaled_dataset = True
        for crystal_dict in self.preprocessed_data:
            for param_name, param_scaler in self.params_scaler.items():
                crystal_dict['parameters'][param_name] = param_scaler.rescale(crystal_dict['parameters'][param_name])
            for prop_name, prop_scaler in self.prop_scaler.items():
                crystal_dict['properties'][prop_name] = prop_scaler.rescale(crystal_dict['properties'][prop_name])


    def scaleup_crystals_accordingly_to_current_scalers(self):
        # move data to its original scale
        self.rescaled_dataset = False
        for crystal_dict in self.preprocessed_data:
            for param_name, param_scaler in self.params_scaler.items():
                crystal_dict['parameters'][param_name] = param_scaler.scaleup(crystal_dict['parameters'][param_name])
            for prop_name, prop_scaler in self.prop_scaler.items():
                crystal_dict['properties'][prop_name] = prop_scaler.scaleup(crystal_dict['properties'][prop_name])


    def __len__(self):
        return len(self.preprocessed_data)


    def __getitem__(self, index):
        crystal_dict = self.preprocessed_data[index]
        crystal_dict['data'] = Data(
            # crystal features
            lattice=crystal_dict['parameters']['lattice'],
            lengths=crystal_dict['parameters']['lengths'],
            angles=crystal_dict['parameters']['angles'],
            frac_coords=crystal_dict['parameters']['frac_coords'],
            atom_types=crystal_dict['parameters']['atom_types'],
            num_atoms=crystal_dict['parameters']['num_atoms'],
            **{ 
                f"prop_{prop_name}" : prop_value 
                for prop_name, prop_value in crystal_dict['properties'].items() 
            },
            # for correct collate
            num_nodes=crystal_dict['parameters']['num_atoms']
        ).cpu()

        # mirage infusion code
        if "miad:add_mirage_atoms_upto" in os.environ['MODIFICATIONS_FIELD']:
            N_m = int(os.environ['MODIFICATIONS_FIELD'].split('miad:add_mirage_atoms_upto')[1].split('+')[0])
            mirage_type = 0
            crystal_dict['data'].frac_coords = torch.vstack((
                crystal_dict['data'].frac_coords,
                torch.rand((N_m - crystal_dict['data'].num_atoms, 3), dtype=CrystalDataset.cont_dtype)
            ))
            crystal_dict['data'].atom_types = torch.hstack((
                crystal_dict['data'].atom_types, 
                mirage_type + torch.zeros((N_m - crystal_dict['data'].num_atoms,), dtype=CrystalDataset.disc_dtype)
            ))
            crystal_dict['data'].num_atoms = N_m
            crystal_dict['data'].num_nodes = N_m

        return crystal_dict['data']