from collections import Counter
import argparse
import os
import json

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd

from torch_geometric.data import Batch
from data.crystal_datasets.cryst_utils import lattices_to_params_shape
from collections import defaultdict

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from pyxtal import pyxtal

import pickle

from metrics.crystal_metrics.metrics_utils import (
    smact_validity, 
    structure_validity,
    #load_config, 
    #load_data, 
    get_crystals_list,
    compute_cov
)
from metrics.compress_results import compress_data
from saving_utils.get_repo_root import get_repo_root




CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20'     : np.array([ -3.17562208,   -2.82196882,   -2.52814761 ]),
    'carbon24' : np.array([ -154.527093, -154.45865733, -154.44206825 ]),
    'perov5'   : np.array([  0.43924842,    0.61202443,     0.7364607 ]),
}

COV_Cutoffs = {
    'mp20'     : { 'struc': 0.4, 'comp' : 10. },
    'carbon24' : { 'struc': 0.2, 'comp' :  4. },
    'perov5'   : { 'struc': 0.2, 'comp' :  4. },
}




class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if (
            np.isnan(self.lengths).any() or 
            np.isnan(self.angles).any() or
            np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'
            

                
    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

        
    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)    
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

        
    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        
        self.struct_fp = np.array(site_fps).mean(axis=0)

        
        

class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

        
    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {
            'match_rate' : match_rate,
            'rms_dist' : mean_rms_dist,
            'list_rms_dist' : rms_dists
        }     

    
    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics

    
    

class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

        
    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        # debug
        full_rms_dists = []
        # debug
        
        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
            # debug
            full_rms_dists.append(tmp_rms_dists)
            # debug
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {
            'match_rate' : match_rate,
            'rms_dist' : mean_rms_dist,
            
            # debug
            'full_rms_dists' : full_rms_dists
            # debug
        }    

    
    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



    
class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.valid_samples = [ c for c in pred_crys if c.valid ]
        n_samples = len(gt_crys) 
        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            self.valid_samples = valid_crys
            print(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}',
                flush=True
            )
            #raise Exception(
            #    f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}'
            #)
        pass

            
    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {
            'comp_valid'   : comp_valid,
            'struct_valid' : struct_valid,
            'valid'        : valid
        }


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {
            'wdist_density' : wdist_density
        }


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species)) for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {
            'wdist_num_elems' : wdist_num_elems
        }

        
    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    
    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics




def get_crystal_array_list(file_path, batch_idx=0):
    #data = load_data(file_path)
    data = torch.load(os.path.join(get_repo_root(), 'results', file_path))
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i]
            )
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'][0],
            data['atom_types'][0],
            data['lengths'][0],
            data['angles'][0],
            data['num_atoms'][0]
        )
        if 'input_data_batch' in data:
            data.pop('input_data_batch')
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx]
        )

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crys_array_list = get_crystals_list(
                batch['frac_coords'], 
                batch['atom_types'], 
                batch['lengths'],
                batch['angles'], 
                batch['num_atoms']
            )
        else:
            true_crys_array_list = get_crystals_list(
                batch.frac_coords, 
                batch.atom_types, 
                batch.lengths,
                batch.angles, 
                batch.num_atoms
            )
    else:
        true_crys_array_list = None

    return crys_array_list, true_crys_array_list




def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 




def evaluate(config):
    """        
    ConfigDict({
        tasks -> tuple = (str, str): name of tasks
        file_path -> str: path to file from <results> dir
        dataset_part -> str: part
        eval_model_name -> str: dataset
        multi_eval -> bool: more than 1 object or not
    })
    """
    fname = compress_data(config.file_path, config.multi_eval)
    
    all_metrics = {}

    if 'gen' in config.tasks:
        crys_array_list, _ = get_crystal_array_list(fname, batch_idx=-2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        csv = pd.read_csv(os.path.join(
            get_repo_root(), 'datasets', config.eval_model_name, config.dataset_part+'.csv'
        ))
        gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        gen_evaluator = GenEval(gen_crys, gt_crys, eval_model_name=config.eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    else:
        batch_idx = -1 if config.multi_eval else 0
        crys_array_list, true_crys_array_list = get_crystal_array_list(fname, batch_idx=batch_idx)
        
        # reorder results
        data_pack = torch.load(fname)
        if 'crystal_indexes' in data_pack:
            index = np.array(data_pack['crystal_indexes'])

            true_crys_array_list = np.array(true_crys_array_list)
            empty = np.zeros_like(true_crys_array_list, dtype=np.dtype('O'))
            empty[index] = true_crys_array_list
            true_crys_array_list = empty.copy()
            
            if config.multi_eval:
                crys_array_list = np.array(crys_array_list).T
            else:
                crys_array_list = np.array(crys_array_list)
            empty = np.zeros_like(crys_array_list, dtype=np.dtype('O'))
            empty[index] = crys_array_list
            crys_array_list = empty.copy()
            if config.multi_eval:
                crys_array_list = np.array(crys_array_list).T
            else:
                crys_array_list = np.array(crys_array_list)
            
            num_uniq_crys = np.unique(index).shape[0]
            true_crys_array_list = true_crys_array_list[:num_uniq_crys]
            crys_array_list = crys_array_list[...,:num_uniq_crys]
            
        gt_crys = p_map(lambda x: Crystal(x), true_crys_array_list)
        if not config.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   
    
        if 'csp' in config.tasks: 
            if config.multi_eval:
                rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
            else:
                rec_evaluator = RecEval(pred_crys, gt_crys)
            recon_metrics = rec_evaluator.get_metrics()
            all_metrics.update(recon_metrics)
            
        new_metrics = { 'match_rate' : all_metrics['match_rate'], 'rms_dist' : all_metrics['rms_dist'] }
        print(new_metrics)
    
    
    metrics_out_file = os.path.join(
        get_repo_root(), 
        'results',
        Path(fname).parent.parent, 
        'eval_metrics.json'
    )

    # only overwrite metrics computed in the new run.
    """
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(new_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(new_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(new_metrics, f)
    """
    return all_metrics

    
