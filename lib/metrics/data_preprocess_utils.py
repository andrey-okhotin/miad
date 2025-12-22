import os

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

from saving_utils.get_repo_root import get_repo_root




def preprocess_folder_with_cif(
    path,
    num_crystals=None,
    print_invalid_crystals=False
):
    if not os.path.exists(path):
        raise Exception(f"Path '{path}' doesn't exist.")
    
    structures = {
        'cif_name'  : [],
        'valid'     : [],
        'structure' : []
    }

    for i, fname in enumerate(os.listdir(path)):
        if not (num_crystals is None) and i == num_crystals:
            break
        if not (fname[-4:] == '.cif'):
            raise TypeError(f"Found not a .cif file: '{fname}'")
        structures['cif_name'].append(fname.replace('.cif', ''))
        try:
            cif_str = CifParser(os.path.join(path, fname))._cif.__dict__['orig_string']
            structures['structure'].append(Structure.from_str(cif_str, fmt='cif'))
            structures['valid'].append(True)
        except Exception as e:
            structures['structure'].append(None)
            structures['valid'].append(False)
            if print_invalid_crystals:
                print(f"while processing file {fname} catch:\n    {e}", flush=True)
    
    return structures
