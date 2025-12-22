import os
import shutil
from pathlib import Path

import json
import time
from ml_collections import ConfigDict

import torch

from models.init_model import save_model_to_path, load_model_from_path
from saving_utils.get_repo_root import get_repo_root




class Checkpoints:
    
    def __init__(self, config, process):
        """
        INPUT:
        
        <>  config = {
                'saving_freq' : (int) every 'saving_freq' epochs Chekpoints objects
                    would save model parameters,
                'path' : (str) path where model config and model parameters
                    will be saved
            }
        
        """
        self.process = process
        self.checkpoints_path = os.path.join(get_repo_root(), 'checkpoints', config['path'])
        if not self.process.is_root_process:
            return None
        period = config['saving_freq']
        self.saving_rule = lambda epoch: isinstance(epoch, int) and (epoch % period == 0)
        if os.path.exists(self.checkpoints_path):
            if 'reset_previous' in config.keys() and config['reset_previous']:
                shutil.rmtree(self.checkpoints_path)
                os.mkdir(self.checkpoints_path)
            else:
                print('Checkpoints path already exists.')
                print('Existing checkpoints in path would be replaced by their duplicates.')
        else:
            os.mkdir(self.checkpoints_path)
        pass
        
    
    def save_checkpoint(self, model, optimizer, epoch, tag='model'):
        if not self.process.is_root_process:
            return None
        if self.saving_rule(epoch):
            optimizer.switch_to_ema()
            model_name = model.model_config.model_name
            save_model_to_path(
                model,
                os.path.join(self.checkpoints_path, f'{model_name}_epoch{epoch}_{tag}.pt')
            )
            torch.save(
                optimizer.state_dict(), 
                os.path.join(self.checkpoints_path, f'Opt_{model_name}_epoch{epoch}_{tag}.pt')
            )
            config_name = model_name + f'_config.json'
            with open(os.path.join(self.checkpoints_path, config_name), 'w') as json_file:
                json.dump(dict(model.model_config.items()), json_file)
            optimizer.switch_from_ema()
        pass

    
    def load_checkpoint(self, model, optimizer, epoch, tag='model'):
        model_name = model.model_config.model_name
        path = os.path.join(self.checkpoints_path, f'{model_name}_epoch{epoch}_{tag}.pt')
        model = load_model_from_path(path)
        state_dict = torch.load(
            os.path.join(self.checkpoints_path, f'Opt_{model_name}_epoch{epoch}_{tag}.pt'),
            map_location='cpu'
        )
        optimizer.load_state_dict(model, state_dict)
        pass
    
        
        
        