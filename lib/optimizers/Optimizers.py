import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist

import numpy as np




class Optimizer:
    
    
    def __init__(self, model, optimization_config):
        self.model = model
        self.optimization_config = optimization_config
        
        optim_method = {
            'Adam' : optim.Adam,
            'SGD' : optim.SGD
        }
        self.optimizer = optim_method[optimization_config.optimizer['method']](
            self.model.parameters(), **optimization_config.optimizer['config'])
        self.iteration = 0

        if 'schedulers' in optimization_config:
            scheduler_method = {
                'Exponential'       : lr_scheduler.ExponentialLR,
                'Linear'            : lr_scheduler.LinearLR,
                'Multiplicative'    : lr_scheduler.MultiplicativeLR,
                'ReduceLROnPlateau' : lr_scheduler.ReduceLROnPlateau,
                'Lambda'            : lr_scheduler.LambdaLR
            }
            self.schedulers = []
            for scheduler_type, scheduler_config, iters_bounds in optimization_config.schedulers:
                self.schedulers.append((
                    iters_bounds,
                    scheduler_method[scheduler_type](self.optimizer, **scheduler_config)
                ))
                
        if 'clip_grad_norm' in optimization_config:
            self.grad_norm = optimization_config.clip_grad_norm
            
        if 'clip_grad_value' in optimization_config:
            self.grad_value = optimization_config.clip_grad_value
        
        if 'ema' in optimization_config:
            ema_config = optimization_config.ema
            self.ema_rate = ema_config['ema_rate']
            self.initial_acceleration = ema_config['initial_acceleration']
            self.ema_parameters = [ 
                p.clone().detach() for p in self.model.parameters() if p.requires_grad
            ]
        pass
    
    
    
    
    def state_dict(self):
        state_dict = {
            'config'                 : self.optimization_config,
            'optimizer_state_dict'   : self.optimizer.state_dict(),
            'iteration'              : self.iteration
        }
        if hasattr(self, 'schedulers'):
            state_dict['schedulers_state_dicts'] = [ sched[1].state_dict() for sched in self.schedulers ]
        if hasattr(self, 'ema_parameters'):
            state_dict['model_parameters_copy'] = self.model_parameters_copy
        return state_dict
    
    
    
    
    def load_state_dict(self, model, state_dict):
        self.__init__(model, state_dict['config'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.iteration = state_dict['iteration']
        if hasattr(self, 'schedulers'):
            for i, sched in enumerate(self.schedulers):
                sched[1].load_state_dict(state_dict['schedulers_state_dicts'][i])
        if hasattr(self, 'ema_parameters'):
            # necessary self.ema_parameters already set in __init__
            self.model_parameters_copy = state_dict['model_parameters_copy']
        pass
                
    
    
    
    def optimizing_step(self, loss, logger):
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.log_grad_norm(logger)
        
        if hasattr(self, 'grad_norm') or hasattr(self, 'grad_value'):
            self.clip_grad()
            
        self.optimizer.step()
        
        self.log_weight_norm(logger)
        
        if hasattr(self, 'ema_rate'):
            self.update_ema_weights(logger)
            
        if hasattr(self, 'schedulers'):
            self.update_lr(logger)
            
        self.iteration += 1
        pass
    
    
    

    def log_grad_norm(self, logger):
        grad = [
            param.grad.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None
        ]
        grad = torch.cat(grad)
        logger.add("grad:l2-norm", grad.norm().item(), stack_after_epoch=True)
        logger.add("grad:max-norm", abs(grad).max().item(), stack_after_epoch=True)
        pass
    
    
    
    
    def clip_grad(self):
        if hasattr(self, 'grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.grad_norm
            )
        if hasattr(self, 'grad_value'):
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                clip_value=self.grad_value
            )
        pass
    
    
    
    
    def log_weight_norm(self, logger):
        weight = [
            param.data.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None
        ]
        weight = torch.cat(weight)
        logger.add("weight:l2-norm", weight.norm().item(), stack_after_epoch=True)
        logger.add("weight:max-norm", abs(weight).max().item(), stack_after_epoch=True)
        pass
    
    
    
    
    def update_ema_weights(self, logger):
        ema_rate = self.ema_rate
        # acceleration ema_rate: [ 0.18, 0.25, 0.30, ..., 0.9999, 0.9999, ... ]
        # until dynamical ema_rate rich initial ema_rate
        if self.initial_acceleration:
            ema_rate = min(
                ema_rate, 
                (1 + self.iteration) / (10 + self.iteration)
            )
        logger.add('ema_rate', ema_rate)
        # update ema params using new model params
        with torch.no_grad():
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            for ema_param, model_param in zip(self.ema_parameters, model_parameters):
                ema_param.to(self.model.device)
                ema_param.sub_((1 - ema_rate) * (ema_param - model_param))
        pass
    
    
    
    
    def switch_to_ema(self):
        if hasattr(self, 'ema_rate'):
            self.model_parameters_copy = [ p.clone().cpu().detach()
                for p in self.model.parameters() if p.requires_grad ]
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            for ema_param, model_param in zip(self.ema_parameters, model_parameters):
                ema_param.to(self.model.device)
                model_param.data.copy_(ema_param.data)
                ema_param.cpu()
        pass
    
    
    
    
    def switch_from_ema(self):
        if hasattr(self, 'ema_rate'):
            model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            for copy_param, model_param in zip(self.model_parameters_copy, model_parameters):
                copy_param.to(self.model.device)
                model_param.data.copy_(copy_param.data)
                copy_param.cpu()
        pass

    
    
    
    def update_lr(self, logger):
        for iters_bounds, scheduler in self.schedulers:
            if iters_bounds[0] <= self.iteration < iters_bounds[1]: 
                scheduler.step()
                logger.add('lr', scheduler.get_lr())
        pass
    
    
    
    
    def cuda(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(device)
        pass
    
    
    
    
    def cpu(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        pass