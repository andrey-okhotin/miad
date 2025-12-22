import os
import torch

from diffusion.crystal_diffusions.utils import (
    SinusoidalTimeEmbeddings,
    TimeDistribution,
    mean_interleave
)

from diffusion.lat_diffusions.ddpm import DDPM
from diffusion.lat_diffusions.flow_matching import FM
from diffusion.lat_diffusions.flow_matching_lenang import FM_LenAng

from diffusion.frac_diffusions.wrapnorm_diff import WrappedNormal
from diffusion.frac_diffusions.periodic_flow_matching import PFM

from diffusion.type_diffusions.ddpm_onehot import DDPM_onehot
from diffusion.type_diffusions.d3pm import D3PM




class CrystalGen:
    
    """
    DESCRIPTION:
        
    INIT:
        diffusion_config = ConfigDict({
            'task' : str - ('csp-...' or 'gen-...')
            'lat_diffusion' : str,
            'lat_diffusion_config' : ConfigDict,
            'frac_diffusion' : str,
            'frac_diffusion_config' : ConfigDict,
            'type_diffusion' : str,
            'type_diffusion_config' : ConfigDict
        })
        
    """

    def __init__(self, diffusion_config, logger):
  
        # base settings
        
        self.config = diffusion_config
        self.logger = logger
        self.device = 'cpu'
        self.cont_time = self.config.cont_time
        self.num_steps = self.config.num_steps
        self.time_distribution = TimeDistribution(self.num_steps, self.cont_time)
        self.time_embedding = SinusoidalTimeEmbeddings(256)
        
        # base diffusion models
        
        switch_lat = {
            'ddpm'      : DDPM,
            'fm'        : FM,
            'fm_lenang' : FM_LenAng
        }
        self.lat_diffusion = switch_lat[self.config.lat_diffusion.method](self.config)
        switch_frac = {
            'wrapped_normal'    : WrappedNormal,
            'pfm'               : PFM
        }
        self.frac_diffusion = switch_frac[self.config.frac_diffusion.method](self.config)
        self.gen_type = ('gen' in self.config.task)
        if self.gen_type:
            switch_type = {
                'ddpm_onehot' : DDPM_onehot,
                'd3pm'        : D3PM
            }
            self.type_diffusion = switch_type[self.config.type_diffusion.method](self.config)
        else:
            self.type_diffusion = None
        pass


    
    
    def forward_step_sample(self, x0, t, batch):
        l0, f0, a0 = x0
        lt = self.lat_diffusion.forward_step_sample(l0, t[0], batch)
        ft = self.frac_diffusion.forward_step_sample(f0, t[1], batch)
        at = self.type_diffusion.forward_step_sample(a0, t[1], batch) if self.gen_type else a0
        xt = [ lt, ft, at ]
        return xt

    
    
    
    def reverse_step_sample(self, xt, t, model, batch):
        """
        Standard form of generation step, but it can differs depend on the particular
        model. As example DiffCSP calls score estimator twice per generation step.
        """
        lt, ft, at = xt
        batch['prediction'] = self.model_prediction(xt, t, model, batch)
        l_pred, f_pred, a_pred = batch['prediction']
        lt_1 = self.lat_diffusion.reverse_step_sample(l_pred, lt, t[0], batch)
        ft_1 = self.frac_diffusion.reverse_step_sample(f_pred, ft, t[1], batch)
        at_1 = self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch) if self.gen_type else at
        xt_1 = [ lt_1, ft_1, at_1 ]
        return xt_1
    
    
    
    
    def prior_sample(self, batch):
        xT = [
            self.lat_diffusion.prior_sample(batch),
            self.frac_diffusion.prior_sample(batch),
            self.type_diffusion.prior_sample(batch) if self.gen_type else batch['batch'].atom_types
        ]
        return xT
    
    
    
    
    def model_prediction(self, xt, t, model, batch):
        lt, ft, at = xt
        nn_pred = model(
            t[0],
            self.time_embedding(1000 * (t[0] / self.num_steps) + 1),
            at,
            ft,
            lt,
            batch['batch'].num_atoms,
            batch['batch'].batch
        )
        l_pred = nn_pred[0]
        f_pred = nn_pred[1]
        a_pred = nn_pred[2] if self.gen_type else None
        pred = [ l_pred, f_pred, a_pred ]
        return pred
    
    
    
    
    def get_x0_prediction(self, pred, xt, t, batch):
        l_pred, f_pred, a_pred = pred
        lt, ft, at = xt
        x0_pred = [
            self.lat_diffusion.get_x0_prediction(l_pred, lt, t[0], batch),
            self.frac_diffusion.get_x0_prediction(f_pred, ft, t[1], batch),
            self.type_diffusion.get_x0_prediction(a_pred, at, t[1], batch) if self.gen_type else at.clone().detach()
        ]
        return x0_pred


    
    
    def train_step(self, batch, model, mode):
        batch['t'] = self.time_distribution.sample(
            batch,
            mode
        )
        batch['xt'] = self.forward_step_sample(
            batch['x0'], 
            batch['t'],
            batch
        )
        batch['prediction'] = self.model_prediction(
            batch['xt'], 
            batch['t'],
            model,
            batch
        )
        loss_lat = self.lat_diffusion.loss(batch)
        loss_frac = self.frac_diffusion.loss(batch)
        
        if "insert_latloss_" in os.environ['MODIFICATIONS_FIELD']:
            lat_coef = float(os.environ['MODIFICATIONS_FIELD'].split("insert_latloss_")[1].split("_coef")[0])
            loss_lat *= lat_coef
        if "insert_fracloss_" in os.environ['MODIFICATIONS_FIELD']:
            frac_coef = float(os.environ['MODIFICATIONS_FIELD'].split("insert_fracloss_")[1].split("_coef")[0])
            loss_frac *= frac_coef
        
        loss_lat_val, loss_frac_val = loss_lat.mean(), loss_frac.mean()
        batch['loss'] = loss_lat_val + loss_frac_val
        if self.gen_type:
            loss_type = self.type_diffusion.loss(batch)
            if "insert_typeloss_" in os.environ['MODIFICATIONS_FIELD']:
                type_coef = float(os.environ['MODIFICATIONS_FIELD'].split("insert_typeloss_")[1].split("_coef")[0])
                loss_type *= type_coef
            loss_type_val = loss_type.mean()
            batch['loss'] += loss_type_val
        
        # logs
        
        t = torch.clip(batch['t'][0].clone().to(torch.long).cpu(), 0, 999)
        # logs - lat
        self.logger.add(f"loss:lattice:{mode}", loss_lat_val.item(), stack_after_epoch=True)
        loss_lat_t = torch.zeros((self.num_steps), dtype=loss_lat.dtype, device='cpu')
        loss_lat_t[t] = loss_lat.clone().cpu()
        self.logger.add(f"loss:lattice4time:{mode}", loss_lat_t,
                   stack_after_epoch=True)
        # logs - frac
        self.logger.add(f"loss:coord:{mode}", loss_frac_val.item(), stack_after_epoch=True)
        loss_frac_t = torch.zeros((self.num_steps), dtype=loss_frac.dtype, device='cpu')
        loss_frac_t[t] = mean_interleave(
            loss_frac.clone().cpu(), batch['batch'].num_atoms)
        self.logger.add(f"loss:coord4time:{mode}", loss_frac_t,
                   stack_after_epoch=True)
        # logs - type
        if self.gen_type:
            self.logger.add(f"loss:type:{mode}", loss_type_val.item(), stack_after_epoch=True)
            loss_type_t = torch.zeros((self.num_steps), dtype=loss_type.dtype, device='cpu')
            loss_type_t[t] = mean_interleave(
                loss_type.clone().cpu(), batch['batch'].num_atoms)
            self.logger.add(f"loss:type4time:{mode}", loss_type_t,
                       stack_after_epoch=True)
        return batch

    
    
    
    def sampling_procedure(self, model, batch, progress_printer):
        batch['xt'] = self.prior_sample(batch)
        reverse_time_iterator = self.time_distribution.reverse_time_iterator(
            batch, start_from=self.num_steps-1
        )
        for t_vector in reverse_time_iterator:
            batch['t'] = self.time_distribution.to_cuda(t_vector, batch)
            t_value = batch['t'][0][0].item()
            progress_printer(t_value)
            batch['xt'] = self.reverse_step_sample(
                batch['xt'],
                batch['t'],
                model,
                batch
            )
        batch['xt'] = self.output_transform(batch['xt'], batch) 
        batch['x0_prediction'] = batch['xt']
        return batch




    def output_transform(self, x0, batch):
        return [
            self.lat_diffusion.output_transform(x0[0], batch),
            self.frac_diffusion.output_transform(x0[1], batch),
            self.type_diffusion.output_transform(x0[2], batch) if self.gen_type else x0[2]
        ]
        
            
    """
    Device utils.
    """
    
    
    def to(self, device):
        if hasattr(self, "rp_config"):
            self.repulsive_potential.to(device)
        submodels = [ self ]
        for model in submodels:
            for attr, value in model.__dict__.items():
                if torch.is_tensor(value):
                    model.__dict__[attr] = value.to(device)
                elif hasattr(value, '__dict__'):
                    submodels.append(value)
        self.device = device
        pass

    
    def cuda(self, device):
        self.to(device)
        pass
    
    
    def cpu(self):
        self.to('cpu')
        pass