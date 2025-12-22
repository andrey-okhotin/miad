import os
import torch

from saving_utils.get_repo_root import get_repo_root




class PFM:
    
    """
    DESCRIPTION:
        Periodic Flow Matching. Used for experiments on Crystal Data
        for fractional coordinates.
        
    """
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.frac_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.final_t = diffusion_config.num_steps
        self.step_coef = torch.ones(self.final_t, dtype=torch.float32)[:,None]
        self.step = torch.tensor(1. / self.final_t)
        self.to_domain = lambda x: x
        self.default_loss_scale = 10
        pass
    

    def output_transform(self, x0, batch):
        return x0
    
    
    def forward_step_sample(self, x0, t, batch):
        xT_minus_x0 = torch.rand_like(x0) - 0.5
        step = (1 + t[:,None]) * self.step
        xt = (x0 + step * xT_minus_x0) % 1.
        self.ut = (-1) * xT_minus_x0
        return xt
    
    
    def reverse_step_sample(self, vt, xt, t, batch):
        xt_1 = (xt + self.step_coef[t] * self.step * vt) % 1.
        return xt_1
    
    
    def prior_sample(self, batch):
        return torch.rand(batch['num_atoms'], 3, device=batch['device'])
        

    def loss(self, batch):
        vt = batch['prediction'][1]
        l2 = ((vt - self.ut)**2).reshape(-1,3).mean(dim=1)
        return l2 * self.default_loss_scale
    
    
    def get_x0_prediction(self, vt, xt, t, batch):
        step = (1 + t[:,None]) * self.step
        return (xt + step * vt) % 1.