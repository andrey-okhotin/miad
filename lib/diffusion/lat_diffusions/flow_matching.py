import os
import torch




class FM:
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.lat_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.device = 'cpu'
        
        self.step = 1. / self.num_steps
        self.parameterization = diffusion_config.lat_diffusion.parameterization
        self.step_coef = torch.ones(self.num_steps, dtype=torch.float32)[:,None,None]
        self.to_domain = lambda x: x
        pass


    def output_transform(self, x0, batch):
        return x0
    
    
    def forward_step_sample(self, x0, t, batch):
        eps = torch.randn_like(x0)
        step = (1 + t[:,None,None]) * self.step
        xt = (1 - step) * x0 + step * eps
        if self.parameterization == 'eps':
            self.ut = (-1) * eps
        elif self.parameterization == 'v':
            self.ut = x0 - eps
        return xt
    
    
    def reverse_step_sample(self, pred, xt, t, batch):
        if self.parameterization == 'eps':
            if t[0] >= 999:
                return torch.randn_like(xt)
            eps_pred = (-1) * pred
            step = (1 + t[:,None,None]) * self.step
            x0_pred = (xt - step * eps_pred) / (1 - step)
            vt = x0_pred - eps_pred
        elif self.parameterization == 'v':
            vt = pred
        xt_1 = xt + self.step_coef[t] * self.step * vt
        return xt_1
    
    
    def prior_sample(self, batch):
        return torch.randn((batch['batch_size'], 3, 3), dtype=torch.float32, device=batch['device'])
    
    
    def loss(self, batch):
        vt = batch['prediction'][0]
        l2 = ((vt - self.ut)**2).reshape(-1,9).mean(dim=1)
        return l2
    
    
    def get_x0_prediction(self, pred, xt, t, batch):
        step = (1 + t[:,None,None]) * self.step
        if self.parameterization == 'eps':
            if t[0] >= 999:
                return torch.randn_like(xt)
            eps_pred = (-1) * pred
            x0_pred = (xt - step * eps_pred) / (1 - step)
            vt = x0_pred - eps_pred
        elif self.parameterization == 'v':
            vt = pred
        return xt + step * vt