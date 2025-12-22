import torch

from diffusion.scheduler import scheduler

import os
from saving_utils.get_repo_root import get_repo_root



class DDPM:
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.lat_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        
        cumprod_alphas_t, сa_t = scheduler(self.config.scheduler, self.num_steps)
        
        eps = 1e-6
        cumprod_alphas_t_1 = cumprod_alphas_t[:-1]
        cumprod_alphas_t = cumprod_alphas_t[1:]
        f_cumprod_alphas_t = lambda t: сa_t(1 + t)
        alphas_t = cumprod_alphas_t / cumprod_alphas_t_1
        betas_t = 1 - alphas_t
        
        self.betas_t            = betas_t            = betas_t.view(-1,1,1)
        self.alphas_t           = alphas_t           = alphas_t.view(-1,1,1)
        self.cumprod_alphas_t   = cumprod_alphas_t   = cumprod_alphas_t.view(-1,1,1)
        self.cumprod_alphas_t_1 = cumprod_alphas_t_1 = cumprod_alphas_t_1.view(-1,1,1)
        self.f_cumprod_alphas_t = lambda t: f_cumprod_alphas_t(t).view(-1,1,1)
        
        self.reverse_c0 = 1 / torch.sqrt(alphas_t)
        self.reverse_c1 = (1 - alphas_t) / torch.sqrt(1 - cumprod_alphas_t) 
        self.reverse_std_coef = ( 
            torch.sqrt(betas_t * (1 - cumprod_alphas_t_1) / (1 - cumprod_alphas_t))  
        )
        
        self.eps_to_x0_c0 = torch.sqrt(1 / self.cumprod_alphas_t)
        self.eps_to_x0_c1 = torch.sqrt(1 / self.cumprod_alphas_t - 1)

        self.to_domain = lambda x: x
        pass
    

    def output_transform(self, x0, batch):
        return x0

    
    def forward_step_sample(self, x0, t, batch):
        if self.cont_time:
            at = self.f_cumprod_alphas_t(t)
        else:
            at = self.cumprod_alphas_t[t]
        mu_xt = torch.sqrt(at) * x0
        std_xt = torch.sqrt(1 - at) * torch.ones_like(x0)
        self.randn_x = torch.randn_like(x0)
        xt = mu_xt + self.randn_x * std_xt
        return xt
    
    
    def reverse_step_sample(self, eps_pred, xt, t, batch):
        mu_xt_1 = self.reverse_c0[t] * (xt - self.reverse_c1[t] * eps_pred)
        if t[0] == 0:
            return mu_xt_1
        return mu_xt_1 + self.reverse_std_coef[t] * torch.randn_like(mu_xt_1)
    
    
    def prior_sample(self, batch):
        return torch.randn((batch['batch_size'], 3, 3), dtype=torch.float32, device=batch['device'])
    
    
    def loss(self, batch):
        eps_pred = batch['prediction'][0]
        l2 = ((eps_pred - self.randn_x)**2).reshape(eps_pred.shape[0],-1).mean(dim=1)
        return l2

    
    def get_x0_prediction(self, eps_pred, xt, t, batch):
        x0_pred = self.eps_to_x0_c0[t] * xt - self.eps_to_x0_c1[t] * eps_pred
        return x0_pred