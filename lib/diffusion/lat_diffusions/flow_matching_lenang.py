import torch
from torch.distributions.gamma import Gamma

from data.crystal_utils import lattice_to_lengths_and_angles, lengths_and_angles_to_lattice



class FM_LenAng:
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.lat_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.device = 'cpu'
        self.step = 1. / self.num_steps
        
        alpha, theta = 1.3, 0.25
        self.len_gamma_prior = Gamma(torch.tensor(alpha), torch.tensor(theta))
        self.angle_difference_bound = 20
        self.default_loss_scale = 0.45
        
        self.lat2lenang = lambda lat: torch.hstack(lattice_to_lengths_and_angles(lat))
        self.lenang2lat = lambda lenang: lengths_and_angles_to_lattice(lenang[:,:3], lenang[:,3:])
        pass


    def output_transform(self, x0, batch):
        return x0
    
    
    def forward_step_sample(self, x0, t, batch):
        xT = self.prior_sample(batch)
        step = (1 + t[:,None,None]) * self.step
        xt = (1 - step) * x0 + step * xT
        self.ut = x0 - xT
        return xt
    
    
    def reverse_step_sample(self, vt, xt, t, batch):
        xt_1 = xt + self.step * vt
        return xt_1
    
    
    def prior_sample(self, batch):
        lenang_xT = torch.zeros((batch['batch_size'], 6), dtype=torch.float32, device=batch['device'])
        # lengths
        lenang_xT[:,:3] = 2 + self.len_gamma_prior.sample(lenang_xT[:,:3].shape)
        # angles
        ang = 60 + 60 * torch.rand((4*batch['batch_size'],3), dtype=torch.float32, device=batch['device'])
        check = lambda i, j, k, x: x[:,i] + x[:,j] - x[:,k] > self.angle_difference_bound
        lenang_xT[:,3:] = ang[check(0,1,2,ang) * check(2,0,1,ang) * check(1,2,0,ang)][:batch['batch_size']]
        # output tranformation
        xT = self.lenang2lat(lenang_xT)
        return xT
    
    
    def loss(self, batch):
        vt = batch['prediction'][0]
        l2 = ((vt - self.ut)**2).reshape(-1,9).mean(dim=1)
        return self.default_loss_scale * l2
    
    
    def get_x0_prediction(self, vt, xt, t, batch):
        step = (1 + t[:,None,None]) * self.step
        return xt + step * vt