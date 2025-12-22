import torch
import torch.nn.functional as F

from diffusion.scheduler import scheduler
from diffusion.lat_diffusions.ddpm import DDPM




class DDPM_onehot(DDPM):
    
    def __init__(self, diffusion_config):
        ddpm_config = diffusion_config.lat_diffusion
        diffusion_config.lat_diffusion = diffusion_config.type_diffusion
        super().__init__(diffusion_config)
        for attr, value in self.__dict__.items():       # redefine shapes
            if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                self.__dict__[attr] = value.reshape(-1,1)
        diffusion_config.lat_diffusion = ddpm_config
        
        self.num_types = 100
        self.to_domain = lambda types: F.one_hot(types - 1, num_classes=self.num_types).float()
        self.from_domain = lambda onehot: onehot.argmax(dim=-1) + 1
        pass
    
    
    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        onehot_xt = super().forward_step_sample(onehot_x0, t, batch)
        return onehot_xt
    
    
    def reverse_step_sample(self, onehot_eps_pred, onehot_xt, t, batch):
        onehot_xt_1 = super().reverse_step_sample(onehot_eps_pred, onehot_xt, t, batch)
        if t[0] == 0:
            xt_1 = self.from_domain(onehot_xt_1) 
            return xt_1
        return onehot_xt_1
    
    
    def prior_sample(self, batch):
        return torch.randn((batch['num_atoms'], self.num_types), dtype=torch.float32, device=batch['device'])
    
    
    def loss(self, batch):
        eps_pred = batch['prediction'][2]
        l2 = ((eps_pred - self.randn_x)**2).reshape(eps_pred.shape[0],-1).mean(dim=1)
        return l2

    
    def get_x0_prediction(self, onehot_eps_pred, onehot_xt, t, batch, x0_format):
        onehot_x0_pred = super().get_x0_prediction(onehot_eps_pred, onehot_xt, t, batch)
        if x0_format == 'onehot':
            return onehot_x0_pred
        elif x0_format == 'disc':
            return self.from_domain(onehot_x0_pred)
        pass