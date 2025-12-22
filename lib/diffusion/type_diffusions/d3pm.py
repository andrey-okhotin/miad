import os

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from diffusion.scheduler import scheduler




class D3PM:
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.type_diffusion
        self.num_steps = diffusion_config.num_steps
        self.num_types = 100
        
        Q_t, cumprod_Q_t = scheduler(self.config.scheduler, self.num_steps)
        
        Q_t_1 = torch.vstack((torch.eye(*Q_t[0].shape)[None,...], Q_t[:-1]))
        self.Q_t = Q_t.reshape(-1,self.num_types,self.num_types)
        self.Q_t_1 = Q_t_1.reshape(-1,self.num_types,self.num_types)
        cumprod_Q_t_1 = torch.vstack((
            torch.eye(*cumprod_Q_t[0].shape)[None,...], cumprod_Q_t[:-1]
        ))
        self.cumprod_Q_t = cumprod_Q_t.reshape(-1,self.num_types,self.num_types)
        self.cumprod_Q_t_1 = cumprod_Q_t_1.reshape(-1,self.num_types,self.num_types)
        
        self.to_domain = lambda types: F.one_hot(types, num_classes=self.num_types).float()
        self.from_domain = lambda onehot: onehot.argmax(dim=-1)
        self.prediction_to_domain = lambda pred: torch.softmax(pred, dim=-1)
        self.default_loss_scale = 1000
        pass


    def output_transform(self, x0, batch):
        return self.from_domain(x0)
    
    
    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0) 
        xt_probs = torch.matmul(onehot_x0[:,None,:], self.cumprod_Q_t[t])[:,0,:]
        xt = Categorical(xt_probs).sample()
        onehot_xt = self.to_domain(xt)
        return onehot_xt
    
    
    def reverse_step_distribution(self, onehot_x0, onehot_xt, t):
        return (
            torch.matmul(onehot_xt[:,None,:], self.Q_t[t].transpose(-1,-2))[:,0,:] * 
            torch.matmul(onehot_x0[:,None,:], self.cumprod_Q_t_1[t])[:,0,:] /
            (torch.matmul(onehot_x0[:,None,:], self.cumprod_Q_t[t])[:,0,:] * onehot_xt).sum(dim=-1)[:,None]
        )
    
    
    def reverse_step_sample(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self.reverse_step_distribution(onehot_x0, onehot_xt, t)
        if t[0] == 0:
            return self.to_domain(self.from_domain(xt_1_probs))
        xt_1 = Categorical(xt_1_probs).sample()
        onehot_xt_1 = self.to_domain(xt_1)
        self.xt_1_probs = xt_1_probs
        return onehot_xt_1
    
    
    def prior_sample(self, batch):
        shape = (batch['num_atoms'], self.num_types)
        xT_probs = torch.ones(shape, dtype=torch.float32, device=batch['device']) / self.num_types
        xT = Categorical(xT_probs).sample()
        onehot_xT = self.to_domain(xT)
        return onehot_xT
    
    
    def loss(self, batch):
        onehot_xt, t = batch['xt'][2], batch['t'][1]
        onehot_x0_pred = self.prediction_to_domain(batch['prediction'][2])
        pred_xt_1_probs = self.reverse_step_distribution(onehot_x0_pred, onehot_xt, t)
        onehot_x0 = self.to_domain(batch['x0'][2])
        orig_xt_1_probs = self.reverse_step_distribution(onehot_x0, onehot_xt, t)
        eps = 1e-4
        kl_loss = (
            orig_xt_1_probs * (torch.log(orig_xt_1_probs + eps) - torch.log(pred_xt_1_probs + eps))
        ).reshape(onehot_xt.shape[0],-1).sum(dim=-1)
        return self.default_loss_scale * kl_loss
    

    def get_x0_prediction(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        return self.from_domain(onehot_x0)
    

    def get_prob_of_nonexistence(self, onehot_pred):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        return onehot_x0[:,0]