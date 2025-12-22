import os
import math

import torch
import torch.nn as nn




class SinusoidalTimeEmbeddings(nn.Module):
    
    """ Attention is all you need. """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings




class TimeDistribution:
    
    """
    DESCRIPTION:
        Typical operations with time in diffusion models.
        
    """
    
    def __init__(self, num_steps, cont_time):
        self.num_steps = num_steps
        self.cont_time = cont_time
        if 'time_log_normal_0_1_clip' in os.environ['MODIFICATIONS_FIELD']:
            print("MODIF: time_log_normal_0_1_clip", flush=True)
        self.eps = 1e-3
        self.atom_expand = lambda batch, t: [ t, t.repeat_interleave(batch['batch'].num_atoms) ]
        pass
    
    
    def sample(self, batch, mode=None):
        # sample time on [0, 1]
        if 'time_log_normal_0_1_clip' in os.environ['MODIFICATIONS_FIELD']:
            s_eps = 2e-2   # - for escaping total vanishing of probabilities of extreme points 
            t = torch.clip(
                (1 + 2 * s_eps) * torch.sigmoid(torch.randn(batch['batch_size'])) - s_eps, 
                0, 1
            )
        else:
            t = torch.rand(batch['batch_size'])

        if 'max_time_' in os.environ['MODIFICATIONS_FIELD']:
            max_time = float(os.environ['MODIFICATIONS_FIELD'].split("max_time_")[1].split("+")[0])
            t *= max_time

        # postprocess time from [0, 1] to [0, num_steps]
        t = self.eps + (self.num_steps - 1 - self.eps) * t
        if not self.cont_time:
            t = t.round().to(torch.long)
        return self.atom_expand(batch, t.to(batch['device']))
    
    
    def get_time_points_tensor(self, batch, t):
        if t == -1:
            t = self.num_steps - 1
        t = t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device'])
        return self.atom_expand(batch, t)
    
    
    def reverse_time_iterator(self, batch, start_from=-1):
        if start_from == -1:
            start_from = self.num_steps - 1
        for t in range(start_from,-1,-1):
            t = t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device']) 
            yield self.atom_expand(batch, t)
    
    
    def forward_time_iterator(self, batch, start_from):
        for t in range(start_from,self.num_steps):
            t = t * torch.ones(batch['batch_size'], dtype=torch.long, device=batch['device'])
            yield self.atom_expand(batch, t)
            
    
    def to_cuda(self, t_vector, batch):
        return [
            t_vector[0].to(batch['device']),
            t_vector[1].to(batch['device'])
        ]




def mean_interleave(t, num_repeats):
    mean_t = torch.zeros((num_repeats.shape[0]), device=t.device, dtype=t.dtype)
    shift = 0
    for i, n in enumerate(num_repeats):
        mean_t[i] = t[shift:shift+n].mean()
        shift += n
    return mean_t