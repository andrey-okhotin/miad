import os
from pathlib import Path

import torch
import math
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

from saving_utils.get_repo_root import get_repo_root
    
    
    
def scheduler(scheduler_name, num_steps):
    if scheduler_name == 'diffcsp_cosine':
        s = 0.008
        discretization = torch.linspace(0, num_steps, num_steps + 1)
        alphas_cumprod = torch.cos(((discretization / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        alphas = 1. - betas
        cumprod_alphas_t = torch.cumprod(alphas, axis=0).to(torch.float32)
        cumprod_alphas_t = torch.hstack((torch.tensor([1.]), cumprod_alphas_t))
        return cumprod_alphas_t, None
    
    elif scheduler_name == 'cosine':
        s = 0.008
        f_t = lambda t: torch.cos((t / (num_steps + 1) + s) / (1 + s) * math.pi / 2)**2
        a_t = lambda t: f_t(t) / f_t(torch.tensor(0., dtype=torch.float64))
        discretization = torch.linspace(0, num_steps, num_steps + 1)
        cumprod_alphas_t = a_t(discretization).to(torch.float32)
        return cumprod_alphas_t, a_t
    
    elif 'default_d3pm' in scheduler_name:
        vocab_size = 100
        def get_uniform_transition_mat(vocab_size, beta_t):
            mat = torch.full((vocab_size, vocab_size), beta_t/float(vocab_size))
            diag_indices = np.diag_indices_from(mat.cpu().numpy())
            diag_val = 1 - beta_t * (vocab_size - 1) / vocab_size
            mat[diag_indices] = diag_val.to(torch.float32)
            return mat
        s = 0.008
        f_t = lambda t: torch.cos((t / (num_steps+1) + s) / (1 + s) * math.pi / 2)
        a_t = lambda t: f_t(t) / f_t(torch.tensor(0., dtype=torch.float64))
        discretization = torch.arange(1, num_steps+1, dtype=torch.float64)
        cumprod_alphas_t = a_t(discretization)
        cumprod_alphas_t_1 = torch.hstack((torch.tensor([1.0]), cumprod_alphas_t[:-1]))
        betas_t = 1 - cumprod_alphas_t / cumprod_alphas_t_1
        Q_t = []
        for t in range(num_steps):
            Q_t.append(get_uniform_transition_mat(vocab_size, betas_t[t]))
        Q_t = torch.stack(Q_t)
        cumprod_Q_t = [ Q_t[0] ]
        for t in range(1, num_steps):
            cumprod_Q_t.append(torch.matmul(cumprod_Q_t[-1], Q_t[t]))
        cumprod_Q_t = torch.stack(cumprod_Q_t)      
        return Q_t, cumprod_Q_t
    
    elif scheduler_name == 'default_wrapped_normal':
        # params
        sigma_begin, sigma_end = 0.005, 0.5
        sigmas = torch.FloatTensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_steps))
        )
        # approx of WrappedNormal(x|sigma)
        def p_wrapped_normal(x, sigma, N=10, T=1.0):
            p_ = 0
            for i in range(-N, N + 1):
                p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
            return p_
        # approx of grad_x(log(WrappedNormal(x|sigma)))
        def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0, device='cpu'):
            x, sigma = x.to(device), sigma.to(device)
            p_ = 0
            for i in range(-N, N + 1):
                p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
            return (p_ / p_wrapped_normal(x, sigma, N, T)).to(device)
        # approx of expectation_x(grad_x(log(WrappedNormal(x|sigma)))^2)
        def sigma_norm(sigma, T=1.0, sn = 10000_0):
            sigmas = sigma[None, :].repeat(sn, 1)
            x_sample = sigma * torch.randn_like(sigmas)
            x_sample = x_sample % T
            normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T, device=torch.get_default_device())
            return (normal_ ** 2).mean(dim = 0).cpu()
        # computation
        sigmas_norm_ = sigma_norm(sigmas)
        sigmas_t = sigmas
        sigmas_norm_t = sigmas_norm_
        d_log_p = d_log_p_wrapped_normal
        return sigmas_t, sigmas_norm_t, sigma_begin, sigma_end, d_log_p
        
    raise NotImplementedError