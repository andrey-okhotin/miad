import os
import torch

from diffusion.scheduler import scheduler

from saving_utils.get_repo_root import get_repo_root




class WrappedNormal:
    
    def __init__(self, diffusion_config):
        self.config = diffusion_config.frac_diffusion
        self.num_steps = diffusion_config.num_steps
        
        sigmas_t, sigmas_norm_t, sb, se, d_log_p = scheduler(self.config.scheduler, self.num_steps)
        
        self.sigmas_t = sigmas_t[:,None]
        self.sigmas_norm_t = sigmas_norm_t[:,None]
        self.sb = sb
        self.se = se
        self.d_log_p = d_log_p
        switch_optimal_gamma = {
            'csp_perov5'    : 5e-7,
            'gen_perov5'    : 5e-7,
            'csp_mp20'      : 1e-5,
            'gen_mp20'      : 1e-5,
            'csp_alex_mp20' : 1e-5,
            'gen_alex_mp20' : 1e-5,
            'csp_mpts52'    : 1e-5,
            'gen_mpts52'    : 1e-5,
            'csp_carbon24'  : 5e-7,
            'gen_carbon24'  : 1e-5
        }
        self.step_lr = switch_optimal_gamma[diffusion_config.task]

        self.drift_step_coef = torch.ones(self.num_steps, dtype=torch.float32)[:,None]
        self.diff_step_coef = torch.ones(self.num_steps, dtype=torch.float32)[:,None]
        self.to_domain = lambda x: x
        pass


    def output_transform(self, x0, batch):
        return x0
    
    
    def forward_step_sample(self, x0, t, batch):
        st = self.sigmas_t[t]
        self.randn_x = torch.randn_like(x0)
        xt = (x0 + st * self.randn_x) % 1.
        return xt
    
    
    def reverse_step_sample_part_1(self, normed_score_pred, xt, t, batch):
        st, snt = self.sigmas_t[t], self.sigmas_norm_t[t]
        step_size = self.step_lr * (st / self.sb)**2
        std_x = torch.sqrt(2 * step_size)
        drift = - step_size * normed_score_pred * torch.sqrt(snt) * self.drift_step_coef[t]
        diffusion = std_x * torch.randn_like(xt) * self.diff_step_coef[t]
        xt_05 = xt + drift + diffusion
        return xt_05
    
    
    def reverse_step_sample_part_2(self, normed_score_pred, xt_05, t, batch):
        st, st_1 = self.sigmas_t[t], self.sigmas_t[torch.clamp(t-1, min=0)]
        snt = self.sigmas_norm_t[t]
        step_size = st**2 - st_1**2
        std_x = torch.sqrt((st_1**2 * (st**2 - st_1**2)) / (st**2))
        drift = - step_size * normed_score_pred * torch.sqrt(snt) * self.drift_step_coef[t]
        diffusion = std_x * torch.randn_like(xt_05) * self.diff_step_coef[t]
        xt_1 = (xt_05 + drift + diffusion) % 1.
        return xt_1
    
    
    def prior_sample(self, batch):
        return torch.rand((batch['num_atoms'], 3), dtype=torch.float32, device=batch['device'])
    
    
    def loss(self, batch):
        st, snt = self.sigmas_t[batch['t'][1]], self.sigmas_norm_t[batch['t'][1]]
        normed_score_pred = batch['prediction'][1]
        normed_score = self.d_log_p(st * self.randn_x, st, device=batch['device']) / torch.sqrt(snt)
        l2 = (
            (normed_score_pred - normed_score)**2
        ).reshape(normed_score_pred.shape[0],-1).mean(dim=1)

        # mirage infusion code
        if "miad:add_mirage_atoms_upto" in os.environ['MODIFICATIONS_FIELD']:
            mirage_type = 0
            atom_types = batch['x0'][2]
            mask = (atom_types != mirage_type).to(l2.dtype).to(l2.device)
            # coef to keep the same loss scale after average over increased number of atoms:
            coef = mask.shape[0] / mask.sum()
            l2 = l2 * mask * coef

        return l2
    
    
    def get_x0_prediction(self, normed_score_pred, xt, t, batch):
        x0_pred = (xt - self.sigmas_t[t] * normed_score_pred) % 1.
        return x0_pred