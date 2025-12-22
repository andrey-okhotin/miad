import os
import torch

from diffusion.crystal_diffusions.general_crystal_diffusion import CrystalGen




class DiffCSP(CrystalGen):

    def reverse_step_sample(self, xt, t, model, batch):
        lt, ft, at = xt
        
        _, f_pred, _ = self.model_prediction(xt, t, model, batch)
        ft_05 = self.frac_diffusion.reverse_step_sample_part_1(f_pred, ft, t[1], batch)
        xt_05 = [ lt, ft_05, at ]
        
        l_pred, f_pred, a_pred = self.model_prediction(xt_05, t, model, batch)
        lt_1 = self.lat_diffusion.reverse_step_sample(l_pred, lt, t[0], batch)
        ft_1 = self.frac_diffusion.reverse_step_sample_part_2(f_pred, ft_05, t[1], batch)
        at_1 = self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch) if self.gen_type else at
        xt_1 = [ lt_1, ft_1, at_1 ]

        return xt_1