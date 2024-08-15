# -*- coding: utf-8 -*-
from abc import ABC, ABCMeta, abstractmethod

from torch import Tensor

from sfm.models.psm.psm_config import PSMConfig
from sfm.utils.register import Register

DIFFUSION_PROCESS_REGISTER = Register("diffusion_process_register")


class DiffusionProcess(ABC, metaclass=ABCMeta):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        self.alpha_cummlative_product = alpha_cummlative_product
        self.psm_config = psm_config

    @abstractmethod
    def sample_step(self, x_t, x_init_pos, predicted_noise, epsilon, t):
        return


@DIFFUSION_PROCESS_REGISTER.register("ddpm")
class DDPM(DiffusionProcess):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        super().__init__(alpha_cummlative_product, psm_config)

    def sample_step(self, x_t, x_init_pos, predicted_noise, epsilon, t, stepsize=1):
        hat_alpha_t = self.alpha_cummlative_product[t]
        hat_alpha_t_1 = 1.0 if t == 0 else self.alpha_cummlative_product[t - 1]
        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = (1 - alpha_t) * stepsize
        beta_tilde_t = (
            0.0
            if t == 0
            else ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
        )

        x_t_minus_1 = (
            x_t
            - x_init_pos
            - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * predicted_noise
        ) / alpha_t.sqrt() + beta_tilde_t * epsilon
        x_t_minus_1 += x_init_pos
        return x_t_minus_1


@DIFFUSION_PROCESS_REGISTER.register("ode")
class ODE(DiffusionProcess):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        super().__init__(alpha_cummlative_product, psm_config)

    def sample_step(self, x_t, x_init_pos, predicted_noise, epsilon, t, stepsize=1):
        hat_alpha_t = self.alpha_cummlative_product[t]
        hat_alpha_t_1 = 1.0 if t == 0 else self.alpha_cummlative_product[t - 1]
        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = (1 - alpha_t) * stepsize
        score = -predicted_noise / (1.0 - self.alpha_cummlative_product[t]).sqrt()

        x_t_minus_1 = (
            (2 - (1.0 - beta_t).sqrt()) * (x_t - x_init_pos)
            + 0.5 * beta_t * (score)
            + x_init_pos
        )
        return x_t_minus_1
