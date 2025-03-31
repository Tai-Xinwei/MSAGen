# -*- coding: utf-8 -*-
import math
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.psm_config import PSMConfig
from sfm.utils.register import Register

DIFFUSION_PROCESS_REGISTER = Register("diffusion_process_register")


class DiffusionProcess(ABC, metaclass=ABCMeta):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        self.alpha_cummlative_product = alpha_cummlative_product
        if (
            self.alpha_cummlative_product is not None
        ):  # for edm, the alpha_cummlative_product is None
            self.alpha_cummlative_product_t_1 = torch.cat(
                [
                    torch.tensor([1.0]).to(alpha_cummlative_product.device),
                    alpha_cummlative_product[:-1],
                ]
            )
        self.psm_config = psm_config

    @abstractmethod
    def sample_step(self, x_t, x_init_pos, predicted_noise, epsilon, t):
        return

    def _extract(self, a, t, x_shape):
        if len(t.shape) == 1:
            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu().long())
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        elif len(t.shape) == 2:
            batch_size, L = t.shape
            # a is in shape of [num_timesteps], t is in shape of [batch_size, L],
            out = torch.gather(a.unsqueeze(0).expand(batch_size, -1), 1, t.cpu().long())
            return out.reshape(batch_size, L, *((1,) * (len(x_shape) - 2))).to(t.device)
        else:
            raise Exception(f"t shape: {t.shape} not supported")


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

    def sample_step_multi_t(
        self, x_t, x_init_pos, predicted_noise, epsilon, t, stepsize=1
    ):
        hat_alpha_t = self._extract(self.alpha_cummlative_product, t, x_t.shape)
        hat_alpha_t_1 = self._extract(self.alpha_cummlative_product_t_1, t, x_t.shape)

        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = (1 - alpha_t) * stepsize

        beta_tilde_t = torch.where(
            t == 0,
            torch.tensor(0.0).to(t.device),
            ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt(),
        )

        temp = (1 - alpha_t) / (1 - hat_alpha_t).sqrt()

        x_t_minus_1 = (
            x_t - x_init_pos - temp * predicted_noise
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


@DIFFUSION_PROCESS_REGISTER.register("sde")
class SDE(DiffusionProcess):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        super().__init__(alpha_cummlative_product, psm_config)

    def sample_step(self, x_t, x_init_pos, predicted_noise, epsilon, t, stepsize=1):
        hat_alpha_t = self.alpha_cummlative_product[t]
        hat_alpha_t_1 = 1.0 if t == 0 else self.alpha_cummlative_product[t - 1]
        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = 1 - alpha_t
        beta_t = beta_t * stepsize
        score = -predicted_noise / (1.0 - self.alpha_cummlative_product[t]).sqrt()

        x_t_minus_1 = (
            (2 - (1.0 - beta_t).sqrt()) * (x_t - x_init_pos)
            + beta_t * (score)
            + x_init_pos
            + beta_t.sqrt() * epsilon
        )
        return x_t_minus_1

    def sample_step_multi_t(
        self, x_t, x_init_pos, predicted_noise, epsilon, t, stepsize=1
    ):
        hat_alpha_t = self._extract(self.alpha_cummlative_product, t, x_t.shape)
        hat_alpha_t_1 = self._extract(self.alpha_cummlative_product_t_1, t, x_t.shape)

        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = (1 - alpha_t) * stepsize

        score = -predicted_noise / (1.0 - hat_alpha_t).sqrt()

        x_t_minus_1 = (
            (2 - (1.0 - beta_t).sqrt()) * (x_t - x_init_pos)
            + beta_t * (score)
            + x_init_pos
            + beta_t.sqrt() * epsilon
        )

        return x_t_minus_1


@DIFFUSION_PROCESS_REGISTER.register("ddim")
class DDIM(DiffusionProcess):
    def __init__(self, alpha_cummlative_product: Tensor, psm_config) -> None:
        super().__init__(alpha_cummlative_product, psm_config)
        self.final_alpha_cumprod = torch.tensor(1.0, device="cuda")

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alpha_cummlative_product[timestep]
        alpha_prod_t_prev = (
            self.alpha_cummlative_product[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def sample_step(
        self,
        x_t,
        x_init_pos,
        predicted_noise,
        epsilon,
        t,
        stepsize=1,
        eta=0,
        use_clipped_pred_epsilon=False,
    ):
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # 1. get previous step value (=t-1)
        prev_timestep = t + self.psm_config.num_timesteps_stepsize

        # 2. compute alphas, betas
        alpha_prod_t = self.alpha_cummlative_product[t]
        alpha_prod_t_prev = (
            self.alpha_cummlative_product[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12)
        pred_original_sample = (
            x_t - beta_prod_t ** (0.5) * predicted_noise
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = predicted_noise

        # 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(t, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_pred_epsilon:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                x_t - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            variance = std_dev_t * epsilon

            prev_sample = prev_sample + variance

        return prev_sample


@DIFFUSION_PROCESS_REGISTER.register("dpm")
class DPM(DiffusionProcess):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        super().__init__(alpha_cummlative_product, psm_config)
        # register a list of the predicted noise for higher order solvers
        self.model_outputs = [None] * self.psm_config.solver_order
        self.lower_order_nums = 0

        # Currently we only support VP-type noise schedule
        # timesteps is not continuous integers
        # TODO: change hard cuda device to a more flexible way
        self.timesteps = torch.tensor(
            range(
                self.psm_config.num_timesteps - 1,
                -1,
                self.psm_config.num_timesteps_stepsize,
            ),
            device="cuda",
        )
        self.alphas_cumprod = alpha_cummlative_product.to("cuda")[self.timesteps]
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

    def t_to_index(self, t):
        # index t to its corresponding index in the timesteps
        t_index = int(torch.where(self.timesteps == t)[0][0])
        return t_index

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def _post_step(self):
        # reset the model_outputs when the last timestep is reached
        if self.step_index == len(self.timesteps) - 1:
            self.lower_order_nums = 0
            self.model_outputs = [None] * self.psm_config.solver_order
            self.step_index = 0

    # TODO: support for other order solvers
    def sample_step(self, x_t, x_init_pos, model_output, epsilon, t, stepsize=1):
        self.step_index = self.t_to_index(t)

        model_output = self.convert_model_output(model_output, sample=x_t)
        for i in range(self.psm_config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]

        self.model_outputs[-1] = model_output
        # TODO: lower_order_final
        if (
            self.psm_config.solver_order == 1
            or self.lower_order_nums < 1
            or self.step_index == len(self.timesteps) - 1
        ):
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=x_t, noise=epsilon
            )
        elif self.psm_config.solver_order == 2 or self.lower_order_nums < 2:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=x_t, noise=epsilon
            )
        else:
            raise NotImplementedError("Higher order solvers are not supported yet.")

        if self.lower_order_nums < self.psm_config.solver_order:
            self.lower_order_nums += 1

        # reset the model_outputs when the last timestep is reached
        self._post_step()

        return prev_sample

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list,
        *args,
        sample: torch.Tensor = None,
        noise=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.psm_config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.psm_config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.psm_config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.psm_config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        sigma_t = (
            torch.tensor(0.0, device=self.sigmas.device)
            if self.step_index == len(self.sigmas) - 1
            else self.sigmas[self.step_index + 1]
        )
        sigma_s = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s

        if self.psm_config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (
                alpha_t * (torch.exp(-h) - 1.0)
            ) * model_output
        elif self.psm_config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (
                sigma_t * (torch.exp(h) - 1.0)
            ) * model_output
        elif self.psm_config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.psm_config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.psm_config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.psm_config.diffusion_mode == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.psm_config.diffusion_mode == "sample":
                x0_pred = model_output
            elif self.psm_config.diffusion_mode == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"diffusion_mode given as {self.psm_config.diffusion_mode} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )
            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.psm_config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.psm_config.diffusion_mode == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                epsilon = model_output
            elif self.psm_config.diffusion_mode == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.psm_config.diffusion_mode == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"diffusion_mode given as {self.psm_config.diffusion_mode} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            return epsilon


@DIFFUSION_PROCESS_REGISTER.register("dpm_edm")
class DPM_EDM(DiffusionProcess):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        super().__init__(alpha_cummlative_product, psm_config)
        # register a list of the predicted noise for higher order solvers
        self.model_outputs = [None] * self.psm_config.solver_order
        self.lower_order_nums = 0
        self.final_sigmas_type = "zero"

        # timesteps is not continuous integers
        # TODO: change hard cuda device to a more flexible way
        self.timesteps = torch.tensor(
            range(
                self.psm_config.num_timesteps - 1,
                -1,
                self.psm_config.num_timesteps_stepsize,
            ),
            device="cuda",
        )

        self.num_inference_steps = int(
            -np.floor(
                self.psm_config.num_timesteps / self.psm_config.num_timesteps_stepsize
            )
        )  # number_timesteps_stepsize is negative
        ramp = torch.linspace(0, 1, self.num_inference_steps, device="cuda")

        sigmas = self._compute_karras_sigmas(
            ramp
        )  # Karras et al. (2022) noise schedule
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = torch.tensor(
            1, device="cuda"
        )  # Inputs are pre-scaled before going into unet, so alpha_t = 1
        sigma_t = sigma

        return alpha_t, sigma_t

    # To align with AF3, we use sigma_data scaled sigmas
    def _compute_karras_sigmas(self, ramp) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""
        sigma_data = self.psm_config.edm_sigma_data
        rho = self.psm_config.edm_sample_rho
        inv_rho = 1.0 / rho
        sigma_min = self.psm_config.edm_sample_sigma_min
        sigma_max = self.psm_config.edm_sample_sigma_max
        min_inv_rho = sigma_min**inv_rho
        max_inv_rho = sigma_max**inv_rho
        sigmas = sigma_data * (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def t_to_index(self, t):
        # index t to its corresponding index in the timesteps
        t_index = int(torch.where(self.timesteps == t)[0][0])
        return t_index

    def t_to_sigma(self, t):
        t_index = self.t_to_index(t)
        sigma = self.sigmas[t_index]
        return sigma

    def _post_step(self):
        # reset the model_outputs when the last timestep is reached
        if self.step_index == len(self.timesteps) - 1:
            self.lower_order_nums = 0
            self.model_outputs = [None] * self.psm_config.solver_order
            self.step_index = 0

    def sample_step_euler(self, x_t, x_init_pos, model_output, epsilon, t, stepsize=1):
        self.step_index = self.t_to_index(t)
        t_cur = self.sigmas[self.step_index + 1]
        t_hat = self.sigmas[self.step_index]
        dt = t_cur - t_hat
        x0_pred = model_output
        delta = (x_t - x0_pred) / t_hat
        prev_sample = x_t + dt * delta
        return prev_sample

    # TODO: support for other order solvers
    def sample_step(self, x_t, x_init_pos, model_output, epsilon, t, stepsize=1):
        self.step_index = self.t_to_index(t)

        for i in range(self.psm_config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]

        self.model_outputs[-1] = model_output
        # TODO: lower_order_final
        if (
            self.psm_config.solver_order == 1
            or self.lower_order_nums < 1
            or self.step_index == len(self.timesteps) - 1
        ):
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=x_t, noise=epsilon
            )
        elif self.psm_config.solver_order == 2 or self.lower_order_nums < 2:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=x_t, noise=epsilon
            )
        else:
            raise NotImplementedError("Higher order solvers are not supported yet.")

        if self.lower_order_nums < self.psm_config.solver_order:
            self.lower_order_nums += 1

        # reset the model_outputs when the last timestep is reached
        self._post_step()
        return prev_sample

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list,
        *args,
        sample: torch.Tensor = None,
        noise=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.psm_config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.psm_config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.psm_config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.psm_config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.psm_config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.psm_config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        sigma_t = (
            torch.tensor(0.0, device=self.sigmas.device)
            if self.step_index == len(self.sigmas) - 1
            else self.sigmas[self.step_index + 1]
        )
        sigma_s = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s

        if self.psm_config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (
                alpha_t * (torch.exp(-h) - 1.0)
            ) * model_output
        elif self.psm_config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (
                sigma_t * (torch.exp(h) - 1.0)
            ) * model_output
        elif self.psm_config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.psm_config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t


class Diffsuion_LM:
    """
    T = number of timesteps to set up diffuser with

    schedule = type of noise schedule to use linear, cosine, gaussian

    noise = type of ditribution to sample from; DEFAULT - normal_gaussian

    """

    def __init__(
        self,
        T=1000,
        schedule="sqrt",
        sample_distribution="normal",
        sample_distribution_gmm_means=[-1.0, 1.0],
        sample_distribution_gmm_variances=[1.0, 1.0],
        F=1,
    ):
        # Use float64 for accuracy.
        betas = np.array(get_named_beta_schedule(schedule, T), dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.F = F

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)

        # sample_distribution_params
        self.sample_distribution = sample_distribution
        self.sample_distribution_gmm_means = [
            float(mean) for mean in sample_distribution_gmm_means
        ]
        self.sample_distribution_gmm_variances = [
            float(variance) for variance in sample_distribution_gmm_variances
        ]

        if self.sample_distribution == "normal":
            self.noise_function = torch.randn_like
        else:
            self.noise_function = self.randnmixture_like

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, mask=None, DEVICE=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """

        # noise_function is determined in init depending on type of noise specified
        noise = self.noise_function(x_start) * (self.F**2)
        if DEVICE is not None:
            noise = noise.to(DEVICE)

        assert noise.shape == x_start.shape
        x_sample = (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if mask is not None:
            x_sample[mask] = x_start[mask]

        return x_sample

    def p_sample(self, x0_pred, x_t, t, mask=None, DEVICE=None):
        (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        ) = self.q_posterior_mean_variance(x0_pred, x_t, t)

        if isinstance(t, torch.Tensor):
            t_scalar = t[0].item() if t.numel() > 1 else t.item()
        else:
            t_scalar = t

        if t_scalar > 0:
            noise = torch.randn_like(x_t)
            # NoiseTerm = sqrt(variance) * noise = exp(0.5 * log_variance) * noise
            x_t_minus_1 = (
                posterior_mean + torch.exp(0.5 * posterior_log_variance_clipped) * noise
            )
        else:
            x_t_minus_1 = posterior_mean

        if mask is not None:
            x_t_minus_1[mask] = x_t_minus_1[mask]

        return x_t_minus_1

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def randnmixture_like(self, tensor_like, number_normal=3, weights_normal=None):
        if (
            self.sample_distribution_gmm_means
            and self.sample_distribution_gmm_variances
        ):
            assert len(self.sample_distribution_gmm_means) == len(
                self.sample_distribution_gmm_variances
            )

        if not weights_normal:
            mix = torch.distributions.Categorical(
                torch.ones(len(self.sample_distribution_gmm_means))
            )  # number_normal
        else:
            assert len(weights_normal) == number_normal
            mix = torch.distributions.Categorical(weights_normal)
        # comp = torch.distributions.Normal(torch.randn(number_normal), torch.rand(number_normal))
        comp = torch.distributions.Normal(
            torch.tensor(self.sample_distribution_gmm_means),
            torch.tensor(self.sample_distribution_gmm_variances),
        )
        # comp = torch.distributions.Normal([-3, 3], [1, 1])
        # comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        # comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return torch.tensor(
            [gmm.sample() for _ in range(np.prod(tensor_like.shape))]
        ).reshape(tensor_like.shape)

    def get_loss_weight(self, t, device):
        alpha_bar = _extract(self.sqrt_alphas_cumprod, t, t.shape).to(device)
        weight = 1.0 / (1.0 - alpha_bar + 1e-8)
        return weight


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class DiffNoise(nn.Module):
    def __init__(self, psm_config: PSMConfig):
        super(DiffNoise, self).__init__()
        self.psm_config = psm_config

        assert psm_config.ddpm_schedule in [
            "linear",
            "quadratic",
            "sigmoid",
            "cosine",
            "sqrt",
        ]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.alphas_cumprod,
            self.beta_list,
        ) = self._beta_schedule(
            psm_config.num_timesteps,
            psm_config.ddpm_beta_start,
            psm_config.ddpm_beta_end,
            psm_config.ddpm_schedule,
        )
        self.unit_noise_scale = psm_config.diffusion_noise_std
        self.torch_generator = None

    def _beta_schedule(
        self, num_timesteps, beta_start, beta_end, schedule_type="sigmoid"
    ):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "sqrt":
            x = 1 - torch.sqrt(torch.linspace(0, 1, num_timesteps + 1) + 0.0001)
            beta_list = 1 - x[1:] / x[:-1]
            beta_list[beta_list > 0.999] = 0.999
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError("only support linear, quadratic, sigmoid, cosine")

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return (
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            alphas_cumprod,
            beta_list,
        )

    def _extract(self, a, t, x_shape):
        if len(t.shape) == 1:
            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu().long())
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        elif len(t.shape) == 2:
            batch_size, L = t.shape
            # a is in shape of [num_timesteps], t is in shape of [batch_size, L],
            out = torch.gather(a.unsqueeze(0).expand(batch_size, -1), 1, t.cpu().long())
            return out.reshape(batch_size, L, *((1,) * (len(x_shape) - 2))).to(t.device)
        else:
            raise Exception(f"t shape: {t.shape} not supported")

    def get_noise(
        self,
        pos,
    ):
        if self.torch_generator is None:
            self.torch_generator = torch.Generator(device=pos.device)
            self.torch_generator.manual_seed(dist.get_rank())
        noise = (
            torch.randn(
                pos.size(),
                device=pos.device,
                dtype=pos.dtype,
                generator=self.torch_generator,
            )
            * self.unit_noise_scale
        )
        noise_center = noise.mean(dim=2, keepdim=True)
        noise -= noise_center
        return noise

    def get_sampling_start(self, init_pos):
        noise = self.get_noise(
            init_pos,
        )
        return init_pos + noise

    def noise_sample(
        self,
        x_start,
        t,
        x_init=None,
        clean_mask: Optional[Tensor] = None,
    ):
        tmpt = t
        t = (t.float() * (self.psm_config.num_timesteps - 1)).long()
        if t >= 1000:
            print(tmpt)
        noise = self.get_noise(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        if x_init is None:
            x_t = (
                sqrt_alphas_cumprod_t * x_start
                + sqrt_one_minus_alphas_cumprod_t * noise
            )
        else:
            x_t = (
                sqrt_alphas_cumprod_t * (x_start - x_init)
                + sqrt_one_minus_alphas_cumprod_t * noise
                + x_init
            )
        if clean_mask is not None:
            if len(clean_mask.shape) == 1:
                x_t = torch.where(
                    clean_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), x_start, x_t
                )
            elif len(clean_mask.shape) == 2:
                x_t = torch.where(clean_mask.unsqueeze(-1).unsqueeze(-1), x_start, x_t)
            elif len(clean_mask.shape) == 3:
                x_t = torch.where(clean_mask.unsqueeze(-1), x_start, x_t)
            else:
                raise ValueError(
                    f"clean_mask should be [B] or [B, L] or [B, D, L]tensor, but it's shape is {clean_mask.shape}"
                )

        return x_t, noise, sqrt_one_minus_alphas_cumprod_t, sqrt_alphas_cumprod_t

    # Here, t is time point in (0, 1]
    def _angle_noise_sample(self, x_start, t):
        T = t
        # T = t / 1000.0
        T = T.unsqueeze(-1).unsqueeze(-1)

        beta_min = 0.01 * torch.pi
        beta_max = 1.0 * torch.pi

        sigma = beta_min ** (1 - T) * beta_max**T  # SMLD (31)

        noise = torch.randn_like(x_start) * sigma

        return x_start + noise, noise, sigma


class TimeStepSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, n_graph, device, dtype, clean_sample_ratio: float = 0.0):
        time_step = torch.rand(size=(n_graph // 2 + 1,), device=device)
        time_step = torch.cat([time_step, 1.0 - time_step], dim=0)[:n_graph]
        time_step = time_step.to(dtype=dtype)

        clean_mask = torch.tensor(
            np.random.rand(n_graph) <= clean_sample_ratio,
            dtype=torch.bool,
            device=device,
        )
        return time_step, clean_mask

    def get_continuous_time_step(self, t, n_graph, device, dtype):
        time_step = torch.zeros(
            [
                n_graph,
            ],
            device=device,
            dtype=dtype,
        )
        time_step = time_step.fill_(t * 1.0 / self.num_timesteps)
        return time_step


from sfm.models.psm.psm_config import DiffusionTimeStepEncoderType, PSMConfig


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "SinusoidalPositionEmbeddings requires dim to be even."
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        time = time * self.max_period
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class TimeStepEncoder(nn.Module):
    def __init__(
        self,
        n_timesteps,
        timestep_emb_dim,
        timestep_emb_type: DiffusionTimeStepEncoderType,
        mlp=True,
    ):
        super(TimeStepEncoder, self).__init__()

        if timestep_emb_type == DiffusionTimeStepEncoderType.POSITIONAL:
            self.time_proj = SinusoidalPositionEmbeddings(timestep_emb_dim)
        elif timestep_emb_type == DiffusionTimeStepEncoderType.DISCRETE_LEARNABLE:
            self.time_proj = nn.Embedding(n_timesteps + 1, timestep_emb_dim)
        else:
            raise NotImplementedError

        if mlp:
            self.time_embedding = nn.Sequential(
                nn.Linear(timestep_emb_dim, timestep_emb_dim),
                nn.GELU(),
                nn.Linear(timestep_emb_dim, timestep_emb_dim),
            )
        else:
            self.time_embedding = None

        self.n_timesteps = n_timesteps
        self.timestep_emb_type = timestep_emb_type

    def forward(self, timesteps, clean_mask: Optional[Tensor]):
        ngraph, nD, nL = timesteps.shape[:3]
        if self.timestep_emb_type == DiffusionTimeStepEncoderType.DISCRETE_LEARNABLE:
            discretized_time_steps = (
                timesteps * self.n_timesteps
            ).long()  # in range [0, n_timesteps - 1]
            if clean_mask is not None:
                discretized_time_steps[
                    clean_mask
                ] = self.n_timesteps  # use last time step embedding for clean samples
            t_emb = self.time_proj(discretized_time_steps).view(ngraph, nD, nL, -1)
        elif self.timestep_emb_type == DiffusionTimeStepEncoderType.POSITIONAL:
            if clean_mask is not None:
                timesteps = timesteps.masked_fill(
                    clean_mask, 0.0
                )  # use t = 0 for clean samples with positional time embedding (which is continuous time embedding)
            t_emb = self.time_proj(timesteps.unsqueeze(-1)).view(ngraph, nD, nL, -1)
        else:
            raise ValueError(f"Unkown timestep_emb_type {self.timestep_emb_type}")
        if self.time_embedding is not None:
            t_emb = self.time_embedding(t_emb)

        return t_emb
