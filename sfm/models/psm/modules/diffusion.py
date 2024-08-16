# -*- coding: utf-8 -*-
from abc import ABC, ABCMeta, abstractmethod

import torch
from torch import Tensor

from sfm.models.psm.psm_config import PSMConfig
from sfm.utils.register import Register

DIFFUSION_PROCESS_REGISTER = Register("diffusion_process_register")


class DiffusionProcess(ABC, metaclass=ABCMeta):
    def __init__(
        self, alpha_cummlative_product: Tensor, psm_config: PSMConfig = None
    ) -> None:
        self.alpha_cummlative_product = alpha_cummlative_product
        self.alpha_cummlative_product_t_1 = [1.0] + alpha_cummlative_product[:-1]
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
        hat_alpha_t_1 = torch.where(
            t == 0,
            torch.tensor(1.0).to(t.device),
            self._extract(self.alpha_cummlative_product_t_1, t, x_t.shape),
        )

        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = (1 - alpha_t) * stepsize

        beta_tilde_t = torch.where(
            t == 0,
            torch.tensor(0.0).to(t.device),
            ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt(),
        )

        temp = (1 - alpha_t) / (1 - hat_alpha_t).sqrt()

        x_t_minus_1 = (
            x_t - x_init_pos - temp.unsqueeze(-1).unsqueeze(-1) * predicted_noise
        ) / alpha_t.sqrt().unsqueeze(-1).unsqueeze(-1) + beta_tilde_t.unsqueeze(
            -1
        ).unsqueeze(
            -1
        ) * epsilon
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
