# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.autograd as autograd

from sfm.logging import logger


# FIXME: define different time step for different time_pos?
def set_time_step(time_pos, hp=None, hm=None):
    if hp is None:
        hp = 0.0005
    if hm is None:
        hm = 0.001
        if torch.min(time_pos) < 2e-3:
            hm = 1e-4
    return hp, hm


# To avoid zero cases, we do not compute gauss values here, instead we compute the nabla q/q and laplace q/q
class MixtureGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = None
        self.mask = None

    def set_sigma(self, sigma):
        self.sigma = sigma  # [B, 1, 1]

    def set_mask(self, mask):
        self.mask = mask  # [B, L, 3]

    def compute_mixgauss_gradient_term(self, mu, q_point):
        assert self.mask is not None

        func = torch.zeros(
            [q_point.shape[0], q_point.shape[1], q_point.shape[2]], device=self.device
        )

        assert (
            mu.shape[0] == q_point.shape[0]
        ), "mu.shape[0] should be the same as q_point.shape[0]"

        for k in range(q_point.shape[0]):
            # func[k, :, :] = torch.where(self.mask, -(q_point[k, :, :] - mu[k, :, :]) / self.sigma[k] ** 2, torch.zeros_like(q_point[k, :, :]))
            func[k, :, :] = torch.where(
                self.mask[k],
                -(q_point[k, :, :] - mu[k, :, :]),
                torch.zeros_like(q_point[k, :, :]),
            )

        return func

    def compute_mixgauss_laplace_term(self, mu, q_point):
        assert self.mask is not None

        func = torch.zeros([q_point.shape[0]], device=self.device)
        for k in range(q_point.shape[0]):
            # diff_squared = ((q_point[k, :, :] - mu[k, :, :]) / self.sigma[k]) ** 2
            # masked_diff_squared = torch.where(self.mask[k], diff_squared, torch.zeros_like(diff_squared))
            # func[k] = -1.0 * torch.sum(self.mask[k]) / self.sigma[k, 0, 0] ** 2 + torch.sum(masked_diff_squared) / self.sigma[k, 0, 0] ** 2

            diff_squared = (q_point[k, :, :] - mu[k, :, :]) ** 2
            masked_diff_squared = torch.where(
                self.mask[k], diff_squared, torch.zeros_like(diff_squared)
            )
            # sigma**4 * laplace q/q
            func[k] = -1.0 * torch.sum(self.mask[k]) * self.sigma**2 + torch.sum(
                masked_diff_squared
            )

        return func

    def sampler(self, x, x_0, sample_number=None):
        """
        Sample from the Mixture Gaussian.
        In our class, the weight of each Gaussian is the same,
        so we can just sample from the standard Gaussian and shift them around to non-standard Gaussians.
        """
        if sample_number is None:
            sample_number = x.shape[0]  # batch size = gauss number = sample number

        # Sample from the standard Gaussian distribution
        data = torch.randn((sample_number,) + x.shape[1:], device=self.device)

        # # random_means = torch.mean(x, dim=1).unsqueeze(1)#.unsqueeze(-1)
        # random_means = torch.rand(1, device=self.device) * torch.pi * 2 - torch.pi
        # random_std_devs = 0.1
        # data = data * random_std_devs + random_means

        data0 = data.clone()

        # random_means = torch.mean(x, dim=1).unsqueeze(1)#.unsqueeze(-1)
        random_means = torch.rand(1, device=self.device) * torch.pi * 2 - torch.pi
        random_std_devs = 0.1
        data = data * random_std_devs + random_means

        data0 = data.clone()

        # # Calculate the number of samples distributed on per Gaussian
        # L = data.shape[0] // x.shape[0]  # Integer division

        # # Shift samples to represent each non-standard Gaussian
        # for k in range(x.shape[0]):
        #     start_idx = k * L
        #     end_idx = (k + 1) * L if (k + 1) * L < data.shape[0] else data.shape[0]

        #     # Scale by the standard deviation and add the mean for each Gaussian
        #     data[start_idx:end_idx] *= torch.sqrt(self.sigma)
        #     data[start_idx:end_idx] += x[k]

        #     data0[start_idx:end_idx] *= torch.sqrt(self.sigma)
        #     data0[start_idx:end_idx] += x_0[k]

        return data, data0

    def forward(self, x, x_0):
        """
        Compute the value, gradient, a67nd laplacian of the mixture Gaussian.
        Compute the value, gradient, a67nd laplacian of the mixture Gaussian.

        Args:
            x (torch.Tensor): Input data points at time t.
            x (torch.Tensor): Input data points at time t.

        Returns:
            tuple: A tuple containing the following:
                - q_point (torch.Tensor): Sampled points from the mixture Gaussian.
                - nabla_phi_term (torch.Tensor): nabla q/q.
                - laplace_phi_term (torch.Tensor): laplace q/q.
        """
        with torch.no_grad():
            # q_point, q_point_0 = self.sampler(x, x_0)
            q_point, q_point_0 = x, x_0

            nabla_phi_term = self.compute_mixgauss_gradient_term(q_point_0, q_point)
            laplace_phi_term = self.compute_mixgauss_laplace_term(q_point_0, q_point)

            assert not torch.isnan(
                nabla_phi_term
            ).any(), "nabla_phi_term should not contain nan"
            assert not torch.isnan(
                laplace_phi_term
            ).any(), "laplace_phi_term should not contain nan"

            # # normalize the laplace_phi_term as a inner product in high dimension space is usually large
            # nabla_phi_term = 1 / (x.shape[1] * x.shape[2]) * nabla_phi_term
            # laplace_phi_term = 1 / (x.shape[1] * x.shape[2]) * laplace_phi_term

            return q_point, q_point_0, nabla_phi_term, laplace_phi_term


def t_finite_diff(q_output, q_output_m, q_output_p, hp, hm):
    assert not torch.isnan(q_output_m).any(), "q_output_mtq should not contain nan"
    assert not torch.isnan(q_output_p).any(), "q_output_ptq should not contain nan"
    # assert not torch.isnan(hp).any(), "hp should not contain nan"
    # assert not torch.isnan(hm).any(), "hm should not contain nan"

    up = hm**2 * q_output_p + (hp**2 - hm**2) * q_output - hp**2 * q_output_m
    low = hm * hp * (hp + hm)  # TODO: check this file
    return up / low


class SingleGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_vector = None
        self.covariance_matrix = None

    def _compute_gradient_term(self, q_point):
        result = torch.zeros((q_point.shape[0], q_point.shape[1]), device=self.device)
        for i in range(q_point.shape[0]):
            result[i] = self.covariance_matrix.inverse() @ (
                q_point[i] - self.mean_vector
            )

        return result

    def _compute_laplace_term(self, q_point):
        result = torch.zeros((q_point.shape[0]), device=self.device)
        for i in range(q_point.shape[0]):
            q_centered = q_point[i] - self.mean_vector
            covariance_matrix_inv = self.covariance_matrix.inverse()
            result[i] = (
                -torch.trace(covariance_matrix_inv)
                + q_centered.T
                @ covariance_matrix_inv
                @ covariance_matrix_inv
                @ q_centered
            )

        return result

    def _sampler(self, batch_size, sampler_number=None):
        # default sampler number is 10x of batch size
        sample_num_tensor = (
            torch.tensor([sampler_number], device=self.device)
            if sampler_number is not None
            else torch.tensor([10 * batch_size], device=self.device)
        )

        multivariate_gaussian = torch.distributions.MultivariateNormal(
            self.mean_vector, self.covariance_matrix
        )

        data = multivariate_gaussian.sample(sample_num_tensor)

        data0 = data.clone()

        return data, data0

    def forward(self, x, x_0):
        inf_mask = x == float("inf")
        x = x.masked_fill(inf_mask, 100.0)
        inf_mask_0 = x_0 == float("inf")
        x_0 = x_0.masked_fill(inf_mask_0, 100.0)

        with torch.no_grad():
            # flatten the input tensor
            x_flattened = x.view(x.shape[0], -1)
            x0_flattened = x_0.view(x_0.shape[0], -1)

            # modify the inf or 100.0 to 0.0
            mask = torch.logical_or(x_flattened == 100.0, x_flattened == float("inf"))
            x_flattened = x_flattened.masked_fill(mask, 0.0)
            mask_0 = torch.logical_or(
                x0_flattened == 100.0, x0_flattened == float("inf")
            )
            x0_flattened = x0_flattened.masked_fill(mask_0, 0.0)

            # maximum likelihood estimation
            self.mean_vector = torch.mean(x_flattened, dim=0)
            x_centered = x_flattened - self.mean_vector
            self.covariance_matrix = (
                (x_centered.T @ x_centered) / (x_flattened.shape[0] - 1)
                if x_flattened.shape[0] > 0
                else torch.zeros_like(x_centered.T @ x_centered)
            )

            # regularize the covariance matrix
            eps = 1e-6
            self.covariance_matrix = self.covariance_matrix + eps * torch.eye(
                self.covariance_matrix.shape[0], device=self.covariance_matrix.device
            )

            # sample from multivariate Gaussian
            q_point, q_point_0 = self._sampler(
                x_flattened.shape[0], 50 * x_flattened.shape[0]
            )
            q_point, q_point_0 = q_point.to(self.device), q_point_0.to(self.device)
            nabla_phi_term = self._compute_gradient_term(q_point)
            laplace_phi_term = self._compute_laplace_term(q_point)
            # logger.debug(f"q_point.shape: {q_point.shape}, q_point_0.shape: {q_point_0.shape},  nabla_phi_term.shape: {nabla_phi_term.shape}, laplace_phi_term.shape: {laplace_phi_term.shape}")

            # reshape q_point, q_point_0, nabla_phi_term, laplace_phi_term to (B, R, 3)
            q_point = q_point.view(q_point.shape[0], x.shape[1], x.shape[2])
            q_point_0 = q_point_0.view(q_point_0.shape[0], x_0.shape[1], x_0.shape[2])
            nabla_phi_term = nabla_phi_term.view(
                nabla_phi_term.shape[0], x.shape[1], x.shape[2]
            )
            laplace_phi_term = laplace_phi_term.view(laplace_phi_term.shape[0])

        return q_point, q_point_0, nabla_phi_term, laplace_phi_term


class VESDE(object):
    def __init__(self, sigma_min=0.01 * torch.pi, sigma_max=torch.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def sde_term(self, q, t):
        drift = 0
        diffusion = (
            self.sigma_min ** (1 - t)
            * self.sigma_max**t
            * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
        )
        return drift, diffusion

    def sigma_term(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


def compute_PDE_q_loss(
    sde: VESDE,
    q_output,
    nabla_phi_term,
    laplace_phi_term,
    q_output_m,
    q_output_p,
    hp,
    hm,
    sigma_t=None,
    is_clip=False,
):
    """
    Computes the partial differential equation (PDE) loss for the given inputs.

    Args:
        sde (SDE): The stochastic differential equation object.
        q_output (torch.Tensor): The output tensor.
        time_pos (torch.Tensor): The time positions tensor.
        q_point (torch.Tensor): The tensor representing the points in the domain.
        nabla_phi_term (torch.Tensor): The tensor representing the nabla phi term.
        laplace_phi_term (torch.Tensor): The tensor representing the laplace phi term.
        q_output_m (torch.Tensor): The tensor representing the output tensor at the previous time step.
        q_output_p (torch.Tensor): The tensor representing the output tensor at the next time step.
        padding_mask (torch.Tensor): The tensor representing the padding mask.
        hp (float): The positive step size.
        hm (float): The negative step size.
        is_clip (bool, optional): Whether to clip the RHS values. Defaults to False.
        sigma_t (torch.Tensor, optional): The tensor representing the sigma term. Defaults to None.

    Returns:
        torch.Tensor: The computed PDE loss.

    Raises:
        AssertionError: If any of the input tensors contain NaN values.

    """
    # LHS = (q_output * math.log(sde.sigma_max / sde.sigma_min) - t_finite_diff(
    #     q_output, q_output_m, q_output_p, hp, hm
    # ))
    LHS = (
        q_output * math.log(sde.sigma_max / sde.sigma_min)
        - t_finite_diff(q_output, q_output_m, q_output_p, hp, hm)
    ) * sigma_t**2

    assert not torch.isnan(
        laplace_phi_term
    ).any(), "laplace_phi_term should not contain nan"
    assert not torch.isnan(
        nabla_phi_term
    ).any(), "nabla_phi_term should not contain nan"
    assert not torch.isnan(q_output).any(), "diffusion should not contain nan"

    # nabla_phi_term = sigma_t^2 * nabla q/q
    # laplace_phi_term = sigma_t^4 * laplace q/q
    RHS = -math.log(sde.sigma_max / sde.sigma_min) * (
        (sigma_t[:, 0, 0] * torch.sum(q_output**2, dim=(1, 2))).reshape(-1, 1, 1)
        * nabla_phi_term
        + q_output * laplace_phi_term.reshape(-1, 1, 1)
    )

    assert not torch.isnan(LHS).any(), "LHS should not contain nan"
    assert not torch.isnan(RHS).any(), "RHS should not contain nan"

    # Clip the RHS values to be within a certain range to avoid very large values
    # When in debugging, we can set is_clip to False to see the original values
    if is_clip:
        clip_min = -1e8
        clip_max = 1e20
        RHS_clipped = torch.clamp(RHS, min=clip_min, max=clip_max)
    else:
        RHS_clipped = RHS

    # Here, we compute the mean on the batched data, and then compute the mse on the spatial dimensions
    res = torch.mean(torch.mean((LHS - RHS_clipped), dim=(0)) ** 2)
    return res


### Terminal loss for score model ###
# Refer the code from https://github.com/ermongroup/sliced_score_matching/blob/master/losses/score_matching.py
# https://github.com/ermongroup/sliced_score_matching/blob/master/losses/sliced_sm.py


# single_sliced_score_matching and sliced_VR_score_matching implement a basic version of SSM
# with only M=1. These are used in density estimation experiments for DKEF.
def compute_pde_control_loss(output, epsilon):
    """
    Compute the control loss for the diffusion.

    Args:
        epsilon_predict: The predicted epsilon, shape is [batch_size, dim1, dim2]
        epsilon_true: The true epsilon, shape is [batch_size, dim1, dim2]

    Returns:
        The computed loss.

    """
    # Now is regular MSE loss, here we approximate it by square
    loss = torch.mean(torch.sum((output * (output - epsilon)) ** 2, dim=(1, 2)))
    return loss


# def compute_pde_control_loss(
#     sde,
#     net_output,
#     samples,
#     time_pos,
#     diffmode="score",
#     noise=None,
#     detach=False,
#     noise_type="radermacher",
# ):
#     """
#     Computes the control loss for the diffusion.

#     Args:
#         sde: The stochastic differential equation (SDE) instance.
#         net_output: The output of the neural network.
#         samples: The samples used for computing the loss.
#         time_pos: The time positions of the samples.
#         diffmode: The type of diffusion term to use. Default is "score".
#         noise: The noise vector used for computing the loss. Default is None.
#         detach: Whether to detach the loss from the computation graph. Default is False.
#         noise_type: The type of noise to use. Default is "radermacher".

#     Returns:
#         The computed loss.

#     Raises:
#         ValueError: If the control type is not supported.

#     """
#     if net_output is None:  # unplugging the control loss
#         return 0
#     else:
#         pass
#     control_type = "running_loss" if time_pos.sum() > 0 else "terminal_loss"
#     if control_type == "running_loss":
#         _, diffusion = sde.sde_term(samples, time_pos)
#     elif control_type == "terminal_loss":
#         pass
#     else:
#         raise ValueError("Only support running_loss and terminal_loss")

#     if diffmode == "score":
#         score_output = net_output
#     elif diffmode == "x0":
#         _, std = sde.marginal_prob(samples, time_pos)
#         score_output = (
#             net_output - samples
#         ) / std**2  # VE case, alpha = 1, diffusion = sigma**2

#     # shape: samples: [batch_size, dim1, dim2]
#     if noise is None:
#         vectors = torch.randn_like(samples)
#         if noise_type == "radermacher":
#             vectors = vectors.sign()
#         elif noise_type == "sphere":
#             vectors = (
#                 vectors
#                 / torch.norm(vectors, dim=-1, keepdim=True)
#                 * np.sqrt(vectors.shape[-1])
#             )
#         elif noise_type == "gaussian":
#             pass
#         else:
#             raise ValueError("Noise type not implemented")
#     else:
#         vectors = noise
#     gradv = torch.sum(score_output * vectors)  # all dimensions are reduced
#     loss1 = torch.sum(score_output * vectors, dim=(1, 2)) ** 2 * 0.5
#     if detach:
#         loss1 = loss1.detach()
#     grad2 = autograd.grad(gradv, samples, create_graph=True)[0]
#     loss2 = torch.sum(vectors * grad2, dim=(1, 2))
#     if detach:
#         loss2 = loss2.detach()

#     if control_type == "running_loss":
#         loss = (diffusion**2 * (2 * loss1 + loss2)).mean()
#     elif control_type == "terminal_loss":
#         loss = (loss1 + loss2).mean()

#     else:
#         raise ValueError("Only support running_loss and terminal_loss")
#     return loss
