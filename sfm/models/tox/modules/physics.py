# -*- coding: utf-8 -*-
import math

import numpy as np
import torch

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
    def __init__(self, sigma=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.tensor(sigma, device=self.device)

    def compute_mixgauss_gradient_term(self, mu, q_point):
        # q_point: [q_batch_size, q_dim1, q_dim2]
        # mu: [gauss_num, q_dim1, q_dim2]
        func = torch.zeros(
            [q_point.shape[0], q_point.shape[1], q_point.shape[2]], device=self.device
        )  # [q_batch_size, q_dim1, q_dim2, gauss_num]
        # FIXME: only support sampling the same number points as the batch size right now
        assert (
            mu.shape[0] == q_point.shape[0]
        ), "mu.shape[0] should be the same as q_point.shape[0]"
        for k in range(q_point.shape[0]):
            func[k, :, :] = -(q_point[k, :, :] - mu[k, :, :]) / self.sigma**2
        return func  # [q_batch_size, q_dim1, q_dim2]

    def compute_mixgauss_laplace_term(self, mu, q_point):
        x_dim = q_point.shape[1] * q_point.shape[2]
        func = torch.zeros([q_point.shape[0]], device=self.device)
        for k in range(q_point.shape[0]):
            func[k] = -1 / (self.sigma**2) * x_dim + torch.sum(
                ((q_point[k, :, :] - mu[k, :, :]) / self.sigma**2) ** 2, dim=(0, 1)
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

        # Calculate the number of samples distributed on per Gaussian
        L = data.shape[0] // x.shape[0]  # Integer division

        # Shift samples to represent each non-standard Gaussian
        for k in range(x.shape[0]):
            start_idx = k * L
            end_idx = (k + 1) * L if (k + 1) * L < data.shape[0] else data.shape[0]

            # Scale by the standard deviation and add the mean for each Gaussian
            data[start_idx:end_idx] *= torch.sqrt(self.sigma)
            data[start_idx:end_idx] += x[k]

            data0[start_idx:end_idx] *= torch.sqrt(self.sigma)
            data0[start_idx:end_idx] += x_0[k]

        return data, data0

    def forward(self, x, x_0):
        """
        Compute the value, gradient, a67nd laplacian of the mixture Gaussian.

        Args:
            x (torch.Tensor): Input data points at time t.

        Returns:
            tuple: A tuple containing the following:
                - q_point (torch.Tensor): Sampled points from the mixture Gaussian.
                - nabla_phi_term (torch.Tensor): nabla q/q.
                - laplace_phi_term (torch.Tensor): laplace q/q.
        """

        # no computational graph
        with torch.no_grad():
            q_point, q_point_0 = self.sampler(x, x_0)
            # assert not torch.isinf(q_point).any(), "q_point should not contain inf"
            # assert not torch.isinf(q_point_0).any(), "q_point should not contain inf"
            # assert not torch.isnan(x).any(), "x should not contain nan"
            # assert not torch.isnan(x_0).any(), "x should not contain nan"
            nabla_phi_term = self.compute_mixgauss_gradient_term(x, q_point_0)
            laplace_phi_term = self.compute_mixgauss_laplace_term(x, q_point_0)
            assert not torch.isnan(
                laplace_phi_term
            ).any(), "laplace_phi_term should not contain nan"
            # normalize the laplace_phi_term as a inner product in high dimension space is usually large
            nabla_phi_term = 1 / (x.shape[1] * x.shape[2]) * nabla_phi_term
            laplace_phi_term = 1 / (x.shape[1] * x.shape[2]) * laplace_phi_term
            return q_point, q_point_0, nabla_phi_term, laplace_phi_term


# This MixtureGaussian will encounter difficulties of zero values
# We would use the approximation to avoid the zero values by re-designing the class
class MixtureGaussian_v0(torch.nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.tensor(sigma).to(self.device)

    def compute_mixgauss_value(self, mu, q_point):
        # q_point: [q_batch_size, q_dim1, q_dim2]
        # mu: [gauss_num, q_dim1, q_dim2]
        n_dim = q_point.shape[1] * q_point.shape[2]
        variance = self.sigma**2
        gauss_values = torch.zeros(
            [q_point.shape[0], mu.shape[0]]
        )  # [q_batch_size, gauss_num]
        for n in range(mu.shape[0]):
            gauss_values[:, n] = (
                2 * torch.tensor(torch.pi).to(self.device) * variance
            ) ** (-n_dim / 2) * torch.exp(
                -0.5 * torch.sum((q_point - mu[n, :, :]) ** 2, dim=(1, 2)) / variance
            )  # broadcasting for q_point - mu[n, :, :]
        mixgauss_value = torch.mean(gauss_values, dim=-1)  # [q_batch_size]
        return gauss_values, mixgauss_value

    def compute_mixgauss_gradient(self, mu, q_point, gauss_values):
        func = torch.zeros(
            [q_point.shape[0], q_point.shape[1], q_point.shape[2], mu.shape[0]]
        )  # [q_batch_size, q_dim1, q_dim2, gauss_num]

        for k in range(mu.shape[0]):
            func[:, :, :, k] = (
                -(q_point - mu[k, :, :])
                / self.sigma**2
                * gauss_values[:, k].reshape(-1, 1, 1)
            )  # broadcasting
        y = torch.mean(func, dim=-1)  # [q_batch_size, q_dim1, q_dim2]
        return y

    def compute_mixgauss_laplace(self, mu, q_point, gauss_values):
        x_dim = q_point.shape[1] * q_point.shape[2]
        func = torch.zeros([q_point.shape[0], mu.shape[0]])
        for k in range(mu.shape[0]):
            func[:, k] = (
                -1 / (self.sigma**2) * x_dim
                + torch.sum(
                    ((q_point - mu[k, :, :]) / self.sigma**2) ** 2, dim=(1, 2)
                )
            ) * gauss_values[:, k]
        y = torch.mean(func, dim=-1)
        return y

    def sampler(self, x, sample_number=None):
        """
        Sample from the Mixture Gaussian.
        In our class, the weight of each Gaussian is the same,
        so we can just sample from the standard Gaussian and shift them around to non-standard Gaussians.
        """
        if sample_number is None:
            sample_number = x.shape[0]  # batch size = gauss number = sample number

        # Sample from the standard Gaussian distribution
        data = torch.randn((sample_number,) + x.shape[1:]).to(self.device)

        # Calculate the number of samples distributed on per Gaussian
        L = data.shape[0] // x.shape[0]  # Integer division

        # Shift samples to represent each non-standard Gaussian
        for k in range(x.shape[0]):
            start_idx = k * L
            end_idx = (k + 1) * L if (k + 1) * L < data.shape[0] else data.shape[0]

            # Scale by the standard deviation and add the mean for each Gaussian
            data[start_idx:end_idx] *= torch.sqrt(self.sigma)
            data[start_idx:end_idx] += x[k]
        return data

    def forward(self, x):
        """
        Compute the value, gradient, and laplacian of the mixture Gaussian.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the following:
                - q_point (torch.Tensor): Sampled points from the mixture Gaussian.
                - phi (torch.Tensor): Mixture Gaussian values.
                - nabla_phi (torch.Tensor): Mixture Gaussian gradients.
                - laplace_phi (torch.Tensor): Mixture Gaussian Laplacians.
        """

        # no computational graph
        with torch.no_grad():
            q_point = self.sampler(x)

            gauss_values, phi = self.compute_mixgauss_value(x, q_point)

            nabla_phi = self.compute_mixgauss_gradient(x, q_point, gauss_values)
            laplace_phi = self.compute_mixgauss_laplace(x, q_point, gauss_values)
            return q_point, phi, nabla_phi, laplace_phi


def t_finite_diff(q_output, q_output_mtq, q_output_ptq, hp, hm):
    assert not torch.isnan(q_output_mtq).any(), "q_output_mtq should not contain nan"
    assert not torch.isnan(q_output_ptq).any(), "q_output_ptq should not contain nan"
    # assert not torch.isnan(hp).any(), "hp should not contain nan"
    # assert not torch.isnan(hm).any(), "hm should not contain nan"

    up = (
        hm**2 * q_output_ptq + (hp**2 - hm**2) * q_output - hp**2 * q_output_mtq
    )
    low = hm * hp * (hp + hm)  # TODO: check this file
    return up / low


# TODO: should have the same config as the model and refer to (30)
def sde(q, t, type="VE"):
    if type == "VE":
        drift = 0
        sigma_min = 0.01 * torch.pi
        sigma_max = torch.pi
        diffusion = (
            sigma_min ** (1 - t)
            * sigma_max**t
            * math.sqrt(2 * math.log(sigma_max / sigma_min))
        )
    elif type == "VP":
        beta_min = 0.1
        beta_max = 20
        beta_t = beta_min + t * (beta_max - beta_min)
        print("beta_t: ", beta_t)
        drift = -0.5 * beta_t[:, None, None] * q
        diffusion = torch.sqrt(beta_t)
    else:
        raise NotImplementedError("only support VE and VP right now")
    return drift, diffusion


def compute_PDEloss(
    q_output,
    time_pos,
    q_point,
    nabla_phi_term,
    laplace_phi_term,
    q_output_mtq,
    q_output_ptq,
    padding_mask,
    hp,
    hm,
    is_clip=False,
):
    _, diffusion = sde(q_point, time_pos, type="VE")
    # FIXME: paddings for q_output_mtq and q_output_ptq
    # TODO: Do we need to consider the padding_mask here? We may not need it. Discuss with the team here.
    # TODO: add more general sde form

    # assert not torch.isinf(q_output).any(), "q_output should not contain inf"
    # assert not torch.isinf(q_output_mtq).any(), "q_output_mtq should not contain inf"
    # assert not torch.isinf(q_output_ptq).any(), "q_output_ptq should not contain inf"

    # As normalizations are used in nabla_phi_term and laplace_phi_term, we need to do the same for LHS
    LHS = (
        t_finite_diff(q_output, q_output_mtq, q_output_ptq, hp, hm)
        / (q_point.shape[1] * q_point.shape[2])
        # * 1000
    )

    assert not torch.isnan(diffusion).any(), "diffusion should not contain nan"
    assert not torch.isnan(
        laplace_phi_term
    ).any(), "laplace_phi_term should not contain nan"
    assert not torch.isnan(
        nabla_phi_term
    ).any(), "nabla_phi_term should not contain nan"
    assert not torch.isnan(q_output).any(), "diffusion should not contain nan"

    RHS = (0.5 * diffusion**2 * torch.sum(q_output**2, dim=(1, 2))).reshape(
        -1, 1, 1
    ) * nabla_phi_term - (0.5 * diffusion**2 * laplace_phi_term).reshape(
        -1, 1, 1
    ) * q_output

    assert not torch.isnan(LHS).any(), "LHS should not contain nan"
    assert not torch.isnan(RHS).any(), "RHS should not contain nan"
    # Clip the RHS values to be within a certain range to avoid very large values
    # When in debugging, we can set is_clip to False to see the original values
    if is_clip:
        clip_min = -1e8
        clip_max = 1e8
        RHS_clipped = torch.clamp(RHS, min=clip_min, max=clip_max)
    else:
        RHS_clipped = RHS
    # logger.info(f"LHS: {LHS}")
    # logger.info(f"RHS_clipped: {RHS_clipped}"); exit()
    # TODO: here we use 1-norm to compute the loss, we may need to use 2-norm to compute the loss
    # res = torch.mean(torch.mean((LHS + RHS_clipped), dim=(1, 2)), dim=0).abs()
    res = torch.mean(torch.mean((LHS + RHS_clipped), dim=0).abs(), dim=(0, 1))

    return res
