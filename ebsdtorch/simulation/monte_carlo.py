"""
This file implements the Monte Carlo simulation of electron scattering from:

Joy, David C. Monte Carlo modeling for electron microscopy and microanalysis.
Vol. 9. Oxford University Press, 1995.

The code is based on the OpenCL implementation from the following file:

https://github.com/EMsoft-org/EMsoftOO/blob/develop/opencl/EMMC.cl

"""

from typing import Tuple
import torch
from torch import Tensor
import math


@torch.jit.script
def rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert projection.

    Args:
        pts: torch tensor of shape (..., 3) containing the points
    Returns:
        torch tensor of shape (..., 2) containing the projected points
    """
    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # floating point error can yield z above 1.0 example for float32 is:
    # xyz = [1.7817265e-04, 2.8403841e-05, 1.0000001e0]
    # so we have to clamp to avoid sqrt of negative number
    factor = torch.sqrt(torch.clamp(2.0 * (1.0 - torch.abs(z)), min=0.0))

    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor * (2.0 / (8.0**0.5))
    arctan_term = (
        torch.where(big < 0, -1, 1)
        * factor
        * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big))
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    # stack them together but flip the order if the condition is false
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    return out


# continuous MC simulation where backscattered events are reset to initial conditions
@torch.jit.script
def bse_sim_mc(
    starting_E_keV: float,
    n_exit_energy_bins: int,
    n_exit_direction_bins: int,
    n_exit_depth_bins: int,
    binsize_exit_energy: float,
    binsize_exit_depth: float,
    atom_num: float,
    unit_cell_density_rho: float,
    atomic_weight_A: float,
    n_electrons: int,
    n_bse_electrons_max: int,
    n_simulations_max: int,
    n_max_steps: int,
    sigma: float,
    omega: float,
    dtype: torch.dtype,
    device: torch.device,
):

    # depth maxval
    depth_max = n_exit_depth_bins * binsize_exit_depth

    # set up the accumulator array for the scattering directions
    accumulator = torch.zeros(
        (
            n_exit_energy_bins,
            n_exit_depth_bins,
            n_exit_direction_bins,
            n_exit_direction_bins,
        ),
        device=device,
        dtype=torch.float32,
    )

    # get all of the precomputed values
    bse_yield = 0
    sims_finished = 0
    ticks = torch.zeros((n_electrons,), device=device, dtype=torch.int32)
    rand_phi = torch.empty((n_electrons,), device=device, dtype=dtype)
    psi = torch.empty((n_electrons,), device=device, dtype=dtype)
    rand_step = torch.empty((n_electrons,), device=device, dtype=dtype)
    energies = torch.full((n_electrons,), starting_E_keV, device=device, dtype=dtype)
    depths = torch.zeros((n_electrons,), device=device, dtype=dtype)
    sigma_rad = sigma * torch.pi / 180.0
    omega_rad = omega * torch.pi / 180.0
    current_directions = torch.tensor(
        [
            [
                math.sin(sigma_rad) * math.cos(omega_rad),
                math.sin(sigma_rad) * math.sin(omega_rad),
                math.cos(sigma_rad),
            ]
        ],
        device=device,
        dtype=dtype,
    ).repeat(n_electrons, 1)

    # norm the current directions
    current_directions /= torch.norm(current_directions, dim=-1, keepdim=True)

    mean_ionization_pot_J = ((9.76 * atom_num) + (58.5 * (atom_num**-0.19))) * 1.0e-3

    while (bse_yield < n_bse_electrons_max) and (sims_finished < n_simulations_max):
        # calculate the screening factor
        # ---------- Equation 3.2 of Joy's book          ----------
        alpha = (3.4e-3) * (atom_num**0.66667) / energies

        # calculate the (energy dependent) screened Ruthford scattering cross section
        # ---------- Equation 3.1 and part of 3.3 of Joy's book          ----------
        sigma_E = (
            5.21
            * 602.2
            * (atom_num / energies) ** 2
            * (4 * torch.pi / (alpha * (1 + alpha)))
            * ((511.0 + energies) / (1024.0 + energies)) ** 2
        )

        # calculate the mean free path (with cm -> nm conversion factor)
        # ---------- Equation 3.3 without N_avagadro of Joy's book          ----------
        mean_free_path_nm = 1.0e7 * atomic_weight_A / (unit_cell_density_rho * sigma_E)

        # sample step
        rand_step.uniform_()

        # calculate the random step size
        step_nm = -1.0 * mean_free_path_nm * torch.log(rand_step)

        # ---------- Equation 3.21 of Joy's book          ----------
        # There is a discrepancy between the book and the code. The book has an offset
        # 0.85 in the log but the original OpenCL code has 0.9911.
        de_ds = (
            -0.00785
            * (atom_num / (atomic_weight_A * energies))
            * torch.log((1.166 * energies / mean_ionization_pot_J) + 0.9911)
        )

        # sample phi
        # ---------- Equation 3.10 and 3.11 of Joy's book ----------
        rand_phi.uniform_()
        phi = torch.acos(1 - ((2 * alpha * rand_phi) / (1 + alpha - rand_phi)))
        # sample psi
        psi.uniform_(0.0, 2 * torch.pi)

        # calculate the new direction cosines of the electrons after scattering event
        # ---------- Equations 3.12a - 3
        cx, cy, cz = current_directions.unbind(dim=-1)

        dir_z_pole = torch.abs(cz) > 0.99999
        dsq = torch.sqrt((1.0 - cz**2).clamp_(min=0.0))
        dsqi = 1.0 / dsq

        ca = torch.where(
            dir_z_pole,
            torch.sin(phi) * torch.cos(psi),
            torch.sin(phi) * (cx * cz * torch.cos(psi) - cy * torch.sin(psi)) * dsqi
            + cx * torch.cos(phi),
        )
        cb = torch.where(
            dir_z_pole,
            torch.sin(phi) * torch.sin(psi),
            (torch.sin(phi) * (cy * cz * torch.cos(psi) + cx * torch.sin(psi)) * dsqi)
            + cy * torch.cos(phi),
        )
        cc = torch.where(
            dir_z_pole,
            (cz / torch.abs(cz)) * torch.cos(phi),
            -1.0 * torch.sin(phi) * torch.cos(psi) * dsq + cz * torch.cos(phi),
        )

        # combine and normalize the new direction cosines
        current_directions = torch.stack((ca, cb, cc), dim=-1)

        # renormalize the direction cosines
        current_directions /= torch.norm(current_directions, dim=-1, keepdim=True)

        # save the escape depth incase we need it after checking new depth
        escape_depth = torch.abs(depths / current_directions[..., 2])

        # calculate the new depth of the electrons after scattering event
        depths += step_nm * current_directions[..., 2]

        # find backscattered electrons
        exited_sample = depths < 0

        # calculate the new energy of the electrons after scattering event
        # ---------- Equation 3.17 of Joy's book ----------
        energies += step_nm * unit_cell_density_rho * de_ds

        # find the bin indices
        exit_depth_indices = (
            ((escape_depth[exited_sample]) / depth_max) * n_exit_depth_bins
        ).to(torch.int32)
        exit_energy_indices = (
            ((starting_E_keV - energies[exited_sample]) / binsize_exit_energy)
        ).to(torch.int32)
        exit_direction_indices = (
            ((rosca_lambert(current_directions[exited_sample]) * 0.499999) + 0.5)
            * (n_exit_direction_bins)
        ).to(torch.int32)

        # mask away points with indices exceeding the max for depth or energy
        valid_mask = (
            (exit_depth_indices < n_exit_depth_bins)
            & (exit_energy_indices < n_exit_energy_bins)
            & (exit_energy_indices >= 0)
            & (exit_depth_indices >= 0)
        )

        exit_depth_indices = exit_depth_indices[valid_mask]
        exit_energy_indices = exit_energy_indices[valid_mask]
        exit_direction_indices = exit_direction_indices[valid_mask]

        # push MC batch into accumulator
        accumulator[
            exit_energy_indices,
            exit_depth_indices,
            exit_direction_indices[..., 0],
            exit_direction_indices[..., 1],
        ] += 1

        # reset if the ticks are greater than the max
        ticks += 1
        reset_mask = (ticks > n_max_steps) | exited_sample

        # reset the energy, depth, and direction of the backscattered electrons
        ticks[reset_mask] = 0
        energies[reset_mask] = starting_E_keV
        depths[reset_mask] = 0
        current_directions[reset_mask, 0] = math.sin(sigma_rad) * math.cos(omega_rad)
        current_directions[reset_mask, 1] = math.sin(sigma_rad) * math.sin(omega_rad)
        current_directions[reset_mask, 2] = math.cos(sigma_rad)

        # norm the current directions
        current_directions[reset_mask] /= torch.norm(
            current_directions[reset_mask], dim=-1, keepdim=True
        )

        # update the bse_yield and sims_finished
        bse_yield += exited_sample.sum().item()
        sims_finished += reset_mask.sum().item()

        if (sims_finished // 1000) % 100 == 0:
            print(
                f"n_bse_electrons: {bse_yield}, n_simulations: {sims_finished}, frac valid {torch.mean(valid_mask.float()).item()}"
            )

    return accumulator


# # run the continuous simulation
# accumulator = bse_sim_mc(
#     starting_E_keV=30.0,
#     n_exit_energy_bins=16,
#     n_exit_direction_bins=51,
#     n_exit_depth_bins=100,
#     binsize_exit_energy=1.0,
#     binsize_exit_depth=1.0,
#     atom_num=28.0,
#     unit_cell_density_rho=8.911,
#     atomic_weight_A=58.6934,
#     n_electrons=10000000,
#     n_bse_electrons_max=1000000000,
#     n_simulations_max=10000000,
#     n_max_steps=300,
#     sigma=70.0,
#     omega=0.0,
#     dtype=torch.float32,
#     device=torch.device("cuda"),
# )

# print(accumulator.shape)
# print(f"mean/std {torch.mean(accumulator).item()}, {torch.std(accumulator).item()}")

# import matplotlib.pyplot as plt

# accumulator = accumulator.cpu().numpy()[::-1]
# depth_integrated = accumulator.sum(axis=1)

# # make a 4x4 grid of image
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# for i in range(16):
#     axs[i // 4, i % 4].imshow(depth_integrated[i])
#     axs[i // 4, i % 4].set_title(f"Energy bin {i}")
#     axs[i // 4, i % 4].axis("off")
# plt.show()
# plt.savefig("monte_carlo_simulation_pytorch.png")
