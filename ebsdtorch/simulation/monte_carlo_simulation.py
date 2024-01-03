from typing import Tuple
import torch
from torch import Tensor


@torch.jit.script
def monte_carlo_batch(
    starting_energy_E_keV: float,
    atomic_number_z: float,
    unit_cell_density_rho: float,
    atomic_weight_A: float,
    mean_ionization_pot_J: float,
    n_electrons: int,
    n_max_collisions: int,
    incident_direction: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """

    Monte Carlo simulation of electron scattering in a scanning electron microscope.

    Args:
        starting_energy_E: The starting energy of the electrons in keV.
        atomic_number_z: The atomic number of the material (can be an approximate derived value)
        unit_cell_density_rho: The density of the material in g/cm^3.
        mean_ionization_pot_J: The mean ionization potential of the material in J.
        n_electrons: The number of electrons to simulate.
        n_max_collisions: The maximum number of collisions to simulate.
        incident_direction: The incident direction of the electrons.
        device: The device to run the simulation on.

    """

    # create a grid of random normalized 3D vectors
    phi_rand = torch.empty((n_electrons,), device=device, dtype=dtype)
    psi_rand = torch.empty((n_electrons,), device=device, dtype=dtype)
    step_rand = torch.empty((n_electrons,), device=device, dtype=dtype)

    # set the loop variables
    energies = torch.full(
        (n_electrons,), starting_energy_E_keV, device=device, dtype=dtype
    )
    depths = torch.zeros((n_electrons,), device=device, dtype=dtype)

    # not_exited_sample = torch.ones((n_electrons,), dtype=torch.bool, device=device)
    current_directions = incident_direction.repeat(n_electrons, 1)

    ones = torch.ones((n_electrons,), device=device, dtype=dtype)

    for _ in range(n_max_collisions):
        # sample all the random variables in place
        phi_rand.uniform_()
        psi_rand.uniform_(0.0, 2 * torch.pi)
        step_rand.uniform_()

        # calculate the screening factor
        alpha = (3.4e-3) * (atomic_number_z**0.66667) / energies

        # calculate the (energy dependent) screened Ruthford scattering cross section
        sigma_E = (
            5.21e-21
            * (atomic_number_z**2 / energies**2)
            * (4 * torch.pi / (alpha * (1 + alpha)))
            * (511 + energies) ** 2
            / (1024 + energies) ** 2
        )

        # calculate the mean free path (with cm -> nm conversion factor)
        mean_free_path_nm = 1e7 * atomic_weight_A / (unit_cell_density_rho * sigma_E)

        # calculate the random step size
        step_nm = -1.0 * mean_free_path_nm * torch.log(step_rand)

        # sample the polar and azimuthal scattering angles phi and psi
        # ---------- Equation 3.10 and 3.11 of Joy's book ----------
        phi = torch.acos(1 - (2 * alpha * phi_rand) / (1 + alpha - phi_rand))

        # calculate the new direction cosines of the electrons after scattering event
        # ---------- Equation 3.21 of Joy's book          ----------
        # There is a discrepancy between the book and the code. The book has an offset
        # 0.85 in the log but the original OpenCL code has 0.9911. I don't know which
        # is correct. Both values are just meant to force a positive value in the log.
        de_ds = (
            -0.00785
            * (atomic_number_z / (atomic_weight_A * energies))
            * torch.log((1.166 * energies / mean_ionization_pot_J) + 0.9911)
        )

        # calculate the new direction cosines of the electrons after scattering event
        # ---------- Equations 3.12a - 3.16 of Joy's book ----------
        x_old, y_old, z_old = current_directions.unbind(dim=-1)

        # # from https://www.theoretical-physics.com/0.1/src/math/other.html
        # ratio_xz = x_old / z_old
        # value_AN = -1.0 * ratio_xz
        # value_AN_dot_AM = -1.0 * ratio_xz * torch.rsqrt(1 + ratio_xz**2)

        # value_V1 = value_AN * torch.sin(phi)
        # value_V2 = value_AN_dot_AM * torch.sin(phi)
        # value_V3 = torch.cos(psi_rand)
        # value_V4 = torch.sin(psi_rand)

        # x_new = (
        #     x_old * torch.cos(phi) + value_V1 * value_V3 + y_old * value_V2 * value_V4
        # )
        # y_new = y_old * torch.cos(phi) + value_V4 * (
        #     z_old * value_V1 - x_old * value_V2
        # )
        # z_new = (
        #     z_old * torch.cos(phi) + value_V2 * value_V3 - y_old * value_V1 * value_V4
        # )

        ratio_xz = x_old / z_old
        ratio_xz_over_sqrt_1_plus_ratio_xz_squared = torch.sin(
            torch.atan2(ratio_xz, ones)
        )

        x_new = (
            x_old * torch.cos(phi)
            + -1.0 * ratio_xz * torch.sin(phi) * torch.cos(psi_rand)
            + y_old
            * -1.0
            * ratio_xz_over_sqrt_1_plus_ratio_xz_squared
            * torch.sin(phi)
            * torch.sin(psi_rand)
        )

        y_new = y_old * torch.cos(phi) + torch.sin(psi_rand) * (
            z_old * -1.0 * ratio_xz * torch.sin(phi)
            - x_old * -1.0 * ratio_xz_over_sqrt_1_plus_ratio_xz_squared * torch.sin(phi)
        )

        z_new = (
            z_old * torch.cos(phi)
            + -1.0
            * ratio_xz_over_sqrt_1_plus_ratio_xz_squared
            * torch.sin(phi)
            * torch.cos(psi_rand)
            - y_old * -1.0 * ratio_xz * torch.sin(phi) * torch.sin(psi_rand)
        )

        # combine and normalize the new direction cosines
        directions_new = torch.stack((x_new, y_new, z_new), dim=-1)
        # directions_new /= torch.norm(directions_new, dim=-1, keepdim=True)

        # calculate the new energy of the electrons after scattering event
        # ---------- Equation 3.17 of Joy's book ----------
        energies += step_nm * unit_cell_density_rho * de_ds

        # calculate the new depth of the electrons after scattering event
        depths += step_nm * z_new

        # set the old direction to the new direction
        current_directions = directions_new

    # turn off the mask for electrons that have exited the sample
    exited_sample = depths < 0

    # now that all of the electrons have been scattered, calculate the Rosca Lambert projection
    # of the current direction of the electrons that have exited and return it along with
    # the depth and energy of the electrons
    return (
        current_directions[exited_sample],
        depths[exited_sample],
        energies[exited_sample],
    )


def monte_carlo(
    starting_energy_E_keV: float,
    energy_bin_size_keV: float,
    energy_bin_count: int,
    atomic_number_z: float,
    unit_cell_density_rho: float,
    atomic_weight_A: float,
    mean_ionization_pot_J: float,
    n_electrons: int,
    vram_limit_GB: float,
    accumulator_radius: int,
    n_max_collisions: int,
    incident_direction: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """

    Monte Carlo simulation of electron scattering in a scanning electron microscope.
    This function uses the batch calculator to accumulate a 2D histogram of the scattering
    directions in the x-y plane from batch calls.

    Args:
        starting_energy_E_keV: The starting energy of the electrons in keV.
        energy_bin_size_keV: The size of the energy bins in keV.
        energy_bin_count: The number of energy bins, counting down from the starting energy.
        atomic_number_z: The atomic number of the material.
        unit_cell_density_rho: The density of the material in g/cm^3.
        mean_ionization_pot_J: The mean ionization potential of the material in J.
        n_electrons: The number of electrons to simulate.
        vram_limit_GB: The maximum amount of VRAM to use in GB.
        accumulator_radius: The radius of the square accumulator array in pixels.
        n_max_collisions: The maximum number of collisions to simulate.
        incident_direction: The incident direction of the electrons.

    Returns:
        A 3D tensor of shape (energy_bin_count, accumulator_radius * 2 + 1, accumulator_radius * 2 + 1)
        with each slice containing the 2D histogram of the scattering directions for that energy bin.

    """

    accumulator_side = accumulator_radius * 2 + 1

    # calculate the number of electrons to simulate per batch
    n_electrons_per_batch = int(vram_limit_GB * 1e9 / (32 * 3))

    # calculate the number of batches to run
    n_batches = int(n_electrons / n_electrons_per_batch) + 1

    # calculate the number of electrons to simulate in the last batch
    n_electrons_last_batch = n_electrons - (n_batches - 1) * n_electrons_per_batch

    # set up the accumulator array for the scattering directions
    accumulator = torch.zeros(
        energy_bin_count * accumulator_side * accumulator_side,
        device=device,
        dtype=torch.int64,
    )

    total_e = 0

    print(f"N batches = {n_batches}")

    # run the batches (skip the last batch)
    for i in range(n_batches):
        # calculate the number of electrons to simulate in this batch
        if i == n_batches - 1:
            n_electrons_this_batch = n_electrons_last_batch
        else:
            n_electrons_this_batch = n_electrons_per_batch

        # run the batch
        scatter_directions_2D, depths, energies = monte_carlo_batch(
            starting_energy_E_keV=starting_energy_E_keV,
            atomic_number_z=atomic_number_z,
            unit_cell_density_rho=unit_cell_density_rho,
            atomic_weight_A=atomic_weight_A,
            mean_ionization_pot_J=mean_ionization_pot_J,
            n_electrons=n_electrons_this_batch,
            n_max_collisions=n_max_collisions,
            incident_direction=incident_direction,
            dtype=dtype,
            device=device,
        )

        # # scattering directions come in as floats pairs in the square [-1, 1] x [-1, 1]
        # # we need them to drop into the accumulator array [0, accumulator_side] x [0, accumulator_side]
        # scatter_directions_2D_discrete = (
        #     (1.0 + (0.5 * scatter_directions_2D)) * accumulator_side
        # ).to(torch.int64)

        # energy_indices = ((energies - starting_energy_E_keV) / energy_bin_size_keV).to(
        #     torch.int64
        # )

        # accumulator_indices = (
        #     energy_indices * accumulator_side * accumulator_side
        #     + scatter_directions_2D_discrete[:, 0] * accumulator_side
        #     + scatter_directions_2D_discrete[:, 1]
        # )

        # accumulator.scatter_add_(
        #     dim=0,
        #     index=accumulator_indices,
        #     src=torch.ones_like(accumulator_indices, device=device),
        # )

        total_e += n_electrons_this_batch

        # print progress
        print(
            f"Batch {i + 1} of {n_batches} complete. Total electrons simulated (in millions) = {total_e / 1e6}"
        )

    # reshape the accumulator into a 3D array
    accumulator = accumulator.reshape(
        (energy_bin_count, accumulator_side, accumulator_side)
    )

    return accumulator


# # profile
# from torch.profiler import profile, ProfilerActivity
# with profile(
#     activities=[
#         ProfilerActivity.CPU,
#         ProfilerActivity.CUDA,
#     ],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
# ) as prof:
#     accumulator = monte_carlo(
#         starting_energy_E_keV=20.0,
#         energy_bin_size_keV=1.0,
#         energy_bin_count=10,
#         atomic_number_z=13.0,
#         unit_cell_density_rho=2.7,
#         atomic_weight_A=26.98,
#         mean_ionization_pot_J=77.0,
#         n_electrons=2000000000,
#         vram_limit_GB=4.0,
#         accumulator_radius=100,
#         n_max_collisions=300,
#         incident_direction=torch.tensor(
#             [0.0, 0.0, 1.0], device=torch.device("cuda"), dtype=torch.float16
#         ),
#         dtype=torch.float16,
#         device=torch.device("cuda"),
#     )

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
