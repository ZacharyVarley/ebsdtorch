"""
:Author: Zachary T. Varley
:Date: 2025
:License: MIT

This module is my attempt to collate the core logic for structure matrix
computations for EBSD dynamical electron scattering simulations. The file
EMEBSDFull in EMsoft (not EMsoftOO) contains the entire computational flow but
ultimately one needs to consult EMMCOpenCL, EMEBSDmasterOpenCL, and a vast
portion of the EMsoft codebase to get a good grasp. Additionally, I know in
advance that the structure matrix approach will have a much better comparitive
speedup on the GPU, as compared to the Bloch wave formalism, in torch due to the
large amount of matrix-matrix and matrix-vector multiplications. 

I am treating every beam as strong to simplify everything for now. I will add
Bethe potentials and double diffraction candidates later. I had erroneously
thought that the matrix exponential had to be evaluated for each k vector but it
is just over voltages for off-diagonal terms so the benefit of having a
non-jagged tensor is not as important as I thought.

The original EMMCOpenCL uses a hand-written matrix multiplication kernel written
in OpenCL dispatched on individual problems. See file "MBmoduleOpenCL.cl". I
hope for a 10x or more speedup using a GEMM from CUDA or metal over OpenCL due
to computing multiple problems at once with tiling and shared memory and all the
works that are automatically done for you in the background when using PyTorch.

I had to use pyOpenCL to implement the Monte Carlo simulation, but I trained a
small neural network to implicitly learn all possible MC simulations for
reasonable densities, SEM accelerating voltages, and angles of incidence. This
is not yet published as of Jan 2025.

I will provide a complete inventory of the computations required below:

1. Note unit cell unique sites in the asymmetric unit, equivalent positions, and
   Debye-Waller factors.
    a. Enumerate the Seitz matrices for the space group
    b. Average density of the unit cell and average atomic number
    c. For rhomobohedral cells use a hexagonal setting.
2. Compute a Monte Carlo simulation or use a fitted model to compute the
   expected histogram over exit energy, depth, and direction (4D histogram).
    a. Leverages equal area bijections between the 2D square and 3D hemisphere
       for equal weighting of histogram bins.
    b. David C. Joy's continuous slowing down approximation for now
    c. Fitted small neural network to all observed entries in Materials Project.
       That is yet to be published as of Jan 2025.
3. Implement complex atomic electron scattering factors from fittings of
   Weickenmeier and Kohl (WK) and of Lobato and Van Dyck (LVD) for each element.
    a. These fit tabulated Relativistic Hartree-Fock computations for each atom
       and given s value.
    b. Need a fast implementation of the exponential integral Ei(x) for WK. I
       ported over Boost's piecewise rational approximation.
    c. WK used exponentials to fit the data for analytical thermal diffuse
       scattering (TDS) integrals. Good for imaginary portion (seems just as
       accurate as manually integrating Lobato and Van Dyck's (LVD) terms).
    d. LVD used a Lorentzian terms to fit the real portion. This is more
       accurate / physical for the real portion (slow for the imaginary portion
       via numerical integration).
    e. These all use the Born approximation and Moliere's treatment is superior,
       but I have not seen it married with TDS computations.
    f. I am unfamiliar with the core loss term from the WK model. I just ported
       it over.
4. Sample a grid of k-vector directions on the 2-sphere that cover the
   fundamental sector for that space group.
    a. For 6-fold requires 2D hexagon to 3D hemisphere equal area bijection
5. Determine the participating hkl for the crystal structure given a minimum
   distance in nm and the centering for a given space group.
6. Compute the structure factor-like Sgh matrix for each unique site.
7. Compute structure matrices A for each voltage and k-vector. Only the diagonal
   terms depend on k-vectors.
    a. Compute a lookup table of Fourier potential coefficients Ucg for each
       valid g-h difference for each voltage. Note that hkl are closed under
       addition & subtraction with respect to validity for a given centering.
    b. Compute excitation errors for a given block of k-vectors and g-vectors
       for each voltage.
    c. Compute the dynamical matrix using the Ucg values for all g-h differences
       for off-diagonal terms and the excitation error and absorption length for
       the diagonal terms. (optionally apply Bethe potentials and check for
       double diffraction candidates). Here, I assume all beams are strong to
       have non-jagged tensors for the next step.
8. Do "Lie Group integration" to obtain the Darwin-Howie-Whelan (DHW) solution
   S(z_0) = exp(iAz_0) S(0). For each depth, the Monte Carlo simulation told us
   the relative amount of electrons originating from that depth for a given
   energy. We use the exp(iAz_i) S(0) for each depth and accumulate the overall
   intensity weighted with the MC data.
    a. OPTION 1: SLICE METHODS. Matrix exp(iAε) advances the wavefunction by a
       small step ε (default 1 nm in EMsoft for EBSD). The off-diagonal terms
       are independent of k so we save compute by separating the structure
       matrix A as A = T + V into a diagonal (T) and off-diagonal (V) matrix.
       Next, exp(i (T + V) ε) = exp(i T ε) exp(i V ε) exp(-0.5 (TV - VT) ε^2)
       ... ≈ exp(i T ε) exp(i V ε) ... first term truncation of the Zassenhaus
       theorem for Lie algebras. We can cache exp(i T ε) for each ε (if there
       are different depth steps or materials ... i.e. lamellar structures). And
       exp(i V ε) is just the simple exp of a diagonal matrix.
    b. OPTION 2: Spectral Decomposition of A. Here we simple compute the
       eigendecomposition of the non-Hermitian complex matrix A. A = U Λ U^-1.
       Then, exp(iAz) = U exp(iΛz) U^-1 for a given thickness z. We have to do
       this for each k and there is no way to use a rank-1 update of the matrix
       exponential etc. as the changing diagonals are full rank unfortunately.
    c. OPTION 3: Bloch Wave Formalism: solve a general complex eigenvalue
       problem as found in Chapter 5 of "Introduction to Conventional
       Transmission Electron Microscopy" by Marc De Graef. Speedup vs CPUs is
       not as stark as for the structure matrix approach.
9. Multiply by the Sgh matrix to get a tensor shaped like: (n_E, n_s, n_k, n_g,
   n_g). Where n_s is the number of unique sites in the asymmetric unit. Sum
   reduce real part on last two or up to three dimensions yielding: (n_E, n_s,
   n_k) or (n_E, n_k). Use symmetry operations to turn n_k to n_k_all and
   reshape to (..., n_k_square, n_k_square) images for saving the entire sphere
   of diffraction to disk for each energy.
   

"""

import torch
from torch import Tensor
import torch.nn.functional as F
from ebsdtorch.crystallography.unit_cell import Cell
from ebsdtorch.crystallography.diffraction import (
    calc_sgh,
    # calc_helper_scatter_matrix,
    calc_helper_bloch_formalism,
    calc_sg,
)
from ebsdtorch.crystallography.kvectors import kvectors_grid
from tqdm import tqdm


# start with a jitted version of the excitation error computation
# n_k x n_g x n_g can be very large so I need a fast batched version
# that occurs before each depth integration loop
@torch.jit.script
def calc_sg_batched(
    reciprocal_metric_tensor: Tensor,
    hkl_valid: Tensor,
    k_vectors: Tensor,
) -> Tensor:
    """
    Compute the excitation error for a batch of k-vectors and g-vectors.

    Args:
        reciprocal_metric_tensor: Reciprocal metric tensor (3x3).
        hkl_valid: Valid reflections (n_g, 3).
        k_vectors: K-vectors (n_k, 3).

    Returns:
        Excitation error tensor (n_k, n_g, n_g).
    """
    # (n_k, n_g, 3) + (n_g, 3) -> (n_k, n_g, n_g, 3)
    kpg = k_vectors + hkl_valid
    tkpg = 2.0 * k_vectors + hkl_valid

    xnom = -torch.einsum("...i,ij,...j->...", hkl_valid, reciprocal_metric_tensor, tkpg)
    q1 = torch.einsum("...i,ij,...j->...", kpg, reciprocal_metric_tensor, kpg) ** 0.5

    k_vectors_lengths = (
        torch.einsum(
            "...i,ij,...j->...",
            k_vectors,
            reciprocal_metric_tensor,
            k_vectors,
        )
        ** 0.5
    )
    kpg_dot_k_vectors = torch.einsum(
        "...i,ij,...j->...",
        kpg,
        reciprocal_metric_tensor,
        k_vectors,
    )
    # q2 = torch.acos((kpg_dot_k_vectors / (q1 * k_vectors_lengths)).clamp_(-1.0, 1.0))
    # xden = 2.0 * q1 * torch.cos(q2)
    xden = 2.0 * q1 * (kpg_dot_k_vectors / (q1 * k_vectors_lengths))

    sg = xnom / xden

    return sg


def ebsd_master_pattern(
    cell: Cell,
    voltages_kV: Tensor,
    depth_step: float,
    mc_weights: Tensor,
    dmin: float,
    k_chunk_size: int = 100,
    grid_half_width: int = 500,
    include_core: bool = True,
    include_phonon: bool = True,
    absorption: bool = True,
    scatter_factor: str = "WK",
) -> Tensor:
    """"""

    Sgh = calc_sgh(cell, dmin)
    n_sites = Sgh.size(0)
    # (n_sites, n_g, n_g) -> (1, n_sites, 1, n_g, n_g) for broadcasting
    Sgh = Sgh[:, None, :, :]
    n_voltages = voltages_kV.size(0)

    # # Get scattering matrix components
    # hkl_valid, qg_diff, xgp_000, mlambdas = calc_helper_scatter_matrix(
    #     cell,
    #     dmin=dmin,
    #     voltages_kV=voltages_kV,
    #     include_core=include_core,
    #     include_phonon=include_phonon,
    #     absorption=absorption,
    #     apply_qg_shift=apply_qg_shift,
    #     scatter_factor=scatter_factor,
    # )

    hkl_valid, ucg_diff, upmod_000, mlambdas = calc_helper_bloch_formalism(
        cell,
        dmin=dmin,
        voltages_kV=voltages_kV,
        include_core=include_core,
        include_phonon=include_phonon,
        absorption=absorption,
        scatter_factor=scatter_factor,
    )

    n_g = hkl_valid.size(0)

    # sample k-vectors on the 2-sphere
    kvectors, kij = kvectors_grid(
        cell,
        grid_half_width=grid_half_width,
    )  # cartesian space
    kvectors = cell.norm_vec(kvectors, "c")  # normalize
    kvectors = kvectors / mlambdas[:, None, None]  # (n_voltages, n_k, 3)
    kvectors = cell.transform_space(kvectors, "c", "r")  # to reciprocal space

    # off diagonal terms of the structure matrix are dynmat_V
    dynmat_V = ucg_diff
    dynmat_V.diagonal(dim1=1, dim2=2).zero_()

    # Initialize output master pattern
    master_pattern = torch.zeros(
        (n_voltages, n_sites, kvectors.size(1)),
        dtype=torch.float32,
        device=kvectors.device,
    )

    # Initial wavefunction [1,0,0,...]
    psi_0 = torch.zeros((n_g,), dtype=dynmat_V.dtype, device=kvectors.device)
    psi_0[0] = 1.0

    # convert integer tensor to float tensor of correct dtype
    hkl_valid_fp = hkl_valid.to(kvectors.dtype)

    # normalize mc weights
    mc_weights /= mc_weights.sum()

    # get depths_95 from the Monte Carlo weights
    energy_marginals_1d = mc_weights / mc_weights.sum(dim=1, keepdim=True)
    energy_cdfs = energy_marginals_1d.cumsum(dim=1)
    depth_cutoffs = torch.argmax((energy_cdfs >= 0.95).float(), dim=1)

    print(f"Depth cutoffs on descending exit keV: {depth_cutoffs}")

    # loop over voltages and then k vectors
    for v in tqdm(range(n_voltages), desc="Processing voltages"):
        for k_start in tqdm(
            range(0, kvectors.size(1), k_chunk_size),
            desc="Processing k-vectors",
            leave=False,
        ):
            k_end = min(k_start + k_chunk_size, kvectors.size(1))
            k_batch = kvectors[v, k_start:k_end]  # (n_voltages, k_chunk, 3)

            # Calculate excitation errors (k_chunk, n_g)
            sg = calc_sg_batched(
                cell.rmt, hkl_valid_fp[None, :, :], k_batch[:, None, :]
            )

            # manually set sg to zero for hkl 000
            sg[:, 0] = 0.0

            # form exp_A exactly instead of composing it via Zassenhaus
            A = (
                dynmat_V[v]
                + torch.diag_embed(torch.complex(2.0 * sg / mlambdas[v], upmod_000))
            ) * (1j * torch.pi * mlambdas[v])
            exp_A = torch.matrix_exp(A)  # (k_chunk, n_g, n_g)

            # Initialize accumulated Lgh matrix
            Lgh = torch.zeros(
                (len(k_batch), n_g, n_g),
                dtype=torch.complex64,
                device=kvectors.device,
            )

            # Initialize wavefunction
            psi = psi_0[None, :].expand(len(k_batch), n_g)

            # Integrate over depths
            for d in range(min(depth_cutoffs[v], 100)):
                # (k_chunk, n_g, n_g) @ (k_chunk, n_g, 1)
                psi = torch.matmul(exp_A, psi[..., None]).squeeze(-1)
                Lgh += mc_weights[v, d] * torch.einsum("ki,kj->kij", psi, psi.conj())

            # Sgh: (n_sites, 1, n_g, n_g) * (1, k_chunk, n_g, n_g)
            master_pattern[v, :, k_start:k_end] += (Sgh * Lgh.unsqueeze(0)).real.sum(
                dim=(-2, -1)
            )  # / float(depth_cutoffs[v])

    return master_pattern


if __name__ == "__main__":
    # Nickel
    cell = Cell(
        sg_num=225,
        atom_data=[(28, 0.0, 0.0, 0.0, 1.0, 0.0035)],
        abc=(0.35236, 0.35236, 0.35236),
        abc_units="nm",
        angles=(90.0, 90.0, 90.0),
        angles_units="deg",
    )

    print(cell)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_dtype = torch.float32

    cell = cell.to(device=device)
    cell = cell.set_fp_dtype(fp_dtype)

    from ebsdtorch.crystallography.ebsd_mc import MonteCarloSimulator

    # Create simulator with log-scale depth bins
    simulator = MonteCarloSimulator(
        cell=cell,
        kV=30.0,
        incidence_angle=70.0,
        binsize_exit_energy=0.5,
        n_energy_bins_extra=4,
        # binsize_exit_depth=0.5,
        # n_depth_bins=200,
        # depth_mode="linear",
        binsize_exit_depth=0.1,
        n_depth_bins=80,
        depth_mode="log",
        min_histogram_mass=10_000_000,
        n_max_steps_override=500,
    )

    # Run simulation
    histogram, total_electrons = simulator.run_simulation()

    bsize = 1.0
    histogram = simulator.resample_to_linear(
        histogram,
        n_linear_bins=100,
        linear_binsize=bsize,
    )

    marginal_energy_depth = histogram.sum(axis=(2, 3))

    # normalize overall
    marginal_energy_depth /= marginal_energy_depth.sum()

    # normalize per energy
    marginal_energy_depth /= marginal_energy_depth.sum(axis=1)[:, None]

    cdf_energy_depth = marginal_energy_depth.cumsum(axis=1)

    # for each energy, find the depth at which 95% of the electrons have escaped
    depths_95 = (cdf_energy_depth > 0.95).float().argmax(axis=1) * bsize

    print(f"Depths 95%: {depths_95}")

    # Get Monte Carlo weights (n_voltages, n_depths)
    mc_weights = histogram.sum(dim=(-2, -1))
    # mc_weights /= mc_weights.sum()

    voltages_kv = torch.tensor(
        # [20.0, 19.0],
        [30.0],
        device=device,
    )

    grid_half_width = 1001  # gigantic grid
    dmin = 0.04  # extra aggressive dmin
    batch_size = 32

    master_pattern = ebsd_master_pattern(
        cell=cell,
        voltages_kV=voltages_kv,
        depth_step=1.0,
        mc_weights=mc_weights,
        dmin=dmin,
        k_chunk_size=batch_size,
        grid_half_width=grid_half_width,
        include_core=True,
        include_phonon=True,
        absorption=True,
        scatter_factor="WK",
    )

    _, k_ij = kvectors_grid(cell, grid_half_width=grid_half_width)
    k_ij = k_ij.cpu().numpy()

    # fill an image using PIL instead
    from PIL import Image
    import numpy as np

    # each k_ij is a 2D vector and plot a point with color intensity
    # # after summing over all energies and sites
    # master_pattern = master_pattern.mean(dim=(0, 1))
    master_pattern = master_pattern[0, 0]

    # weighted sum over all energies then uniform sum over all sites
    # master_pattern = (master_pattern * energy_marginal[:2, None, None]).mean(dim=(0, 1))

    print(f"Intensity range: {master_pattern.min()} to {master_pattern.max()}")

    # normalize to 0-255
    master_pattern = (master_pattern - master_pattern.min()) / (
        master_pattern.max() - master_pattern.min()
    )
    master_pattern = (master_pattern * 255).byte().cpu().numpy()

    # create a new image
    arr = np.zeros((2 * grid_half_width + 1, 2 * grid_half_width + 1), dtype=np.uint8)

    # fill the image with the master pattern
    arr[grid_half_width + k_ij[:, 0], grid_half_width + k_ij[:, 1]] = master_pattern

    # rotate it by 90 degrees counterclockwise
    arr = np.rot90(arr, 1)

    # convert to PIL image
    img = Image.fromarray(arr).convert("L")

    img.save("test.png")
