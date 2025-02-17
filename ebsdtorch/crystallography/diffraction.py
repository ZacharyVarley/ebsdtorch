"""
:Author: Zachary T. Varley
:Date: 2025
:License: MIT

This module is my attempt to collate the core logic for structure matrix
computations for EBSD dynamical electron scattering simulations. The file
EMEBSDFull in EMsoft (not the more recent EMsoftOO) contains the entire
computational flow but ultimately one needs to consult EMMCOpenCL,
EMEBSDmasterOpenCL, and a vast portion of the EMsoft codebase to get a good
grasp. Additionally, I know in advance that the structure matrix approach will
have a much better comparitive speedup on the GPU, as compared to the Bloch wave
formalism, in torch due to the large amount of matrix-matrix and matrix-vector
multiplications.

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
import math
from torch import Tensor
from typing import Tuple, Optional, List
import torch.nn.functional as F
from ebsdtorch.crystallography.unit_cell import Cell
from ebsdtorch.crystallography.scattering_factors.scattering_factors_Hybrid import (
    scatter_factor_hybrid,
)
from ebsdtorch.crystallography.scattering_factors.scattering_factors_WK import (
    scatter_factor_WK,
)


def calc_lambda(voltage_kV: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate the relativistic electron wavelength and related parameters.

    Args:
        voltage: Accelerating voltage in kV as a torch tensor of shape (...,)

    Returns:
        :wavelength: Electron wavelength in nm
        :rel_corr: Relativistic correction factor (gamma)
        :psi_hat: Relativistic acceleration potential

    """
    # # Physical constants
    # h = 6.626070e-34  # Planck's constant in J⋅s
    # m0 = 9.109383e-31  # Electron rest mass in kg
    # e = 1.602177e-19  # Elementary charge in C
    # c = 2.997925e8  # Speed of light in m/s

    # Calculate wavelength using relativistic formula
    # temp1 = 1.0e9 * h / math.sqrt(2.0 * m0 * e)  # Convert to nm
    temp1 = 1.22642584554
    # temp2 = e * 0.5 * voltage_kV * 1000.0 / (m0 * c * c)
    temp2 = 0.00097847561649 * voltage_kV

    # Relativistic correction factor (gamma)
    rel_corr = 1.0 + 2.0 * temp2

    # Relativistic acceleration voltage
    psi_hat = voltage_kV * (1.0 + temp2) * 1000.0

    # Final wavelength
    wavelength = temp1 / torch.sqrt(psi_hat)

    return wavelength, rel_corr, psi_hat


def calc_sgh(
    cell: Cell,
    dmin: float,
) -> Tensor:
    """
    Compute structure factor-like Sgh matrix for EBSD simulations.

    Args:
        cell: Cell instance containing crystal structure info
        g_vectors: Tensor of shape (nns, 3) containing h,k,l indices of strong beams

    Returns:
        Tensor of shape (n_sites, nns, nns) containing Sgh matrix elements
    """
    # Get dimensions
    n_sites = cell.atom_types.shape[0]  # Number of atom sites
    device = cell.device

    # Get g-vectors (nns, 3)
    g_vectors = cell.get_reflections(dmin=dmin, difference_table=False)

    # prepended 000 to the list of reflections for diffraction calculations
    g_vectors = torch.cat(
        [
            torch.zeros(1, 3, device=g_vectors.device, dtype=g_vectors.dtype),
            g_vectors,
        ],
        dim=0,
    )

    nns = g_vectors.shape[0]

    # Compute g-h difference vectors (nns, nns, 3)
    g_minus_h = g_vectors[None, :, :] - g_vectors[:, None, :]
    g_minus_h_fp = g_minus_h.to(cell.apos[0].dtype)

    # Initialize output tensor
    Sgh = torch.zeros((n_sites, nns, nns), dtype=torch.complex64, device=device)

    # For each atom site
    for ip in range(n_sites):
        # Get Z^2 * occupancy for this site (scalar)
        Znsq = (cell.atom_types[ip].float() ** 2) * cell.atom_data[ip, 3]

        # Get positions for this orbit (equivalent positions)
        pos = cell.apos[ip]  # Shape: (n_equiv_pos, 3)

        # Compute squared length of g-h vectors (nns, nns)
        s_squared = 0.25 * cell.calc_length(g_minus_h, "r") ** 2

        # Compute Debye-Waller factor (nns, nns)
        dwf = Znsq * torch.exp(-cell.atom_data[ip, 4] * s_squared)

        # Compute phase factors for all positions and g-h combinations
        # reshape g_minus_h to (nns*nns, 3) and pos to (n_equiv_pos, 3)
        # Result: (nns*nns, n_equiv_pos)
        phase = 2 * torch.pi * torch.matmul(g_minus_h_fp.reshape(-1, 3), pos.T)

        # Complex exponential and sum over equivalent positions
        # Shape: (nns*nns)
        structure_factor = torch.sum(torch.cos(phase) + 1j * torch.sin(phase), dim=-1)

        # Multiply by DWF and reshape to (nns, nns)
        Sgh[ip] = structure_factor.reshape(nns, nns) * dwf

    return Sgh


def calc_helper_bloch_formalism(
    cell: Cell,
    dmin: float,
    voltages_kV: Tensor,
    include_core: bool = True,
    include_phonon: bool = True,
    absorption: bool = True,
    scatter_factor: str = "WK",
) -> Tuple[Tensor, Tensor, Tensor]:
    """

    Calculate the complex structure factor Ucg for EBSD simulations.

    Args:
        :cell: Cell instance containing crystal structure info
        :dmin: Minimum d-spacing for reflections in nm
        :voltages_kV: Tensor ccelerating voltages in kV (n_voltages,)
        :include_core: core loss effects. Default True.
        :include_phonon: thermal diffuse scattering (TDS). Default True.
        :absorption: Include absorption effects. Default True.
        :scatter_factor: Either "WK" or "Hybrid". Default "WK".

    Returns:
        :Ucg: (n_voltages, n_g, n_g) Fourier coefficients Ucg
        :upmod_diag: (n_voltages,)
        :mlambdas: (n_voltages,) relativistic wavelength in nm


    """
    # Get the atomic numbers and Debye-Waller factors... there is room for
    # improvement if two different sites have the same atomic number AND the
    # same DWF.
    device = cell.device
    zs = cell.atom_types
    dwf = cell.atom_data[:, 4]
    occupancy = cell.atom_data[:, 3]

    # get table of valid reflection differences
    hkl_diff_valid = cell.get_reflections(
        dmin=dmin,
        difference_table=True,
    )

    # prepend 000 to the list of reflections for diffraction calculations
    hkl_diff_valid = torch.cat(
        [
            torch.zeros(1, 3, device=hkl_diff_valid.device, dtype=hkl_diff_valid.dtype),
            hkl_diff_valid,
        ],
        dim=0,
    )

    hkl_diff_valid_fp = hkl_diff_valid.to(cell.apos[0].dtype)

    g = cell.calc_length(hkl_diff_valid, "r")  # in nm^-1

    # speed up scatter factor calculation by using unique g values
    g_unique, g_unique_indices = torch.unique(g, return_inverse=True)

    # scattering factors are in angstroms with a 2pi physicist's prefactor
    g_angstrom_twopi = 0.1 * 2.0 * torch.pi * g_unique
    u_angstrom_twopi = 10.0 * torch.sqrt(dwf / (8 * torch.pi**2))

    # Calculate the scattering factors
    if scatter_factor == "WK":
        scatter_factors = scatter_factor_WK(
            g=g_angstrom_twopi.float(),
            Z=zs,
            thermal_sigma=u_angstrom_twopi.float(),
            voltage_kV=voltages_kV,
            include_core=include_core,
            include_phonon=include_phonon,
        ).to(torch.complex64)
    elif scatter_factor == "Hybrid":
        scatter_factors = scatter_factor_hybrid(
            g=g_angstrom_twopi.float(),
            Z=zs,
            thermal_sigma=u_angstrom_twopi.float(),
            voltage_kV=voltages_kV,
            include_core=include_core,
            include_phonon=include_phonon,
        ).to(
            torch.complex64
        )  # shape: (n_voltages, n_sites, n_g)
    else:
        raise ValueError(
            f"Invalid scatter factor type {scatter_factor}. Must be 'WK' or 'Hybrid'."
        )

    # undo the unique operation
    scatter_factors = scatter_factors[:, :, g_unique_indices]

    # original Fortran code:
    # pref = 0.04787801/cell % vol/(4.0*cPi)
    # but I never included 4.0*cPi when porting WK scattering factors
    pref = 0.04787801 / cell.vol  # in nm^3 but 0.04787801 includes Å^3 conversion
    preg = 0.664840340614319  # 2.0 * sngl(cRestmass*cCharge/cPlanck**2)*1.0E-18
    pre = pref * preg

    # initialize ff and gg
    ff = torch.zeros((len(voltages_kV), len(g)), dtype=torch.complex64, device=device)
    gg = torch.zeros((len(voltages_kV), len(g)), dtype=torch.complex64, device=device)

    # loop over the sites and add terms into the structure factor
    # do it parallelized over the voltages
    # looped because "cell.apos" is a jagged list depending on site multiplicity
    # could be improved by padding the apos out to the maximum site multiplicity
    # and making occupancy 2D
    for site in range(len(zs)):
        # do all the orbit positions at once with a matmul
        # this is the same as the original Fortran code
        # i don't know why it is called p1 and it has shape (n_g,)
        p1 = torch.exp(-2j * torch.pi * (hkl_diff_valid_fp @ cell.apos[site].T)).sum(
            dim=1
        )

        # use p1 to scale ff and gg
        ff += p1[None, :] * scatter_factors[:, site].real * occupancy[site]
        gg += p1[None, :] * scatter_factors[:, site].imag * occupancy[site]

    vmod = pref * torch.abs(ff)

    # prepare return tensors (Ucg and qg)
    if absorption:
        upmod = preg * pref * torch.abs(gg)
        # complex Ucg = U_g + i Uprime_g = U_g,r-Uprime_g,i + i(U_g,i+Uprime_g,r)
        ucg = pre * torch.complex(ff.real - gg.imag, ff.imag + gg.real)
    else:
        upmod = torch.zeros_like(ff)
        ucg = pre * ff

    # cell%mLambda is the relativistic wavelength in the orignal f90 code...
    # I don't want it to be an attribute of the cell object because it is defined
    # by the voltage and the speed of light so we will compute it here

    # Ucg[0] is the relevant Ucg for 000
    # temp1 = 1.0D+9*cPlanck/dsqrt(2.D0*cRestmass*cCharge) = 1.22642584554
    temp1 = 1.22642584554

    # temp2 = 1.602177e-19 * 0.5 * voltages_kV * 1000.0 / (9.109383e-31 * 2.997925e8**2)
    temp2 = 0.00097847561649 * voltages_kV

    psi_hats = vmod[:, 0] + voltages_kV * (1.0 + temp2) * 1000.0  # (n_voltages,)
    # also emulate: cell%mPsihat = cell%mPsihat + dble(rlp%Vmod)

    mlambdas = temp1 / torch.sqrt(psi_hats)  # (n_voltages,)

    # now we have a (n_voltages, n_g_difs) tensor of Ucg values. We need to use
    # this to fill in a table of Ucg values for all possible g-h differences
    # where g and h are each a valid reflection when also filtering for dmin
    # those values are used in the dynamical matrix calculation for off-diagonal
    # terms.
    hkl_valid = cell.get_reflections(
        dmin=dmin,
        difference_table=False,
    )

    # prepend 000 to the list of reflections for diffraction calculations
    hkl_valid = torch.cat(
        [
            torch.zeros(1, 3, device=hkl_valid.device, dtype=hkl_valid.dtype),
            hkl_valid,
        ],
        dim=0,
    )

    hkl_diff_outer = hkl_valid[None, :, :] - hkl_valid[:, None, :]
    hkl_diff_outer = hkl_diff_outer.reshape(-1, 3)

    # now we just need the index for each hkl_dif that corresponds to the index
    # we take all of the differences that could have been, ignoring the
    # centering filtering, and then enumerate them and fill a flattened table
    # only where the reflection is allowed by centering (little sparse for FCC)
    hmax, kmax, lmax = cell.get_hkl_limits(dmin=dmin)
    ucg_table = torch.empty(
        (len(voltages_kV), (4 * hmax + 1) * (4 * kmax + 1) * (4 * lmax + 1)),
        device=device,
        dtype=torch.complex64,
    )

    hkl_diff_valid_indices = (
        (hkl_diff_valid[:, 0] + 2 * hmax) * (4 * kmax + 1) * (4 * lmax + 1)
        + (hkl_diff_valid[:, 1] + 2 * kmax) * (4 * lmax + 1)
        + (hkl_diff_valid[:, 2] + 2 * lmax)
    )

    # now we know the indices of the valid reflection differences in the
    # unfiltered table. We could have used a more complicated enumeration of the
    # reflection differences that inherently skipped invalid reflections for a
    # given centering. I will work that out later for a given centering to
    # remove table sparsity.
    ucg_table[:, hkl_diff_valid_indices] = ucg

    # now we use the table to fill a difference tensor using tensor indexing
    hkl_diff_outer_indices = (
        (hkl_diff_outer[:, 0] + 2 * hmax) * (4 * kmax + 1) * (4 * lmax + 1)
        + (hkl_diff_outer[:, 1] + 2 * kmax) * (4 * lmax + 1)
        + (hkl_diff_outer[:, 2] + 2 * lmax)
    )

    ucg_diff = ucg_table[:, hkl_diff_outer_indices].reshape(
        len(voltages_kV), len(hkl_valid), len(hkl_valid)
    )

    # need upmod for 000 and mlambdas for the diagonal of the dynamical matrix
    upmod_000 = upmod[:, 0]  # (n_v,)

    return hkl_valid, ucg_diff, upmod_000, mlambdas


def calc_helper_scatter_matrix(
    cell: Cell,
    dmin: float,
    voltages_kV: Tensor,
    include_core: bool = True,
    include_phonon: bool = True,
    absorption: bool = True,
    apply_qg_shift: bool = True,
    scatter_factor: str = "WK",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Calculate the complex structure factor Ucg for EBSD simulations.

    Args:
        :cell: Cell instance containing crystal structure info
        :dmin: Minimum d-spacing for reflections in nm
        :voltages_kV: Tensor ccelerating voltages in kV (n_voltages,)
        :include_core: Include core loss effects. Default is True.
        :include_phonon: Include phonon effects. Default is True.
        :absorption: Include absorption effects. Default is True.
        :apply_qg_shift: Apply qg shift. Default is True.
        :scatter_factor: Either "WK" or "Hybrid". Default is "WK".

    Returns:
        :qg: (n_voltages, n_g, n_g) Fourier coefficients
        :xgp_000: (n_voltages,) xgp for 000 for each voltage
        :mlambdas: (n_voltages,) relativistic wavelength in nm


    """
    # Get the atomic numbers and Debye-Waller factors... there is room for
    # improvement if two different sites have the same atomic number AND the
    # same DWF.
    device = cell.device
    zs = cell.atom_types
    dwf = cell.atom_data[:, 4]
    occupancy = cell.atom_data[:, 3]

    # get table of valid reflection differences
    hkl_diff_valid = cell.get_reflections(
        dmin=dmin,
        difference_table=True,
    )

    # prepend 000 to the list of reflections for diffraction calculations
    hkl_diff_valid = torch.cat(
        [
            torch.zeros(1, 3, device=hkl_diff_valid.device, dtype=hkl_diff_valid.dtype),
            hkl_diff_valid,
        ],
        dim=0,
    )

    hkl_diff_valid_fp = hkl_diff_valid.to(cell.apos[0].dtype)

    g = cell.calc_length(hkl_diff_valid, "r")  # in nm^-1

    # speed up scatter factor calculation by using unique g values
    g_unique, g_unique_indices = torch.unique(g, return_inverse=True)

    # scattering factors are in angstroms with a 2pi physicist's prefactor
    g_angstrom_twopi = 0.1 * 2.0 * torch.pi * g_unique
    u_angstrom_twopi = 10.0 * torch.sqrt(dwf / (8 * torch.pi**2))

    # Calculate the scattering factors
    if scatter_factor == "WK":
        scatter_factors = scatter_factor_WK(
            g=g_angstrom_twopi.float(),
            Z=zs,
            thermal_sigma=u_angstrom_twopi.float(),
            voltage_kV=voltages_kV,
            include_core=include_core,
            include_phonon=include_phonon,
        ).to(torch.complex64)
    elif scatter_factor == "Hybrid":
        scatter_factors = scatter_factor_hybrid(
            g=g_angstrom_twopi.float(),
            Z=zs,
            thermal_sigma=u_angstrom_twopi.float(),
            voltage_kV=voltages_kV,
            include_core=include_core,
            include_phonon=include_phonon,
        ).to(
            torch.complex64
        )  # shape: (n_voltages, n_sites, n_g)
    else:
        raise ValueError(
            f"Invalid scatter factor type {scatter_factor}. Must be 'WK' or 'Hybrid'."
        )

    # undo the unique operation
    scatter_factors = scatter_factors[:, :, g_unique_indices]

    # original Fortran code:
    # pref = 0.04787801/cell % vol/(4.0*cPi)
    # but I never included 4.0*cPi when porting WK scattering factors
    pref = 0.04787801 / cell.vol  # in nm^3 but 0.04787801 includes Å^3 conversion
    preg = 0.664840340614319  # 2.0 * sngl(cRestmass*cCharge/cPlanck**2)*1.0E-18

    # initialize ff and gg
    ff = torch.zeros((len(voltages_kV), len(g)), dtype=torch.complex64, device=device)
    gg = torch.zeros((len(voltages_kV), len(g)), dtype=torch.complex64, device=device)

    # loop over the sites and add terms into the structure factor
    # do it parallelized over the voltages
    # looped because "cell.apos" is a jagged list depending on site multiplicity
    # could be improved by padding the apos out to the maximum site multiplicity
    # and making occupancy 2D
    for site in range(len(zs)):
        # do all the orbit positions at once with a matmul
        # this is the same as the original Fortran code
        # i don't know why it is called p1 and it has shape (n_g,)
        p1 = torch.exp(-2j * torch.pi * (hkl_diff_valid_fp @ cell.apos[site].T)).sum(
            dim=1
        )

        # use p1 to scale ff and gg
        ff += p1[None, :] * scatter_factors[:, site].real * occupancy[site]
        gg += p1[None, :] * scatter_factors[:, site].imag * occupancy[site]

    vmod = pref * torch.abs(ff)
    vphase = torch.atan2(ff.imag, ff.real)
    umod = preg * vmod

    # prepare return tensors (Ucg and qg)
    if absorption:
        vpphase = torch.atan2(gg.imag, gg.real)
        upmod = preg * pref * torch.abs(gg)
    else:
        vpphase = torch.zeros_like(ff)
        upmod = torch.zeros_like(ff)

    # cell%mLambda is the relativistic wavelength in the orignal f90 code...
    # I don't want it to be an attribute of the cell object because it is defined
    # by the voltage and the speed of light so we will compute it here

    # Ucg[0] is the relevant Ucg for 000
    # temp1 = 1.0D+9*cPlanck/dsqrt(2.D0*cRestmass*cCharge) = 1.22642584554
    temp1 = 1.22642584554

    # temp2 = 1.602177e-19 * 0.5 * voltages_kV * 1000.0 / (9.109383e-31 * 2.997925e8**2)
    temp2 = 0.00097847561649 * voltages_kV

    psi_hats = vmod[:, 0] + voltages_kV * (1.0 + temp2) * 1000.0  # (n_voltages,)
    # also emulate: cell%mPsihat = cell%mPsihat + dble(rlp%Vmod)

    mlambdas = temp1 / torch.sqrt(psi_hats)  # (n_voltages,)

    # xg
    xg = torch.where(
        torch.abs(umod) > 0.0,
        1.0 / torch.abs(umod) / mlambdas[:, None],
        1.0e8,
    )

    # xgp
    xgp = torch.where(
        torch.abs(upmod) > 0.0,
        1.0 / torch.abs(upmod) / mlambdas[:, None],
        1.0e8,
    )

    # ar
    if absorption:
        if apply_qg_shift:
            qg = torch.complex(
                torch.cos(vphase) / xg - torch.sin(vpphase) / xgp,
                torch.cos(vpphase) / xgp + torch.sin(vphase) / xg,
            )
        else:
            arg = vpphase - vphase
            qg = torch.complex(
                1.0 / xg - torch.sin(arg) / xgp,
                torch.cos(arg) / xgp,
            )
    else:
        qg = torch.complex(1.0 / xg, torch.zeros_like(xg))

    # now we have a (n_voltages, n_g_difs) tensor of Ucg values. We need to use
    # this to fill in a table of Ucg values for all possible g-h differences
    # where g and h are each a valid reflection when also filtering for dmin
    # those values are used in the dynamical matrix calculation for off-diagonal
    # terms.
    hkl_valid = cell.get_reflections(
        dmin=dmin,
        difference_table=False,
    )

    # prepend 000 to the list of reflections for diffraction calculations
    hkl_valid = torch.cat(
        [
            torch.zeros(1, 3, device=hkl_valid.device, dtype=hkl_valid.dtype),
            hkl_valid,
        ],
        dim=0,
    )

    hkl_diff_outer = hkl_valid[None, :, :] - hkl_valid[:, None, :]
    hkl_diff_outer = hkl_diff_outer.reshape(-1, 3)

    # now we just need the index for each hkl_dif that corresponds to the index
    # we take all of the differences that could have been, ignoring the
    # centering filtering, and then enumerate them and fill a flattened table
    # only where the reflection is allowed by centering (little sparse for FCC)
    hmax, kmax, lmax = cell.get_hkl_limits(dmin=dmin)
    ucg_table = torch.empty(
        (len(voltages_kV), (4 * hmax + 1) * (4 * kmax + 1) * (4 * lmax + 1)),
        device=device,
        dtype=torch.complex64,
    )
    qg_table = torch.empty_like(ucg_table)

    hkl_diff_valid_indices = (
        (hkl_diff_valid[:, 0] + 2 * hmax) * (4 * kmax + 1) * (4 * lmax + 1)
        + (hkl_diff_valid[:, 1] + 2 * kmax) * (4 * lmax + 1)
        + (hkl_diff_valid[:, 2] + 2 * lmax)
    )

    # now we know the indices of the valid reflection differences in the
    # unfiltered table. table sparsity is not ideal here.
    qg_table[:, hkl_diff_valid_indices] = qg

    # now we use the table to fill a difference tensor using tensor indexing
    hkl_diff_outer_indices = (
        (hkl_diff_outer[:, 0] + 2 * hmax) * (4 * kmax + 1) * (4 * lmax + 1)
        + (hkl_diff_outer[:, 1] + 2 * kmax) * (4 * lmax + 1)
        + (hkl_diff_outer[:, 2] + 2 * lmax)
    )

    qg_diff = qg_table[:, hkl_diff_outer_indices].reshape(
        len(voltages_kV), len(hkl_valid), len(hkl_valid)
    )

    # xpg for scatter matrix approach
    xgp_000 = xgp[:, 0]  # (n_v,)

    return hkl_valid, qg_diff, xgp_000, mlambdas


def calc_sg(
    cell: Cell,
    gg: Tensor,
    kk: Tensor,
) -> Tensor:
    """

    Compute the sg values for EBSD simulations.

    Args:
        cell: Cell instance containing crystal structure info
        hkl_valid: Tensor of shape (n_g, 3) containing valid reflections
        k_vectors: Tensor of shape (n_k, 3) containing k-vectors

    Returns:
        Tensor of shape (n_k, n_g) containing sg values

    f90 code for reference:
    type(unitcell)                  :: cell
    real(kind=sgl),INTENT(IN)       :: gg(3)                !< reciprocal lattice point
    real(kind=sgl),INTENT(IN)       :: kk(3)                !< wave vector
    real(kind=sgl),INTENT(IN)       :: FN(3)                !< foil normal
    kpg=kk+gg
    tkpg=2.0*kk+gg
    xnom = -CalcDot(cell,gg,tkpg,'r')
    ! 2|k0+g|cos(alpha) = 2(k0+g).Foilnormal
    q1 = CalcLength(cell,kpg,'r')
    q2 = CalcAngle(cell,kpg,FN,'r')
    xden = 2.0*q1*cos(q2)
    sg = xnom/xden
    end function CalcsgSingle

    kk and FN were always the exact same when called by Initialize_ReflectionList in
    initializers.f90 for EMEBSDmasterOpenCL.f90, but I don't know why.

    """
    # assume the passed tensors are broadcastable
    gg = gg.to(kk.dtype)
    kpg = kk + gg  # (..., n_k, n_g, 3)
    tkpg = 2.0 * kk + gg  # (..., n_k, n_g, 3)

    # use equation of Ewald sphere
    xnom = -cell.calc_dot(gg, tkpg, "r")  # (..., n_k, n_g)

    # all three are shape (..., n_k, n_g)
    q1 = cell.calc_length(kpg, "r")
    q2 = cell.calc_angle(kpg, kk, "r")
    xden = 2.0 * q1 * torch.cos(q2)

    sg = xnom / xden

    return sg


# # test and visualize using matplotlib an example Sgh
# # NaCl
# cell = Cell(
#     sg_num=225,
#     atom_data=[
#         (11, 0.0, 0.0, 0.0, 1.0, 0.005),
#         (17, 0.0, 0.0, 0.5, 1.0, 0.005),
#     ],
#     abc=(0.5641, 0.5641, 0.5641),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )

# # Regular pyrite (https://next-gen.materialsproject.org/materials/mp-226)
# # has the 3-fold along the (x, x, x) diagonal so it doesn't give the wrong answer
# cell = Cell(
#     sg_num=205,
#     atom_data=[
#         (26, 0.0, 0.5, 0.5, 1.0, 0.005),  # 4a Wyckoff position
#         (16, 0.38538, 0.11462, 0.88538, 1.0, 0.005),  # 16c Wyckoff position
#     ],
#     abc=(0.540, 0.540, 0.540),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )

# # Nickel
# cell = Cell(
#     sg_num=225,
#     atom_data=[
#         (28, 0.0, 0.0, 0.0, 1.0, 0.00328),
#     ],
#     abc=(0.3524, 0.3524, 0.3524),
#     abc_units="nm",
#     angles=(90.0, 90.0, 90.0),
#     angles_units="deg",
# )


# # Magnesium structured Ti (https://next-gen.materialsproject.org/materials/mp-46)
# cell = Cell(
#     sg_num=194,
#     atom_data=[
#         (22, 1.0 / 3.0, 2.0 / 3.0, 0.25, 1.0, 0.005),
#     ],
#     abc=(0.2951, 0.2951, 0.464),
#     abc_units="nm",
#     angles=(90.0, 90.0, 120.0),
#     angles_units="deg",
# )

# # Fe2P2O7
# cell = Cell(
#     sg_num=1,
#     atom_data=[
#         (26, 0.505455, 0.255512, 0.708082, 1.0, 0.005),
#         (26, 0.544846, 0.606076, 0.302714, 1.0, 0.005),
#         (15, 0.938616, 0.1433, 0.209556, 1.0, 0.005),
#         (15, 0.120705, 0.718928, 0.787498, 1.0, 0.005),
#         (8, 0.239998, 0.308523, 0.371617, 1.0, 0.005),
#         (8, 0.75986, 0.982369, 0.351896, 1.0, 0.005),
#         (8, 0.749722, 0.309053, 0.084447, 1.0, 0.005),
#         (8, 0.034505, 0.945726, 0.988598, 1.0, 0.005),
#         (8, 0.304393, 0.558993, 0.924587, 1.0, 0.005),
#         (8, 0.306299, 0.865284, 0.636169, 1.0, 0.005),
#         (8, 0.817001, 0.549336, 0.634235, 1.0, 0.005),
#     ],
#     abc=(0.458, 0.530, 0.563),
#     abc_units="nm",
#     angles=(103.46, 98.57, 99.32),
#     angles_units="deg",
# )

if __name__ == "__main__":

    # Nickel
    cell = Cell(
        sg_num=225,
        atom_data=[
            (28, 0.0, 0.0, 0.0, 1.0, 0.00328),
        ],
        abc=(0.3524, 0.3524, 0.3524),
        abc_units="nm",
        angles=(90.0, 90.0, 90.0),
        angles_units="deg",
    )

    print(cell)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = cell.to(device)

    # Compute Sgh
    Sgh = calc_sgh(cell, dmin=0.05)

    print(f"Sgh shape: {Sgh.shape}")

    # Visualize
    import matplotlib.pyplot as plt

    # do 4 plots for each site and each real and imaginary part
    fig, axs = plt.subplots(2, len(Sgh), figsize=(15, 6))

    if len(Sgh) > 1:
        for i, (site_Sgh, ax1, ax2) in enumerate(zip(Sgh, axs[0], axs[1])):
            # Plot real part
            ax1.imshow(
                site_Sgh.real.cpu().numpy(),
                cmap="viridis",
                interpolation="none",
                origin="lower",
            )
            ax1.set_title(f"Real, Voltage {i}")
            ax1.axis("off")

            # Plot imaginary part
            ax2.imshow(
                site_Sgh.imag.cpu().numpy(),
                cmap="viridis",
                interpolation="none",
                origin="lower",
            )
            ax2.set_title(f"Imag, Voltage {i}")
            ax2.axis("off")
    else:
        # Plot real part
        axs[0].imshow(
            Sgh[0].real.cpu().numpy(),
            cmap="viridis",
            interpolation="none",
            origin="lower",
        )
        axs[0].set_title("Real")
        axs[0].axis("off")

        # Plot imaginary part
        axs[1].imshow(
            Sgh[0].imag.cpu().numpy(),
            cmap="viridis",
            interpolation="none",
            origin="lower",
        )
        axs[1].set_title("Imag")
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # calculate Ucg
    voltages_kV = torch.tensor([10.0, 100.0], device=device)

    # ucg, qg, upmod_diag = calc_dynmat_helper(cell, dmin=0.05, voltages_kV=voltages_kV)
    ucg, upmod_diag, mlambdas = calc_helper_bloch_formalism(
        cell=cell,
        dmin=0.05,
        voltages_kV=voltages_kV,
        include_core=True,
        include_phonon=True,
        absorption=True,
        scatter_factor="WK",
    )

    # visualize
    # do 4 plots for each site and each real and imaginary part
    fig, axs = plt.subplots(2, len(ucg), figsize=(15, 6))
    logscale = True

    if logscale:
        ucg.real = torch.log10(ucg.real.abs() + 1e-10)
        ucg.imag = torch.log10(ucg.imag.abs() + 1e-10)
        norm = plt.Normalize(vmin=-5, vmax=1)  # set the color scale limits
        cmap = plt.cm.viridis
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

    if len(ucg) > 1:
        for i, (site_ucg, ax1, ax2) in enumerate(zip(ucg, axs[0], axs[1])):
            # Plot real part
            if logscale:
                ax1.imshow(
                    site_ucg.real.cpu().numpy(),
                    cmap=cmap,
                    interpolation="none",
                    origin="lower",
                    norm=norm,
                )
            else:
                ax1.imshow(
                    site_ucg.real.cpu().numpy(),
                    cmap="viridis",
                    interpolation="none",
                    origin="lower",
                )
            ax1.set_title(f"Real, voltage {i}")
            ax1.axis("off")

            # Plot imaginary part
            if logscale:
                ax2.imshow(
                    site_ucg.imag.cpu().numpy(),
                    cmap=cmap,
                    interpolation="none",
                    origin="lower",
                    norm=norm,
                )
            else:
                ax2.imshow(
                    site_ucg.imag.cpu().numpy(),
                    cmap="viridis",
                    interpolation="none",
                    origin="lower",
                )
            ax2.set_title(f"Imag, voltage {i}")
            ax2.axis("off")
    else:
        # Plot real part
        if logscale:
            axs[0].imshow(
                ucg[0].real.cpu().numpy(),
                cmap=cmap,
                interpolation="none",
                origin="lower",
                norm=norm,
            )
        else:
            axs[0].imshow(
                ucg[0].real.cpu().numpy(),
                cmap="viridis",
                interpolation="none",
                origin="lower",
            )
        axs[0].set_title("Real")
        axs[0].axis("off")

        # Plot imaginary part
        if logscale:
            axs[1].imshow(
                ucg[0].imag.cpu().numpy(),
                cmap=cmap,
                interpolation="none",
                origin="lower",
                norm=norm,
            )
        else:
            axs[1].imshow(
                ucg[0].imag.cpu().numpy(),
                cmap="viridis",
                interpolation="none",
                origin="lower",
            )
        axs[1].set_title("Imag")
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()
