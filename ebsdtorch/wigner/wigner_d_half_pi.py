"""

    This file contains faster implementations for the case of BETA = π/2.

    Numerically stable calculation of the Wigner d-coefficients up to 10^9
    integer degree and order using virtual extended precision. You would
    probably have to modify the code for half integers. See the following
    publications for an explanation of the recursive algorithm:

    "Numerical computation of spherical harmonics of arbitrary degree and order
    by extending exponent of floating point numbers"

    https://doi.org/10.1007/s00190-011-0519-2

    "Numerical computation of Wigner's d-function of arbitrary high degree and
    orders by extending exponent of floating point numbers"

    http://dx.doi.org/10.13140/RG.2.2.31922.20160
    https://www.researchgate.net/publication/309652602

    Both by Toshio Fukushima.

    ---------------------------------------------
    -----**All Notation Follows Fukushima**------
    ---------------------------------------------

"""

from typing import Tuple
import torch
from torch import Tensor
from ebsdtorch.wigner.x_numbers import x2f, xlsum2, x_norm


@torch.jit.script
def f_powers_half_pi(
    largest_power: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """

    Instead of finding sin(pi/4)^n and cos(pi/4)^n, we will always be accessing
    that table for c(k+m) * s(k-m) = (2**(-1/2))**(2k) = 2**(-k)

    So we just need x numbers for powers of 1/2.

    """

    # powers up to 2^53 can be exactly represented in float64
    powers = torch.arange(0, largest_power + 1, dtype=torch.float64, device=device)

    half = torch.tensor(0.5, dtype=torch.float64, device=device)

    # calculate the powers in log space
    lhalf = torch.log2(half)

    lhalf_sign = torch.sign(lhalf)

    # calculate the powers in log space
    lhalf_powers = lhalf * powers

    # split according to 2**960
    lhalf_powers_x = torch.abs(lhalf_powers) % 960

    # calculate the x-number exponents
    half_powers_x_i = ((torch.abs(lhalf_powers) // 960) * lhalf_sign).to(torch.int32)

    # calculate the x-number mantissas
    half_powers_x = torch.pow(2.0, lhalf_powers_x * lhalf_sign)

    # call x_norm to normalize the x-numbers
    half_powers_x, half_powers_x_i = x_norm(half_powers_x, half_powers_x_i)

    return half_powers_x, half_powers_x_i


@torch.jit.script
def build_km_seeds_half_pi(
    order_max: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Build the recursion seed table for the Wigner d-coefficients. The values
    d_k_km and d_k+1_km can be used in a double recurrence to calculate the rest
    of the table up to d_jmax_km.

    Args:
        order_max: Maximum azimuthal order in k or m.
        device: The device to build the table on.

    Returns:
        (d_k_km, d_k_km_i, d_kp1_km, d_kp1_km_i): The mantissa and exponent of
        d_k_km and d_k+1_km each of shape (order_max + 1, order_max + 1).

    """

    output_shape = (order_max + 1, order_max + 1)

    output_dk_km = torch.zeros(output_shape, dtype=torch.float64, device=device)
    output_dk_km_i = torch.zeros(output_shape, dtype=torch.int32, device=device)

    output_d_kp1_km = torch.zeros(output_shape, dtype=torch.float64, device=device)
    output_d_kp1_km_i = torch.zeros(output_shape, dtype=torch.int32, device=device)

    # calculate the powers of 1/2
    half_powers_x, half_powers_x_i = f_powers_half_pi(
        largest_power=order_max + 1, device=device
    )

    e_km_x_col = torch.ones(output_shape[0], dtype=torch.float64, device=device)
    e_km_x_i_col = torch.zeros(output_shape[0], dtype=torch.int32, device=device)

    coords = torch.arange(
        0, 2 * order_max + 2, step=2, dtype=torch.int32, device=device
    )

    coords_fp = coords.double()

    k2_coords, m2_coords = torch.meshgrid(coords, coords, indexing="ij")
    k2_coords_fp = k2_coords.double()

    kpm = (k2_coords + m2_coords) // 2
    kmm = (k2_coords - m2_coords) // 2

    kpm_fp = kpm.double()
    kmm_fp = kmm.double()

    # removing the bottleneck by iterating column by column using vectorized operations
    for k_index in range(order_max + 1, dtype=torch.int32, device=device):

        # find where k2 is equal to m2 or greater than m2
        # these masks only split the update of the e_km terms
        mask_k2_eq_m2 = coords[k_index] == coords
        mask_k2_gr_m2 = coords[k_index] > coords

        # find valid mask
        mask_valid = mask_k2_eq_m2 | mask_k2_gr_m2

        # find valid m coords
        valid_m_coords_fp = coords_fp[mask_valid]

        # update the e_km terms for k2 == m2 case
        e_km_x_col[mask_k2_eq_m2] = 1.0
        e_km_x_i_col[mask_k2_eq_m2] = 0

        # for k2 > m2 case update the e_km terms
        # f = (k2.double() * (k2.double() - 1)) / (kpm.double() * kmm.double())
        f = (
            k2_coords_fp[k_index, mask_k2_gr_m2]
            * (k2_coords_fp[k_index, mask_k2_gr_m2] - 1)
        ) / (kpm_fp[k_index, mask_k2_gr_m2] * kmm_fp[k_index, mask_k2_gr_m2])

        # update the e_km terms
        e_km_x_col[mask_k2_gr_m2] = e_km_x_col[mask_k2_gr_m2] * f**0.5
        e_km_x_col[mask_k2_gr_m2], e_km_x_i_col[mask_k2_gr_m2] = x_norm(
            e_km_x_col[mask_k2_gr_m2], e_km_x_i_col[mask_k2_gr_m2]
        )

        # dk_km (c_kpm * s_kmm) only depends on k for BETA = π/2
        dk_km = half_powers_x[k_index]
        dk_km_i = half_powers_x_i[k_index]

        # multiply by the e_km term (depends on k and m)
        dk_km = dk_km * e_km_x_col[mask_valid]
        dk_km_i = dk_km_i + e_km_x_i_col[mask_valid]
        dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

        # add to the table
        output_dk_km[k_index, mask_valid] = dk_km
        output_dk_km_i[k_index, mask_valid] = dk_km_i

        a_km = (
            (k2_coords_fp[k_index, mask_valid] + 1)
            / (
                (k2_coords_fp[k_index, mask_valid] + valid_m_coords_fp + 2)
                * (k2_coords_fp[k_index, mask_valid] - valid_m_coords_fp + 2)
            )
        ) ** 0.5 * (-1.0 * valid_m_coords_fp)

        # multiply the first seed term by a_km and turn X-number into float
        d_kp1_km = dk_km * a_km
        d_kp1_km_i = dk_km_i
        d_kp1_km, d_kp1_km_i = x_norm(d_kp1_km, d_kp1_km_i)

        # add to the table
        output_d_kp1_km[k_index, mask_valid] = d_kp1_km
        output_d_kp1_km_i[k_index, mask_valid] = d_kp1_km_i

    return output_dk_km, output_dk_km_i, output_d_kp1_km, output_d_kp1_km_i


@torch.jit.script
def build_a_b_jkm_volume_half_pi(
    degree_max: int,
    order_max: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """

    Build the volume of the Wigner d-coefficient table for the case when
    degree_max >= j >= order_max >= k >= m >= 0. These are the coefficients for the double
    recurrence relation between d_k+2_km and both d_k_km and d_k+1_km.

    Args:
        degree_max: The maximum degree to calculate d_jkm for.
        order_max: The maximum order to calculate d_jkm for.

    Returns:
        a_volume: The a coefficients for all j, k, m: shape (degree_max + 1, order_max + 1, order_max + 1)
        b_volume: The b coefficients for all j, k, m: shape (degree_max + 1, order_max + 1, order_max + 1)

    """

    # make volumes for a and b coefficients for all j, k, m
    output_shape = (degree_max + 1, order_max + 1, order_max + 1)
    a_volume = torch.zeros(output_shape, dtype=torch.float64, device=device)
    b_volume = torch.zeros(output_shape, dtype=torch.float64, device=device)

    jkm = torch.stack(
        torch.meshgrid(
            torch.arange(0, degree_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)

    j, k, m = torch.unbind(jkm.double(), dim=-1)

    valid_mask_flat = (j >= k + 2) & (k >= m)
    valid_mask = valid_mask_flat.reshape(
        int(degree_max + 1), int(order_max + 1), int(order_max + 1)
    )

    j = j[valid_mask_flat]
    k = k[valid_mask_flat]
    m = m[valid_mask_flat]

    j2 = 2 * j
    k2 = 2 * k
    m2 = 2 * m

    v_jkm = j2 * ((j2 + k2 - 2) * (j2 - k2 - 2) * (j2 + m2 - 2) * (j2 - m2 - 2)) ** 0.5
    w_jkm = 1.0 / ((j2 - 2) * ((j2 + k2) * (j2 - k2) * (j2 + m2) * (j2 - m2)) ** 0.5)

    a_volume[valid_mask] = (2.0 * j2 - 2.0) * (-1.0 * (k2 * m2)) * w_jkm
    b_volume[valid_mask] = v_jkm * w_jkm

    return a_volume, b_volume


@torch.jit.script
def recurse_km_table_seed_table(
    d_k_km_x: Tensor,
    d_k_km_x_i: Tensor,
    d_kp1_km_x: Tensor,
    d_kp1_km_x_i: Tensor,
    a_jkm_volume: Tensor,
    b_jkm_volume: Tensor,
    degree_max: int,
    device: torch.device,
):
    """

    Recursively calculate the Wigner d-coefficients for a given seed value of k and m,
    for all j values from k to j_max.

    Args:
        d_k_km_x: The mantissa of the Wigner d-coefficients for k and m.
        d_k_km_x_i: The exponent of the Wigner d-coefficients for k and m.
        d_kp1_km_x: The mantissa of the Wigner d-coefficients for k+1 and m.
        d_kp1_km_x_i: The exponent of the Wigner d-coefficients for k+1 and m.
        a_jkm_volume: The a coefficients for all j, k, m: shape (degree_max + 1, order_max + 1, order_max + 1)
        b_jkm_volume: The b coefficients for all j, k, m: shape (degree_max + 1, order_max + 1, order_max + 1)
        degree_max: The maximum degree to calculate d_jkm for.

    Returns:
        d_jkm_volume: The Wigner d-coefficient quarter table: shape (degree_max + 1, order_max + 1, order_max + 1)

    """

    order_max = d_k_km_x.shape[0] - 1

    # make volumes for a and b coefficients for all j, k, m
    d_jkm_volume_x = torch.full(
        (degree_max + 1, order_max + 1, order_max + 1),
        dtype=torch.float64,
        fill_value=torch.inf,
        device=device,
    )
    d_jkm_volume_x_i = torch.full(
        (degree_max + 1, order_max + 1, order_max + 1),
        dtype=torch.int32,
        fill_value=0,
        device=device,
    )

    jkm = torch.stack(
        torch.meshgrid(
            torch.arange(0, degree_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    j_3D, k_3D, m_3D = torch.unbind(jkm, dim=-1)

    km = torch.stack(
        torch.meshgrid(
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    k_2D, m_2D = torch.unbind(km, dim=-1)

    first_seed_mask_3D = ((j_3D == k_3D) & (k_3D >= m_3D)).reshape(
        degree_max + 1, order_max + 1, order_max + 1
    )
    first_seed_mask_2D = (k_2D >= m_2D).reshape(order_max + 1, order_max + 1)

    d_jkm_volume_x[first_seed_mask_3D] = d_k_km_x[first_seed_mask_2D].view(-1)
    d_jkm_volume_x_i[first_seed_mask_3D] = d_k_km_x_i[first_seed_mask_2D].view(-1)

    second_seed_mask_3D = (
        (j_3D == (k_3D + 1)) & (k_3D >= m_3D) & ((k_3D + 1) <= degree_max)
    ).reshape(degree_max + 1, order_max + 1, order_max + 1)
    second_seed_mask_2D = ((k_2D >= m_2D) & ((k_2D + 1) <= degree_max)).reshape(
        order_max + 1, order_max + 1
    )

    d_jkm_volume_x[second_seed_mask_3D] = d_kp1_km_x[second_seed_mask_2D].view(-1)
    d_jkm_volume_x_i[second_seed_mask_3D] = d_kp1_km_x_i[second_seed_mask_2D].view(-1)

    for j in torch.arange(2, degree_max + 1, dtype=torch.int32):
        valid_indices_mask = ((j >= (k_2D + 2)) & (k_2D >= m_2D)).reshape(
            order_max + 1, order_max + 1
        )

        b = b_jkm_volume[j, valid_indices_mask]
        a = a_jkm_volume[j, valid_indices_mask]

        if valid_indices_mask.sum() == 0:
            pass

        d_2_terms_prior_x = d_jkm_volume_x[j - 2, valid_indices_mask]
        d_2_terms_prior_x_i = d_jkm_volume_x_i[j - 2, valid_indices_mask]

        d_1_term_prior_x = d_jkm_volume_x[j - 1, valid_indices_mask]
        d_1_term_prior_x_i = d_jkm_volume_x_i[j - 1, valid_indices_mask]

        d_j_km_x, d_j_km_x_i = xlsum2(
            -b,
            a,
            d_2_terms_prior_x,
            d_2_terms_prior_x_i,
            d_1_term_prior_x,
            d_1_term_prior_x_i,
        )

        d_jkm_volume_x[j, valid_indices_mask] = d_j_km_x
        d_jkm_volume_x_i[j, valid_indices_mask] = d_j_km_x_i

    d_jkm_volume = x2f(d_jkm_volume_x, d_jkm_volume_x_i)

    return d_jkm_volume


@torch.jit.script
def build_jkm_volume_half_pi(
    degree_max: int,
    order_max: int,
    device: torch.device,
):
    """

    Build the volume of the Wigner d-coefficient table for the case when
    degree_max >= j >= order_max >= k >= m >= 0.

    Args:
        degree_max: The maximum degree to calculate d_jkm for.
        order_max: The maximum order to calculate d_jkm for.

    Returns:
        d_jkm_volume: The Wigner d-coefficient quarter table: shape (degree_max + 1, order_max + 1, order_max + 1)

    """

    # make volumes for a and b coefficients for all j, k, m
    a_volume, b_volume = build_a_b_jkm_volume_half_pi(
        degree_max=degree_max,
        order_max=order_max,
        device=device,
    )

    # build the seed table for the recursion
    dk_km, dk_km_i, dkp1_km, dkp1_km_i = build_km_seeds_half_pi(
        order_max=order_max, device=device
    )

    # recursively calculate the rest of the table
    d_jkm_volume = recurse_km_table_seed_table(
        d_k_km_x=dk_km,
        d_k_km_x_i=dk_km_i,
        d_kp1_km_x=dkp1_km,
        d_kp1_km_x_i=dkp1_km_i,
        a_jkm_volume=a_volume,
        b_jkm_volume=b_volume,
        degree_max=degree_max,
        device=device,
    )

    return d_jkm_volume


@torch.jit.script
def access_d_beta_half_pi_ts2kit(
    j: Tensor, k: Tensor, m: Tensor, quarter_table: Tensor
) -> Tensor:
    """

    Index into the quarter table of Wigner d-coefficients. The quarter table
    has the following shape: (degree_max + 1, order_max + 1, order_max + 1),
    and it is only filled in where j >= k, and j >= m. However, m and k are
    allowed to be negative. We use symmetry relations to calculate any values
    outside of the quarter table.

    There is an anomalous factor at the end because the Wigner d-coefficients
    are defined in TS2Kit with the first ordinal corresponding to the exponentiation
    of the first angle in the (Z, Y, Z) pair. Both Toshio and EMSphInx have them
    flipped so that the first ordinal corresponds to the exponentiation of the
    third angle in the (Z, Y, Z) pair.

    Args:
        j: The degree index.
        k: The order index.
        m: The order index.
        quarter_table: The quarter table of Wigner d-coefficients.

    Returns:
        d_jkm: The Wigner d-coefficient for the given indices.

    d^j_-k_-m = (-1)^(k-m) d^j_k_m

    d^j_k_-m(BETA) = (-1)^(j + k + 2m) d^j_k_m(π - BETA)

    d^j_-k_m(BETA) = (-1)^(j + 2k + 3m) d^j_k_m(π - BETA)

    For BETA = π/2, we have: d^j_k_m(π - π/2) = (-1)^(j + k + m) d^j_k_m(π/2)

    So that...

    d^j_-k_-m = (-1)^(k-m) d^j_k_m

    d^j_k_-m = (-1)^(j + 2k + 3m) d^j_k_m

    d^j_-k_m = (-1)^(j + k + 2m) d^j_k_m

    d^j_m_k = (-1)^(k-m) d^j_k_m

    """

    # calculate the symmetry relation for the Wigner d-coefficients
    mask_need_swap = torch.abs(k) < torch.abs(m)

    k_ind = torch.where(mask_need_swap, m, k)
    m_ind = torch.where(mask_need_swap, k, m)

    k_ind_abs = torch.abs(k_ind)
    m_ind_abs = torch.abs(m_ind)

    mask_both_neg = (k_ind < 0) & (m_ind < 0)
    mask_only_k_neg = (k_ind < 0) & (m_ind >= 0)
    mask_only_m_neg = (m_ind < 0) & (k_ind >= 0)

    prefactor = (
        torch.where(
            mask_both_neg,
            (-1.0) ** (k_ind - m_ind),
            1.0,
        )
        * torch.where(
            mask_only_k_neg,
            (-1.0) ** (j - m_ind),
            1.0,
        )
        * torch.where(
            mask_only_m_neg,
            (-1.0) ** (j + k_ind),
            1.0,
        )
        * torch.where(
            mask_need_swap,
            (-1.0) ** (k_ind - m_ind),
            1.0,
        )
    )

    # Note here: (-1.0) ** (k_ind - m_ind)
    # Swap the meaning of k and m because that is what TS2KIT has and it is
    # more sensible to me as well that the first subscript corresponds to the
    # exponentiation of the first angle amongst (Z, Y, Z)

    return (
        quarter_table[j, k_ind_abs, m_ind_abs] * prefactor * (-1.0) ** (k_ind - m_ind)
    )


# @torch.jit.script
def wigner_d_SHT_weights_half_pi(
    B: int,
    device: torch.device = torch.device("cpu"),
):
    """

    Args:
        degree_max: The maximum degree to calculate d_jkm for.
        order_max: The maximum order to calculate d_jkm for.
        device: The device to build the table on.

    Returns:
        zeta: The Wigner d-coefficients for the spherical harmonic transform weights.

    """

    order_max = B - 1

    # build the volume of the Wigner d-coefficients
    d_jkm_volume = build_jkm_volume_half_pi(
        degree_max=order_max + 2,
        order_max=order_max + 2,
        device=device,
    )

    """
    prefactors (doesn't clearly correspond to any equation in the PDF... Yikes!)

    N = 2 * degree_max

    if (k % 2) == 0:
        if m == 0:
            c = np.sqrt((2 * j + 1) / 2.0) * np.sqrt(N)
        else:
            c = np.sqrt((2 * j + 1) / 2.0) * np.sqrt(2.0 * N)
        if (k % 4) == 2:
            c *= -1.0
        coeff = (
            c * d[j, m + (B - 1), -k + (B - 1)] * d[j, m + (B - 1), B - 1]
        )
    else:
        if m == j:
            coeff = 0.0
        else:
            c = np.sqrt((2 * j + 1) / 2.0) * np.sqrt(2.0 * N)
            if (k % 4) == 1:
                c *= -1.0
            coeff = (
                c
                * d[j, (m + 1) + (B - 1), -k + (B - 1)]
                * d[j, (m + 1) + (B - 1), B - 1]
            )

    """

    # calculate the prefactor
    jkm = torch.stack(
        torch.meshgrid(
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            torch.arange(-order_max, order_max + 1, dtype=torch.int32, device=device),
            torch.arange(0, order_max + 1, dtype=torch.int32, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)

    j_all, k_all, m_all = torch.unbind(jkm, dim=-1)

    valid_wrt_data = (j_all >= torch.abs(k_all)) & (j_all >= m_all)

    j = j_all[valid_wrt_data]
    k = k_all[valid_wrt_data]
    m = m_all[valid_wrt_data]

    # use torch.where to calculate the terms
    k_mod_2_mask = (k % 2) == 0
    k_mod_4_eq_1_mask = (k % 4) == 1
    k_mod_4_eq_2_mask = (k % 4) == 2
    m_eq_0_mask = m == 0
    m_eq_j_mask = m == j

    N = 2 * B

    term = torch.where(
        k_mod_2_mask,
        torch.where(
            m_eq_0_mask,
            torch.sqrt((N * (2 * j.double() + 1) / 2.0))
            * torch.where(k_mod_4_eq_2_mask, -1.0, 1.0),
            torch.sqrt(2 * N * ((2 * j.double() + 1) / 2.0))
            * torch.where(k_mod_4_eq_2_mask, -1.0, 1.0),
        ),
        torch.where(
            m_eq_j_mask,
            0,
            torch.sqrt(2 * N * ((2 * j.double() + 1) / 2.0))
            * torch.where(k_mod_4_eq_1_mask, -1.0, 1.0),
        ),
    )

    # zero indices
    z = torch.zeros_like(j, dtype=torch.int32)

    coeff = term * torch.where(
        k_mod_2_mask,
        access_d_beta_half_pi_ts2kit(j, m, -k, d_jkm_volume)
        * access_d_beta_half_pi_ts2kit(j, m, z, d_jkm_volume),
        access_d_beta_half_pi_ts2kit(j, m + 1, -k, d_jkm_volume)
        * access_d_beta_half_pi_ts2kit(j, m + 1, z, d_jkm_volume),
    )

    # find the indices that yield coefficients with magnitude greater than 1e-15
    valid_indices = torch.where(torch.abs(coeff) > 1.0e-15)

    # grab the valid coefficients using the valid indices mask
    j_valid, k_valid, m_valid = j[valid_indices], k[valid_indices], m[valid_indices]

    # coeff_valid = output_dense[j_valid, k_valid + order_max, m_valid]
    coeff_valid = coeff[valid_indices]

    # get the sparse tensor indices
    first_indices = (k_valid + order_max) * B + j_valid
    second_indices = N * (k_valid + order_max) + m_valid

    return torch.sparse_coo_tensor(
        torch.stack([first_indices, second_indices], dim=0),
        coeff_valid,
        [B * (2 * B - 1), 2 * B * (2 * B - 1)],
        dtype=torch.float64,
        device=device,
    )
