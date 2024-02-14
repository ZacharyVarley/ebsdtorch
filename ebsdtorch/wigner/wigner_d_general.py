"""

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
    -----**All Notations Follow Fukushima**------
    ---------------------------------------------

    -------------------

    I ended up directly porting the F90 code from the second paper, including
    the X-number implementation found in the second paper. See
    wigner_d_coeff_reference.py for a reference for a detailed explanation.

    wiki: https://en.wikipedia.org/wiki/Wigner_D-matrix

    Our end goal is calculating d_jkm for all j, k, m in a given range with 

    j = 0, 1, 2, ..., j_max

    k = -order_max, ..., 0, ..., order_max

    m = -order_max, ..., 0, ..., order_max

    j >= |k| >= |m|

    where j_max is the maximum degree and order_max is the maximum order.

    Overall we start by calculating the Wigner d-coefficients for the case when
    j is starting at k, from d_k_km, d_k+1_km, d_k+2_km, ..., d_jmax_km. We only
    do this for the case j_max >= j >= order_max >= k >= m >= 0. We then fill in
    the rest of the table by symmetry (e.g. negative k and m value
    combinations).

    I ended up directly porting the F90 code from the second paper, including
    the X-number implementation found in the second paper.

    ---------------------------------------------

    Note the following symmetry relations of the Wigner d-function:

    1) Negating BETA is equivalent to swapping k and m

    d^j_k_m(-BETA) = d^j_m_k(BETA)


    2) Negating k and m yields -1 to (k-m) power prefactor

    d^j_-k_-m = (-1)^(k-m) d^j_k_m


    3) Negating m yields -1 to (j + k + 2m) power prefactor / angle supplement

    d^j_k_-m(BETA) = (-1)^(j + k + 2m) d^j_k_m(π - BETA)


    4) Negating k yields -1 to (j + 2k + 3m) power prefactor / angle supplement

    d^j_-k_m(BETA) = (-1)^(j + 2k + 3m) d^j_k_m(π - BETA)


    5) Swapping k and m yields -1 to (k-m) power prefactor
    d^j_m_k = (-1)^(k-m) d^j_k_m

"""

from typing import Tuple
import torch
from torch import Tensor
from ebsdtorch.wigner.x_numbers import x2f, xlsum2, x_norm


@torch.jit.script
def trig_powers(
    largest_power: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Calculate the powers of sin(beta/2) and cos(beta/2) from 0 up to the largest power.
    This is done in logspace instead of original slow recursion that should never be used.

    Args:
        largest_power: The largest power to calculate.
        beta: The angle to calculate the powers for 0 <= beta < pi.

    Returns:
        cos_powers_x: mantissa of cosine(beta/2)^n for n from 0 to largest_power: torch.float64
        cos_powers_x_i: exponent of cosine(beta/2)^n for n from 0 to largest_power: torch.int32
        sin_powers_x: mantissa of sine(beta/2)^n for n from 0 to largest_power: torch.float64
        sin_powers_x_i: exponent of sine(beta/2)^n for n from 0 to largest_power: torch.int32

    """

    # values up to 2^53 can be exactly represented in float64
    powers = torch.arange(0, largest_power + 1, dtype=torch.float64, device=device)

    # beta is in the range [0, pi] so sh and ch are always non-negative
    ch = torch.cos(beta / 2.0)
    sh = torch.sin(beta / 2.0)

    # log2 of magnitude of cos(beta/2) and sin(beta/2)
    lmch = torch.log2(torch.abs(ch))
    lmsh = torch.log2(torch.abs(sh))

    lmch_sign = torch.sign(lmch)
    lmsh_sign = torch.sign(lmsh)

    # calculate the powers in log space
    log_cos_powers = lmch * powers
    log_sin_powers = lmsh * powers

    # split according to 2**960
    log_cos_powers_x = torch.abs(log_cos_powers) % 960
    log_sin_powers_x = torch.abs(log_sin_powers) % 960

    # calculate the x-number exponents
    cos_powers_x_i = ((torch.abs(log_cos_powers) // 960) * lmch_sign).to(torch.int32)
    sin_powers_x_i = ((torch.abs(log_sin_powers) // 960) * lmsh_sign).to(torch.int32)

    # calculate the x-number mantissas
    cos_powers_x = torch.pow(2.0, log_cos_powers_x * lmch_sign)
    sin_powers_x = torch.pow(2.0, log_sin_powers_x * lmsh_sign)

    # call x_norm to normalize the x-numbers
    cos_powers_x, cos_powers_x_i = x_norm(cos_powers_x, cos_powers_x_i)
    sin_powers_x, sin_powers_x_i = x_norm(sin_powers_x, sin_powers_x_i)

    return cos_powers_x, cos_powers_x_i, sin_powers_x, sin_powers_x_i


@torch.jit.script
def build_km_seed_table(
    order_max: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Build the recursion seed table for the Wigner d-coefficients. The values
    d_k_km and d_k+1_km can be used in a double recurrence to calculate the rest
    of the table up to d_jmax_km.

    Args:
        order_max: Maximum azimuthal order in k or m.

        beta: The angle to calculate the d_k_km for.

    Returns:
        (d_k_km, d_k_km_i, d_kp1_km, d_kp1_km_i): The mantissa and exponent of
        d_k_km and d_k+1_km each of shape (order_max + 1, order_max + 1).

    """

    output_shape = (order_max + 1, order_max + 1)

    output_dk_km = torch.zeros(output_shape, dtype=torch.float64, device=device)
    output_dk_km_i = torch.zeros(output_shape, dtype=torch.int32, device=device)

    output_d_kp1_km = torch.zeros(output_shape, dtype=torch.float64, device=device)
    output_d_kp1_km_i = torch.zeros(output_shape, dtype=torch.int32, device=device)

    # start by calculating a table of sine(beta/2)^n and cosine(beta/2)^n
    cos_powers_x, cos_powers_x_i, sin_powers_x, sin_powers_x_i = trig_powers(
        largest_power=(order_max + 1) * (order_max + 1),
        beta=beta,
        device=device,
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

        # start by multiplying by the cosine power and sine power
        dk_km = (
            cos_powers_x[kpm[k_index, mask_valid]]
            * sin_powers_x[kmm[k_index, mask_valid]]
        )
        dk_km_i = (
            cos_powers_x_i[kpm[k_index, mask_valid]]
            + sin_powers_x_i[kmm[k_index, mask_valid]]
        )
        dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

        # multiply by the e_km term
        dk_km = dk_km * e_km_x_col[mask_valid]
        dk_km_i = dk_km_i + e_km_x_i_col[mask_valid]
        dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

        # add to the table
        output_dk_km[k_index, mask_valid] = dk_km
        output_dk_km_i[k_index, mask_valid] = dk_km_i

        # do the first iteration outside of the loop because it is a special case
        if 0 <= beta < (torch.pi / 2.0):
            tc = 2.0 * torch.sin(beta / 2.0) ** 2
            u_km = (k2_coords_fp[k_index, mask_valid] - valid_m_coords_fp + 2) - (
                k2_coords_fp[k_index, mask_valid] + 2
            ) * tc
        elif beta == (torch.pi / 2.0):
            u_km = -1.0 * valid_m_coords_fp
        else:
            u_km = (k2_coords_fp[k_index, mask_valid] + 2) * torch.cos(
                beta
            ) - valid_m_coords_fp

        # a_km = ( (k2 + 1).double() / ((k2 + m2 + 2).double() * (k2 - m2 + 2).double())) ** 0.5 * u_km
        a_km = (
            (k2_coords_fp[k_index, mask_valid] + 1)
            / (
                (k2_coords_fp[k_index, mask_valid] + valid_m_coords_fp + 2)
                * (k2_coords_fp[k_index, mask_valid] - valid_m_coords_fp + 2)
            )
        ) ** 0.5 * u_km

        # multiply the first seed term by a_km and turn X-number into float
        d_kp1_km = dk_km * a_km
        d_kp1_km_i = dk_km_i
        d_kp1_km, d_kp1_km_i = x_norm(d_kp1_km, d_kp1_km_i)

        # add to the table
        output_d_kp1_km[k_index, mask_valid] = d_kp1_km
        output_d_kp1_km_i[k_index, mask_valid] = d_kp1_km_i

    print("Computed build_km_seed_table")
    print("output_dk_km", output_dk_km)
    print("output_dk_km_i", output_dk_km_i)
    print("output_d_kp1_km", output_d_kp1_km)
    print("output_d_kp1_km_i", output_d_kp1_km_i)

    return output_dk_km, output_dk_km_i, output_d_kp1_km, output_d_kp1_km_i


@torch.jit.script
def build_a_b_jkm_volume(
    degree_max: int,
    order_max: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """

    Build the volume of the Wigner d-coefficient table for the case when
    degree_max >= j >= order_max >= k >= m >= 0. These are the coefficients for the double
    recurrence relation between d_k+2_km and both d_k_km and d_k+1_km.

    Args:
        degree_max: The maximum degree to calculate d_jkm for.
        order_max: The maximum order to calculate d_jkm for.
        beta: The angle to calculate the d_jkm for.

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

    if 0 <= beta < (torch.pi / 2.0):
        tc = 2.0 * torch.sin(beta / 2.0) ** 2
        j2_times_j2m2 = j2 * (j2 - 2)
        u_jkm = (j2_times_j2m2 - (m2 * k2)) - j2_times_j2m2 * tc
    elif beta == (torch.pi / 2.0):
        u_jkm = -1.0 * (k2 * m2)
    else:
        t = torch.cos(beta)
        j2_times_j2m2 = j2 * (j2 - 2)
        u_jkm = j2_times_j2m2 * t - (m2 * k2)

    a_volume[valid_mask] = (2.0 * j2 - 2.0) * u_jkm * w_jkm
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
def build_jkm_volume(
    degree_max: int,
    order_max: int,
    beta: Tensor,
    device: torch.device,
):
    """

    Build the volume of the Wigner d-coefficient table for the case when
    degree_max >= j >= order_max >= k >= m >= 0.

    Args:
        degree_max: The maximum degree to calculate d_jkm for.
        order_max: The maximum order to calculate d_jkm for.
        beta: The angle to calculate the d_jkm for.

    Returns:
        d_jkm_volume: The Wigner d-coefficient quarter table: shape (degree_max + 1, order_max + 1, order_max + 1)

    """

    # make volumes for a and b coefficients for all j, k, m
    a_volume, b_volume = build_a_b_jkm_volume(
        degree_max=degree_max,
        order_max=order_max,
        beta=beta,
        device=device,
    )

    # build the seed table for the recursion
    dk_km, dk_km_i, dkp1_km, dkp1_km_i = build_km_seed_table(
        order_max=order_max, beta=beta, device=device
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
