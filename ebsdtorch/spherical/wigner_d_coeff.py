from typing import Tuple
import torch
from torch import Tensor
from extended_precision import x2f, xlsum2, x_norm

"""
    -------------------

    See the following publications for an explanation of the recursive algorithm:

    "Numerical computation of spherical harmonics of arbitrary degree 
    and order by extending exponent of floating point numbers"

    https://doi.org/10.1007/s00190-011-0519-2

    "Numerical computation of Wigner's d-function of arbitrary high 
    degree and orders by extending exponent of floating point numbers"

    http://dx.doi.org/10.13140/RG.2.2.31922.20160 
    https://www.researchgate.net/publication/309652602

    Both by Toshio Fukushima.

    -------------------

    I ended up directly porting the F90 code from the second paper, including
    the X-number implementation found in the second paper. See wigner_d_coeff_reference.py 
    for a reference for a detailed explanation.

    The overall goal here is to relate the product of two spherical harmonics such
    that an inverse Fourier transform yields the convolution of two spherical harmonics
    over SO(3). The constant that should be multiplied by each spherical harmonic 
    coefficient is the Wigner d-coefficient. If you choose to use the Wigner d-coefficient
    to be Z-X-Z then half of them are purely imaginary, while if you choose to 
    use Z-Y-Z then they are all real entries.

    wiki: https://en.wikipedia.org/wiki/Wigner_D-matrix

    Our end goal is calculating d_jkm for all j, k, m in a given range with 

    j = 0, 1, 2, ..., j_max
    k = -order_max, ..., 0, ..., order_max
    m = -order_max, ..., 0, ..., order_max

    j >= |k| >= |m|

    where j_max is the maximum degree and order_max is the maximum order.

    Overall we start by calculating the Wigner d-coefficients for the case when
    j is starting at k, from d_k_km, d_k+1_km, d_k+2_km, ..., d_jmax_km. We only do 
    this for the case j_max >= j >= order_max >= k >= m >= 0. We then fill in 
    the rest of the table by symmetry (e.g. negative k and m value combinations).

    Steps:
    1) Build the quarter table of seeds for recursion: shape (order_max + 1, order_max + 1)

    2) Recursively find d_k_km, d_k+1_km, d_k+2_km, ..., d_jmax_km

    3) Fill in the rest of the table by symmetry: shape (degree_max + 1, order_max + 1, order_max + 1)

    If the maximum degree exceeds around 500, then floating point 64 precision
    becomes insufficient, and we use Toshio's X-number formulation to define a 
    new float using a 64 bit float and an integer, as the new mantissa and exponent. 
    He originally used a 32 bit integer, but I am using a 64 bit integer just in case.

"""


@torch.jit.script
def trig_powers_array(
    largest_power: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Calculate the powers of sin(beta/2) and cos(beta/2) from 0 up to the largest power.
    This is done with X-numbers to avoid underflow.

    Args:
        largest_power: The largest power to calculate.
        beta: The angle to calculate the powers for.

    Returns:
        cos_powers_x: mantissa of cosine(beta/2)^n for n from 0 to largest_power: torch.float64
        cos_powers_x_i: exponent of cosine(beta/2)^n for n from 0 to largest_power: torch.int32
        sin_powers_x: mantissa of sine(beta/2)^n for n from 0 to largest_power: torch.float64
        sin_powers_x_i: exponent of sine(beta/2)^n for n from 0 to largest_power: torch.int32

    """

    cos_powers_x = torch.ones(largest_power + 1, dtype=torch.float64, device=device)
    sin_powers_x = torch.ones(largest_power + 1, dtype=torch.float64, device=device)
    cos_powers_x_i = torch.zeros(largest_power + 1, dtype=torch.int32, device=device)
    sin_powers_x_i = torch.zeros(largest_power + 1, dtype=torch.int32, device=device)

    ch = torch.cos(beta / 2.0)
    sh = torch.sin(beta / 2.0)

    cn = torch.ones(1, dtype=torch.float64, device=device)
    icn = torch.zeros(1, dtype=torch.int32, device=device)
    sn = torch.ones(1, dtype=torch.float64, device=device)
    isn = torch.zeros(1, dtype=torch.int32, device=device)

    for n in range(1, largest_power + 1):
        # cosine calculation
        cn = ch * cn
        cn, icn = x_norm(cn, icn)
        cos_powers_x[n] = cn[0]
        cos_powers_x_i[n] = icn[0]

        # sine calculation
        sn = sh * sn
        sn, isn = x_norm(sn, isn)
        sin_powers_x[n] = sn[0]
        sin_powers_x_i[n] = isn[0]

    return cos_powers_x, cos_powers_x_i, sin_powers_x, sin_powers_x_i


@torch.jit.script
def build_km_seed_row(
    m2: int,
    order_max: int,
    cos_powers_x: Tensor,
    cos_powers_x_i: Tensor,
    sin_powers_x: Tensor,
    sin_powers_x_i: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Build a row of the recursion seed table for the Wigner d-coefficients for the case when
    j, k, and m are all non-negative and k <= j and m <= k.

    Args:
        k2_start: The starting value of k2.
        order_max: The maximum order to calculate d_jkm for.
        cos_powers_x: mantissa of cosine(beta/2)^n for n from 0 to largest_power: torch.float64
        cos_powers_x_i: exponent of cosine(beta/2)^n for n from 0 to largest_power: torch.int32
        sin_powers_x: mantissa of sine(beta/2)^n for n from 0 to largest_power: torch.float64
        sin_powers_x_i: exponent of sine(beta/2)^n for n from 0 to largest_power: torch.int32
        beta: The angle to calculate the d_jkm for.

    Returns:
        output: The row of the recursion seed table: shape (order_max + 1)

    """

    output_shape = order_max + 1

    m2_float = torch.tensor(
        [
            m2,
        ],
        dtype=torch.float64,
        device=cos_powers_x.device,
    )

    arr_dk_km = torch.empty(
        output_shape, dtype=torch.float64, device=cos_powers_x.device
    )
    arr_dk_km_i = torch.empty(
        output_shape, dtype=torch.int32, device=cos_powers_x.device
    )

    arr_dk_kp1_km = torch.empty(
        output_shape, dtype=torch.float64, device=cos_powers_x.device
    )
    arr_dk_kp1_km_i = torch.empty(
        output_shape, dtype=torch.int32, device=cos_powers_x.device
    )

    e_km_x = torch.ones(1, dtype=torch.float64)
    e_km_x_i = torch.zeros(1, dtype=torch.int32)

    for k2 in torch.arange(
        m2,
        2 * order_max + 2,
        step=2,
        dtype=torch.int32,
        device=cos_powers_x.device,
    ):
        kpm = (k2 + m2) // 2
        kmm = (k2 - m2) // 2

        if m2 == k2:
            e_km_x = torch.ones(1, dtype=torch.float64, device=cos_powers_x.device)
            e_km_x_i = torch.zeros(1, dtype=torch.int32, device=cos_powers_x.device)
        else:
            f = (k2.double() * (k2.double() - 1)) / (kpm.double() * kmm.double())
            e_km_x = e_km_x * f**0.5
            e_km_x, e_km_x_i = x_norm(e_km_x, e_km_x_i)

        # start by multiplying by the cosine power and sine power
        dk_km = cos_powers_x[kpm] * sin_powers_x[kmm]
        dk_km_i = cos_powers_x_i[kpm] + sin_powers_x_i[kmm]
        dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

        # multiply by the e_km term
        dk_km = dk_km * e_km_x
        dk_km_i = dk_km_i + e_km_x_i
        dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

        # add to the table
        arr_dk_km[k2 // 2] = dk_km[0]
        arr_dk_km_i[k2 // 2] = dk_km_i[0]

        # do the first iteration outside of the loop because it is a special case
        if 0 <= beta < (torch.pi / 2.0):
            tc = 2.0 * torch.sin(beta / 2.0) ** 2
            u_km = (k2 - m2_float + 2).double() - (k2 + 2).double() * tc
        elif beta == (torch.pi / 2.0):
            u_km = -1.0 * m2_float.double()
        else:
            u_km = (k2 + 2).double() * torch.cos(beta) - m2_float.double()

        a_km = (
            (k2 + 1).double()
            / ((k2 + m2_float + 2).double() * (k2 - m2_float + 2).double())
        ) ** 0.5 * u_km

        # multiply the first seed term by a_km and turn X-number into float
        d_kp1_km = dk_km * a_km
        d_kp1_km_i = dk_km_i

        d_kp1_km, d_kp1_km_i = x_norm(d_kp1_km, d_kp1_km_i)

        # add to the table
        arr_dk_kp1_km[k2 // 2] = d_kp1_km[0]
        arr_dk_kp1_km_i[k2 // 2] = d_kp1_km_i[0]

        # print(
        #     f"all end variables for k = {k2.item() // 2}, m = {m2 // 2}: {dk_km.item()}, {dk_km_i.item()}, {d_kp1_km.item()}, {d_kp1_km_i.item()}"
        # )

        # print(
        #     f"all intermediate variables for k = {k2.item() // 2}, m = {m2 // 2}: {e_km_x.item()}, {e_km_x_i.item()}, {dk_km.item()}, {dk_km_i.item()}, {a_km.item()}, {d_kp1_km.item()}, {d_kp1_km_i.item()}"
        # )
        # break

    return arr_dk_km, arr_dk_km_i, arr_dk_kp1_km, arr_dk_kp1_km_i


@torch.jit.script
def build_km_seed_table(
    order_max: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    Build the recursion seed table for the Wigner d-coefficients for the case when
    j, k, and m are all non-negative and k <= j and m <= k.

    Args:
        order_max: The maximum order to calculate d_jkm for.
        beta: The angle to calculate the d_jkm for.


    """

    output_shape = (order_max + 1, order_max + 1)

    output_dk_km = torch.empty(output_shape, dtype=torch.float64, device=device)
    output_dk_km_i = torch.empty(output_shape, dtype=torch.int32, device=device)

    output_d_kp1_km = torch.empty(output_shape, dtype=torch.float64, device=device)
    output_d_kp1_km_i = torch.empty(output_shape, dtype=torch.int32, device=device)

    # start by calculating a table of sine(beta/2)^n and cosine(beta/2)^n
    cos_powers_x, cos_powers_x_i, sin_powers_x, sin_powers_x_i = trig_powers_array(
        largest_power=(order_max + 1) * (order_max + 1),
        beta=beta,
        device=device,
    )

    # This is the main bottleneck by a wide margin
    for m2 in torch.arange(
        0, 2 * order_max + 2, step=2, dtype=torch.int32, device=device
    ):
        dk_km, dk_km_i, d_kp1_km, d_kp1_km_i = build_km_seed_row(
            m2=m2,
            order_max=order_max,
            cos_powers_x=cos_powers_x,
            cos_powers_x_i=cos_powers_x_i,
            sin_powers_x=sin_powers_x,
            sin_powers_x_i=sin_powers_x_i,
            beta=beta,
        )

        # add to the table
        output_dk_km[:, m2 // 2] = dk_km
        output_dk_km_i[:, m2 // 2] = dk_km_i
        output_d_kp1_km[:, m2 // 2] = d_kp1_km
        output_d_kp1_km_i[:, m2 // 2] = d_kp1_km_i

    return output_dk_km, output_dk_km_i, output_d_kp1_km, output_d_kp1_km_i


# @torch.jit.script
# def build_km_seed_table(
#     order_max: int,
#     beta: Tensor,
#     device: torch.device,
# ):
#     """

#     Build the recursion seed table for the Wigner d-coefficients for the case when
#     j, k, and m are all non-negative and k <= j and m <= k.

#     Args:
#         order_max: The maximum order to calculate d_jkm for.
#         beta: The angle to calculate the d_jkm for.


#     """

#     output_shape = (order_max + 1, order_max + 1)

#     output_dk_km = torch.empty(output_shape, dtype=torch.float64, device=device)
#     output_dk_km_i = torch.empty(output_shape, dtype=torch.int32, device=device)

#     output_d_kp1_km = torch.empty(output_shape, dtype=torch.float64, device=device)
#     output_d_kp1_km_i = torch.empty(output_shape, dtype=torch.int32, device=device)

#     # start by calculating a table of sine(beta/2)^n and cosine(beta/2)^n
#     cos_powers_x, cos_powers_x_i, sin_powers_x, sin_powers_x_i = trig_powers_array(
#         largest_power=(order_max + 1) * (order_max + 1),
#         beta=beta,
#         device=device,
#     )

#     e_km_x = torch.ones(1, dtype=torch.float64, device=device)
#     e_km_x_i = torch.zeros(1, dtype=torch.int32, device=device)

#     # This is the main bottleneck by a wide margin
#     for m2 in torch.arange(
#         0, 2 * order_max + 2, step=2, dtype=torch.int32, device=device
#     ):
#         e_km_x = torch.ones(1, dtype=torch.float64)
#         e_km_x_i = torch.zeros(1, dtype=torch.int32)
#         for k2 in torch.arange(
#             int(m2), 2 * order_max + 2, step=2, dtype=torch.int32, device=device
#         ):
#             kpm = (k2 + m2) // 2
#             kmm = (k2 - m2) // 2
#             if m2 == k2:
#                 e_km_x = torch.ones(1, dtype=torch.float64, device=device)
#                 e_km_x_i = torch.zeros(1, dtype=torch.int32, device=device)
#             else:
#                 f = (k2.double() * (k2.double() - 1)) / (kpm.double() * kmm.double())
#                 e_km_x = e_km_x * f**0.5
#                 e_km_x, e_km_x_i = x_norm(e_km_x, e_km_x_i)

#             # start by multiplying by the cosine power and sine power
#             dk_km = cos_powers_x[kpm] * sin_powers_x[kmm]
#             dk_km_i = cos_powers_x_i[kpm] + sin_powers_x_i[kmm]
#             dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

#             # multiply by the e_km term
#             dk_km = dk_km * e_km_x
#             dk_km_i = dk_km_i + e_km_x_i
#             dk_km, dk_km_i = x_norm(dk_km, dk_km_i)

#             # print(f'    dk_km for k = {k2.item() // 2}, m = {m2.item() // 2}: {dk_km.item()}')

#             # add to the table
#             output_dk_km[k2 // 2, m2 // 2] = dk_km[0]
#             output_dk_km_i[k2 // 2, m2 // 2] = dk_km_i[0]

#             # do the first iteration outside of the loop because it is a special case
#             if 0 <= beta < (torch.pi / 2.0):
#                 tc = 2.0 * torch.sin(beta / 2.0) ** 2
#                 u_km = (k2 - m2 + 2).double() - (k2 + 2).double() * tc
#             elif beta == (torch.pi / 2.0):
#                 u_km = -1.0 * m2.double()
#             else:
#                 u_km = (k2 + 2).double() * torch.cos(beta) - m2.double()

#             a_km = (
#                 (k2 + 1).double() / ((k2 + m2 + 2).double() * (k2 - m2 + 2).double())
#             ) ** 0.5 * u_km

#             # print(f'      a_km for k = {k2.item() // 2}, m = {m2.item() // 2}: {a_km.item()}')

#             # multiply the first seed term by a_km and turn X-number into float
#             d_kp1_km = dk_km * a_km
#             d_kp1_km_i = dk_km_i
#             d_kp1_km, d_kp1_km_i = x_norm(d_kp1_km, d_kp1_km_i)

#             # print(f'  d_kp1_km for k = {k2.item() // 2}, m = {m2.item() // 2}: {d_kp1_km.item()}')

#             # add to the table
#             output_d_kp1_km[k2 // 2, m2 // 2] = d_kp1_km[0]
#             output_d_kp1_km_i[k2 // 2, m2 // 2] = d_kp1_km_i[0]

#     return output_dk_km, output_dk_km_i, output_d_kp1_km, output_d_kp1_km_i


@torch.jit.script
def build_a_b_jkm_volume(
    degree_max: int,
    order_max: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """

    Build the volume of the Wigner d-coefficient table for the case when
    degree_max >= j >= order_max >= k >= m >= 0.

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

    j, k, m = torch.unbind(jkm, dim=-1)

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

    v_jkm = (
        j2
        * (
            (j2 + k2 - 2).double()
            * (j2 - k2 - 2).double()
            * (j2 + m2 - 2).double()
            * (j2 - m2 - 2).double()
        )
        ** 0.5
    )
    w_jkm = 1.0 / (
        (j2 - 2).double()
        * (
            (j2 + k2).double()
            * (j2 - k2).double()
            * (j2 + m2).double()
            * (j2 - m2).double()
        )
        ** 0.5
    )

    if 0 <= beta < (torch.pi / 2.0):
        tc = 2.0 * torch.sin(beta / 2.0) ** 2
        j2_times_j2m2 = (j2 * (j2 - 2)).double()
        u_jkm = (j2_times_j2m2 - (m2 * k2).double()) - j2_times_j2m2 * tc
    elif beta == (torch.pi / 2.0):
        u_jkm = -1.0 * (k2 * m2).double()
    else:
        t = torch.cos(beta)
        j2_times_j2m2 = (j2 * (j2 - 2)).double()
        u_jkm = j2_times_j2m2 * t - (m2 * k2).double()

    a_volume[valid_mask] = (2.0 * j2 - 2.0).double() * u_jkm * w_jkm
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
        dkkm_x: mantissa of d_kkm
        dkkm_x_i: exponent of d_kkm
        j2_max: twice the maximum degree
        k2: twice the order
        m2: twice the order
        beta: angle in radians

    Returns:
        output: float64 tensor for d_k_km, d_k+1_km, ..., d_jmax_km: shape (j_max - k + 1)

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

    print(
        f"Building d_jkm_volume for degree_max = {degree_max}, order_max = {order_max}, beta = {beta.item()}"
    )

    # make volumes for a and b coefficients for all j, k, m
    a_volume, b_volume = build_a_b_jkm_volume(
        degree_max=degree_max,
        order_max=order_max,
        beta=beta,
        device=device,
    )

    print(f"  a_volume: {a_volume.shape}")

    # build the seed table for the recursion
    dk_km, dk_km_i, dkp1_km, dkp1_km_i = build_km_seed_table(
        order_max=order_max, beta=beta, device=device
    )

    print(f"  dk_km: {dk_km.shape}")

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


test_parameters = [
    (365, 102, 20, -4.23570250037880395095020243575390e-02, 8161.0 / 16384.0),
    (294, 247, 188, -1.11943794723176255836019618855372e-01, 7417.0 / 16384.0),
    (6496, 141, 94, 1.91605798359216822133779150869763e-03, 10134.0 / 16384.0),
]

for test_parameter in test_parameters:
    j = test_parameter[0]
    k = test_parameter[1]
    m = test_parameter[2]
    correct_value = test_parameter[3]
    beta = torch.tensor(
        [
            torch.pi * test_parameter[4],
        ],
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    volume = build_jkm_volume(j, k, beta, torch.device("cpu"))
    print(volume.shape)
    d_value = volume[j, k, m]
    print("Correct   : " + "{:.20f}".format(correct_value))
    print("Computed  : " + "{:.20f}".format(d_value))
    print("Difference: " + "{:.20f}".format(abs(d_value - correct_value)))
    print(" ------------------- ")
