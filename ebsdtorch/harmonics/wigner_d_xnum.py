"""

Virtual extended precision arithmetic useful for Wigner d recursion for very
high band limits.

"Numerical computation of spherical harmonics of arbitrary degree and order by
extending exponent of floating point numbers"

https://doi.org/10.1007/s00190-011-0519-2

"Numerical computation of Wigner's d-function of arbitrary high degree and
orders by extending exponent of floating point numbers"

http://dx.doi.org/10.13140/RG.2.2.31922.20160

https://www.researchgate.net/publication/309652602

"""

from typing import Tuple
import torch
from torch import Tensor
import math


@torch.jit.script
def xnum_norm(x_num: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    """

    Normalize a virtual extended precision number with FP64 mantissa and INT32
    exponent, all pivoted around an exponent of 2^460.

    """
    x, ix = x_num

    # Constants
    IND = torch.tensor(460, dtype=torch.int32)
    BIG = torch.pow(2.0, IND.double())
    BIGI = torch.pow(2.0, -IND.double())
    BIGS = torch.pow(2.0, IND.double() / 2.0)
    BIGSI = torch.pow(2.0, -IND.double() / 2.0)

    w = torch.abs(x)
    x = torch.where(w >= BIGS, x * BIGI, x)
    x = torch.where(w < BIGSI, x * BIG, x)
    ix = torch.where(w >= BIGS, ix + 1, ix)
    ix = torch.where(w < BIGSI, ix - 1, ix)

    return x, ix


@torch.jit.script
def xnum_to_fp(x_num: Tuple[Tensor, Tensor]) -> torch.Tensor:
    """

    Convert a virtual extended precision number to a floating point number.

    Args:
        x_num: Tuple of Tensor, the virtual extended precision number.

    Returns:
        x_fp: Tensor, the floating point number.
    """

    x, ix = x_num

    # Constants
    IND = torch.tensor(460, dtype=torch.int32)
    BIG = 2.0**IND
    BIGI = 2.0 ** (-IND)
    return torch.where(ix == 0, x, torch.where(ix < 0, x * BIGI, x * BIG))


@torch.jit.script
def xnum_sum(
    xnum_a: Tuple[Tensor, Tensor],
    xnum_b: Tuple[Tensor, Tensor],
    factor_a: float,
    factor_b: float,
) -> Tuple[Tensor, Tensor]:
    """
    Return the linear combination of two virtual extended precision numbers,
    xnum_a and xnum_b, according to multiplicative prefactors factor_a and
    factor_b.

    Args:
        xnum_a: Tuple of Tensor, the first virtual extended precision number.
        factor_a: Tensor, the prefactor for the first number.
        xnum_b: Tuple of Tensor, the second virtual extended precision number.
        factor_b: Tensor, the prefactor for the second number.

    """

    x, ix = xnum_a
    y, iy = xnum_b

    # Constants
    IND = torch.tensor(460, dtype=torch.int32)
    BIGI = 2.0 ** (-IND)

    id = ix - iy
    z = torch.where(
        id == 0,
        factor_a * x + factor_b * y,
        torch.where(
            id == 1,
            factor_a * x + factor_b * y * BIGI,
            torch.where(
                id == -1,
                factor_b * y + factor_a * x * BIGI,
                torch.where(id > 1, factor_a * x, factor_b * y),
            ),
        ),
    )
    iz = torch.where(
        id == 0,
        ix,
        torch.where(
            id == 1, ix, torch.where(id == -1, iy, torch.where(id > 1, ix, iy))
        ),
    )
    z, iz = xnum_norm((z, iz))
    return z, iz


@torch.jit.script
def trig_powers_xnum(
    largest_power: int,
    beta: Tensor,
    device: torch.device,
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
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

    # split according to 2**460
    log_cos_powers_x = torch.abs(log_cos_powers) % 460
    log_sin_powers_x = torch.abs(log_sin_powers) % 460

    # calculate the x-number exponents
    cos_powers_x_i = ((torch.abs(log_cos_powers) // 460) * lmch_sign).to(torch.int32)
    sin_powers_x_i = ((torch.abs(log_sin_powers) // 460) * lmsh_sign).to(torch.int32)

    # calculate the x-number mantissas
    cos_powers_x = torch.pow(2.0, log_cos_powers_x * lmch_sign)
    sin_powers_x = torch.pow(2.0, log_sin_powers_x * lmsh_sign)

    # call x_norm to normalize the x-numbers
    cos_powers_x, cos_powers_x_i = xnum_norm((cos_powers_x, cos_powers_x_i))
    sin_powers_x, sin_powers_x_i = xnum_norm((sin_powers_x, sin_powers_x_i))

    return (cos_powers_x, cos_powers_x_i), (sin_powers_x, sin_powers_x_i)


@torch.jit.script
def half_powers_xnum(
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

    # split according to 2**460
    lhalf_powers_x = torch.abs(lhalf_powers) % 460

    # calculate the x-number exponents
    half_powers_x_i = ((torch.abs(lhalf_powers) // 460) * lhalf_sign).to(torch.int32)

    # calculate the x-number mantissas
    half_powers_x = torch.pow(2.0, lhalf_powers_x * lhalf_sign)

    # call x_norm to normalize the x-numbers
    half_powers_x, half_powers_x_i = xnum_norm((half_powers_x, half_powers_x_i))

    return half_powers_x, half_powers_x_i


@torch.jit.script
def index_from_coords(
    coords: Tensor,
) -> Tensor:
    """
    Returns the index into the Wigner-d 1D array for a given l, m, and n.
    Only positive integers with l >= m >= n are considered.

    Args:
        coords: Tensor
            coordinates (..., 3) orderd (l, m, n) to convert to an index.
        precision: str
            The precision of the output either "single" or "double".

    First few entries (in order):
    ---
    l = 0, m = 0, n = 0
    ---
    l = 1, m = 0, n = 0

    l = 1, m = 1, n = 0
    l = 1, m = 1, n = 1
    ---
    l = 2, m = 0, n = 0

    l = 2, m = 1, n = 0
    l = 2, m = 1, n = 1

    l = 2, m = 2, n = 0
    l = 2, m = 2, n = 1
    l = 2, m = 2, n = 2
    ---

    Each l block begins at a tetrahedral number, while each m block begins at a
    triangular number. The n block is just a linear sequence.

    Args:
        coords: Tensor
            coordinates (..., 3) orderd (l, m, n) to convert to an index.

    Notes:
        This will silently fail if int32 overflows.

    """
    l, m, n = coords.unbind(-1)
    # note the parantheses are important to try to avoid int32 overflow with
    # the l * (l + 1) * (l + 2) // 6 term ... either l or l + 1 will be even
    return (((l * (l + 1)) // 2) * (l + 2)) // 3 + (m * (m + 1)) // 2 + n


@torch.jit.script
def wigner_d_xnum(
    beta: Tensor,
    deg_max: int,
    device: torch.device,
) -> Tensor:
    """

    Wigner d^l_{m,n}(β) for 0 <= β < π for l >= m >= n >= 0.

    Args:
        beta: The angle to calculate the Wigner D function for 0 <= beta < pi.
        deg_max: The maximum degree of the Wigner D function.
        device: The device to run the computations on.

    Returns:
        d_xnum: The Wigner D function as a virtual extended precision number.
        d_xnum_i: The exponent of the Wigner D function as a virtual extended precision number.

    """

    # compute the table size
    table_size = (((deg_max + 1) * (deg_max + 2)) // 2) * (deg_max + 3) // 3

    # calculate the powers of sin(beta/2) and cos(beta/2)
    cos_powers_xnum, sin_powers_xnum = trig_powers_xnum(2 * (deg_max + 1), beta, device)
    half_power_xnum = half_powers_xnum(2 * deg_max, device)

    # set output xnumber
    wigner_xnum = (
        torch.empty(table_size, dtype=torch.float64, device=device),
        torch.empty(table_size, dtype=torch.int32, device=device),
    )

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(deg_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(deg_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1)
    coords = coords.view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)

    # make the coordinates the proper datatype
    mmn_coords_fp = mmn_coords.to(torch.float64)

    # d_m_mn = c_{m+n} s_{m-n} e_mn
    # for m >= n, e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    if beta.item() == torch.pi:
        wigner_xnum[0][first_seed_indices] = (
            torch.exp(
                0.5
                * (
                    torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
                    - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
                    - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
                )
            )
            * half_power_xnum[0][mmn_coords[:, 1]]
        )
        wigner_xnum[1][first_seed_indices] = half_power_xnum[1][mmn_coords[:, 1]]
    else:
        wigner_xnum[0][first_seed_indices] = (
            torch.exp(
                0.5
                * (
                    torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
                    - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
                    - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
                )
            )
            * cos_powers_xnum[0][mmn_coords[:, 1] + mmn_coords[:, 2]]
            * sin_powers_xnum[0][mmn_coords[:, 1] - mmn_coords[:, 2]]
        )
        wigner_xnum[1][first_seed_indices] = (
            cos_powers_xnum[1][mmn_coords[:, 1] + mmn_coords[:, 2]]
            + sin_powers_xnum[1][mmn_coords[:, 1] - mmn_coords[:, 2]]
        )

    wigner_xnum[0][first_seed_indices], wigner_xnum[1][first_seed_indices] = xnum_norm(
        (wigner_xnum[0][first_seed_indices], wigner_xnum[1][first_seed_indices])
    )

    coords = torch.stack((mm, mm - 1, nn - 1), dim=-1)
    coords = coords.view(-1, 3)
    mask = (coords[:, 1] >= coords[:, 2]) & (coords[:, 1] >= 0) & (coords[:, 2] >= 0)
    mp1_mn_coords = coords[mask]
    mp1_mn_coords_fp = mp1_mn_coords.to(torch.float64)
    second_seed_indices = index_from_coords(mp1_mn_coords)
    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]

    # d_m+1_mn = a_mn * d_m_mn
    # a_mn = sqrt((2 * (2*m + 1)) / ((2m + 2n + 2) * (2m - 2n + 2))) * u_mn
    # u_nm for β in [0, pi/2] is (2m - 2n - 2) - (2m - 2) * tc, tc = 2 * sin^2(beta/2)
    # u_nm for β equal to pi/2: -4(m)(n)
    # u_nm for β in [0, pi/2] is (2m - 2) * t - 2n, where t = cos(beta)

    if beta.item() < torch.pi / 2.0:
        tc = 2 * torch.sin(beta / 2.0) ** 2
        u_mn = (2 * mp1_mn_coords_fp[:, 1] - 2 * mp1_mn_coords_fp[:, 2] + 2) - (
            2 * mp1_mn_coords_fp[:, 1] + 2
        ) * tc
    elif beta.item() == torch.pi / 2.0:
        u_mn = -2 * mp1_mn_coords_fp[:, 2]
    else:
        t = torch.cos(beta)
        u_mn = (2 * mp1_mn_coords_fp[:, 1] + 2) * t - 2 * mp1_mn_coords_fp[:, 2]

    wigner_xnum[0][second_seed_indices] = (
        wigner_xnum[0][first_seed_indices_trunc]
        * torch.sqrt(
            (2 * mp1_mn_coords_fp[:, 1] + 1)
            / (
                (2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
                * (2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
            )
        )
        * u_mn
    )
    wigner_xnum[1][second_seed_indices] = wigner_xnum[1][first_seed_indices_trunc]

    # normalize the x-numbers
    wigner_xnum[0][second_seed_indices], wigner_xnum[1][second_seed_indices] = (
        xnum_norm(
            (wigner_xnum[0][second_seed_indices], wigner_xnum[1][second_seed_indices])
        )
    )

    # get the starting recursion coords and indices
    curr_coords = mmn_coords[: -(2 * deg_max + 1)]
    curr_coords_fp = curr_coords.to(torch.float64)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * deg_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, deg_max - 1):
        # define w in logspace
        w_log = -1.0 * torch.log(2 * curr_coords_fp[:, 0] - 2) - 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2])
        )
        # v in logspace
        v_log = torch.log(2 * curr_coords_fp[:, 0]) + 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2] - 2)
        )

        # b is w * v
        b = torch.exp(w_log + v_log)

        # u in logspace (it is negative): -(2m)*(2n)
        if beta.item() < torch.pi / 2.0:
            # u = 2l(2l−2)−(2m)(2n)]−2l(2l −2)*tc, tc = 2 * sin^2(beta/2)
            tc = 2 * torch.sin(beta / 2.0) ** 2
            u = (
                2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2)
                - 2 * curr_coords_fp[:, 1] * 2 * curr_coords_fp[:, 2]
            ) - 2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2) * tc
        elif beta.item() == torch.pi / 2.0:
            u = -4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2]
        else:
            # u = 2l(2l −2)t −(2m)(2n), t = cos(beta)
            t = torch.cos(beta)
            u = (
                2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2) * t
                - 2 * curr_coords_fp[:, 1] * 2 * curr_coords_fp[:, 2]
            )

        # a in logspace
        a = (4 * curr_coords_fp[:, 0] - 2) * u * torch.exp(w_log)

        # get a * d_l-1_mn
        x_num_a_by_d_lm1_nm = (
            a
            * wigner_xnum[0][
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ],
            wigner_xnum[1][
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ],
        )
        x_num_a_by_d_lm1_nm = xnum_norm(x_num_a_by_d_lm1_nm)

        # get (b * d_l-2_mn)
        x_num_b_by_d_lm2_nm = (
            b * wigner_xnum[0][curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])],
            wigner_xnum[1][curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])],
        )

        x_num_b_by_d_lm2_nm = xnum_norm(x_num_b_by_d_lm2_nm)

        # linear combination of a * d_l-1_mn and - b * d_l-2_mn
        wigner_xnum[0][curr_indices], wigner_xnum[1][curr_indices] = xnum_sum(
            x_num_a_by_d_lm1_nm,
            x_num_b_by_d_lm2_nm,
            1.0,
            -1.0,
        )

        # update the indices for the next iteration
        curr_coords = curr_coords[: -(deg_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(deg_max - 1 - step), :]
        curr_indices = curr_indices[: -(deg_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    return xnum_to_fp(wigner_xnum)


@torch.jit.script
def retrieve_wigner_d(
    wigner_d_values: Tensor,
    coords: Tensor,
    swap_mn: bool = False,
) -> Tensor:
    """
    Wigner d functions for the case where the angle is pi/2 and l >= 0, l > |m|,
    l > |n|. Use symmetry relations for cases where it is not true that l >= m
    >= n >= 0.

    Args:
        wigner_d_values: Tensor
            Wigner d values in a flattened array.
        coords: Tensor
            The coordinates (..., 3) ordered (l, m, n) to lookup.
        swap_mn: bool
            Some libraries have the m and n indices swapped in meaning.

    Returns:
        wignerd: Tensor
            Wigner d matrix entries

    Notes:
        To understand what I mean by swapped, look at equation (11) of: Lenthe,
        W. C., S. Singh, and M. De Graef. "A spherical harmonic transform
        approach to the indexing of electron back-scattered diffraction
        patterns." Ultramicroscopy 207 (2019): 112841.

        which reads:

        D^ℓ_{k,m}(α, β, γ) = d^ℓ_{k,n}(β) exp(i m α) exp(i k γ)

        and note that k is associated with γ while m is associated with α.

        Now compare it to equation (2.6) of: Kostelec, Peter J., and Daniel N.
        Rockmore. "FFTs on the rotation group." Journal of Fourier analysis and
        applications 14 (2008): 145-179.

        which reads (notation brought into line with the above):

        D^ℓ_{k,m}(α, β, γ) = exp(-i k α) d^ℓ_{k,m}(β) exp(-i m γ)

        and note that k is now associated with α while m is associated with γ.
        Complex numbers commute, so I don't think the order matters, but I have
        difficultly reconciling the negative signs in the two exponentials.

        TS2Kit used the second convention, so I have to swap m and n in the
        lookup.
    """

    l, m, n = coords.unbind(-1)

    mask_mn_swap = torch.abs(m) < torch.abs(n)
    m_new = torch.where(mask_mn_swap, n, m)
    n_new = torch.where(mask_mn_swap, m, n)

    prefactor = torch.where(
        mask_mn_swap,
        (-1.0) ** ((m_new - n_new) % 2),
        1.0,
    )

    if swap_mn:
        prefactor *= (-1.0) ** ((m_new - n_new) % 2)

    return (
        wigner_d_values[
            (((l * (l + 1)) // 2 * (l + 2)) // 3) + ((m_new * (m_new + 1)) // 2) + n_new
        ]
        * prefactor
    )


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # test it against spherical for 10 random angles in [0, pi] and 10 random l, m, n
# deg_max_test = 750
# n_ex_to_print = 10
# angle = torch.rand(1, device=device, dtype=torch.float64) * math.pi / 2.0
# l = torch.randint(deg_max_test - 100, deg_max_test, (n_ex_to_print,), device=device)
# m = (l * torch.rand(n_ex_to_print, device=device)).to(torch.int64)
# n = (m * torch.rand(n_ex_to_print, device=device)).to(torch.int64)
# coords = torch.stack((l, m, n), dim=-1)
# # coords = coords_from_index(torch.arange(10, device=device))

# from spherical import Wigner
# import numpy as np
# from ebsdtorch.harmonics.wigner_d_logspace import (
#     wigner_d_lt_half_pi,
#     wigner_d_gt_half_pi,
# )

# wig_xnum = wigner_d_xnum(
#     angle,
#     deg_max_test,
#     device,
# )
# logwig = wigner_d_lt_half_pi(
#     beta=angle,
#     order_max=deg_max_test,
#     dtype=torch.float64,
#     device=device,
# )

# for i in range(n_ex_to_print):
#     l, m, n = coords[i]
#     w = Wigner(l.item(), ell_min=l.item())
#     wignerd_sph = w.d(np.exp(-1j * angle.item()))
#     index = index_from_coords(
#         torch.tensor([[l.item(), m.item(), n.item()]], dtype=torch.int64)
#     )
#     print(
#         f"angle: {angle.item() * 180 / np.pi}, l = {l.item()}, m = {m.item()}, n = {n.item()}"
#     )
#     print(f"wlog value: {logwig[index].item():.17f}")
#     print(f" wig value: {wig_xnum[index].item():.17f}")
#     print(f" sph value: {wignerd_sph[w.dindex(l.item(), m.item(), n.item())]:.17f}")


# # angle += math.pi / 2.0
# angle = torch.tensor(0.01, device=device, dtype=torch.float64)

# wig_xnum = wigner_d_xnum(
#     angle,
#     deg_max_test,
#     device,
# )

# logwig = wigner_d_gt_half_pi(
#     beta=angle,
#     order_max=deg_max_test,
#     dtype=torch.float64,
#     device=device,
# )

# for i in range(n_ex_to_print):
#     l, m, n = coords[i]
#     w = Wigner(l.item(), ell_min=l.item())
#     wignerd_sph = w.d(np.exp(-1j * angle.item()))
#     index = index_from_coords(
#         torch.tensor([[l.item(), m.item(), n.item()]], dtype=torch.int64)
#     )
#     print(
#         f"angle: {angle.item() * 180 / np.pi}, l = {l.item()}, m = {m.item()}, n = {n.item()}"
#     )
#     print(f"wlog value: {logwig[index].item():.17f}")
#     print(f" wig value: {wig_xnum[index].item():.17f}")
#     print(f" sph value: {wignerd_sph[w.dindex(l.item(), m.item(), n.item())]:.17f}")


# # lastly for case of pi / 2

# angle = torch.tensor(math.pi / 2.0, device=device, dtype=torch.float64)

# wig_xnum = wigner_d_xnum(
#     angle,
#     deg_max_test,
#     device,
# )

# logwig = wigner_d_lt_half_pi(
#     beta=angle,
#     order_max=deg_max_test,
#     dtype=torch.float64,
#     device=device,
# )

# for i in range(n_ex_to_print):
#     l, m, n = coords[i]
#     w = Wigner(l.item(), ell_min=l.item())
#     wignerd_sph = w.d(np.exp(-1j * angle.item()))
#     index = index_from_coords(
#         torch.tensor([[l.item(), m.item(), n.item()]], dtype=torch.int64)
#     )
#     print(
#         f"angle: {angle.item() * 180 / np.pi}, l = {l.item()}, m = {m.item()}, n = {n.item()}"
#     )
#     print(f"wlog value: {logwig[index].item():.17f}")
#     print(f" wig value: {wig_xnum[index].item():.17f}")
#     print(f" sph value: {wignerd_sph[w.dindex(l.item(), m.item(), n.item())]:.17f}")
