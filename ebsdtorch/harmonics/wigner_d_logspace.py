"""

Wigner d recursions in logspace useful for moderate bandwidths: L < 1500.

Algorithms based on:

"Numerical computation of spherical harmonics of arbitrary degree and order by
extending exponent of floating point numbers"

https://doi.org/10.1007/s00190-011-0519-2

"Numerical computation of Wigner's d-function of arbitrary high degree and
orders by extending exponent of floating point numbers"

http://dx.doi.org/10.13140/RG.2.2.31922.20160

https://www.researchgate.net/publication/309652602

Both by Toshio Fukushima.

Instead of using Toshio's x-numbers (fp64 mantissas with int32 exponents),
totaling 96 bits, we instead do everything in logspace with fp32 or fp64. As far
as I can tell this is a completely new approach to avoiding numerical
instability. I do not think I've ever seen fp32 Wigner d recursions. The
drawback comes with taking a bunch of logs and exps as the recursion is
additive. One big benefit is almost half the memory usage of double precision.
(almost half because I have to store the sign of the result separately). I have
checked the results against the double precision versions up to l_max = 1000.

"""

from typing import Tuple
import torch
from torch import Tensor
import math


@torch.jit.script
def signedlogsumexp(
    ln_a: Tensor,
    ln_b: Tensor,
    sign_a: Tensor,
    sign_b: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the log of the sum of the exponentials of the inputs and the sign of the result.

    Args:

    ln_a: Tensor
        The log of the magnitude of the first input.
    ln_b: Tensor
        The log of the magnitude of the second input.
    sign_a: Tensor
        The sign of the first input.
    sign_b: Tensor
        The sign of the second input.

    Returns:

    logsum: Tensor
        log of the magnitude of the signed sum of the exponentials.

    sign: Tensor
        sign of the sum of the exponentials of the inputs.

    """

    # Get the maximum and minimum of the inputs
    max_ln = torch.max(ln_a, ln_b)

    # only subtract where not -inf
    ln_a_shifted = torch.where(~torch.isinf(ln_a), ln_a - max_ln, ln_a)
    ln_b_shifted = torch.where(~torch.isinf(ln_b), ln_b - max_ln, ln_b)

    # exponentiate the shifted inputs and sum them with the correct signs
    exp_a = torch.exp(ln_a_shifted)
    exp_b = torch.exp(ln_b_shifted)
    exp_sum = exp_a * (2 * sign_a.to(exp_a.dtype) - 1) + exp_b * (
        2 * sign_b.to(exp_b.dtype) - 1
    )
    sign_out = exp_sum >= 0
    log_exp_sum_shifted = torch.log(torch.abs(exp_sum))

    # add the maximum back to the result
    logsum = log_exp_sum_shifted + max_ln

    return logsum, sign_out


@torch.jit.script
def log_powers_of_half(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Returns the powers of 0.5 up to n.

    Args:

    n: int
        The number of powers to compute.

    device: torch.device

    Returns:

    powers of sin/cos of pi/4: Tensor

    """

    # Initialize the powers of sin and cos of pi/4.
    powers = torch.arange(0, n, device=device, dtype=dtype)

    # Compute the powers
    return math.log(0.5) * powers


@torch.jit.script
def log_powers_trig_half_beta(
    beta: float,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the powers of sin and cos of beta/2 up to n.

    Args:

    beta: float
        Rotation angle beta in radians in the range [0, pi]
    n: int
        number of powers to compute 0, 1, ..., n - 1.
    dtype: torch.dtype
        datatype of the output.
    device: torch.device

    Returns:
        Tuple[Tensor, Tensor]
        :log of powers of cos
        :log of powers of sin

    """

    # Initialize the powers of sin and cos of beta/2.
    powers = torch.arange(0, n, device=device, dtype=dtype)

    # Compute the powers
    cos_powers = powers * math.log(math.cos(beta / 2.0))
    sin_powers = powers * math.log(math.sin(beta / 2.0))
    return cos_powers, sin_powers


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
def coords_from_index(
    indices: Tensor,
) -> Tensor:
    """

    Returns the coordinates (..., 3) ordered (l, m, n) based on the index into
    the Wigner-d 1D array. Only positive integers with l >= m >= n are
    considered.

    Args:
        indices: Tensor
            The indices into the Wigner-d 1D array.

    """
    # automatically determine the required datatype based on the maximum index
    dtype_int = torch.int32 if torch.max(indices).item() < 2**28 else torch.int64
    dtype_fp = torch.float32 if torch.max(indices).item() < 2**28 else torch.float64

    # this has to be done in double precision to avoid numerical instability
    indices_fp = indices.to(dtype_fp)

    # Compute coordinates via inverse tetrahedral formula:
    # https://en.wikipedia.org/wiki/Tetrahedral_number
    # inverse of x = l * (l + 1) * (l + 2) // 6
    # l = (3x + sqrt(9x^2 - 1/27))^(1/3) + (3x - sqrt(9x^2 - 1/27))^(1/3) - 1
    l_fp = (
        (
            3 * indices_fp
            + torch.sqrt(torch.abs(9 * indices_fp * indices_fp - (1.0 / 27.0)))
        ).pow(1.0 / 3.0)
        + (
            torch.abs(
                3 * indices_fp
                - torch.sqrt(torch.abs(9 * indices_fp * indices_fp - (1.0 / 27.0)))
            )
        ).pow(1.0 / 3.0)
        - 1.0
    )

    # overestimate of l, which will be corrected
    l = torch.floor(l_fp).to(dtype_int) + 1

    # we failed if indices - l * (l + 1) * (l + 2) // 6 is negative
    # need a while loop on the correction
    overestimated_l = indices - l * (l + 1) * (l + 2) // 6 < 0
    while torch.any(overestimated_l):
        l = torch.where(
            overestimated_l,
            l - 1,
            l,
        )
        overestimated_l = indices - l * (l + 1) * (l + 2) // 6 < 0

    # residual can very likely be done with 32 bits
    first_residual = (indices - l * (l + 1) * (l + 2) // 6).to(torch.int32)

    # l can now be cast back down to int32 as the correction is done
    l = l.to(torch.int32)

    # inverse of x = m * (m + 1) // 2
    # m = (-1 + sqrt(1 + 8 x)) // 2
    m = torch.floor((-1 + torch.sqrt(1 + 8 * first_residual)) // 2).to(torch.int32) + 1

    # again, we failed if indices - l * (l + 1) * (l + 2) // 6 - m * (m + 1) // 2 is negative
    # need a while loop on the correction
    overestimated_m = first_residual - m * (m + 1) // 2 < 0
    while torch.any(overestimated_m):
        m = torch.where(overestimated_m, m - 1, m)
        overestimated_m = first_residual - m * (m + 1) // 2 < 0

    # n is whatever is left
    n = first_residual - m * (m + 1) // 2

    return torch.stack((l, m, n), dim=-1)


@torch.jit.script
def wigner_d_lt_half_pi(
    beta: float,
    order_max: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """

    Returns Wigner d coefficients for β in [0, pi/2) for l >= m >= n >= 0

    Args:
        beta: float
            The angle beta in radians.
        order_max: int
            The maximum order / degree of the Wigner d functions.
        dtype: torch.dtype
            The datatype of the output.
        device: torch.device
            The device to run the computations on.

    Returns:
        wignerd: Tensor
            Log of the magnitude of Wigner d matrix entries
        wignerd_sign: Tensor
            Sign of Wigner d matrix entries

    """

    # initialize the output array
    wigner_d_logmag = torch.empty(
        size=(
            ((order_max * (order_max + 1)) // 2) * (order_max + 2) // 3
            + order_max * (order_max + 1) // 2
            + order_max
            + 1,
        ),
        # fill_value=torch.nan,
        dtype=dtype,
        device=device,
    )

    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    # initialize the powers of sin and cos of pi/4
    powers_of_cos_logscale, powers_of_sin_logscale = log_powers_trig_half_beta(
        beta, 2 * (order_max + 1), dtype, device
    )

    # get cos(beta) which Toshio denotes as t and its complement is tc = 1 - t = 1 - 2 sin(beta/2)^2
    tc = 2.0 * math.sin(beta / 2.0) ** 2

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1)
    coords = coords.view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)

    # make the coordinates the proper datatype
    mmn_coords_fp = mmn_coords.to(dtype)

    # d_m_mn = c_{m+n} s_{m-n} e_mn
    # for m >= n, e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    wigner_d_logmag[first_seed_indices] = (
        0.5
        * (
            torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
        )
        + powers_of_cos_logscale[mmn_coords[:, 1] + mmn_coords[:, 2]]
        + powers_of_sin_logscale[mmn_coords[:, 1] - mmn_coords[:, 2]]
    )

    # sign is positive
    wigner_d_sign[first_seed_indices] = True

    coords = torch.stack((mm, mm - 1, nn - 1), dim=-1)
    coords = coords.view(-1, 3)
    mask = (coords[:, 1] >= coords[:, 2]) & (coords[:, 1] >= 0) & (coords[:, 2] >= 0)
    mp1_mn_coords = coords[mask]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    # d_m+1_mn = a_mn * d_m_mn
    # a_mn = sqrt((2 * (2*m + 1)) / ((2m + 2n + 2) * (2m - 2n + 2))) * u_mn
    # u_mn for β in [0, pi/2] is (2m - 2n - 2) - (2m - 2) * tc
    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    u_mn = (2 * mp1_mn_coords_fp[:, 1] - 2 * mp1_mn_coords_fp[:, 2] + 2) - (
        2 * mp1_mn_coords_fp[:, 1] + 2
    ) * tc
    wigner_d_logmag[second_seed_indices] = (
        wigner_d_logmag[first_seed_indices_trunc]
        + 0.5
        * (
            torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
        )
        + torch.log(torch.abs(u_mn))
    )

    # this will be the sign of u_mn only
    wigner_d_sign[second_seed_indices] = ~torch.logical_xor(
        wigner_d_sign[first_seed_indices_trunc], u_mn >= 0
    )

    # get the starting recursion coords and indices
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
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
        b_log = w_log + v_log

        # u in logspace: 2*l (2*l - 2) - 4mn - 2l (2l - 2) * tc
        u = (
            (2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2))
            - (4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2])
            - (2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2) * tc)
        )

        u_log = torch.log(torch.abs(u))
        u_sign = u >= 0

        # a in logspace
        a_log = torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log

        # get a * d_l-1_mn
        term1_logmag = (
            a_log
            + wigner_d_logmag[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ]
        )
        # # term1_sign is the sign of (u times d_l-1_mn) as w is always positive
        term1_sign = ~torch.logical_xor(
            wigner_d_sign[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ],
            u_sign,
        )

        # get - (b * d_l-2_mn)
        term2_logmag = (
            b_log
            + wigner_d_logmag[curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])]
        )
        # for signs, b is always positive so term2_sign is negated
        # because we are subtracting with a call to logSUMexp
        term2_sign = ~wigner_d_sign[
            curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        ]

        # do logsumexp trick to find a * d_l-1_mn - b * d_l-2_mn
        wigner_d_logmag[curr_indices], wigner_d_sign[curr_indices] = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )

        # update the indices for the next iteration
        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d


@torch.jit.script
def wigner_d_eq_half_pi(
    order_max: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Returns Wigner d coefficients for β = pi/2 for l >= m >= n >= 0

    Args:
        order_max: int
            The maximum order / degree of the Wigner d functions.
        dtype: torch.dtype
            The datatype of the output.
        device: torch.device
            The device to run the computations on.

    Returns:
        wignerd: Tensor
            Log of the magnitude of Wigner d matrix entries
        wignerd_sign: Tensor
            Sign of Wigner d matrix entries

    """
    # initialize the output array
    wigner_d_logmag = torch.empty(
        size=((((order_max + 1) * (order_max + 2)) // 2) * (order_max + 3) // 3,),
        dtype=dtype,
        device=device,
    )

    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    # initialize the powers of sin and cos of pi/4
    powers_of_half_logscale = log_powers_of_half(order_max + 1, dtype, device)

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1)
    coords = coords.view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)

    # make the coordinates the proper datatype
    mmn_coords_fp = mmn_coords.to(dtype)

    # d_m_mn = (0.5 ** m) * e_mn
    # for m >= n, e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    # in logspace thats 0.5 * (lngamma(2m + 1) - lngamma(m + n + 1) - lngamma(m - n + 1))
    wigner_d_logmag[first_seed_indices] = (
        0.5
        * (
            torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
        )
        + powers_of_half_logscale[mmn_coords[:, 0]]
    )

    coords = torch.stack((mm, mm - 1, nn - 1), dim=-1)
    coords = coords.view(-1, 3)
    mask = (coords[:, 1] >= coords[:, 2]) & (coords[:, 1] >= 0) & (coords[:, 2] >= 0)
    mp1_mn_coords = coords[mask]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    # d_m+1_mn = a_mn * d_m_mn
    # a_mn = sqrt((2 * (2*m + 1)) / ((2m + 2n + 2) * (2m - 2n + 2))) * (-2n)
    # magnitude in logspace is 0.5 * (ln(2 * (2 * m + 1)) - ln(2 * m + 2 * n + 2) - ln(2 * m - 2 * n + 2)) + ln(2n)
    # sign is negative
    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    # mmn_coords_trunc = mmn_coords[: len(second_seed_indices)]

    wigner_d_logmag[second_seed_indices] = (
        wigner_d_logmag[first_seed_indices_trunc]
        + 0.5
        * (
            torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
        )
        + torch.log(2 * mp1_mn_coords_fp[:, 2])
    )
    wigner_d_sign[second_seed_indices] = False

    # can be tricky to index into flattened arrays for previous values
    # also I have to implement EXPSUMLOG (torch.logsumexp doesn't have a sign option)

    # get the starting recursion coords and indices
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
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
        b_log = w_log + v_log

        # u in logspace (it is negative): -(2m)*(2n)
        u_log = (
            math.log(4)
            + torch.log(curr_coords_fp[:, 1])
            + torch.log(curr_coords_fp[:, 2])
        )

        # a in logspace
        a_log = torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log

        # get a * d_l-1_mn
        term1_logmag = (
            a_log
            + wigner_d_logmag[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ]
        )
        # for signs, a is always negative so term1_sign flips
        term1_sign = ~wigner_d_sign[
            curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
        ]

        # get - (b * d_l-2_mn)
        term2_logmag = (
            b_log
            + wigner_d_logmag[curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])]
        )
        # for signs, b is always positive so term2_sign is negated
        # because we are subtracting with a call to logSUMexp
        term2_sign = ~wigner_d_sign[
            curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        ]

        # do logsumexp trick to find a * d_l-1_mn - b * d_l-2_mn
        wigner_d_logmag[curr_indices], wigner_d_sign[curr_indices] = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )

        # update the indices for the next iteration
        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )
    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d


@torch.jit.script
def wigner_d_gt_half_pi(
    beta: float,
    order_max: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """

    Returns Wigner d coefficients for β in (pi/2, pi] for l >= m >= n >= 0

    Args:
        beta: float
            The angle beta in radians.
        order_max: int
            The maximum order / degree of the Wigner d functions.
        dtype: torch.dtype
            The datatype of the output.
        device: torch.device
            The device to run the computations on.

    Returns:
        wignerd: Tensor
            Log of the magnitude of Wigner d matrix entries
        wignerd_sign: Tensor
            Sign of Wigner d matrix entries

    """

    # initialize the output array
    wigner_d_logmag = torch.empty(
        size=(
            ((order_max * (order_max + 1)) // 2) * (order_max + 2) // 3
            + order_max * (order_max + 1) // 2
            + order_max
            + 1,
        ),
        # fill_value=torch.nan,
        dtype=dtype,
        device=device,
    )

    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    # initialize the powers of sin and cos of pi/4
    powers_of_cos_logscale, powers_of_sin_logscale = log_powers_trig_half_beta(
        beta, 2 * (order_max + 1), dtype, device
    )

    # get cos(beta) which Toshio denotes as t and its complement is tc = 1 - t = 1 - 2 sin(beta/2)^2
    t = math.cos(beta)

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1)
    coords = coords.view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = index_from_coords(mmn_coords)

    # make the coordinates the proper datatype
    mmn_coords_fp = mmn_coords.to(dtype)

    # d_m_mn = c_{m+n} s_{m-n} e_mn
    # for m >= n, e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    wigner_d_logmag[first_seed_indices] = (
        0.5
        * (
            torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
        )
        + powers_of_cos_logscale[mmn_coords[:, 1] + mmn_coords[:, 2]]
        + powers_of_sin_logscale[mmn_coords[:, 1] - mmn_coords[:, 2]]
    )

    # adjust the sign which only depends on c_{m+n}
    wigner_d_sign[first_seed_indices] = True

    coords = torch.stack((mm, mm - 1, nn - 1), dim=-1)
    coords = coords.view(-1, 3)
    mask = (coords[:, 1] >= coords[:, 2]) & (coords[:, 1] >= 0) & (coords[:, 2] >= 0)
    mp1_mn_coords = coords[mask]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = index_from_coords(mp1_mn_coords)

    # d_m+1_mn = a_mn * d_m_mn
    # a_mn = sqrt((2 * (2*m + 1)) / ((2m + 2n + 2) * (2m - 2n + 2))) * u_mn
    # u_mn for β in [pi/2, pi] is (2m − 2)*t − (2n), where t = cos(β)
    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    u_mn = (2 * mp1_mn_coords_fp[:, 1] + 2) * t - 2 * mp1_mn_coords_fp[:, 2]
    wigner_d_logmag[second_seed_indices] = (
        wigner_d_logmag[first_seed_indices_trunc]
        + 0.5
        * (
            torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
        )
        + torch.log(torch.abs(u_mn))
    )

    # this will be the sign of u_mn only
    wigner_d_sign[second_seed_indices] = ~torch.logical_xor(
        wigner_d_sign[first_seed_indices_trunc], u_mn >= 0
    )

    # get the starting recursion coords and indices
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
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
        b_log = w_log + v_log

        # u in logspace: 2l(2l −2)t −(2m)(2n)
        u = (2 * curr_coords_fp[:, 0] * (2 * curr_coords_fp[:, 0] - 2) * t) - (
            4 * curr_coords_fp[:, 1] * curr_coords_fp[:, 2]
        )

        u_log = torch.log(torch.abs(u))
        u_sign = u >= 0

        # a in logspace
        a_log = torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log

        # get a * d_l-1_mn
        term1_logmag = (
            a_log
            + wigner_d_logmag[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ]
        )
        # # term1_sign is the sign of (u times d_l-1_mn) as w is always positive
        term1_sign = ~torch.logical_xor(
            wigner_d_sign[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ],
            u_sign,
        )

        # get - (b * d_l-2_mn)
        term2_logmag = (
            b_log
            + wigner_d_logmag[curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])]
        )
        # for signs, b is always positive so term2_sign is negated
        # because we are subtracting with a call to logSUMexp
        term2_sign = ~wigner_d_sign[
            curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        ]

        # do logsumexp trick to find a * d_l-1_mn - b * d_l-2_mn
        wigner_d_logmag[curr_indices], wigner_d_sign[curr_indices] = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )

        # update the indices for the next iteration
        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )

    wigner_d = torch.exp(wigner_d_logmag) * (
        2 * wigner_d_sign.to(wigner_d_logmag.dtype) - 1
    )
    return wigner_d


@torch.jit.script
def read_lmn_wigner_d_half_pi_table(
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

    both_neg_mask = (m_new < 0) & (n_new < 0)
    only_m_neg_mask = (m_new < 0) & (n_new >= 0)
    only_n_neg_mask = (n_new < 0) & (m_new >= 0)

    prefactor = (
        torch.where(
            both_neg_mask,
            (-1.0) ** ((m_new - n_new) % 2),
            1.0,
        )
        * torch.where(
            only_m_neg_mask,
            (-1.0) ** ((l - n_new) % 2),
            1.0,
        )
        * torch.where(
            only_n_neg_mask,
            (-1.0) ** ((l + m_new) % 2),
            1.0,
        )
        * torch.where(
            mask_mn_swap,
            (-1.0) ** ((m_new - n_new) % 2),
            1.0,
        )
    )

    if swap_mn:
        prefactor *= (-1.0) ** ((m_new - n_new) % 2)

    return (
        wigner_d_values[
            (((l * (l + 1)) // 2 * (l + 2)) // 3)
            + ((torch.abs(m_new) * (torch.abs(m_new) + 1)) // 2)
            + torch.abs(n_new)
        ]
        * prefactor
    )


@torch.jit.script
def read_mn_wigner_d_half_pi_table(
    wigner_d_values: Tensor,
    m_coords: Tensor,
    n_coords: Tensor,
    l: int,
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

    m, n = m_coords, n_coords

    mask_mn_swap = torch.abs(m) < torch.abs(n)
    m_new = torch.where(mask_mn_swap, n, m)
    n_new = torch.where(mask_mn_swap, m, n)

    both_neg_mask = (m_new < 0) & (n_new < 0)
    only_m_neg_mask = (m_new < 0) & (n_new >= 0)
    only_n_neg_mask = (n_new < 0) & (m_new >= 0)

    prefactor = (
        torch.where(
            both_neg_mask,
            (-1.0) ** ((m_new - n_new) % 2),
            1.0,
        )
        * torch.where(
            only_m_neg_mask,
            (-1.0) ** ((l - n_new) % 2),
            1.0,
        )
        * torch.where(
            only_n_neg_mask,
            (-1.0) ** ((l + m_new) % 2),
            1.0,
        )
        * torch.where(
            mask_mn_swap,
            (-1.0) ** ((m_new - n_new) % 2),
            1.0,
        )
    )

    if swap_mn:
        prefactor *= (-1.0) ** ((m_new - n_new) % 2)

    return (
        wigner_d_values[
            (((l * (l + 1)) // 2 * (l + 2)) // 3)
            + ((torch.abs(m_new) * (torch.abs(m_new) + 1)) // 2)
            + torch.abs(n_new)
        ]
        * prefactor
    )


def csht_weights_half_pi(
    L: int,
    device: torch.device,
    precision: str = "double",
    batch_size: int = int(1e7),
):
    """
    Returns the spherical harmonic transform weights for the case where the angle is pi/2.

    Args:
        L: int
            The maximum degree to calculate d_jkm for.
        device: torch.device
            The device to run the computations on.

    Returns:
        wig: Tensor of expanded Wigner d coefficients

    """
    dtype_fp = torch.float64 if precision == "double" else torch.float32
    dtype_int = torch.int64 if precision == "double" else torch.int32

    wigner_d_table = wigner_d_eq_half_pi(L, dtype_fp, device)
    max_index = (((L - 1) * L) // 2 * (L + 1)) // 3 + ((L - 1) * L) // 2 + L
    indices = []
    values = []

    # here we do batches because we can't necessarily store the 8x
    # larger table with m < 0 and |m| < |n| both allowed in memory
    for batch_start in range(0, max_index, batch_size):
        indices_batch = torch.arange(
            batch_start,
            min(batch_start + batch_size, max_index),
            device=device,
            dtype=dtype_int,
        )
        lmk = coords_from_index(indices_batch)

        # quarter clone will first be swapped where m != n
        lmk_quarter_clone = lmk.clone()
        mask_m_neq_n = lmk_quarter_clone[:, 1] != lmk_quarter_clone[:, 2]
        lmk_quarter_clone = lmk_quarter_clone[mask_m_neq_n]
        lmk_quarter_clone = torch.stack(
            [lmk_quarter_clone[:, 0], lmk_quarter_clone[:, 2], lmk_quarter_clone[:, 1]],
            dim=-1,
        )

        # half table is formed by concatenating the original and the quarter clone
        lmk_half = torch.cat([lmk, lmk_quarter_clone], dim=0)

        # clone half lmk tensor and negate m where m != 0
        lmk_half_clone = lmk_half.clone()
        mask_m_neq_0 = lmk_half_clone[:, 1] != 0
        lmk_half_clone = lmk_half_clone[mask_m_neq_0]
        lmk_half_clone[:, 1] = -lmk_half_clone[:, 1]

        # full lmk tensor is formed by concatenating the half and half clone
        lmk_full = torch.cat([lmk_half, lmk_half_clone], dim=0)

        l, m, n = torch.unbind(lmk_full, dim=-1)

        m_mod_2_mask = (m % 2) == 0
        m_mod_4_eq_1_mask = (m % 4) == 1
        m_mod_4_eq_2_mask = (m % 4) == 2
        n_eq_0_mask = n == 0
        n_eq_l_mask = n == l

        N = 2 * L

        term = torch.where(
            m_mod_2_mask,
            torch.where(
                n_eq_0_mask,
                torch.sqrt((N * (2 * l.to(dtype_fp) + 1) / 2.0)),
                torch.sqrt(N * (2 * l.to(dtype_fp) + 1)),
            )
            * torch.where(m_mod_4_eq_2_mask, -1.0, 1.0),
            torch.where(
                n_eq_l_mask,
                0,
                torch.sqrt(N * (2 * l.to(dtype_fp) + 1))
                * torch.where(m_mod_4_eq_1_mask, -1.0, 1.0),
            ),
        )

        z = torch.zeros_like(l, dtype=dtype_int)

        # the swap in n and m in the access coordinates (e.g. [l, n, -m]) is
        # from Edmonds (l, -m, 0) expansion of the Wigner d function into cosine
        # and sine series. The swap mn flag called "swap_mn" is due to TS2Kit
        # having a different convention (the first Euler angle is associated
        # with the first order instead of the second order).
        wig_prod = torch.where(
            m_mod_2_mask,
            read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n, -m], dim=-1), swap_mn=True
            )
            * read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n, z], dim=-1), swap_mn=True
            ),
            read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n + 1, -m], dim=-1), swap_mn=True
            )
            * read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n + 1, z], dim=-1), swap_mn=True
            ),
        )

        coeff = term * wig_prod

        # # for debugging:
        # for i in range(len(coeff)):
        #     print(
        #         f"l: {l[i]}, m: {m[i]}, n: {n[i]}, coeff: {coeff[i]} = {term[i]} * {wig_prod[i]}"
        #     )

        first_indices = (m + L - 1) * L + l
        second_indices = N * (m + L - 1) + n

        # filter indices and values out where the magnitude is less than 1e-15
        mask = torch.abs(coeff) > 1.0e-14
        first_indices = first_indices[mask]
        second_indices = second_indices[mask]
        coeff = coeff[mask]

        indices_batch = torch.stack([first_indices, second_indices], dim=0)
        indices.append(indices_batch)
        values.append(coeff)

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values, dim=0)

    wig = torch.sparse_coo_tensor(
        indices,
        values,
        [L * (2 * L - 1), 2 * L * (2 * L - 1)],
        dtype=dtype_fp,
        device=device,
    )
    return wig


def rsht_weights_half_pi(
    L: int,
    device: torch.device,
    precision: str = "double",
    batch_size: int = int(1e7),
):
    """
    Returns the spherical harmonic transform weights for the case where the
    angle is pi/2 and the signal on the sphere is real.

    Args:
        L: int
            The maximum degree to calculate d_jkm for.
        device: torch.device
            The device to run the computations on.

    Returns:
        wig: Tensor of expanded Wigner d coefficients

    """
    dtype_fp = torch.float64 if precision == "double" else torch.float32
    dtype_int = torch.int64 if precision == "double" else torch.int32

    wigner_d_table = wigner_d_eq_half_pi(L, dtype_fp, device)
    max_index = (((L - 1) * L) // 2 * (L + 1)) // 3 + ((L - 1) * L) // 2 + L

    indices = []
    values = []

    # here we do batches because we can't necessarily store the 8x
    # larger table with m < 0 and |m| < |n| both allowed in memory
    for batch_start in range(0, max_index, batch_size):
        indices_batch = torch.arange(
            batch_start,
            min(batch_start + batch_size, max_index),
            device=device,
            dtype=dtype_int,
        )
        lmk = coords_from_index(indices_batch)

        # quarter clone will first be swapped where m != n
        lmk_half_clone = lmk.clone()
        mask_m_neq_n = lmk_half_clone[:, 1] != lmk_half_clone[:, 2]
        lmk_half_clone = lmk_half_clone[mask_m_neq_n]
        lmk_half_clone = torch.stack(
            [lmk_half_clone[:, 0], lmk_half_clone[:, 2], lmk_half_clone[:, 1]],
            dim=-1,
        )

        # half table is formed by concatenating the original and the quarter clone
        lmk_full = torch.cat([lmk, lmk_half_clone], dim=0)

        l, m, n = torch.unbind(lmk_full, dim=-1)

        m_mod_2_mask = (m % 2) == 0
        m_mod_4_eq_1_mask = (m % 4) == 1
        m_mod_4_eq_2_mask = (m % 4) == 2
        n_eq_0_mask = n == 0
        n_eq_l_mask = n == l

        N = 2 * L

        term = torch.where(
            m_mod_2_mask,
            torch.where(
                n_eq_0_mask,
                torch.sqrt((N * (2 * l.to(dtype_fp) + 1) / 2.0)),
                torch.sqrt(N * (2 * l.to(dtype_fp) + 1)),
            )
            * torch.where(m_mod_4_eq_2_mask, -1.0, 1.0),
            torch.where(
                n_eq_l_mask,
                0,
                torch.sqrt(N * (2 * l.to(dtype_fp) + 1))
                * torch.where(m_mod_4_eq_1_mask, -1.0, 1.0),
            ),
        )
        z = torch.zeros_like(l, dtype=dtype_int)

        # the swap in n and m in the access coordinates (e.g. [l, n, -m]) is
        # from Edmonds (l, -m, 0) expansion of the Wigner d function into cosine
        # and sine series. The swap mn flag called "swap_mn" is due to TS2Kit
        # having a different convention (the first Euler angle is associated
        # with the first order instead of the second order).
        wig_prod = torch.where(
            m_mod_2_mask,
            read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n, -m], dim=-1), swap_mn=True
            )
            * read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n, z], dim=-1), swap_mn=True
            ),
            read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n + 1, -m], dim=-1), swap_mn=True
            )
            * read_lmn_wigner_d_half_pi_table(
                wigner_d_table, torch.stack([l, n + 1, z], dim=-1), swap_mn=True
            ),
        )
        coeff = term * wig_prod

        # # for debugging:
        # for i in range(len(coeff)):
        #     print(
        #         f"l: {l[i]}, m: {m[i]}, n: {n[i]}, coeff: {coeff[i]} = {term[i]} * {wig_prod[i]}"
        #     )

        first_indices = m * L + l
        second_indices = N * m + n

        # filter indices and values out where the magnitude is less than 1e-15
        mask = torch.abs(coeff) > 1.0e-14
        first_indices = first_indices[mask]
        second_indices = second_indices[mask]
        coeff = coeff[mask]

        indices_batch = torch.stack([first_indices, second_indices], dim=0)
        indices.append(indices_batch)
        values.append(coeff)

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values, dim=0)

    wig = torch.sparse_coo_tensor(
        indices,
        values,
        [L * L, 2 * L * L],
        dtype=dtype_fp,
        device=device,
    )
    return wig
