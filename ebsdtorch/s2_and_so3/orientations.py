"""

Routines for orientation representations adopted from PyTorch3D and from EMsoft

https://github.com/facebookresearch/pytorch3d

https://github.com/marcdegraef/3Drotations

Abbreviations used in the code:

cu: cubochoric
ho: homochoric
ax: axis-angle
qu: quaternion
om: orientation matrix
bu: Bunge ZXZ Euler angles
cl: Clifford Torus
ro: Rodrigues-Frank vector
zh: 6D continuous representation of orientation

"""

import torch
from torch import Tensor


@torch.jit.script
def qu2ho(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to homochoric coordinates.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4).

    Returns:
        Homochoric coordinates as tensor of shape (..., 3).
    """
    if qu.size(-1) != 4:
        raise ValueError(f"Invalid quaternion shape {qu.shape}.")
    ho = torch.empty_like(qu[..., :3])
    # get the angle
    angle = 2 * torch.acos(qu[..., 0:1].clamp_(min=-1.0, max=1.0))
    # get the unit vector
    unit = qu[..., 1:] / torch.norm(qu[..., 1:], dim=-1, keepdim=True)
    ho = unit * (3.0 * (angle - torch.sin(angle)) / 4.0) ** (1 / 3)
    # fix the case where the angle is zero
    ho[(angle.squeeze(-1) < 1e-8)] = 0.0
    return ho


@torch.jit.script
def om2qu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Notes:

    Farrell, J.A., 2015. Computation of the Quaternion from a Rotation Matrix.
    University of California, 2.

    "Converting a Rotation Matrix to a Quaternion" by Mike Day, Insomniac Games

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m10 = matrix[..., 1, 0]

    mask_A = m22 < 0
    mask_B = m00 > m11
    mask_C = m00 < -m11
    branch_1 = mask_A & mask_B
    branch_2 = mask_A & ~mask_B
    branch_3 = ~mask_A & mask_C
    branch_4 = ~mask_A & ~mask_C

    branch_1_t = 1 + m00[branch_1] - m11[branch_1] - m22[branch_1]
    branch_1_t_rsqrt = 0.5 * torch.rsqrt(branch_1_t)
    branch_2_t = 1 - m00[branch_2] + m11[branch_2] - m22[branch_2]
    branch_2_t_rsqrt = 0.5 * torch.rsqrt(branch_2_t)
    branch_3_t = 1 - m00[branch_3] - m11[branch_3] + m22[branch_3]
    branch_3_t_rsqrt = 0.5 * torch.rsqrt(branch_3_t)
    branch_4_t = 1 + m00[branch_4] + m11[branch_4] + m22[branch_4]
    branch_4_t_rsqrt = 0.5 * torch.rsqrt(branch_4_t)

    qu = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    qu[branch_1, 1] = branch_1_t * branch_1_t_rsqrt
    qu[branch_1, 2] = (m01[branch_1] + m10[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 3] = (m20[branch_1] + m02[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 0] = (m12[branch_1] - m21[branch_1]) * branch_1_t_rsqrt

    qu[branch_2, 1] = (m01[branch_2] + m10[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 2] = branch_2_t * branch_2_t_rsqrt
    qu[branch_2, 3] = (m12[branch_2] + m21[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 0] = (m20[branch_2] - m02[branch_2]) * branch_2_t_rsqrt

    qu[branch_3, 1] = (m20[branch_3] + m02[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 2] = (m12[branch_3] + m21[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 3] = branch_3_t * branch_3_t_rsqrt
    qu[branch_3, 0] = (m01[branch_3] - m10[branch_3]) * branch_3_t_rsqrt

    qu[branch_4, 1] = (m12[branch_4] - m21[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 2] = (m20[branch_4] - m02[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 3] = (m01[branch_4] - m10[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 0] = branch_4_t * branch_4_t_rsqrt

    # guarantee the correct axis signs
    qu[..., 0] = torch.abs(qu[..., 0])
    qu[..., 1] = qu[..., 1].copysign((m21 - m12))
    qu[..., 2] = qu[..., 2].copysign((m02 - m20))
    qu[..., 3] = qu[..., 3].copysign((m10 - m01))

    return qu


@torch.jit.script
def om2ax(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        axis-angle representation as tensor of shape (..., 4).

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    # set the output with the same batch dimensions as the input
    axis = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    # Get the trace of the matrix
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    # find the angles
    acos_arg = 0.5 * (trace - 1.0)
    acos_arg = torch.clamp(acos_arg, -1.0, 1.0)
    theta = torch.acos(acos_arg)

    # where the angle is small, treat theta/sin(theta) as 1
    stable = theta > 0.001
    axis[..., 0] = matrix[..., 2, 1] - matrix[..., 1, 2]
    axis[..., 1] = matrix[..., 0, 2] - matrix[..., 2, 0]
    axis[..., 2] = matrix[..., 1, 0] - matrix[..., 0, 1]
    factor = torch.where(stable, 0.5 / torch.sin(theta), 0.5)
    axis[..., :3] = factor[:, None] * axis[:, :3]

    # normalize the axis
    axis[..., :3] /= torch.norm(axis[:, :3], dim=-1, keepdim=True)

    # set the angle
    axis[..., 3] = theta

    return axis.view(batch_dim + (4,))


@torch.jit.script
def ax2om(axis_angle: Tensor) -> Tensor:
    """
    Convert axis-angle representation to rotation matrices.

    Args:
        axis_angle: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if axis_angle.size(-1) != 4:
        raise ValueError(f"Invalid axis-angle shape {axis_angle.shape}.")

    batch_dim = axis_angle.shape[:-1]
    data_n = int(torch.prod(torch.tensor(batch_dim)))

    # set the output
    matrices = torch.empty(
        batch_dim + (3, 3), dtype=axis_angle.dtype, device=axis_angle.device
    )

    theta = axis_angle[..., 3:4]
    omega = axis_angle[..., :3] * theta

    matrices = torch.zeros((data_n, 3, 3), dtype=omega.dtype, device=omega.device)
    matrices[..., 0, 1] = -omega[..., 2]
    matrices[..., 0, 2] = omega[..., 1]
    matrices[..., 1, 2] = -omega[..., 0]
    matrices[..., 1, 0] = omega[..., 2]
    matrices[..., 2, 0] = -omega[..., 1]
    matrices[..., 2, 1] = omega[..., 0]

    skew_sq = torch.matmul(matrices, matrices)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.05).squeeze()

    theta_unstable = theta[~stable].unsqueeze(-1)

    # This prefactor is only used for the calculation of exp(skew)
    # sin(theta) / theta
    # expression: 1 - theta^2 / 6 + theta^4 / 120 - theta^6 / 5040 ...
    prefactor1 = 1 - theta_unstable**2 / 6

    # This prefactor is shared between calculations of exp(skew) and v
    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    prefactor2 = 1 / 2 - theta_unstable**2 / 24

    theta_stable = theta[stable].unsqueeze(-1)
    matrices[stable] = (
        torch.eye(3, dtype=matrices.dtype, device=matrices.device)
        + (torch.sin(theta_stable) / theta_stable) * matrices[stable]
        + (1 - torch.cos(theta_stable)) / theta_stable**2 * skew_sq[stable]
    )
    matrices[~stable] = (
        torch.eye(3, dtype=matrices.dtype, device=matrices.device)
        + prefactor1 * matrices[~stable]
        + prefactor2 * skew_sq[~stable]
    )

    return matrices.view(batch_dim + (3, 3))


# @torch.jit.script
# def cu2ho(cu: Tensor) -> Tensor:
#     """
#     Converts cubochoric vector representation to homochoric vector representation.

#     Args:
#         cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

#     Returns:
#         Homochoric vectors as tensor of shape (..., 3).
#     """

#     # step 1: scale by (pi/6)^(1/6)
#     ho = cu * (torch.pi / 6) ** (1 / 6)

#     # step 2: apply the squircle function T to x and y
#     # this is simplified by reordering xyz so that only Appendix A eq (27) is needed
#     indices = torch.argsort(torch.abs(ho), dim=-1, descending=False)
#     sorted = torch.gather(ho, -1, indices)
#     sml, med, big = sorted.unbind(dim=-1)

#     # this is nan if at [+/-1, 0, 0]
#     trig_arg_xy = sml * torch.pi / (12.0 * med)
#     trig_arg_xy[torch.isnan(trig_arg_xy)] = 0
#     factor_xy_s2 = (
#         (2**0.25)
#         * med
#         / (((torch.pi / 6.0) ** 0.5) * torch.sqrt((2**0.5) - torch.cos(trig_arg_xy)))
#     )

#     # note that big is z_s2 but it is unchanged unlike x and y
#     x_s2 = factor_xy_s2 * ((2**0.5) * torch.sin(trig_arg_xy))
#     y_s2 = factor_xy_s2 * ((2**0.5) * torch.cos(trig_arg_xy) - 1)

#     # step 3
#     factor_xy_s3 = torch.sqrt(1 - torch.pi * ((x_s2**2) + (y_s2**2)) / (24 * big**2))
#     x_s3 = x_s2 * factor_xy_s3
#     y_s3 = y_s2 * factor_xy_s3
#     z_s3 = ((6 / torch.pi) ** 0.5) * big - (
#         ((torch.pi / 24.0) ** 0.5) * (x_s2**2 + y_s2**2) / big
#     )

#     # undo the argsort with an in-place scatter
#     ho.scatter_(-1, indices, torch.stack((x_s3, y_s3, z_s3), dim=-1))

#     # replace any nans with 0
#     ho[torch.isnan(ho)] = 0
#     return ho


@torch.jit.script
def cu2ho(cu: torch.Tensor) -> torch.Tensor:
    """
    Converts cubochoric vector representation to homochoric vector representation.

    Args:
        cu: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """

    # Sort components
    indices = torch.argsort(torch.abs(cu), dim=-1, descending=False)
    sorted = torch.gather(cu, -1, indices)
    s, m, b = sorted.unbind(dim=-1)

    # Calculate trigonometric argument and avoid indeterminate forms
    trig_arg_xy = s * torch.pi / (12.0 * m)
    trig_arg_xy[torch.isnan(trig_arg_xy)] = 0

    factor_xy = (
        2 ** (1 / 12)
        * 3 ** (1 / 3)
        * m
        * torch.sqrt(
            (
                4 * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
                + (2**0.5) * m**2 * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
            )
            / (torch.cos(trig_arg_xy) - (2**0.5))
        )
        / (torch.pi ** (1 / 3) * b * torch.sqrt(-torch.cos(trig_arg_xy) + (2**0.5)))
    )

    # Compute x_s3, y_s3, and z_s3 using the provided equations
    x_s3 = factor_xy * torch.sin(trig_arg_xy)
    y_s3 = factor_xy * 2**-0.5 * ((2**0.5) * torch.cos(trig_arg_xy) - 1)
    z_s3 = (
        2 * 6 ** (1 / 3) * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
        + 2 ** (5 / 6)
        * 3 ** (1 / 3)
        * m**2
        * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
    ) / (2 * torch.pi ** (1 / 3) * b * (torch.cos(trig_arg_xy) - (2**0.5)))

    # Reassemble the vector and undo the argsort
    ho = torch.stack((x_s3, y_s3, z_s3), dim=-1)
    ho = torch.scatter(ho, -1, indices, ho)

    # replace any nans with 0
    ho[torch.isnan(ho)] = 0

    # copy the sign of the original cubochoric vector
    ho.copysign_(cu)

    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to cubochoric vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).

    """

    # inverse steps in reverse order of cu2ho
    # start with an argsort on the magnitudes
    indices = torch.argsort(torch.abs(ho), dim=-1, descending=False)
    sorted = torch.gather(ho, -1, indices)
    x_s3, y_s3, z_s3 = torch.abs(sorted).unbind(dim=-1)

    # step 3 inverse
    r_s = torch.norm(ho, dim=-1, keepdim=False)
    prefactor_xy_s3 = torch.sqrt(2 * r_s / (r_s + z_s3))
    x_s2 = x_s3 * prefactor_xy_s3
    y_s2 = y_s3 * prefactor_xy_s3
    z_s2 = (torch.pi / 6) ** 0.5 * r_s

    # # step 2 inverse from Appendix A eq (29)
    # prefactor_xy_s2 = (torch.pi / 6)**0.5 * torch.sqrt((x_s2**2 + 2 * y_s2**2) * (x_s2**2 + y_s2**2)) / (
    #     (2 ** 0.5) * torch.sqrt(x_s2**2 + 2 * y_s2**2 - (torch.abs(y_s2) * torch.sqrt(x_s2**2 + 2 * y_s2**2)))
    # )
    # the above equation in the publication can be dramatically simplified if you assume x and y are positive:
    # ((x^2 + 2 * y^2) * (x^2 + y^2)) / (x^2 + 2 * y^2 - |y| * sqrt(x^2 + 2*y^2)) is
    # the same as:
    # x^2 + y*(sqrt(x^2 + 2*y^2) + 2*y)
    # for x and y positive
    # these are the inverse squircle functions
    # this also avoids an annoying 0/0 case that slows down the calculation
    prefactor_xy_s2 = (
        (torch.pi / 6) ** 0.5
        * torch.sqrt(x_s2**2 + y_s2 * (torch.sqrt(x_s2**2 + 2 * y_s2**2) + 2 * y_s2))
        / (2**0.5)
    )
    # z_s1 is unchanged from z_s2 while x_s1 and y_s1 are found from inverse squircle function
    x_s1 = (prefactor_xy_s2 * 12.0 * torch.sign(x_s2) / torch.pi) * (
        torch.arccos(
            (
                (x_s2**2 + y_s2 * torch.sqrt(x_s2**2 + 2 * y_s2**2))
                / ((2**0.5) * (x_s2**2 + y_s2**2))
            ).clamp_(-1.0, 1.0)
        )
    )
    y_s1 = prefactor_xy_s2 * torch.sign(y_s2)

    # undo the argsort with an in-place scatter
    cu = torch.empty_like(ho)
    cu.scatter_(-1, indices, torch.stack((x_s1, y_s1, z_s2), dim=-1))
    cu /= (torch.pi / 6) ** (1 / 6)

    # copy the sign of the original homochoric vector
    cu.copysign_(ho)

    # replace any nans with 0
    cu[torch.isnan(cu)] = 0

    return cu


@torch.jit.script
def ho2ax(ho: Tensor, fast: bool = True) -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)
        fast (bool): by default skip Newton iteration for FP64 only

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    These are Chebyshev fits on the modified homochoric inverse
    fitted by Zachary Varley on 05/24/2024. The modified homochoric
    inverse ties the square of the homochoric vector back to the
    cosine of the half rotation angle.

    """
    if ho.dtype == torch.float32 or ho.dtype == torch.float16:
        fit_parameters = torch.tensor(
            [
                # 8 terms to reach FP32 machine eps
                1.0000000000000009e00,
                -4.9999943403867775e-01,
                -2.5015165060149020e-02,
                -3.8120131548551729e-03,
                -1.2106188330642162e-03,
                4.9329295993155416e-04,
                -7.0089385526450620e-04,
                3.0979774923589078e-04,
                -7.3023474963298843e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    else:
        # 10 term loss mean abs error 5e-10 instead of 15 term's 1e-11
        # 1 iteration of Newton's method is needed for double precision
        # machine error... 20 terms somehow is stuck at 1e-11 error
        fit_parameters = torch.tensor(
            [
                # 10 term polyfit
                1.0000000000000000e00,
                -4.9999997124013285e-01,
                -2.5001181866044025e-02,
                -3.9144209820521038e-03,
                -8.9320268104539483e-04,
                3.1181024286083695e-05,
                -4.3961032788396477e-04,
                3.9657471727506439e-04,
                -2.6379945050586932e-04,
                9.1185355979587159e-05,
                -1.4875867805692529e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    # ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)
    # # makes out of memory error doing all at once
    # s = torch.zeros_like(ho_norm_sq[..., 0])
    # for i in range(len(fit_parameters)):
    #     s += fit_parameters[i] * ho_norm_sq[..., 0] ** i

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=False)
    # makes out of memory error doing all at once
    s = torch.zeros_like(ho_norm_sq)
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq**i

    if ho.dtype == torch.float64 and not fast:
        w = 2 * torch.arccos(torch.clamp(s, -1.0, 1.0))
        # do 1 iteration of Newton's method
        f_w = ((3 / 4) * (w - torch.sin(w))) ** (1 / 3) - torch.sqrt(ho_norm_sq)
        f_p_w = (1 - torch.cos(w)) / (6 ** (2 / 3) * (w - torch.sin(w)) ** (2 / 3))
        update = f_w / f_p_w
        # remove any nans
        update[torch.isnan(update)] = 0
        w -= update
    else:
        w = 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))

    ax = torch.concat(
        [
            ho * torch.rsqrt(ho_norm_sq).unsqueeze(-1),
            w.unsqueeze(-1),
        ],
        dim=-1,
    )
    rot_is_identity = torch.abs(ho_norm_sq) < 1e-6
    # set the identity rotation
    ax[rot_is_identity] = 0
    ax[rot_is_identity, ..., 2] = 1.0
    return ax


@torch.jit.script
def ho2ax_reference(ho: Tensor, coeffs: str = "kikuchipy") -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    I have seen two polynomial fits for this conversion, one from EMsoft
    and the other from Kikuchipy. The Kikuchipy one is used here.


    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    f(w) = [(3/4) * (w - sin(w))]^(1/3) -> no inverse -> polynomial fit it

    """
    if coeffs == "kikuchipy":
        fit_parameters = torch.tensor(
            [
                # Kikuchipy polyfit coeffs
                1.0000000000018852,
                -0.5000000002194847,
                -0.024999992127593126,
                -0.003928701544781374,
                -0.0008152701535450438,
                -0.0002009500426119712,
                -0.00002397986776071756,
                -0.00008202868926605841,
                0.00012448715042090092,
                -0.0001749114214822577,
                0.0001703481934140054,
                -0.00012062065004116828,
                0.000059719705868660826,
                -0.00001980756723965647,
                0.000003953714684212874,
                -0.00000036555001439719544,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    elif coeffs == "EMsoft":
        fit_parameters = torch.tensor(
            [
                # EMsoft polyfit coeffs
                0.9999999999999968,
                -0.49999999999986866,
                -0.025000000000632055,
                -0.003928571496460683,
                -0.0008164666077062752,
                -0.00019411896443261646,
                -0.00004985822229871769,
                -0.000014164962366386031,
                -1.9000248160936107e-6,
                -5.72184549898506e-6,
                7.772149920658778e-6,
                -0.00001053483452909705,
                9.528014229335313e-6,
                -5.660288876265125e-6,
                1.2844901692764126e-6,
                1.1255185726258763e-6,
                -1.3834391419956455e-6,
                7.513691751164847e-7,
                -2.401996891720091e-7,
                4.386887017466388e-8,
                -3.5917775353564864e-9,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)
    else:
        raise ValueError(f"Invalid fit parameters {coeffs}.")

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)

    # makes out of memory error doing all at once
    s = torch.zeros_like(ho_norm_sq[..., 0])
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq[..., 0] ** i

    ax = torch.empty(ho.shape[:-1] + (4,), dtype=ho.dtype, device=ho.device)

    rot_is_identity = torch.abs(ho_norm_sq.squeeze(-1)) < 1e-8
    ax[rot_is_identity, 0:1] = 0.0
    ax[rot_is_identity, 1:2] = 0.0
    ax[rot_is_identity, 2:3] = 1.0

    ax[~rot_is_identity, :3] = ho[~rot_is_identity, :] * torch.rsqrt(
        ho_norm_sq[~rot_is_identity]
    )
    ax[..., 3] = torch.where(
        ~rot_is_identity,
        2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0)),
        0,
    )
    return ax


@torch.jit.script
def ho2ax_newton(ho: Tensor) -> Tensor:
    """
    Converts homochoric coordinates to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)

    Notes:
        Newton's method

    """

    # initial guess for ang given h
    h = torch.norm(ho, dim=-1)

    # where zero return zero
    mask_zero = h == 0
    ax = torch.empty(ho.shape[:-1] + (4,), dtype=ho.dtype, device=ho.device)
    ax[mask_zero, 0] = 0.0
    ax[mask_zero, 1] = 0.0
    ax[mask_zero, 2] = 1.0
    ax[mask_zero, 3] = 0.0

    # Newton's method
    # initial guess for w given h is an inverted Pade approximation
    w_newton = (15 - torch.sqrt(225 - 60 * h[~mask_zero] ** 2)) / h[~mask_zero]

    # Newton's method
    for _ in range(2 if h.dtype == torch.float32 else 3):
        f_w = ((3 / 4) * (w_newton - torch.sin(w_newton))) ** (1 / 3) - h[~mask_zero]
        f_p_w = (1 - torch.cos(w_newton)) / (
            6 ** (2 / 3) * (w_newton - torch.sin(w_newton)) ** (2 / 3)
        )
        update = f_w / f_p_w
        # remove any nans
        update[torch.isnan(update)] = 0
        w_newton -= update

    ax[~mask_zero, 0:3] = ho[~mask_zero] * torch.rsqrt(h[~mask_zero].unsqueeze(-1))
    ax[~mask_zero, 3] = w_newton

    return ax


@torch.jit.script
def ax2ho(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to homochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).

    """
    return (0.75 * (ax[..., 3:4] - torch.sin(ax[..., 3:4]))) ** (1.0 / 3.0) * ax[
        ..., :3
    ]


@torch.jit.script
def ax2ro(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to Rodrigues vector representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle (x, y, z, angle)

    Returns:
        torch.Tensor: shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2))
    """
    ro = ax.clone()
    ro[..., 3] = torch.tan(ax[..., 3] / 2)
    return ro


@torch.jit.script
def ro2ax(ro: Tensor) -> Tensor:
    """
    Converts a rotation vector to an axis-angle representation.

    Args:
        ro (Tensor): shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2)).

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle).
    """
    ax = torch.empty_like(ro)
    mask_zero_ro = torch.abs(ro[..., 3]) == 0
    ax[mask_zero_ro] = torch.tensor([0, 0, 1, 0], dtype=ro.dtype, device=ro.device)

    mask_inf_ro = torch.isinf(ro[..., 3])
    ax[mask_inf_ro, :3] = ro[mask_inf_ro, :3]
    ax[mask_inf_ro, 3] = torch.pi

    mask_else = ~(mask_zero_ro | mask_inf_ro)
    ax[mask_else, :3] = ro[mask_else, :3] / torch.norm(
        ro[mask_else, :3], dim=-1, keepdim=True
    )
    ax[mask_else, 3] = 2 * torch.atan(ro[mask_else, 3])
    return ax


@torch.jit.script
def ax2qu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to quaternion representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle in the format (x, y, z, angle).

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty_like(ax)
    cos_half_ang = torch.cos(ax[..., 3] / 2.0)
    sin_half_ang = torch.sin(ax[..., 3:4] / 2.0)
    qu[..., 0] = cos_half_ang
    qu[..., 1:] = ax[..., :3] * sin_half_ang
    return qu


@torch.jit.script
def qu2ax(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to axis-angle representation.

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 4) axis-angle in the format (x, y, z, angle).
    """

    ax = torch.empty_like(qu)
    angle = 2 * torch.acos(torch.clamp(qu[..., 0], min=-1.0, max=1.0))

    s = torch.where(
        qu[..., 0:1] != 0,
        torch.sign(qu[..., 0:1]) / torch.norm(qu[..., 1:], dim=-1, keepdim=True),
        1.0,
    )

    ax[..., :3] = qu[..., 1:] * s
    ax[..., 3] = angle

    # fix identity quaternions to be about z axis
    mask_identity = angle == 0.0
    ax[mask_identity, 0] = 0.0
    ax[mask_identity, 1] = 0.0
    ax[mask_identity, 2] = 1.0
    ax[mask_identity, 3] = 0.0

    return ax


@torch.jit.script
def qu2ro(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to Rodrigues-Frank vector representation.

    Args:
        qu: shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        Tensor: shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2))
    """
    ro = torch.empty_like(qu)

    # Handle general case
    ro[..., :3] = qu[..., 1:] * torch.rsqrt(
        torch.sum(qu[..., 1:] ** 2, dim=-1, keepdim=True)
    )
    ro[..., 3] = torch.tan(torch.acos(torch.clamp(qu[..., 0], min=-1.0, max=1.0)))

    # w < 1e-8 for float32 / w < 1e-10 for float64 -> infinite tan
    eps = 1e-8 if qu.dtype == torch.float32 else 1e-10
    mask_zero = torch.abs(qu[..., 0]) < eps
    ro[mask_zero, 3] = float("inf")
    return ro


@torch.jit.script
def qu2bu(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to Bunge angles (ZXZ Euler angles).

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 3) Bunge angles in radians.
    """

    bu = torch.empty(qu.shape[:-1] + (3,), dtype=qu.dtype, device=qu.device)

    q03 = qu[..., 0] ** 2 + qu[..., 3] ** 2
    q12 = qu[..., 1] ** 2 + qu[..., 2] ** 2
    chi = torch.sqrt((q03 * q12))

    mask_chi_zero = chi == 0
    mA = (mask_chi_zero) & (q12 == 0)
    mB = (mask_chi_zero) & (q03 == 0)
    mC = ~mask_chi_zero

    bu[mA, 0] = torch.atan2(-2 * qu[mA, 0] * qu[mA, 3], qu[mA, 0] ** 2 - qu[mA, 3] ** 2)
    bu[mA, 1] = 0
    bu[mA, 2] = 0

    bu[mB, 0] = torch.atan2(2 * qu[mB, 1] * qu[mB, 2], qu[mB, 1] ** 2 - qu[mB, 2] ** 2)
    bu[mB, 1] = torch.pi
    bu[mB, 2] = 0

    bu[mC, 0] = torch.atan2(
        (qu[mC, 1] * qu[mC, 3] - qu[mC, 0] * qu[mC, 2]) / chi[mC],
        (-qu[mC, 0] * qu[mC, 1] - qu[mC, 2] * qu[mC, 3]) / chi[mC],
    )
    bu[mC, 1] = torch.atan2(2 * chi[mC], q03[mC] - q12[mC])
    bu[mC, 2] = torch.atan2(
        (qu[mC, 0] * qu[mC, 2] + qu[mC, 1] * qu[mC, 3]) / chi[mC],
        (qu[mC, 2] * qu[mC, 3] - qu[mC, 0] * qu[mC, 1]) / chi[mC],
    )

    # add 2pi to negative angles for first and last angles
    bu[..., 0] = torch.where(bu[..., 0] < 0, bu[..., 0] + 2 * torch.pi, bu[..., 0])
    bu[..., 2] = torch.where(bu[..., 2] < 0, bu[..., 2] + 2 * torch.pi, bu[..., 2])

    return bu


@torch.jit.script
def bu2qu(bu: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to quaternions.

    Args:
        bu (Tensor): shape (..., 3) Bunge angles in radians.

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty(bu.shape[:-1] + (4,), dtype=bu.dtype, device=bu.device)

    sigma = 0.5 * (bu[..., 0] + bu[..., 2])
    delta = 0.5 * (bu[..., 0] - bu[..., 2])

    c = torch.cos(0.5 * bu[..., 1])
    s = torch.sin(0.5 * bu[..., 1])

    qu[..., 0] = c * torch.cos(sigma)
    qu[..., 1] = -s * torch.cos(delta)
    qu[..., 2] = -s * torch.sin(delta)
    qu[..., 3] = -c * torch.sin(sigma)

    # correct for negative real part of quaternion
    return qu * torch.where(qu[..., 0] < 0, -1, 1).unsqueeze(-1)


@torch.jit.script
def qu2cl(qu: Tensor) -> Tensor:
    """
    Convert rotations given as unit quaternions to Clifford Torus coordinates.
    The coordinates are in the format (X, Z_y, Y, X_z, Z, Y_x)

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 6) Clifford Torus coordinates.

    """

    cl = torch.empty(qu.shape[:-1] + (6,), dtype=qu.dtype, device=qu.device)

    cl[..., 0] = torch.atan(qu[..., 1] / qu[..., 0])
    cl[..., 1] = torch.atan(qu[..., 3] / qu[..., 2])
    cl[..., 2] = torch.atan(qu[..., 2] / qu[..., 0])
    cl[..., 3] = torch.atan(qu[..., 1] / qu[..., 3])
    cl[..., 4] = torch.atan(qu[..., 3] / qu[..., 0])
    cl[..., 5] = torch.atan(qu[..., 2] / qu[..., 1])

    return cl


@torch.jit.script
def qu2om(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: Tensor of quaternions (real part first) of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    q_bar = qu[..., 0] ** 2 - torch.sum(qu[..., 1:] ** 2, dim=-1)
    matrix = torch.empty(qu.shape[:-1] + (3, 3), dtype=qu.dtype, device=qu.device)

    matrix[..., 0, 0] = q_bar + 2 * qu[..., 1] ** 2
    matrix[..., 0, 1] = 2 * (qu[..., 1] * qu[..., 2] - qu[..., 0] * qu[..., 3])
    matrix[..., 0, 2] = 2 * (qu[..., 1] * qu[..., 3] + qu[..., 0] * qu[..., 2])

    matrix[..., 1, 0] = 2 * (qu[..., 2] * qu[..., 1] + qu[..., 0] * qu[..., 3])
    matrix[..., 1, 1] = q_bar + 2 * qu[..., 2] ** 2
    matrix[..., 1, 2] = 2 * (qu[..., 2] * qu[..., 3] - qu[..., 0] * qu[..., 1])

    matrix[..., 2, 0] = 2 * (qu[..., 3] * qu[..., 1] - qu[..., 0] * qu[..., 2])
    matrix[..., 2, 1] = 2 * (qu[..., 3] * qu[..., 2] + qu[..., 0] * qu[..., 1])
    matrix[..., 2, 2] = q_bar + 2 * qu[..., 3] ** 2

    return matrix


@torch.jit.script
def zh2om(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = zh[..., :3], zh[..., 3:]
    b1 = a1 / torch.norm(a1, p=2, dim=-1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    b3 = b3 / torch.norm(b3, p=2, dim=-1, keepdim=True)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def om2zh(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@torch.jit.script
def zh2qu(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to quaternion
    representation.

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of quaternions of size (*, 4)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035

    """

    return om2qu(zh2om(zh))


@torch.jit.script
def qu2zh(quaternions: Tensor) -> Tensor:
    """
    Converts quaternion representation to 6D rotation representation by Zhou et al.

    Args:
        quaternions: batch of quaternions of size (*, 4)

    Returns:
        6D rotation representation, of size (*, 6)
    """

    return om2zh(qu2om(quaternions))


@torch.jit.script
def qu2cu(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to cubochoric vectors.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ho2cu(qu2ho(quaternions))


@torch.jit.script
def om2ho(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to homochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return ax2ho(om2ax(matrix))


@torch.jit.script
def ax2cu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to cubochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return qu2cu(ax2qu(ax))


@torch.jit.script
def ro2qu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to quaternions.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ro2ax(ro))


@torch.jit.script
def ro2om(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to rotation matrices.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ro2ax(ro))


@torch.jit.script
def ro2cu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to cubochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ax2cu(ro2ax(ro))


@torch.jit.script
def ro2ho(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to homochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return ax2ho(ro2ax(ro))


@torch.jit.script
def cu2ax(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to axis-angle representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return ho2ax(cu2ho(cubochoric_vectors))


@torch.jit.script
def cu2qu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to quaternions.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2om(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to rotation matrices.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2ro(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def ho2qu(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to quaternions.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2om(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to rotation matrices.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2ro(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(homochoric_vectors))


@torch.jit.script
def om2ro(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to Rodrigues-Frank vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return qu2ro(om2qu(matrix))


@torch.jit.script
def om2cu(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to cubochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Cubochoric vector representation as tensor of shape (..., 3).
    """
    return qu2cu(om2qu(matrix))


@torch.jit.script
def bu2om(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to rotation matrices.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return qu2om(bu2qu(bunge_angles))


@torch.jit.script
def bu2ax(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to axis-angle representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return qu2ax(bu2qu(bunge_angles))


@torch.jit.script
def bu2ro(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to Rodrigues-Frank vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return qu2ro(bu2qu(bunge_angles))


@torch.jit.script
def bu2cu(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to cubochoric vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Cubochoric vector representation as tensor of shape (..., 3).
    """
    return qu2cu(bu2qu(bunge_angles))


@torch.jit.script
def bu2ho(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to homochoric vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return qu2ho(bu2qu(bunge_angles))


@torch.jit.script
def om2bu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to Bunge angles (ZXZ Euler angles).

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(om2qu(matrix))


@torch.jit.script
def ax2bu(axis_angle: Tensor) -> Tensor:
    """
    Convert rotations given as axis-angle representation to Bunge angles (ZXZ Euler angles).

    Args:
        axis_angle: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ax2qu(axis_angle))


@torch.jit.script
def ro2bu(rodrigues_frank: Tensor) -> Tensor:
    """
    Convert rotations given as Rodrigues-Frank vector representation to Bunge angles (ZXZ Euler angles).

    Args:
        rodrigues_frank: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ro2qu(rodrigues_frank))


@torch.jit.script
def cu2bu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Convert rotations given as cubochoric vectors to Bunge angles (ZXZ Euler angles).

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(cu2qu(cubochoric_vectors))


@torch.jit.script
def ho2bu(homochoric_vectors: Tensor) -> Tensor:
    """
    Convert rotations given as homochoric vectors to Bunge angles (ZXZ Euler angles).

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ho2qu(homochoric_vectors))
