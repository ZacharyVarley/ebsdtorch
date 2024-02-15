"""

Adopted from PyTorch3D (project from Meta Research - formerly Facebook Research) and from EMsoft

https://github.com/facebookresearch/pytorch3d

https://github.com/marcdegraef/3Drotations

List of acronyms used in the code:

cu: cubochoric
ho: homochoric
ax: axis-angle
qu: quaternion
om: orientation matrix
eu: Euler angles
ro: Rodrigues-Frank vector
zh: 6D continuous representation or orientation

"""

import torch
from torch import Tensor


@torch.jit.script
def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non-negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@torch.jit.script
def quaternion_raw_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


@torch.jit.script
def quaternion_real_of_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions and return the positive real part of the product.


    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The positive real part of the product of a and b, a tensor of shape (..., 1).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def quaternion_invert(quaternion: Tensor) -> Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


@torch.jit.script
def quaternion_apply(quaternion: Tensor, point: Tensor) -> Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point. Usual torch rules
    for broadcasting apply.

    Args:
        quaternion: shape (..., 4) of quaternions in the form (w, x, y, z)
        point: shape (..., 3) of 3D points.

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


@torch.jit.script
def normalize_quaternion(quaternion: Tensor) -> Tensor:
    """
    Normalize a quaternion to a unit quaternion.

    Args:
        quaternion: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized quaternions.
    """
    return quaternion / torch.norm(quaternion, dim=-1, keepdim=True)


@torch.jit.script
def norm_standard_quaternion(quaternion: Tensor) -> Tensor:
    """
    Normalize a quaternion to a unit quaternion and standardize it.

    Args:
        quaternion: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized and standardized quaternions.
    """
    return standardize_quaternion(normalize_quaternion(quaternion))


@torch.jit.script
def quaternion_rotate_sets_sphere(points_start: Tensor, points_finish) -> Tensor:
    """
    Determine the quaternions that rotate the points_start to the points_finish.
    All points are assumed to be on the unit sphere. The cross product is used
    as the axis of rotation, but there are an infinite number of quaternions that
    fulfill the requirement as the points can be rotated around their axis by
    an arbitrary angle, and they will still have the same latitude and longitude.

    Args:
        points_start: Starting points as tensor of shape (..., 3).
        points_finish: Ending points as tensor of shape (..., 3).

    Returns:
        The quaternions, as tensor of shape (..., 4).

    """
    # determine mask for numerical stability
    valid = torch.abs(torch.sum(points_start * points_finish, dim=-1)) < 0.999999
    # get the cross product of the two sets of points
    cross = torch.cross(points_start[valid], points_finish[valid], dim=-1)
    # get the dot product of the two sets of points
    dot = torch.sum(points_start[valid] * points_finish[valid], dim=-1)
    # get the angle
    angle = torch.atan2(torch.norm(cross, dim=-1), dot)
    # add tau to the angle if the cross product is negative
    angle[angle < 0] += 2 * torch.pi
    # set the output
    out = torch.empty(
        (points_start.shape[0], 4), dtype=points_start.dtype, device=points_start.device
    )
    out[valid, 0] = torch.cos(angle / 2)
    out[valid, 1:] = torch.sin(angle / 2).unsqueeze(-1) * (
        cross / torch.norm(cross, dim=-1, keepdim=True)
    )
    out[~valid, 0] = 1
    out[~valid, 1:] = 0
    return out


@torch.jit.script
def misorientation_angle(quaternion: Tensor) -> Tensor:
    """
    Compute the misorientation angle for a quaternion.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The misorientation angle, a tensor of shape (...).
    """
    return 2 * torch.acos(quaternion[..., 0])


# -------------------------------------------------------------------
# ------------------------ conversion functions ---------------------
# -------------------------------------------------------------------


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

    batch_dim = qu.shape[:-1]

    ho = torch.empty(batch_dim + (3,), dtype=qu.dtype, device=qu.device)

    # get the angle
    angle = 2 * torch.acos(qu[..., 0])

    # get mask of zero angles
    nonzero_mask = angle != 0

    # get the non-zero angles
    nonzero_angles = angle[nonzero_mask][:, None]

    ho[~nonzero_mask] = 0
    ho[nonzero_mask] = (
        qu[nonzero_mask, 1:] / torch.norm(qu[nonzero_mask, 1:], dim=-1, keepdim=True)
    ) * (3.0 * (nonzero_angles - torch.sin(nonzero_angles)) / 4.0) ** (1 / 3)

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


@torch.jit.script
def cu2ho(cu: Tensor) -> Tensor:
    """
    Convert cubochoric coordinates to homochoric coordinates.

    Args:
        cu: Cubochoric coordinates as tensor of shape (B, 3)

    Returns:
        Homochoric coordinates as tensor of shape (B, 3)
    """
    ho = cu.clone()
    cu_abs = torch.abs(ho)
    x_abs, y_abs, z_abs = torch.unbind(cu_abs, dim=-1)

    # Determine pyramid
    pyramid = torch.zeros(ho.shape[0], dtype=torch.uint8)
    pyramid[(x_abs <= z_abs) & (y_abs <= z_abs)] = 1
    pyramid[(x_abs <= -z_abs) & (y_abs <= -z_abs)] = 2
    pyramid[(z_abs <= x_abs) & (y_abs <= x_abs)] = 3
    pyramid[(z_abs <= -x_abs) & (y_abs <= -x_abs)] = 4
    pyramid[(x_abs <= y_abs) & (z_abs <= y_abs)] = 5
    pyramid[(x_abs <= -y_abs) & (z_abs <= -y_abs)] = 6

    # move everything to correct pyramid
    mask_34 = (pyramid == 3) | (pyramid == 4)
    mask_56 = (pyramid == 5) | (pyramid == 6)
    ho[mask_34] = torch.roll(ho[mask_34], shifts=-1, dims=1)
    ho[mask_56] = torch.roll(ho[mask_56], shifts=1, dims=1)

    # Scale
    ho = ho * torch.pi ** (1.0 / 6.0) / 6.0 ** (1.0 / 6.0)

    # Process based on conditions
    x, y, z = torch.unbind(ho, dim=-1)
    prefactor = (
        (3 * torch.pi / 4) ** (1 / 3)
        * 2 ** (1 / 4)
        / (torch.pi ** (5 / 6) / 6 ** (1 / 6) / 2)
    )
    sqrt2 = 2**0.5

    # abs(y) <= abs(x) condition
    mask_y_leq_x = torch.abs(y) <= torch.abs(x)
    q_y_leq_x = (torch.pi / 12.0) * y[mask_y_leq_x] / x[mask_y_leq_x]
    cosq_y_leq_x = torch.cos(q_y_leq_x)
    sinq_y_leq_x = torch.sin(q_y_leq_x)
    q_val_y_leq_x = prefactor * x[mask_y_leq_x] / torch.sqrt(sqrt2 - cosq_y_leq_x)
    t1_y_leq_x = (sqrt2 * cosq_y_leq_x - 1) * q_val_y_leq_x
    t2_y_leq_x = sqrt2 * sinq_y_leq_x * q_val_y_leq_x
    c_y_leq_x = t1_y_leq_x**2 + t2_y_leq_x**2
    s_y_leq_x = torch.pi * c_y_leq_x / (24 * z[mask_y_leq_x] ** 2)
    c_y_leq_x = torch.pi**0.5 * c_y_leq_x / 24**0.5 / z[mask_y_leq_x]
    q_y_leq_x = torch.sqrt(1 - s_y_leq_x)

    ho[mask_y_leq_x, 0] = t1_y_leq_x * q_y_leq_x
    ho[mask_y_leq_x, 1] = t2_y_leq_x * q_y_leq_x
    ho[mask_y_leq_x, 2] = (6 / torch.pi) ** 0.5 * z[mask_y_leq_x] - c_y_leq_x

    # abs(y) > abs(x) condition
    mask_y_gt_x = ~mask_y_leq_x
    q_y_gt_x = (torch.pi / 12.0) * x[mask_y_gt_x] / y[mask_y_gt_x]
    cosq_y_gt_x = torch.cos(q_y_gt_x)
    sinq_y_gt_x = torch.sin(q_y_gt_x)
    q_val_y_gt_x = prefactor * y[mask_y_gt_x] / torch.sqrt(sqrt2 - cosq_y_gt_x)
    t1_y_gt_x = sqrt2 * sinq_y_gt_x * q_val_y_gt_x
    t2_y_gt_x = (sqrt2 * cosq_y_gt_x - 1) * q_val_y_gt_x
    c_y_gt_x = t1_y_gt_x**2 + t2_y_gt_x**2
    s_y_gt_x = torch.pi * c_y_gt_x / (24 * z[mask_y_gt_x] ** 2)
    c_y_gt_x = torch.pi**0.5 * c_y_gt_x / 24**0.5 / z[mask_y_gt_x]
    q_y_gt_x = torch.sqrt(1 - s_y_gt_x)

    ho[mask_y_gt_x, 0] = t1_y_gt_x * q_y_gt_x
    ho[mask_y_gt_x, 1] = t2_y_gt_x * q_y_gt_x
    ho[mask_y_gt_x, 2] = (6 / torch.pi) ** 0.5 * z[mask_y_gt_x] - c_y_gt_x

    # Roll the array based on the pyramid values
    ho[mask_34] = torch.roll(ho[mask_34], shifts=1, dims=1)
    ho[mask_56] = torch.roll(ho[mask_56], shifts=-1, dims=1)

    # wherever cu had all zeros, ho should be set to be (0, 0, 0)
    mask_zero = torch.abs(cu).sum(dim=1) == 0
    ho[mask_zero] = 0

    # wherever cu had (0, 0, z) ho should be set to be (0, 0, np.sqrt(6 / np.pi) * z)
    mask_z = torch.abs(cu[:, :2]).sum(dim=1) == 0
    ho[mask_z, :2] = 0
    ho[mask_z, 2] = (6.0 / torch.pi) ** 0.5 * cu[mask_z, 2]

    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """
    Homochoric vector to cubochoric vector.

    Args:
        ho: Homochoric coordinates as tensor of shape (..., 3)

    Returns:
        Cubochoric coordinates as tensor of shape (..., 3)

    """

    cu = torch.empty_like(ho)

    # get the magnitude of the homochoric vector
    ho_norm = torch.norm(ho, dim=-1)

    mask_pyramids_34 = (
        (torch.abs(ho[..., 2]) <= torch.abs(ho[..., 0]))
        & (torch.abs(ho[..., 1]) <= torch.abs(ho[..., 0]))
    ) | (
        (torch.abs(ho[..., 2]) <= -torch.abs(ho[..., 0]))
        & (torch.abs(ho[..., 1]) <= -torch.abs(ho[..., 0]))
    )
    mask_pyramids_56 = (
        (torch.abs(ho[..., 0]) <= torch.abs(ho[..., 1]))
        & (torch.abs(ho[..., 2]) <= torch.abs(ho[..., 1]))
    ) | (
        (torch.abs(ho[..., 0]) <= -torch.abs(ho[..., 1]))
        & (torch.abs(ho[..., 2]) <= -torch.abs(ho[..., 1]))
    )
    mask_pyramids_12 = ~mask_pyramids_34 & ~mask_pyramids_56

    # move everything to correct pyramid
    cu[mask_pyramids_12] = ho[mask_pyramids_12]
    cu[mask_pyramids_34] = torch.roll(ho[mask_pyramids_34], -1, dims=-1)
    cu[mask_pyramids_56] = torch.roll(ho[mask_pyramids_56], 1, dims=-1)

    cu[..., 0] *= (2 * ho_norm / (ho_norm + torch.abs(cu[..., 2]))) ** 0.5
    cu[..., 1] *= (2 * ho_norm / (ho_norm + torch.abs(cu[..., 2]))) ** 0.5
    cu[..., 2] = (torch.sign(cu[..., 2]) * ho_norm) / ((6 / torch.pi) ** 0.5)

    qxy = cu[..., 0] ** 2 + cu[..., 1] ** 2

    sx = torch.where(cu[..., 0] != 0, torch.sign(cu[..., 0]), 1)
    sy = torch.where(cu[..., 1] != 0, torch.sign(cu[..., 1]), 1)

    mask_h2_leq_h1 = torch.abs(cu[..., 1]) <= torch.abs(cu[..., 0])

    h1_top_h2_bot = torch.where(
        mask_h2_leq_h1,
        cu[..., 0],
        cu[..., 1],
    )
    h2_top_h1_bot = torch.where(
        mask_h2_leq_h1,
        cu[..., 1],
        cu[..., 0],
    )

    q2xy = h1_top_h2_bot**2 + qxy
    sq2xy = torch.sqrt(q2xy)
    q_new = (
        torch.pi ** (5 / 6)
        * 4 ** (1 / 3)
        / (2 * (2 ** (1 / 2)) * (3 * torch.pi) ** (1 / 3) * 6 ** (1 / 6))
    ) * torch.sqrt(q2xy * qxy / (q2xy - torch.abs(h1_top_h2_bot) * sq2xy))
    arccos = torch.acos(
        (h2_top_h1_bot**2 + torch.abs(h1_top_h2_bot) * sq2xy) / (2**0.5) / qxy
    )

    t1_inv = torch.where(
        mask_h2_leq_h1,
        q_new * sx,
        12 * q_new * sx * arccos / torch.pi,
    )
    t2_inv = torch.where(
        mask_h2_leq_h1,
        12 * q_new * sy * arccos / torch.pi,
        q_new * sy,
    )

    cu[..., 0] = t1_inv
    cu[..., 1] = t2_inv
    cu *= (6 / torch.pi) ** (1 / 6)

    # roll back to the original pyramid order
    cu[mask_pyramids_34] = torch.roll(cu[mask_pyramids_34], 1, dims=-1)
    cu[mask_pyramids_56] = torch.roll(cu[mask_pyramids_56], -1, dims=-1)

    # where the magnitude exceeds the homochoric ball, fill nan
    error_mask = ho_norm > (3 * torch.pi / 4) ** (1 / 3)
    cu[error_mask] = torch.nan

    # mask off where the magnitude is zero
    mask_zero = ho_norm == 0
    cu[mask_zero] = 0

    return cu


# @torch.jit.script
# def ho2ax(ho: Tensor) -> Tensor:
#     """
#     Converts a set of homochoric vectors to axis-angle representation.

#     Args:
#         ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)

#     Returns:
#         torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)

#     """
#     fit_parameters = torch.tensor(
#         [
# 0.9999999999999968,
# -0.49999999999986866,
# -0.025000000000632055,
# -0.003928571496460683,
# -0.0008164666077062752,
# -0.00019411896443261646,
# -0.00004985822229871769,
# -0.000014164962366386031,
# -1.9000248160936107e-6,
# -5.72184549898506e-6,
# 7.772149920658778e-6,
# -0.00001053483452909705,
# 9.528014229335313e-6,
# -5.660288876265125e-6,
# 1.2844901692764126e-6,
# 1.1255185726258763e-6,
# -1.3834391419956455e-6,
# 7.513691751164847e-7,
# -2.401996891720091e-7,
# 4.386887017466388e-8,
# -3.5917775353564864e-9,
#         ],
#         dtype=torch.float64,
#         device=ho.device,
#     ).to(ho.dtype)

#     ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)

#     s = torch.sum(
#         fit_parameters
#         * ho_norm_sq ** torch.arange(len(fit_parameters), dtype=ho.dtype),
#         dim=-1,
#     )

#     ax = torch.empty(ho.shape[:-1] + (4,), dtype=ho.dtype, device=ho.device)

#     mask_identity = torch.abs(ho_norm_sq.squeeze(-1)) < 1e-8
#     ax[mask_identity, 0:1] = 0.0
#     ax[mask_identity, 1:2] = 0.0
#     ax[mask_identity, 2:3] = 1.0

#     mask_large = ~mask_identity
#     ax[mask_large, :3] = ho[mask_large, :] * torch.rsqrt(ho_norm_sq[mask_large])

#     ax[..., 3] = torch.where(
#         mask_large,
#         2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0)),
#         0,
#     )

#     return ax


@torch.jit.script
def ho2ax(ho: Tensor) -> Tensor:
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
            # # EMsoft polyfit coeffs
            # 0.9999999999999968,
            # -0.49999999999986866,
            # -0.025000000000632055,
            # -0.003928571496460683,
            # -0.0008164666077062752,
            # -0.00019411896443261646,
            # -0.00004985822229871769,
            # -0.000014164962366386031,
            # -1.9000248160936107e-6,
            # -5.72184549898506e-6,
            # 7.772149920658778e-6,
            # -0.00001053483452909705,
            # 9.528014229335313e-6,
            # -5.660288876265125e-6,
            # 1.2844901692764126e-6,
            # 1.1255185726258763e-6,
            # -1.3834391419956455e-6,
            # 7.513691751164847e-7,
            # -2.401996891720091e-7,
            # 4.386887017466388e-8,
            # -3.5917775353564864e-9,
        ],
        dtype=torch.float64,
        device=ho.device,
    ).to(ho.dtype)

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)

    s = torch.sum(
        fit_parameters
        * ho_norm_sq ** torch.arange(len(fit_parameters), dtype=ho.dtype),
        dim=-1,
    )

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
def ax2ho(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to homochoric representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle (x, y, z, angle)

    Returns:
        torch.Tensor: shape (..., 3) homochoric coordinates (x, y, z)
    """
    f = (0.75 * (ax[..., 3:4] - torch.sin(ax[..., 3:4]))) ** (1.0 / 3.0)
    ho = ax[..., :3] * f
    return ho


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
        qu (Tensor): Tensor of shape (..., 4) where N is the number of quaternions.
            Each row represents a quaternion in the format (w, x, y, z).

    Returns:
        torch.Tensor: Tensor of shape (..., 4) where N is the number of Rodrigues-Frank vectors.
            Each row represents a Rodrigues-Frank vector representation.
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
    return standardize_quaternion(qu)


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
def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


@torch.jit.script
def eu2om(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.jit.script
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


@torch.jit.script
def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


@torch.jit.script
def om2eu(matrix: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).

    Raises:
        ValueError: If the convention is invalid or the matrix is not a
            rotation matrix shape (..., 3, 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


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
    representation using Gram--Schmidt orthogonalization per Section B of [1].
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
    Converts quaternion representation to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
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
def ax2ho(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to homochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return qu2ho(ax2qu(ax))


@torch.jit.script
def ax2eu(ax: Tensor, convention: str) -> Tensor:
    """
    Converts axis-angle representation to Euler angles in radians.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ax2om(ax), convention)


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
def ro2eu(ro: Tensor, convention: str) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to Euler angles in radians.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ro2om(ro), convention)


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
def cu2eu(cubochoric_vectors: Tensor, convention: str) -> Tensor:
    """
    Converts cubochoric vector representation to Euler angles in radians.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(cu2om(cubochoric_vectors), convention)


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
def ho2eu(homochoric_vectors: Tensor, convention: str) -> Tensor:
    """
    Converts homochoric vector representation to Euler angles in radians.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(ho2om(homochoric_vectors), convention)


@torch.jit.script
def eu2ax(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to axis-angle representation.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return om2ax(eu2om(euler_angles, convention))


@torch.jit.script
def eu2qu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to quaternions.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return om2qu(eu2om(euler_angles, convention))


@torch.jit.script
def eu2ro(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to Rodrigues-Frank vector representation.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(eu2ax(euler_angles, convention))


@torch.jit.script
def eu2cu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to cubochoric vectors.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ax2cu(eu2ax(euler_angles, convention))


@torch.jit.script
def qu2eu(quaternions: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as quaternions to Euler angles in radians.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    return om2eu(qu2om(quaternions), convention)


@torch.jit.script
def eu2qu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to quaternions.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return om2qu(eu2om(euler_angles, convention))


@torch.jit.script
def eu2cu(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to cubochoric vectors.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return qu2cu(eu2qu(euler_angles, convention))


@torch.jit.script
def eu2ho(euler_angles: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as Euler angles in radians to homochoric vectors.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return qu2ho(eu2qu(euler_angles, convention))


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
