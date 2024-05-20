"""

Unit normal quaternions (points that sit on the surface of the 3-sphere with
unit radius in 4D Euclidean space) are used to represent 3D rotations. This
module provides a set of operations for working with quaternions in general.
Often times only the angle of the rotation is needed for comparison amongst
quaternions, so separate functions are provided for accelerating this common
operation. The quaternion (w, x, y, z) is used to represent a rotation that is
indistinguishable from the quaternion (-w, x, y, z), so the standardization
function is provided to make the real part non-negative by conjugation, limiting
the hypervolume we work with to the positive w hemisphere of the 3-sphere.

For more information on quaternions, see:

https://en.wikipedia.org/wiki/Quaternion

Adopted from PyTorch3D

https://github.com/facebookresearch/pytorch3d

"""

import torch
from torch import Tensor


@torch.jit.script
def qu_std(qu: Tensor) -> Tensor:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(qu[..., 0:1] >= 0, qu, -qu)


@torch.jit.script
def qu_norm(qu: Tensor) -> Tensor:
    """
    Normalize quaternions to unit norm.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized quaternions.
    """
    return qu / torch.norm(qu, dim=-1, keepdim=True)


@torch.jit.script
def qu_prod_raw(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def qu_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)


@torch.jit.script
def qu_slerp(a: Tensor, b: Tensor, t: float) -> Tensor:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        t: interpolation parameter between 0 and 1

    Returns:
        The interpolated quaternions, a tensor of shape (..., 4).
    """
    a = qu_norm(a)
    b = qu_norm(b)
    cos_theta = torch.sum(a * b, dim=-1)
    angle = torch.acos(cos_theta)
    sin_theta = torch.sin(angle)
    w1 = torch.sin((1 - t) * angle) / sin_theta
    w2 = torch.sin(t * angle) / sin_theta
    return (a.unsqueeze(-1) * w1 + b.unsqueeze(-1) * w2).squeeze(-1)


@torch.jit.script
def qu_prod_pos_real(a: Tensor, b: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def qu_triple_prod_pos_real(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion triple product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        c: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b*c Tensor shape (..., ) of quaternion triple product real part magnitudes.
    """
    return qu_prod_pos_real(a, qu_prod(b, c))


@torch.jit.script
def qu_prod_axis(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the axis of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 3) of quaternion product axes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ox, oy, oz), -1)


@torch.jit.script
def qu_conj(qu: Tensor) -> Tensor:
    """
    Get the unit quaternions for the inverse action.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=qu.device, dtype=qu.dtype)
    return qu * scaling


@torch.jit.script
def qu_apply(qu: Tensor, point: Tensor) -> Tensor:
    """
    Rotate 3D points by unit quaternions.

    Args:
        qu: shape (..., 4) of quaternions in the form (w, x, y, z)
        point: shape (..., 3) of 3D points.

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    aw, ax, ay, az = qu[..., 0], qu[..., 1], qu[..., 2], qu[..., 3]
    bx, by, bz = point[..., 0], point[..., 1], point[..., 2]

    # need qu_prod_axis(qu_prod_raw(qu, point_as_quaternion), qu_conj(qu))
    # do qu_prod_raw(qu, point_as_quaternion) first to get intermediate values
    iw = aw - ax * bx - ay * by - az * bz
    ix = aw * bx + ax + ay * bz - az * by
    iy = aw * by - ax * bz + ay + az * bx
    iz = aw * bz + ax * by - ay * bx + az

    # next qu_prod_axis(qu_prod_raw(qu, point_as_quaternion), qu_conj(qu))
    ox = -iw * ax + ix * aw - iy * az + iz * ay
    oy = -iw * ay + ix * az + iy * aw - iz * ax
    oz = -iw * az - ix * ay + iy * ax + iz * aw

    return torch.stack((ox, oy, oz), -1)


@torch.jit.script
def qu_norm_std(qu: Tensor) -> Tensor:
    """
    Normalize a quaternion to unit norm and make real part non-negative.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized and standardized quaternions.
    """
    return qu_std(qu_norm(qu))


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
def qu_angle(qu: Tensor) -> Tensor:
    """
    Compute angles of rotation for quaternions.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        tensor of shape (..., ) of rotation angles.
    """
    return 2 * torch.acos(qu[..., 0])
