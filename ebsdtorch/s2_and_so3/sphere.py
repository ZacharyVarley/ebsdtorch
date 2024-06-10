"""

The Rosca-Lambert projection is a bijection between the unit sphere and the
square, scaled here to be between (-1, 1) X (-1, 1) for compatibility with the
PyTorch function "grid_sample".

For more information, see: "Roşca, D., 2010. New uniform grids on the sphere.
Astronomy & Astrophysics, 520, p.A63."

"""

import torch
from torch import Tensor


@torch.jit.script
def theta_phi_to_xyz(theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates to cartesian coordinates.

    Args:
        theta (Tensor): shape (..., ) of polar declination angles
        phi (Tensor): shape (..., ) of azimuthal angles

    Returns:
        Tensor: torch tensor of shape (..., 3) containing the cartesian
        coordinates
    """
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=1,
    )


@torch.jit.script
def xyz_to_theta_phi(xyz: Tensor) -> Tensor:
    """
    Convert cartesian coordinates to latitude and longitude.

    Args:
        xyz (Tensor): torch tensor of shape (..., 3) of cartesian coordinates

    Returns:
        Tensor: torch tensor of shape (..., 2) of declination from z-axis and
        azimuthal angle

    """
    return torch.stack(
        (
            torch.atan2(torch.norm(xyz[:, :2], dim=1), xyz[:, 2]),
            torch.atan2(xyz[:, 1], xyz[:, 0]),
        ),
        dim=1,
    )


@torch.jit.script
def rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert projection.

    Args:
        pts: torch tensor of shape (..., 3) containing the points
    Returns:
        torch tensor of shape (..., 2) containing the projected points
    """
    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # floating point error can yield z above 1.0 example for float32 is:
    # xyz = [1.7817265e-04, 2.8403841e-05, 1.0000001e0]
    # so we have to clamp to avoid sqrt of negative number
    factor = torch.sqrt(torch.clamp(2.0 * (1.0 - torch.abs(z)), min=0.0))

    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor * (2.0 / (8.0**0.5))
    arctan_term = (
        torch.where(big < 0, -1, 1)
        * factor
        * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big))
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    # stack them together but flip the order if the condition is false
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    return out


@torch.jit.script
def rosca_lambert_side_by_side(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert
    projection. Points with a positive z-coordinate are projected to the left
    side of the square, while points with a negative z-coordinate are projected
    to the right side of the square.

    Args:
        pts: torch tensor of shape (..., 3) containing the points

    Returns:
        torch tensor of shape (..., 2) containing the projected points

    """
    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # floating point error can yield z above 1.0 example for float32 is:
    # xyz = [1.7817265e-04, 2.8403841e-05, 1.0000001e0]
    # so we have to clamp to avoid sqrt of negative number
    factor = torch.sqrt(torch.clamp(2.0 * (1.0 - torch.abs(z)), min=0.0))

    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor * (2.0 / (8.0**0.5))
    arctan_term = (
        torch.where(big < 0, -1, 1)
        * factor
        * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big))
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    # stack them together but flip the order if the condition is false
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    # halve the x index for all points then subtract 0.5 to move to [-1, 0]
    # then add 1 to the j coordinate if z is negative to move to [0, 1]
    # note that torch's grid_sample has flipped coordinates from ij indexing
    out[..., 0] = (out[..., 0] / 2.0) - 0.5 + torch.where(z < 0, 1.0, 0)
    return out


@torch.jit.script
def inv_rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square lambert projection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points
    Returns:
        torch tensor of shape (..., 3) containing the projected points

    """

    # get shape of input
    shape_in = pts.shape[:-1]
    n_pts = int(torch.prod(torch.tensor(shape_in)))

    pts = pts.view(-1, 2)

    pi = torch.pi

    a = pts[:, 0] * (pi / 2) ** 0.5
    b = pts[:, 1] * (pi / 2) ** 0.5

    # mask for branch
    cond = torch.abs(b) <= torch.abs(a)

    output = torch.empty((n_pts, 3), dtype=pts.dtype, device=pts.device)

    output[cond, 0] = (
        (2 * a[cond] / pi)
        * torch.sqrt(pi - a[cond] ** 2)
        * torch.cos((pi * b[cond]) / (4 * a[cond]))
    )
    output[cond, 1] = (
        (2 * a[cond] / pi)
        * torch.sqrt(pi - a[cond] ** 2)
        * torch.sin((pi * b[cond]) / (4 * a[cond]))
    )
    output[cond, 2] = 1 - (2 * a[cond] ** 2 / pi)

    output[~cond, 0] = (
        (2 * b[~cond] / pi)
        * torch.sqrt(pi - b[~cond] ** 2)
        * torch.sin((pi * a[~cond]) / (4 * b[~cond]))
    )
    output[~cond, 1] = (
        (2 * b[~cond] / pi)
        * torch.sqrt(pi - b[~cond] ** 2)
        * torch.cos((pi * a[~cond]) / (4 * b[~cond]))
    )
    output[~cond, 2] = 1 - (2 * b[~cond] ** 2 / pi)

    return output.reshape(shape_in + (3,))
