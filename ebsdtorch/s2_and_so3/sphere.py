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
def inv_rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square
    lambert projection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    This version is more efficient than the previous one, as it just plops
    everything into the first quadrant, then swaps the x and y coordinates
    if needed so that we always have x >= y. Then swaps back at the end and
    copy the sign of the original x and y coordinates.

    """
    pi = torch.pi
    # map to first quadrant and swap x and y if needed
    x_abs, y_abs = (
        torch.abs(pts[..., 0]) * (pi / 2) ** 0.5,
        torch.abs(pts[..., 1]) * (pi / 2) ** 0.5,
    )
    cond = x_abs >= y_abs
    x_new = torch.where(cond, x_abs, y_abs)
    y_new = torch.where(cond, y_abs, x_abs)

    # only one case now
    x_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.cos(pi * y_new / (4 * x_new))
    )
    y_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.sin(pi * y_new / (4 * x_new))
    )
    z_out = 1 - (2 * x_new**2 / pi)

    # swap back and copy sign
    x_out = torch.where(cond, x_hs, y_hs)
    y_out = torch.where(cond, y_hs, x_hs)
    x_out = x_out.copysign_(pts[..., 0])
    y_out = y_out.copysign_(pts[..., 1])

    return torch.stack((x_out, y_out, z_out), dim=-1)


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
