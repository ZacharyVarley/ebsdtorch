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
    # get shape of input
    shape_in = pts.shape[:-1]
    n_pts = int(torch.prod(torch.tensor(shape_in)))

    # symbolically reshape pts
    pts = pts.view(-1, 3)

    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # Define output tensor
    out = torch.empty((n_pts, 2), dtype=pts.dtype, device=pts.device)

    # Define conditions and calculations
    cond = torch.abs(y) <= torch.abs(x)
    factor = torch.sqrt(2.0 * (1.0 - torch.abs(z)))

    # instead of precalcuating each branch, just use the condition to select the correct branch
    out[cond, 0] = torch.sign(x[cond]) * factor[cond] * (2.0 / (8.0**0.5))
    out[cond, 1] = (
        torch.sign(x[cond])
        * factor[cond]
        * torch.atan2(
            y[cond] * torch.sign(x[cond]),
            x[cond] * torch.sign(x[cond]),
        )
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    out[~cond, 0] = (
        torch.sign(y[~cond])
        * factor[~cond]
        * torch.atan2(
            x[~cond] * torch.sign(y[~cond]),
            y[~cond] * torch.sign(y[~cond]),
        )
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    out[~cond, 1] = torch.sign(y[~cond]) * factor[~cond] * (2.0 / (8.0**0.5))

    return out.reshape(shape_in + (2,))


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
