"""
The Rosca-Lambert projection is a bijection between the unit sphere and the
square (-1, 1) X (-1, 1). 

For more information, see: "Roşca, D., 2010. New uniform grids on the sphere.
Astronomy & Astrophysics, 520, p.A63."

"""

import torch
from torch import Tensor


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
    go = torch.abs(b) <= torch.abs(a)

    output = torch.empty((n_pts, 3), dtype=pts.dtype, device=pts.device)

    output[go, 0] = (
        (2 * a[go] / pi)
        * torch.sqrt(pi - a[go] ** 2)
        * torch.cos((pi * b[go]) / (4 * a[go]))
    )
    output[go, 1] = (
        (2 * a[go] / pi)
        * torch.sqrt(pi - a[go] ** 2)
        * torch.sin((pi * b[go]) / (4 * a[go]))
    )
    output[go, 2] = 1 - (2 * a[go] ** 2 / pi)

    output[~go, 0] = (
        (2 * b[~go] / pi)
        * torch.sqrt(pi - b[~go] ** 2)
        * torch.sin((pi * a[~go]) / (4 * b[~go]))
    )
    output[~go, 1] = (
        (2 * b[~go] / pi)
        * torch.sqrt(pi - b[~go] ** 2)
        * torch.cos((pi * a[~go]) / (4 * b[~go]))
    )
    output[~go, 2] = 1 - (2 * b[~go] ** 2 / pi)

    return output.reshape(shape_in + (3,))
