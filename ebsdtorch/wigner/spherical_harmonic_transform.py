import torch
from typing import Tuple


@torch.jit.script
def sphere_to_square(coords_3d: torch.Tensor) -> torch.Tensor:
    """
    Square Lambert projection from unit hemisphere to unit square.

    Args:
    coords_3d: Tensor of shape (..., 3) representing [x, y, z] coordinates on the unit sphere.

    Returns:
    coords_2d: Tensor of shape (..., 2) representing [X, Y] coordinates in the unit square (0,1).
    """
    x, y, z = coords_3d.unbind(-1)
    kPi_4 = torch.tensor(0.7853981633974483096156608458199)
    fZ = torch.abs(z)

    # Initialize output tensors
    X = torch.zeros_like(x)
    Y = torch.zeros_like(y)

    # Handle the case where fZ == 1.0
    pole_mask = fZ == 1.0
    X[pole_mask] = 0.5
    Y[pole_mask] = 0.5

    # Compute X and Y for other cases
    non_pole_mask = ~pole_mask
    abs_x_greater = torch.abs(x) >= torch.abs(y)

    # First mask where abs(x) >= abs(y)
    mask1 = non_pole_mask & abs_x_greater
    X[mask1] = torch.copysign(torch.sqrt(1 - fZ[mask1]), x[mask1]) * 0.5 + 0.5
    Y[mask1] = X[mask1] * torch.atan(y[mask1] / x[mask1]) / kPi_4 + 0.5

    # Second mask where abs(y) > abs(x)
    mask2 = non_pole_mask & ~abs_x_greater
    Y[mask2] = torch.copysign(torch.sqrt(1 - fZ[mask2]), y[mask2]) * 0.5 + 0.5
    X[mask2] = Y[mask2] * torch.atan(x[mask2] / y[mask2]) / kPi_4 + 0.5

    return torch.stack([X, Y], dim=-1)


@torch.jit.script
def square_to_sphere(coords_2d: torch.Tensor) -> torch.Tensor:
    """
    Square Lambert projection from unit square to unit hemisphere.

    Args:
    coords_2d: Tensor of shape (..., 2) representing [X, Y] coordinates in unit square (0,1).

    Returns:
    coords_3d: Tensor of shape (..., 3) representing [x, y, z] coordinates on unit sphere.
    """
    X, Y = coords_2d.unbind(-1)
    kPi_4 = torch.tensor(0.7853981633974483096156608458199)
    sX = 2 * X - 1
    sY = 2 * Y - 1

    # Initialize output tensors
    x = torch.zeros_like(X)
    y = torch.zeros_like(Y)
    z = torch.zeros_like(X)

    # Compute x, y, z
    aX = torch.abs(sX)
    aY = torch.abs(sY)
    vMax = torch.max(aX, aY)

    center_mask = vMax <= 1e-10
    z[center_mask] = 1.0

    # Masks for different conditions
    valid_mask = ~center_mask
    aX_le_aY = aX <= aY

    # First mask where aX <= aY
    mask1 = valid_mask & aX_le_aY
    q1 = sY[mask1] * torch.sqrt(2 - sY[mask1] ** 2)
    qq1 = kPi_4 * sX[mask1] / sY[mask1]
    x[mask1] = q1 * torch.sin(qq1)
    y[mask1] = q1 * torch.cos(qq1)

    # Second mask where aY < aX
    mask2 = valid_mask & ~aX_le_aY
    q2 = sX[mask2] * torch.sqrt(2 - sX[mask2] ** 2)
    qq2 = kPi_4 * sY[mask2] / sX[mask2]
    x[mask2] = q2 * torch.cos(qq2)
    y[mask2] = q2 * torch.sin(qq2)

    z[valid_mask] = 1 - vMax[valid_mask] ** 2
    mag = torch.sqrt(x**2 + y**2 + z**2)
    x /= mag
    y /= mag
    z /= mag

    return torch.stack([x, y, z], dim=-1)


@torch.jit.script
def cos_lats_optimized(dim: int) -> torch.Tensor:
    count = (dim + 1) // 2
    even = dim % 2 == 0
    denom = (dim - 1) ** 2
    numer_start = denom - (1 if even else 0)
    delta_start = 8 if even else 4
    deltas = torch.arange(delta_start, delta_start + 8 * count, 8, dtype=torch.float64)
    deltas = torch.cat([torch.tensor([0], dtype=torch.float64), deltas[:-1]])
    numers = numer_start - torch.cumsum(deltas, dim=0)
    lats = numers / denom
    return lats
