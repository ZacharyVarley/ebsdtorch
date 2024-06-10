from typing import Optional
import torch


@torch.jit.script
def get_radial_mask(
    shape: tuple[int, int],
    radius: Optional[float] = None,
    center: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """
    Get a radial mask.

    Args:
        shape (tuple[int, int]): The shape of the mask.
        radius (float): The radius of the mask.
        center (tuple[float, float]): The center of the mask in fractional coordinates.

    Returns:
        torch.Tensor: The radial mask.

    """
    # get the shape of the mask
    h, w = shape

    # if the radius is not provided, set it to half the minimum dimension
    if radius is None:
        radius = min(h, w) / 2.0

    # create the grid
    ii, jj = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )

    mask = (
        (ii - center[0] * (h - 1)) ** 2 + (jj - center[1] * (w - 1)) ** 2
    ) < radius**2
    return mask.to(torch.bool)
