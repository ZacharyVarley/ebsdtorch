"""
PyTorch implementation of 2D contrast limited adaptive histogram equalization.

Heavily inspired by Kornia's CLAHE implementation:
https://github.com/kornia/kornia

"""

from typing import Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import math


@torch.jit.script
def clahe_grayscale(
    x: torch.Tensor,
    clip_limit: float = 40.0,
    n_bins: int = 64,
    grid_shape: Tuple[int, int] = (8, 8),
) -> Tensor:
    """
    Calculates the tile CDFs for CLAHE on a 2D tensor.

    Args:
        x (Tensor): Input grayscale images of shape (B, 1, H, W).
        clip_limit (float): The clip limit for the histogram equalization.
        n_bins (int): The number of bins to use for the histogram equalization.
        grid_shape (Tuple[int, int]): The shape of the grid to divide the image into.

    Returns:
        Tensor: Output images of shape (B, 1, H, W).


    """
    # get shapes
    h_img, w_img = x.shape[-2:]
    signal_shape = x.shape[:-2]
    B = int(torch.prod(torch.tensor(signal_shape)).item())

    h_grid, w_grid = grid_shape
    n_tiles = h_grid * w_grid
    h_tile = math.ceil(h_img / h_grid)
    w_tile = math.ceil(w_img / w_grid)
    voxels_per_tile = h_tile * w_tile
    # pad the input to be divisible by the tile counts in each dimension
    pad_w = w_grid - (w_img % w_grid) if w_img % w_grid != 0 else 0
    pad_h = h_grid - (h_img % h_grid) if h_img % h_grid != 0 else 0
    paddings = (0, pad_w, 0, pad_h)
    # torch.nn.functional.pad uses last dimension to first in pairs
    x_padded = torch.nn.functional.pad(
        x,
        paddings,
        mode="reflect",
    )
    # unfold the input into tiles of shape (B, voxels_per_tile, -1)
    tiles = torch.nn.functional.unfold(
        x_padded, kernel_size=(h_tile, w_tile), stride=(h_tile, w_tile)
    )
    tiles = tiles.view((B, voxels_per_tile, n_tiles))
    # permute from (B, voxels_per_tile, n_tiles) to (B, n_tiles, voxels_per_tile)
    tiles = tiles.swapdims(-1, -2)
    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, n_tiles, n_bins), device=x.device, dtype=torch.float32)
    # use scatter to do an inplace histogram calculation per tile
    # large memory usage because scatter only supports int64 indices
    x_discrete = (tiles * (n_bins - 1)).to(torch.int64)
    pdf.scatter_(
        dim=-1,
        index=x_discrete,
        value=1,
        reduce="add",
    )
    pdf = pdf / pdf.sum(dim=-1, keepdim=True)
    # pdf is handled in pixel counts for OpenCV equivalence
    histos = (pdf * voxels_per_tile).view(-1, n_bins)
    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)
        histos.clamp_(max=limit)
        # calculate the clipped pdf of shape (B, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(dim=-1)
        # calculate the excess of shape (B, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]
        # trick to avoid using a loop to assign the residual
        v_range = torch.arange(n_bins, device=histos.device)
        mat_range = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)
    # cdf (B, n_tiles, n_bins)
    cdfs = torch.cumsum(histos, dim=-1) / voxels_per_tile
    cdfs = cdfs.clamp_(min=0.0, max=1.0)
    cdfs = cdfs.view(
        B,
        h_grid,
        w_grid,
        n_bins,
    )
    coords = torch.meshgrid(
        [
            torch.linspace(-1.0, 1.0, w_img, device=x.device),
            torch.linspace(-1.0, 1.0, h_img, device=x.device),
        ],
        indexing="xy",
    )
    coords = torch.stack(coords, dim=-1)

    # we use grid_sample with border padding to handle the extrapolation
    # we have to use trilinear as tri-cubic is not available
    coords_into_cdfs = torch.cat(
        [
            x[..., None].view(B, 1, h_img, w_img, 1) * 2.0 - 1.0,
            coords[None, None].repeat(B, 1, 1, 1, 1),
        ],
        dim=-1,
    )
    x = F.grid_sample(
        cdfs[:, None, :, :, :],  # (B, 1, n_bins, GH, GW)
        coords_into_cdfs,  # (B, 1, H, W, 3)
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return x[:, 0].reshape(
        signal_shape + (h_img, w_img)
    )  # (B, C, D, H, W) of (B, 1, 1, H, W) -> (B, 1, H, W) -> in shape
