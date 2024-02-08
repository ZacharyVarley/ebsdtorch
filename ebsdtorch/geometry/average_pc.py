from typing import Optional, Tuple
import torch
from torch import Tensor

from ebsdtorch.geometry.square_projection import rosca_lambert
from ebsdtorch.s2_and_so3.orientations import quaternion_apply


@torch.jit.script
def average_pc(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    signal_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Return sets of direction cosines for varying projection centers.

    This should be viewed as a transformation of coordinates specified by n_rows and
    n_cols in the detector plane to points on the sphere.

    Args:
        pcs: Projection centers. Shape (n_pcs, 3)
        n_rows: Number of detector rows.
        n_cols: Number of detector columns.
        tilt: Detector tilt from horizontal in degrees.
        azimuthal: Sample tilt about the sample RD axis in degrees.
        sample_tilt: Sample tilt from horizontal in degrees.
        signal_mask: 1D signal mask with ``True`` values for pixels to get direction

    Returns:
        The direction cosines for each detector pixel for each PC. Shape (n_pcs, n_det_pixels, 3)

    """
    # Generate row and column coordinates
    nrows_array = torch.arange(n_rows - 1, -1, -1, device=pcs.device).float()
    ncols_array = torch.arange(n_cols, device=pcs.device).float()

    # Calculate cosines and sines
    alpha_rad = torch.tensor(
        [(torch.pi / 2.0) + (tilt - sample_tilt) * (torch.pi / 180.0)],
        device=pcs.device,
    )
    azimuthal_rad = torch.tensor([azimuthal * (torch.pi / 180.0)], device=pcs.device)
    cos_alpha = torch.cos(alpha_rad)
    sin_alpha = torch.sin(alpha_rad)
    cos_omega = torch.cos(azimuthal_rad)
    sin_omega = torch.sin(azimuthal_rad)

    # Extract pcx, pcy, pcz from the pc tensor
    pcx_bruker, pcy_bruker, pcz_bruker = torch.unbind(pcs, dim=-1)

    # Convert to detector coordinates
    pcx_ems = n_cols * (0.5 - pcx_bruker)
    pcy_ems = n_rows * (0.5 - pcy_bruker)
    pcz_ems = n_rows * pcz_bruker

    # det_x is shape (n_pcs, n_cols)
    det_x = pcx_ems[:, None] + (1 - n_cols) * 0.5 + ncols_array[None, :]
    det_y = pcy_ems[:, None] - (1 - n_rows) * 0.5 - nrows_array[None, :]

    # Calculate Ls (n_pcs, n_cols)
    Ls = -sin_omega * det_x + pcz_ems[:, None] * cos_omega
    # Calculate Lc (n_pcs, n_rows)
    Lc = cos_omega * det_x + pcz_ems[:, None] * sin_omega

    # Generate 2D grid indices
    row_indices, col_indices = torch.meshgrid(
        torch.arange(n_rows, device=pcs.device),
        torch.arange(n_cols, device=pcs.device),
        indexing="ij",
    )

    # Flatten the 2D grid indices to 1D
    rows_flat = row_indices.flatten()
    cols_flat = col_indices.flatten()

    # Apply signal mask if it exists
    if signal_mask is not None:
        rows = rows_flat[signal_mask]
        cols = cols_flat[signal_mask]
    else:
        rows = rows_flat
        cols = cols_flat

    # Vectorize the computation
    r_g_x = det_y[:, rows] * cos_alpha + sin_alpha * Ls[:, cols]
    r_g_y = Lc[:, cols]
    r_g_z = -sin_alpha * det_y[:, rows] + cos_alpha * Ls[:, cols]

    # Stack and reshape
    r_g_array = torch.stack([r_g_x, r_g_y, r_g_z], dim=-1)

    # Normalize
    r_g_array = r_g_array / torch.linalg.norm(r_g_array, dim=-1, keepdim=True)

    return r_g_array


@torch.jit.script
def avg_pc_proj_to_det(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    quaternions: Tensor,
    direction_cosines: Tensor,
) -> Tensor:
    """
    Project a direction cosine to the detector plane.

    Args:
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)
        quaternions: Quaternions for each crystalline orientation. Shape (n_orientations, 4)
        direction_cosines: Direction cosines for each pixel in the detector. Shape (n_det_pixels, 3)

    Returns:
        The projected master pattern. Shape (n_orientations, n_det_pixels)

    """
    # sanitize inputs
    assert master_pattern_MSLNH.ndim == 2
    assert master_pattern_MSLSH.ndim == 2
    assert quaternions.ndim == 2
    assert direction_cosines.ndim == 2
    assert direction_cosines.shape[-1] == 3

    n_orientations = quaternions.shape[0]
    n_det_pixels = direction_cosines.shape[0]

    output = torch.empty(
        (n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(
        quaternions[:, None, :], direction_cosines[None, :, :]
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # where the z component is negative, use the Southern Hemisphere projection
    coords_within_square = rosca_lambert(rotated_vectors)

    # where the z component is positive, use the Northern Hemisphere projection
    output[mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLNH[None, None, ...],
        coords_within_square[mask][None, None, :],
        align_corners=True,
    ).squeeze()

    # where the z component is negative, use the Southern Hemisphere projection
    output[~mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLSH[None, None, ...],
        coords_within_square[~mask][None, None, :],
        align_corners=True,
    ).squeeze()

    return output


@torch.jit.script
def project_pattern_multiple_geometry(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    quaternions: Tensor,
    direction_cosines: Tensor,
) -> Tensor:
    """
    Args:
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)
        quaternions: Quaternions for each crystalline orientation. Shape (n_orientations, 4)
        direction_cosines: Direction cosines for each pixel in the detector. Shape (n_pcs, n_det_pixels, 3)

    Returns:
        The projected master pattern. Shape (n_pcs, n_orientations, n_det_pixels)

    """
    # sanitize inputs
    assert master_pattern_MSLNH.ndim == 2
    assert master_pattern_MSLSH.ndim == 2
    assert quaternions.ndim == 2
    assert direction_cosines.ndim == 3
    assert direction_cosines.shape[-1] == 3

    n_orientations = quaternions.shape[0]
    n_pcs, n_det_pixels, _ = direction_cosines.shape

    output = torch.empty(
        (n_pcs, n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(
        quaternions[None, :, None, :], direction_cosines[:, None, :, :]
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # where the z component is negative, use the Southern Hemisphere projection
    coords_within_square = rosca_lambert(rotated_vectors)

    # where the z component is positive, use the Northern Hemisphere projection
    output[mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLNH[None, None, ...],
        coords_within_square[mask][None, :, None],
        align_corners=True,
    ).squeeze()

    # where the z component is negative, use the Southern Hemisphere projection
    output[~mask] = torch.nn.functional.grid_sample(
        master_pattern_MSLSH[None, None, ...],
        coords_within_square[~mask][None, :, None],
        align_corners=True,
    ).squeeze()

    return output
