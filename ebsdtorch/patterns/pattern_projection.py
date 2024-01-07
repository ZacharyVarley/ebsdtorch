from typing import Optional
import torch
from torch import Tensor

from ebsdtorch.patterns.square_hemisphere_bijection import square_lambert
from ebsdtorch.s2_and_so3.orientations import quaternion_apply

# @torch.jit.script
# def pattern_center_to_camera_matrix(
#     pcs: Tensor,

# )


@torch.jit.script
def detector_coords_to_ksphere_via_pc(
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
def project_pattern_multiple_geometry(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    quaternions: Tensor,
    direction_cosines: Tensor,
) -> Tensor:
    """

    This function projects the master pattern onto the detector for each crystalline orientation.
    It is called "paired" because each orientation is paired with another pattern center triplet of
    direction cosines. This function would make sense to use in the context of indexing a map of
    EBSD patterns. Each crystalline orientation would be paired with a pattern center triplet that
    corresponds to that location on the sample.

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
    coords_within_square = square_lambert(rotated_vectors)

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
def project_pattern_single_geometry(
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
    coords_within_square = square_lambert(rotated_vectors)

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


# import numpy as np

# def _get_cosine_sine_of_alpha_and_azimuthal(
#     sample_tilt: float, tilt: float, azimuthal: float
# ) -> Tuple[float, float, float, float]:
#     alpha = (np.pi / 2) - np.deg2rad(sample_tilt) + np.deg2rad(tilt)
#     azimuthal = np.deg2rad(azimuthal)
#     return np.cos(alpha), np.sin(alpha), np.cos(azimuthal), np.sin(azimuthal)


# def _get_direction_cosines_for_varying_pc_np(
#     pcx: np.ndarray,
#     pcy: np.ndarray,
#     pcz: np.ndarray,
#     nrows: int,
#     ncols: int,
#     tilt: float,
#     azimuthal: float,
#     sample_tilt: float,
#     signal_mask: np.ndarray,
# ) -> np.ndarray:
#     """Return sets of direction cosines for varying projection centers
#     (PCs).

#     Algorithm adapted from EMsoft, see :cite:`callahan2013dynamical`.

#     Parameters
#     ----------
#     pcx
#         PC x coordinates. Must be a 1D array.
#     pcy
#         PC y coordinates. Must be a 1D array.
#     pcz
#         PC z coordinates. Must be a 1D array.
#     nrows
#         Number of detector rows.
#     ncols
#         Number of detector columns.
#     tilt
#         Detector tilt from horizontal in degrees.
#     azimuthal
#         Sample tilt about the sample RD axis in degrees.
#     sample_tilt
#         Sample tilt from horizontal in degrees.
#     signal_mask
#         1D signal mask with ``True`` values for pixels to get direction
#         cosines for.

#     Returns
#     -------
#     r_g_array
#         Direction cosines for each detector pixel for each PC, of shape
#         (n PCs, n_pixels, 3) and data type of 64-bit floats.

#     See Also
#     --------
#     kikuchipy.detectors.EBSDDetector

#     Notes
#     -----
#     This function is optimized with Numba, so care must be taken with
#     array shapes and data types.
#     """
#     nrows_array = np.arange(nrows - 1, -1, -1)
#     ncols_array = np.arange(ncols)

#     ca, sa, cw, sw = _get_cosine_sine_of_alpha_and_azimuthal(
#         sample_tilt=sample_tilt,
#         tilt=tilt,
#         azimuthal=azimuthal,
#     )

#     det_x_factor = (1 - ncols) * 0.5
#     det_y_factor = (1 - nrows) * 0.5

#     idx_1d = np.arange(nrows * ncols)[signal_mask]
#     rows = idx_1d // ncols
#     cols = np.mod(idx_1d, ncols)

#     rows = rows.flatten()
#     cols = cols.flatten()

#     n_pcs = pcx.size
#     n_pixels = idx_1d.size
#     r_g_array = np.zeros((n_pcs, n_pixels, 3), dtype=np.float64)

#     for i in range(n_pcs):
#         # Bruker to EMsoft's v5 PC convention
#         xpc = ncols * (0.5 - pcx[i])
#         ypc = nrows * (0.5 - pcy[i])
#         zpc = nrows * pcz[i]

#         det_x = xpc + det_x_factor + ncols_array
#         det_y = ypc - det_y_factor - nrows_array

#         Ls = -sw * det_x + zpc * cw
#         Lc = cw * det_x + zpc * sw

#         for j in range(n_pixels):
#             r_g_array[i, j, 0] = det_y[rows[j]] * ca + sa * Ls[cols[j]]
#             r_g_array[i, j, 1] = Lc[cols[j]]
#             r_g_array[i, j, 2] = -sa * det_y[rows[j]] + ca * Ls[cols[j]]

#     # Normalize
#     norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
#     norm = np.expand_dims(norm, axis=-1)
#     r_g_array = np.true_divide(r_g_array, norm)

#     return r_g_array


# # test the function
# pattern_centers = torch.tensor(
#     [
#         [0.0, 0.0, 100],
#     ]
# )
# n_rows = 2
# n_cols = 2
# tilt = 0.0
# azimuthal = 0.0
# sample_tilt = 90.0
# signal_mask = None
# cosines = detector_coords_to_ksphere_via_pc(
#     pattern_centers,
#     n_rows,
#     n_cols,
#     tilt,
#     azimuthal,
#     sample_tilt,
#     signal_mask=signal_mask,
# )
# print(cosines.cpu().numpy())

# # compare with numpy
# # test the function
# pattern_centers_np = pattern_centers.numpy()
# cosines_np = _get_direction_cosines_for_varying_pc_np(
#     pattern_centers_np[:, 0],
#     pattern_centers_np[:, 1],
#     pattern_centers_np[:, 2],
#     n_rows,
#     n_cols,
#     tilt,
#     azimuthal,
#     sample_tilt,
#     signal_mask,
# )
# print(cosines_np)


# # -------------------------------------------

# import kikuchipy as kp
# import matplotlib.pyplot as plt
# import numpy as np
# from orix.quaternion import Orientation, Rotation
# from orix.vector import Vector3d
# from orix import sampling


# s = kp.data.nickel_ebsd_small()

# det = kp.detectors.EBSDDetector(
#     shape=s.axes_manager.signal_shape[::-1],
#     pc=[0.4221, 0.2179, 0.4954],
#     px_size=70,  # Microns
#     binning=8,
#     tilt=0,
#     sample_tilt=70,
# )

# mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")

# ni = mp.phase

# Gr = sampling.get_sample_fundamental(
#     method="cubochoric", resolution=15, point_group=ni.point_group
# )

# sim_kp = mp.get_patterns(
#     rotations=Gr,
#     detector=det,
#     energy=20,
#     dtype_out=np.float32,
#     compute=True,
# )

# mLPNH = mp.data[0, :, :]
# mLPSH = mp.data[1, :, :]

# cos_directions = detector_coords_to_ksphere_via_pc(
#     torch.from_numpy(det.pc).to(torch.float64),
#     det.shape[0],
#     det.shape[1],
#     det.tilt,
#     det.azimuthal,
#     det.sample_tilt,
#     signal_mask=None,
# )

# print(Gr.data.shape)
# print(cos_directions.shape)

# sim_torchebsd = project_master_pattern_single_geometry(
#     torch.tensor(mLPNH, dtype=torch.float64),
#     torch.tensor(mLPSH, dtype=torch.float64),
#     torch.from_numpy(Gr.data).to(torch.float64),
#     cos_directions.squeeze().to(torch.float64),
# )

# # normalize each row to -1 to 1
# sim_torchebsd = (
#     2.0
#     * (sim_torchebsd - sim_torchebsd.min(dim=-1, keepdim=True)[0])
#     / (
#         sim_torchebsd.max(dim=-1, keepdim=True)[0]
#         - sim_torchebsd.min(dim=-1, keepdim=True)[0]
#     )
# ) - 1.0

# sim_torchebsd = sim_torchebsd.numpy().reshape(sim_kp.data.shape)

# # compare with kikuchipy (mse)
# print(np.mean((sim_torchebsd - sim_kp.data) ** 2))

# # error was 1e-09 between all the patterns (265 x 3600) and machine epsilon is around 1e-08
# # for float 64, error was 1e-14 which slightly exceeds machine epsilon of 1e-16

# # save the first pattern from each library
# plt.imsave("kikuchipy.png", sim_kp.data[0, :, :])
# plt.imsave("torchebsd.png", sim_torchebsd[0, :, :])

# # print the min and max values
# print(sim_kp.data.min(), sim_kp.data.max())
# print(sim_torchebsd.min(), sim_torchebsd.max())
