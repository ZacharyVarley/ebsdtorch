"""
This module implements geometry fitters for EBSD data. The geometry fitters are
used to fit the transformation between the detector and sample reference frames.

The following geometry fitters are implemented:

1. Spatial FFT PC fitter: This fitter searches for the best pattern center
and orientation by brute force. It first generates a 1D grid of pc_z values and
applies the corresponding rescaling of the experimental pattern, and takes their
zero-padded FFT. Then we search via enumeration of a dictionary over the
fundamental zone of enlarged patterns, which allow for FFT's to be used to
densely check pc_x and pc_y values. 2 of the 6 degrees of freedom are searched
via FFTs, while the other 4 are searched via brute force enumeration (pc_z and 3
DOF for orientation).

2. (PLANNED) SO3 FFT PC fitter: This fitter uses FFTs over SO3 to search
for the best orientation given a fixed pattern center. Here, 3 of the 6 degrees
of freedom are densely searched via FFTs, while the other 3 are searched via
brute force enumeration (pc_x, pc_y and pc_z). I expect it to be faster than
(1), but it requires a lot of memory at high band limits. We will see. In both
(1) and (2) it is crucial to apply the pc_z to the experimental pattern before
the search to drastically speed up the search.

3. (PLANNED) Gradient descent geometry fitter: This fitter uses gradient descent
on the Lie algebra se3 with multiple patterns with known scan coordinates to fit
the full 6-DOF geometry which includes an in plane rotation of the detector.
Several metrics including cosine distance and mutual information will be
deployed. This fitter is expected to be the most accurate, but also requires
very close initialization.

"""

import torch
from torch import Tensor
from ebsdtorch.ebsd.master_pattern import MasterPattern
from torch import Tensor
from ebsdtorch.preprocessing.clahe import clahe_grayscale
from ebsdtorch.s2_and_so3.orientations import bu2qu, cu2qu, qu2bu
from ebsdtorch.s2_and_so3.quaternions import qu_apply, qu_norm_std, qu_conj, qu_prod
from ebsdtorch.s2_and_so3.laue_fz_ori import sample_ori_fz_laue_angle
from ebsdtorch.s2_and_so3.laue_fz_ori import ori_to_fz_laue
from ebsdtorch.lie_algebra.se3 import se3_exp_map_split, se3_log_map_split
from ebsdtorch.io.read_master_pattern import read_master_pattern
from typing import Optional, Tuple
from PIL import Image
import numpy as np
from torch.fft import rfft2, irfft2, fftshift
import matplotlib.pyplot as plt
from ebsdtorch.utils.mutual_information import mutual_information, cmif
from ebsdtorch.utils.progressbar import progressbar


@torch.jit.script
def bruker_geometries_to_SE3(
    pattern_centers: Tensor,
    primary_tilt_deg: Tensor,
    secondary_tilt_deg: Tensor,
    detector_shape: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    """
    Convert pattern centers in Bruker coordinates to SE3 transformation matrix.

    Args:
        :pattern_centers (Tensor): The pattern centers.
        :primary_tilt_deg (Tensor): The primary tilt in degrees.
        :secondary_tilt_deg (Tensor): The secondary tilt in degrees.
        :detector_shape (Tuple[int, int]): The detector shape.

    Returns:
        Rotation matrices (..., 3, 3) and translation vectors (..., 3).

    """

    pcx, pcy, pcz = torch.unbind(pattern_centers, dim=-1)
    rows, cols = detector_shape
    rows, cols = float(rows), float(cols)

    # convert to radians
    dt_m_sy = torch.deg2rad(primary_tilt_deg)
    sx = torch.deg2rad(secondary_tilt_deg)

    rotation_matrix = torch.stack(
        [
            -torch.sin(dt_m_sy),
            -torch.sin(sx) * torch.cos(dt_m_sy),
            torch.cos(sx) * torch.cos(dt_m_sy),
            torch.zeros_like(sx),
            torch.cos(sx),
            torch.sin(sx),
            -torch.cos(dt_m_sy),
            torch.sin(sx) * torch.sin(dt_m_sy),
            -torch.sin(dt_m_sy) * torch.cos(sx),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    tx = (
        pcx * cols * torch.sin(sx) * torch.cos(dt_m_sy)
        + pcy * rows * torch.sin(dt_m_sy)
        + pcz * rows * torch.cos(sx) * torch.cos(dt_m_sy)
        - torch.sin(sx) * torch.cos(dt_m_sy) / 2
        - torch.sin(dt_m_sy) / 2
    )
    ty = -cols * pcx * torch.cos(sx) + pcz * rows * torch.sin(sx) + torch.cos(sx) / 2
    tz = (
        -cols * pcx * torch.sin(sx) * torch.sin(dt_m_sy)
        + pcy * rows * torch.cos(dt_m_sy)
        - pcz * rows * torch.sin(dt_m_sy) * torch.cos(sx)
        + torch.sin(sx) * torch.sin(dt_m_sy) / 2
        - torch.cos(dt_m_sy) / 2
    )
    translation_vector = torch.stack([tx, ty, tz], dim=-1)

    return rotation_matrix, translation_vector


@torch.jit.script
def bruker_geometry_from_SE3(
    rotation_matrix: Tensor,
    translation_vector: Tensor,
    detector_shape: Tuple[int, int],
):
    """
    Args:
        rotation_matrix (torch.Tensor): The rotation matrix (..., 3, 3).
        translation_vector (torch.Tensor): The translation vector (..., 3).
        detector_shape (Tuple[int, int]): Pattern shape in pixels, (H, W) with 'ij' indexing.

    Returns:
        torch.Tensor: Pattern center parameters (pcx, pcy, pcz).
    """
    tx, ty, tz = torch.unbind(translation_vector, dim=-1)
    rows, cols = detector_shape
    rows, cols = float(rows), float(cols)

    cos_sx = rotation_matrix[..., 1, 1]
    cos_dt_minus_sy = rotation_matrix[..., 0, 2] / cos_sx
    dt_minus_sy = torch.acos(cos_dt_minus_sy.clamp_(min=-1, max=1))
    sx = torch.acos(cos_sx.clamp_(min=-1, max=1))

    pcx = (
        tx * torch.sin(sx) * torch.cos(dt_minus_sy)
        - ty * torch.cos(sx)
        - tz * torch.sin(dt_minus_sy) * torch.sin(sx)
        + 0.5
    ) / cols
    pcy = (tx * torch.sin(dt_minus_sy) + tz * torch.cos(dt_minus_sy) + 0.5) / rows
    pcz = (
        tx * torch.cos(dt_minus_sy) * torch.cos(sx)
        + ty * torch.sin(sx)
        - tz * torch.sin(dt_minus_sy) * torch.cos(sx)
    ) / rows

    pattern_centers = torch.stack([pcx, pcy, pcz], dim=-1)

    return pattern_centers, torch.rad2deg(dt_minus_sy), torch.rad2deg(sx)


@torch.jit.script
def get_3D_pixel_coords(
    detector_shape: Tuple[int, int],
    device: torch.device,
    binning: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # get binned shape
    binned_shape = (
        detector_shape[0] // binning,
        detector_shape[1] // binning,
    )

    binning_fp = (float(binning), float(binning))

    # create the pixel coordinates
    # these are the i indices for 4x4 detector:
    # 0, 0, 0, 0
    # 1, 1, 1, 1
    # 2, 2, 2, 2
    # 3, 3, 3, 3
    # For binning, we don't want every other pixel. We need fractional pixel coordinates.
    # 0,  0,  0,  0
    #   x       x
    # 1,  1,  1,  1
    #
    # 2,  2,  2,  2
    #   x       x
    # 3,  3,  3,  3
    # ... x marks the spots where we want the coordinates.
    # create the pixel coordinates
    pixel_coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0.5 * (binning_fp[0] - 1),
                detector_shape[0] - (0.5 * (binning_fp[0] - 1)),
                binned_shape[0],
                dtype=dtype,
                device=device,
            ),
            torch.linspace(
                0.5 * (binning_fp[1] - 1),
                detector_shape[1] - (0.5 * (binning_fp[1] - 1)),
                binned_shape[1],
                dtype=dtype,
                device=device,
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    # convert to 3D coordinates
    pixel_coordinates = torch.cat(
        [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
    )

    return pixel_coordinates


@torch.jit.script
def get_3D_pixel_coords_with_periphery(
    detector_shape: Tuple[int, int],
    device: torch.device,
    pc_x_min: float,
    pc_x_max: float,
    pc_y_min: float,
    pc_y_max: float,
    binning: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tuple[int, int, int, int], Tuple[int, int]]:
    """

    Args:
        :detector_shape (Tuple[int, int]): The detector shape.
        :device (torch.device): The device to use.
        :pc_x_min (float): min pattern center x (< 0.5)
        :pc_x_max (float): max pattern center x (> 0.5)
        :pc_y_min (float): min pattern center y (< 0.5)
        :pc_y_max (float): max pattern center y (> 0.5)

    Returns:
        :pixel_coordinates (Tensor): 3D pixel coordinates.
        :padding (Tuple[int, int, int, int]): Computed padding.

    """
    # get binned shape
    binning_fp = float(binning)

    w_pad_prior = int((0.5 - pc_x_min) * detector_shape[1] / binning_fp)
    w_pad_after = int((pc_x_max - 0.5) * detector_shape[1] / binning_fp)
    h_pad_prior = int((0.5 - pc_y_min) * detector_shape[0] / binning_fp)
    h_pad_after = int((pc_y_max - 0.5) * detector_shape[0] / binning_fp)

    binned_shape = (
        (detector_shape[0] // binning) + h_pad_prior + h_pad_after,
        (detector_shape[1] // binning) + w_pad_prior + w_pad_after,
    )

    pixel_coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0.5 * (binning_fp - 1) - h_pad_prior,
                detector_shape[0] + h_pad_after - (0.5 * (binning_fp - 1)),
                binned_shape[0],
                dtype=dtype,
                device=device,
            ),
            torch.linspace(
                0.5 * (binning_fp - 1) - w_pad_prior,
                detector_shape[1] + w_pad_after - (0.5 * (binning_fp - 1)),
                binned_shape[1],
                dtype=dtype,
                device=device,
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    # convert to 3D coordinates
    pixel_coordinates = torch.cat(
        [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
    )

    return (
        pixel_coordinates,
        (w_pad_prior, w_pad_after, h_pad_prior, h_pad_after),
        binned_shape,
    )


@torch.jit.script
def apply_geometry(
    pixel_coords: Tensor,
    rotation_matrix: Tensor,
    translation: Tensor,
) -> Tensor:
    """
    Args:
        :pixel_coords (Tensor): Detector pixel coordinates (detector ref frame).
        :quats (Tensor): The quaternions for crystal orientations.
        :rotation_matrix (Tensor): Geometric rotation matrix.
        :translation (Tensor): Geometric translation.

    Returns:
        :detector_coords (Tensor): The detector coordinates in the sample reference frame.

    """
    # rotate the pixel coordinates
    detector_coords = (
        torch.matmul(rotation_matrix, pixel_coords[..., None]).squeeze(-1) + translation
    )

    # normalize the detector coordinates
    # (N_PIX, 3)
    detector_coords = detector_coords / torch.norm(
        detector_coords, dim=-1, keepdim=True
    )

    # (N_PIX, 3)
    return detector_coords


@torch.jit.script
def generate_dictionary_coords(
    pixel_coords: Tensor,
    quats: Tensor,
    rotation_matrix: Tensor,
    translation: Tensor,
) -> Tensor:
    """
    Args:
        :pixel_coords (Tensor): Detector pixel coordinates (detector ref frame).
        :quats (Tensor): The quaternions for crystal orientations.
        :rotation_matrix (Tensor): Geometric rotation matrix.
        :translation (Tensor): Geometric translation.

    Returns:
        :detector_coords (Tensor): The detector coordinates in the sample reference frame.

    """
    # rotate the pixel coordinates
    detector_coords = (
        torch.matmul(rotation_matrix, pixel_coords[..., None]).squeeze(-1) + translation
    )

    # normalize the detector coordinates
    # (N_PIX, 3)
    detector_coords = detector_coords / torch.norm(
        detector_coords, dim=-1, keepdim=True
    )

    # apply quaternions to the detector coordinates
    # (N_PATS, 4) & (N_PIX, 3) -> (N_PATS, N_PIX, 3)
    detector_coords = qu_apply(quats[:, None, :], detector_coords[None, :, :])

    return detector_coords


def spatial_fft_pc_fitter(
    mp: MasterPattern,
    patterns: Tensor,
    pc_x_min: float = 0.0,
    pc_x_max: float = 1.0,
    pc_y_min: float = 0.0,
    pc_y_max: float = 1.0,
    pc_z_min: float = 0.3,
    pc_z_max: float = 0.7,
    pc_z_steps: int = 16,
    primary_tilt_deg: float = 70.0,
    secondary_tilt_deg: float = 0.0,
    dictionary_resolution_deg: float = 1.0,
    dictionary_indexing_batch_size: int = 1024,
    virtual_binning: int = 1,
    metric: str = "dot",
) -> Tuple[Tensor, Tensor]:
    if patterns.dim() != 4:
        raise ValueError("Pattern must be 4D (B, C, H, W).")

    B, C, H, W = patterns.shape
    assert C == 1, "Only single channel patterns are supported."

    H_bin, W_bin = H // virtual_binning, W // virtual_binning
    assert H_bin * virtual_binning == H, "H must be divisible by virtual binning."
    assert W_bin * virtual_binning == W, "W must be divisible by virtual binning."

    if virtual_binning > 1:
        patterns = torch.nn.functional.avg_pool2d(
            patterns,
            kernel_size=virtual_binning,
            stride=virtual_binning,
        )

    patterns -= patterns.view(B, -1).min(dim=-1, keepdim=True).values.view(B, 1, 1, 1)
    patterns /= patterns.view(B, -1).max(dim=-1, keepdim=True).values.view(B, 1, 1, 1)

    if metric == "dot":
        patterns -= patterns.view(B, -1).mean(dim=-1, keepdim=True).view(B, 1, 1, 1)

    pixel_coords_det_ref, padding, padded_shape = get_3D_pixel_coords_with_periphery(
        detector_shape=(H, W),
        device=patterns.device,
        binning=virtual_binning,
        pc_x_min=pc_x_min,
        pc_x_max=pc_x_max,
        pc_y_min=pc_y_min,
        pc_y_max=pc_y_max,
    )

    pattern_pad = torch.nn.functional.pad(patterns, padding, mode="constant", value=0.0)
    pattern_pad = pattern_pad.view(1, B, padded_shape[0], padded_shape[1])

    if metric == "dot":
        pattern_fft = rfft2(pattern_pad).conj()

    quats_dict = sample_ori_fz_laue_angle(
        laue_id=mp.laue_group,
        angular_resolution_deg=dictionary_resolution_deg,
        device=patterns.device,
    )

    primary_tilt_deg = torch.tensor(primary_tilt_deg, device=patterns.device)
    secondary_tilt_deg = torch.tensor(secondary_tilt_deg, device=patterns.device)

    pc_z_vals = torch.linspace(pc_z_min, pc_z_max, pc_z_steps, device=patterns.device)

    best_dots = torch.zeros(
        (pc_z_steps, B, padded_shape[0], padded_shape[1]), device=patterns.device
    )
    best_orientations = torch.zeros(
        (pc_z_steps, B, padded_shape[0], padded_shape[1]),
        dtype=torch.long,
        device=patterns.device,
    )

    src_ind = torch.nn.functional.pad(
        torch.ones((1, 1, H_bin, W_bin), device=patterns.device),
        padding,
        mode="constant",
        value=0.0,
    ).repeat(B, 1, 1, 1)

    # get a helper function to sync devices
    if patterns.device.type == "cuda":
        sync = torch.cuda.synchronize
    elif patterns.device.type == "mps":
        sync = torch.mps.synchronize
    elif patterns.device.type == "xpu" or patterns.device.type == "xla":
        sync = torch.xpu.synchronize
    else:
        sync = lambda: None

    for z_idx, pc_z in enumerate(pc_z_vals):
        print(f"Processing PC Z {pc_z.item():.5f} ({z_idx+1}/{pc_z_steps})")

        rotation_matrix, translation = bruker_geometries_to_SE3(
            pattern_centers=torch.tensor([[0.5, 0.5, pc_z]], device=patterns.device),
            primary_tilt_deg=-1.0 * primary_tilt_deg,
            secondary_tilt_deg=secondary_tilt_deg,
            detector_shape=(H, W),
        )

        # for j in range(0, quats_dict.shape[0], dictionary_indexing_batch_size):
        pb = progressbar(range(0, quats_dict.shape[0], dictionary_indexing_batch_size))
        for j in pb:
            batch_size = min(dictionary_indexing_batch_size, quats_dict.shape[0] - j)
            dictionary_coords = generate_dictionary_coords(
                pixel_coords=pixel_coords_det_ref,
                quats=quats_dict[j : j + dictionary_indexing_batch_size],
                rotation_matrix=rotation_matrix,
                translation=translation,
            )

            sim_pats = mp.interpolate(
                dictionary_coords,
                mode="bilinear",
                align_corners=False,
                normalize_coords=False,
                virtual_binning=virtual_binning,
            )

            sim_pats = sim_pats.view(batch_size, 1, padded_shape[0], padded_shape[1])

            if metric == "dot":
                sim_pats_fft = rfft2(sim_pats, dim=(-2, -1))
                dots_full = fftshift(irfft2(sim_pats_fft * pattern_fft), dim=(-2, -1))
            elif metric == "mi":
                dots_full = cmif(
                    images_dst=sim_pats,
                    images_src=pattern_pad,
                    src_ind=src_ind,
                    bins_dst=4,
                    bins_src=4,
                    constant_marginal=False,
                )
            else:
                raise ValueError("Invalid metric.")

            # dots_full is shape (D, B, padded_shape[0], padded_shape[1])
            # need to max-reduce over the D batch dimension and track which orientation won
            batch_values, batch_indices = torch.max(dots_full, dim=0)

            best_dots[z_idx] = torch.maximum(best_dots[z_idx], batch_values)

            best_orientations[z_idx] = torch.where(
                best_dots[z_idx] == batch_values,
                batch_indices,
                best_orientations[z_idx],
            )

            sync()

        # print(f"Current best point: {best_dots.mean(dim=1).max().item()}")
        print(f"Best point for that PC Z: {best_dots[z_idx].mean(dim=0).max().item()}")

    # Average over batch of experimental patterns
    avg_best_dots = best_dots.mean(dim=1)

    # Find global maximum
    best_z_idx, best_y_idx, best_x_idx = torch.unravel_index(
        avg_best_dots.argmax(), avg_best_dots.shape
    )

    best_pc_z = pc_z_vals[best_z_idx]
    best_pc_y = 0.5 + float(best_y_idx - padding[2] - (H_bin + 1) // 2) / H_bin
    best_pc_x = 0.5 + float(best_x_idx - padding[0] - (W_bin + 1) // 2) / W_bin

    # Get the best orientation for each pattern in the batch
    best_orientations_batch = best_orientations[best_z_idx, :, best_y_idx, best_x_idx]
    best_quats = quats_dict[best_orientations_batch]

    best_pc = torch.tensor([best_pc_x, best_pc_y, best_pc_z], device=patterns.device)

    print(f"Best PC: {best_pc}")
    print(f"Best orientations shape: {best_quats.shape}")

    # # Visualize results
    # visualize_results(
    #     avg_best_dots.cpu(),
    #     best_z_idx.cpu(),
    #     best_y_idx.cpu(),
    #     best_x_idx.cpu(),
    #     padded_shape,
    #     H_bin,
    #     W_bin,
    #     padding,
    # )

    return best_pc, best_quats


# def visualize_results(
#     avg_best_dots,
#     best_z_idx,
#     best_y_idx,
#     best_x_idx,
#     padded_shape,
#     H_bin,
#     W_bin,
#     padding,
# ):
#     # cut through Z
#     plt.figure(figsize=(10, 8))
#     plt.imshow(avg_best_dots[best_z_idx], cmap="viridis")
#     plt.colorbar(label="Normalized Cross-Correlation")
#     plt.title(f"Cross-Correlation Surface for Best PC Z (idx: {best_z_idx})")
#     plt.axhline(y=best_y_idx, color="r", linestyle="--")
#     plt.axvline(x=best_x_idx, color="r", linestyle="--")
#     plt.tight_layout()
#     plt.savefig("cc_surface_best_pcz.png")
#     plt.close()

#     # cut through Y
#     plt.figure(figsize=(10, 8))
#     plt.imshow(avg_best_dots[:, best_y_idx], cmap="viridis")
#     plt.colorbar(label="Normalized Cross-Correlation")
#     plt.title(f"Cross-Correlation Surface for Best PC Y (idx: {best_y_idx})")
#     plt.axhline(y=best_z_idx, color="r", linestyle="--")
#     plt.axvline(x=best_x_idx, color="r", linestyle="--")
#     plt.tight_layout()
#     plt.savefig("cc_surface_best_pcy.png")
#     plt.close()

#     # cut through X
#     plt.figure(figsize=(10, 8))
#     plt.imshow(avg_best_dots[:, :, best_x_idx], cmap="viridis")
#     plt.colorbar(label="Normalized Cross-Correlation")
#     plt.title(f"Cross-Correlation Surface for Best PC X (idx: {best_x_idx})")
#     plt.axhline(y=best_z_idx, color="r", linestyle="--")
#     plt.axvline(x=best_y_idx, color="r", linestyle="--")
#     plt.tight_layout()
#     plt.savefig("cc_surface_best_pcx.png")
#     plt.close()

#     # all three in one
#     fig, axs = plt.subplots(1, 3, figsize=(20, 8))
#     axs[0].imshow(avg_best_dots[best_z_idx], cmap="viridis")
#     axs[0].set_title(f"Best PC Z (idx: {best_z_idx})")
#     axs[0].axhline(y=best_y_idx, color="r", linestyle="--")
#     axs[0].axvline(x=best_x_idx, color="r", linestyle="--")

#     axs[1].imshow(avg_best_dots[:, best_y_idx], cmap="viridis")
#     axs[1].set_title(f"Best PC Y (idx: {best_y_idx})")
#     axs[1].axhline(y=best_z_idx, color="r", linestyle="--")
#     axs[1].axvline(x=best_x_idx, color="r", linestyle="--")

#     axs[2].imshow(avg_best_dots[:, :, best_x_idx], cmap="viridis")
#     axs[2].set_title(f"Best PC X (idx: {best_x_idx})")
#     axs[2].axhline(y=best_z_idx, color="r", linestyle="--")
#     axs[2].axvline(x=best_y_idx, color="r", linestyle="--")

#     plt.tight_layout()
#     plt.savefig("cc_surface_best_all.png")
#     plt.close()

#     # Print additional information
#     print(f"Best match coordinates (including padding): y={best_y_idx}, x={best_x_idx}")
#     print(
#         f"Best match coordinates (original image): y={best_y_idx - padding[2]}, x={best_x_idx - padding[0]}"
#     )
#     print(f"Padded shape: {padded_shape}")
#     print(f"Original shape: H={H_bin}, W={W_bin}")


# mp_fname = "../EMs/EMplay/Ti-alpha-master-20kV.h5"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp = read_master_pattern(mp_fname).to(device)
# mp.normalize("minmax")
# # mp.normalize("zeromean")
# # mp.apply_clahe()

# # visualize the first pattern
# import matplotlib.pyplot as plt

# # visualize the master pattern using PIL
# img = mp.master_pattern
# img -= img.min()
# img /= img.max()
# img = (img * 255).byte().cpu()
# img = Image.fromarray(img.numpy())
# img.save("mp_ti.png")

# # use loop instead
# x_vals = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
# y_vals = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
# # x_vals = [200, 220, 240, 260, 280]
# # y_vals = [150, 170, 190, 210, 230]
# # x_vals = [200, 240, 280]
# # y_vals = [150, 190, 230]
# # x_vals = [200, 280]
# # y_vals = [150, 230]
# # x_vals = [
# #     200,
# # ]
# # y_vals = [
# #     150,
# # ]
# # do all combinations
# xy_pairs = [(x, y) for x in x_vals for y in y_vals]

# patterns = [
#     torch.tensor(
#         plt.imread(f"./Slice_38/Slice_38_{x}_{y}_EBSD.tif")[:, :].astype(np.float32),
#         dtype=torch.float32,
#         device=device,
#     )
#     # do all combinations
#     for x, y in xy_pairs
# ]

# H, W = patterns[0].shape
# virtual_bin_amount = 2
# hp_kernel = 5

# # make into a (B, 1, H, W) tensor
# patterns = torch.stack(patterns, dim=0)[:, None]

# # # subtract the mean pattern
# patterns -= patterns.mean(dim=0, keepdim=True)

# # clean up the experimental pattern with a high pass filter
# # do the mask in real space with avg_pool2d and a mask
# pattern_padded = torch.nn.functional.pad(
#     patterns, (hp_kernel,) * 4, mode="constant", value=0.0
# )
# avg_vals = torch.nn.functional.avg_pool2d(pattern_padded, 2 * hp_kernel + 1, stride=1)

# mask = torch.ones_like(patterns)
# mask_padded = torch.nn.functional.pad(
#     mask, (hp_kernel,) * 4, mode="constant", value=0.0
# )
# mask_avg = torch.nn.functional.avg_pool2d(mask_padded, 2 * hp_kernel + 1, stride=1)

# actual_avg = avg_vals / mask_avg
# patterns = patterns - actual_avg


# # per pattern normalization instead of global
# patterns -= (
#     patterns.view(patterns.shape[0], -1)
#     .min(dim=-1, keepdim=True)
#     .values.view(patterns.shape[0], 1, 1, 1)
# )
# patterns /= (
#     patterns.view(patterns.shape[0], -1)
#     .max(dim=-1, keepdim=True)
#     .values.view(patterns.shape[0], 1, 1, 1)
# )

# # and CLAHE
# # pattern = clahe_grayscale(pattern, clip_limit=40.0, n_bins=64, grid_shape=(4, 4))
# # pattern -= pattern.min()
# # pattern /= pattern.max()
# # pattern = pattern[0, 0]

# # instead make gif of all patterns
# imgs = []
# for i in range(patterns.shape[0]):
#     img = patterns[i].squeeze()
#     img -= img.min()
#     img /= img.max()
#     img = (img * 255).byte().cpu()
#     img = Image.fromarray(img.numpy())
#     img = img.resize((W // virtual_bin_amount, H // virtual_bin_amount))
#     imgs.append(img)

# imgs[0].save(
#     "zsim_all_pats.gif",
#     save_all=True,
#     append_images=imgs[1:],
#     duration=100,
#     loop=0,
# )

# # fit the pattern center via brute force
# pc, quat = spatial_fft_pc_fitter(
#     mp,
#     patterns,
#     pc_z_min=0.35,
#     pc_z_max=0.65,
#     pc_z_steps=16,
#     primary_tilt_deg=70.0,
#     secondary_tilt_deg=0.0,
#     dictionary_indexing_batch_size=1024,
#     dictionary_resolution_deg=1.0,
#     virtual_binning=virtual_bin_amount,
#     metric="dot",
# )

# # print the pattern with the fitted pattern center and orientation
# print(f"Pattern center: {pc}")
# print(f"Orientation: {quat}")

# # # get the rotation matrix and translation vector
# # rotation_matrix, translation = bruker_geometries_to_SE3(
# #     pc[None, :],
# #     torch.tensor(-70.0, device=device),
# #     torch.tensor(0.0, device=device),
# #     detector_shape=(H, W),
# # )

# # # apply the geometry
# # pixel_coords = get_3D_pixel_coords(
# #     detector_shape=(H, W),
# #     device=device,
# #     binning=virtual_bin_amount,
# # )

# # # apply the geometry
# # detector_coords = apply_geometry(
# #     pixel_coords=pixel_coords,
# #     rotation_matrix=rotation_matrix,
# #     translation=translation,
# # )

# # # apply the orientation
# # detector_coords = qu_apply(quat[None, :], detector_coords)

# # # interpolate the master pattern
# # sim_pats = mp.interpolate(
# #     detector_coords,
# #     mode="bilinear",
# #     align_corners=False,
# #     normalize_coords=True,
# #     virtual_binning=virtual_bin_amount,
# # )

# # # minmax
# # sim_pats -= sim_pats.min()
# # sim_pats /= sim_pats.max()

# # # save the simulated pattern via PIL
# # img_best = Image.fromarray(
# #     (sim_pats * 255)
# #     .byte()
# #     .cpu()
# #     .numpy()
# #     .reshape(
# #         patterns.shape[0] // virtual_bin_amount, patterns.shape[1] // virtual_bin_amount
# #     )
# # )
# # img_best.save("simulated_pattern_ti.png")
