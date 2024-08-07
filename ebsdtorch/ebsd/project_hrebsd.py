from typing import Optional, Tuple
import torch
from torch import Tensor
from ebsdtorch.ebsd.master_pattern import MasterPattern
from ebsdtorch.ebsd.geometry import bruker_geometry_to_SE3, bruker_geometry_from_SE3
from ebsdtorch.io.read_master_pattern import read_master_pattern
from ebsdtorch.lie_algebra.se3 import se3_exp_map_split, se3_log_map_split
from ebsdtorch.s2_and_so3.quaternions import qu_apply


@torch.jit.script
def hrebsd_coords(
    se3_mask: Tensor,  # (6,) SE(3) mask (rotx, roty, rotz, tx, ty, tz)
    se3_vector: Tensor,  # (B, 3) se(3) Lie algebra vector
    quaternions: Tensor,  # (B, 4) (w, x, y, z) Current rotations
    f_inverse: Tensor,  # (B, 3, 3) inv of deformation gradient tensor
    detector_shape: Tuple[int, int],
    binning: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Tensor:  # (B, H*W, 3) coordinates of detector pixels in sample frame
    """
    Project the master pattern to the given se3 mask and se3 vector.

    Args:
        se3_mask (Tensor): The se(3) mask.
        se3_vector (Tensor): The se(3) vector.
        quaternions (Tensor): The quaternions.
        f_inverse (Tensor): The inverse deformation gradient tensor.
        detector_shape (Tuple[int, int]): The detector shape.
        binning (int): The binning factor.
        device (torch.device): The device to use.
        dtype (torch.dtype, optional): The data type. Defaults to torch.float32.

    Returns:
        Tensor: The coordinates of the detector in the sample frame.

    """

    # check the binning evenly divides the detector shape
    if detector_shape[0] % binning != 0:
        raise ValueError(
            f"A height binning of {binning} does not evenly divide the detector height of {detector_shape[0]}."
        )
    if detector_shape[1] % binning != 0:
        raise ValueError(
            f"A width binning of {binning} does not evenly divide the detector width of {detector_shape[1]}."
        )

    # get binned shape
    binned_shape = (
        detector_shape[0] // binning,
        detector_shape[1] // binning,
    )

    binning_fp = (float(binning), float(binning))

    # create the pixel coordinates
    pixel_coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0.5 * (binning_fp[0] - 1),
                detector_shape[0] - (0.5 * (binning_fp[0] - 1)),
                binned_shape[0],
                dtype=dtype,
                device=device,  # is always on model device
            ),
            torch.linspace(
                0.5 * (binning_fp[1] - 1),
                detector_shape[1] - (0.5 * (binning_fp[1] - 1)),
                binned_shape[1],
                dtype=dtype,
                device=device,  # is always on model device
            ),
            indexing="ij",
        ),
        dim=-1,
    ).view(-1, 2)

    # map se(3) to SE(3)
    rotation_matrix, translation_vector = se3_exp_map_split(se3_mask * se3_vector)

    # convert to 3D coordinates
    pixel_coordinates = torch.cat(
        [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
    )

    # apply the transformation
    sample_coordinates = (
        # (B, 1, 3, 3) @ (1, H*W, 3, 1) + (B, 1, 3, 1)
        ((rotation_matrix[:, None, :, :] @ pixel_coordinates[None, :, :, None]))
        + translation_vector[:, None, :, None]
    ).squeeze(-1)

    # normalize the coordinates to unit sphere
    sample_coordinates = sample_coordinates / torch.norm(
        sample_coordinates, dim=-1, keepdim=True
    )

    # apply the quaternions (B, 1, 4), (B, H*W, 3) -> (B, H*W, 3)
    sample_coordinates = qu_apply(quaternions[:, None, :], sample_coordinates)

    # apply the inverse deformation gradient tensor
    detector_coords = (
        # (B, 1, 3, 3) @ (B, H*W, 3, 1)
        f_inverse[:, None, :, :]
        @ sample_coordinates[..., None]
    ).squeeze(-1)

    # renomalize the coordinates to unit sphere
    detector_coords = detector_coords / torch.norm(
        detector_coords, dim=-1, keepdim=True
    )

    return detector_coords


def project_hrebsd(
    master_pattern: MasterPattern,
    detector_coords: Tensor,
    mode: str = "bilinear",
    virtual_binning: int = 1,
) -> Tensor:
    # use master pattern to interpolate
    return master_pattern.interpolate(
        detector_coords,
        mode=mode,
        padding_mode="border",
        align_corners=False,
        normalize_coords=False,  # already normalized
        virtual_binning=virtual_binning,
    )


# # test it out
# device = torch.device("cuda")
# dtype = torch.float32
# mp_fname = "../EMs/EMplay/old/Si-master-20kV.h5"
# mp = read_master_pattern(mp_fname).to(device).to(dtype)

# se3_mask = torch.ones(6, device=device, dtype=dtype)
# # se3_vector = torch.zeros(100, 6, device=device, dtype=dtype)
# rotation, translation = bruker_geometry_to_SE3(
#     pattern_centers=torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype),
#     primary_tilt_deg=torch.tensor([70.0], device=device, dtype=dtype),
#     secondary_tilt_deg=torch.tensor([0.0], device=device, dtype=dtype),
#     detector_shape=(512, 512),
# )
# se3_vector = se3_log_map_split(rotation, translation)
# se3_vector = se3_vector.repeat(100, 1)

# # print top se3_vector
# print(se3_vector[0].cpu().numpy())

# quaternions = torch.zeros(100, 4, device=device, dtype=dtype)
# quaternions[:, 0] = 1.0

# f_inverse = torch.eye(3, device=device, dtype=dtype).repeat(100, 1, 1)
# f_inverse[:, 2, 2] = 0.8
# detector_shape = (512, 512)


# # convert all inputs to nn.Parameter
# se3_mask = torch.nn.Parameter(se3_mask)
# se3_vector = torch.nn.Parameter(se3_vector)
# quaternions = torch.nn.Parameter(quaternions)
# f_inverse = torch.nn.Parameter(f_inverse)

# # feed Parameters to Adam optimizer
# optim = torch.optim.Adam([se3_mask, se3_vector, quaternions, f_inverse], lr=1e-2)

# coords = hrebsd_coords(
#     se3_mask, se3_vector, quaternions, f_inverse, detector_shape, 1, device, dtype
# )
# print(f"coords shape: {coords.shape}")
# pats = project_hrebsd(mp, coords)
# print(f"pats shape: {pats.shape}")

# # save the top pattern as png
# from PIL import Image

# img = pats[0].reshape(512, 512)
# img = (img - img.min()) / (img.max() - img.min())
# img = (img * 255).byte().cpu().numpy()
# Image.fromarray(img).save("test.png")
