"""
This file implements conventional HREBSD projection, which is the projection of
EBSD patterns onto a detector for a given set of crystalline orientations,
pattern projection center, and deformation gradient matrices. In the future, I
would like to define the projection center in terms of a camera matrix model as
is done in the rest of computer vision. This would allow for geometrically
consistent projection of patterns onto a detector from different sample
locations.

"""

from typing import Optional
import torch
from torch import Tensor

from ebsdtorch.geometry.average_pc import average_pc
from ebsdtorch.s2_and_so3.square_projection import rosca_lambert
from ebsdtorch.s2_and_so3.orientations import quaternion_apply


@torch.jit.script
def project_HREBSD_pattern(
    pcs: Tensor,
    n_rows: int,
    n_cols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    quaternions: Tensor,
    deformation_gradients: Tensor,
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    signal_mask: Optional[Tensor] = None,
) -> Tensor:
    """

    This function projects the master pattern onto the detector for each crystalline orientation.
    It is called "paired" because each orientation is paired with another pattern center triplet of
    direction cosines. This function would make sense to use in the context of indexing a map of
    EBSD patterns. Each crystalline orientation would be paired with a pattern center triplet that
    corresponds to that location on the sample.

    Args:
        pcs: Projection centers. Shape (n, 3)
        n_rows: Number of detector rows.
        n_cols: Number of detector columns.
        tilt: Detector tilt from horizontal in degrees.
        azimuthal: Sample tilt about the sample RD axis in degrees.
        sample_tilt: Sample tilt from horizontal in degrees.
        quaternions: Quaternions for each crystalline orientation. Shape (n, 4)
        deformation_gradients: Deformation gradients for each crystalline orientation. Shape (n, 3, 3)
        master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
        master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)


    Returns:
        The projected master pattern. Shape (n, n_det_pixels)

    """
    # sanitize inputs
    if not pcs.ndim == 2 or not pcs.shape[1] == 3:
        raise ValueError(f"pcs must be shape (1, 3) or (n, 3) but got {pcs.shape}")
    if pcs.ndim == 1:
        pcs = pcs[None, :]
    if not quaternions.ndim == 2 or not quaternions.shape[1] == 4:
        raise ValueError(
            f"quaternions must be shape (1, 4) or (n, 4) but got {quaternions.shape}"
        )
    if quaternions.ndim == 1:
        quaternions = quaternions[None, :]
    if not deformation_gradients.ndim == 3 or not deformation_gradients.shape[1:] == (
        3,
        3,
    ):
        raise ValueError(
            f"deformation_gradients must be shape (1, 3, 3) or (n, 3, 3) but got {deformation_gradients.shape}"
        )
    if deformation_gradients.ndim == 2:
        deformation_gradients = deformation_gradients[None, ...]

    # check that the shapes are broadcastable
    if (
        (not pcs.shape[0] == quaternions.shape[0])
        and pcs.shape[0] != 1
        and quaternions.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: pcs shaped {pcs.shape} and quaternions {quaternions.shape}"
        )
    if (
        (not pcs.shape[0] == deformation_gradients.shape[0])
        and pcs.shape[0] != 1
        and deformation_gradients.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: pcs shaped {pcs.shape} and deformation gradient {deformation_gradients.shape}"
        )
    if (
        (not quaternions.shape[0] == deformation_gradients.shape[0])
        and quaternions.shape[0] != 1
        and deformation_gradients.shape[0] != 1
    ):
        raise ValueError(
            f"Not broadcastable: quaternions shaped {quaternions.shape} and deformation gradient {deformation_gradients.shape}"
        )

    if not master_pattern_MSLNH.ndim == 2:
        raise ValueError(
            f"master_pattern_MSLNH must be shape (H, W) but got {master_pattern_MSLNH.shape}"
        )
    if not master_pattern_MSLSH.ndim == 2:
        raise ValueError(
            f"master_pattern_MSLSH must be shape (H, W) but got {master_pattern_MSLSH.shape}"
        )

    # get direction cosines
    direction_cosines = average_pc(
        pcs,
        n_rows,
        n_cols,
        tilt,
        azimuthal,
        sample_tilt,
        signal_mask=signal_mask,
    )

    n_orientations = quaternions.shape[0]
    n_det_pixels = direction_cosines.shape[1]

    output = torch.empty(
        (n_orientations, n_det_pixels),
        dtype=master_pattern_MSLNH.dtype,
        device=master_pattern_MSLNH.device,
    )

    # rotate the outgoing vectors on the K-sphere according to the crystal orientations
    rotated_vectors = quaternion_apply(quaternions[:, None, :], direction_cosines)

    # apply the inverse of the deformation gradients to the rotated vectors
    rotated_vectors = torch.matmul(
        torch.inverse(deformation_gradients), rotated_vectors[:, :, :, None]
    ).squeeze(-1)

    # renormalize the rotated vectors
    rotated_vectors = rotated_vectors / torch.linalg.norm(
        rotated_vectors, dim=-1, keepdim=True
    )

    # mask for positive z component
    mask = rotated_vectors[..., 2] > 0

    # get the coordinates within the image square
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
