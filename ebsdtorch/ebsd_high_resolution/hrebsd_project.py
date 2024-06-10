# """
# This file implements conventional HREBSD projection, which is the projection of
# EBSD patterns onto a detector for a given set of crystalline orientations,
# pattern projection center, and deformation gradient matrices. In the future, I
# would like to define the projection center in terms of a camera matrix model as
# is done in the rest of computer vision. This would allow for geometrically
# consistent projection of patterns onto a detector from different sample
# locations.

# """

# from typing import Optional, Tuple
# import torch
# from torch import Tensor
# from torch.nn import Module

# from ebsdtorch.geometry.average_pc import average_pc
# from ebsdtorch.geometry.rigid_planar import RigidPlanar
# from ebsdtorch.s2_and_so3.sphere import rosca_lambert
# from ebsdtorch.s2_and_so3.quaternions import qu_apply


# @torch.jit.script
# def project_HREBSD_pattern(
#     pcs: Tensor,
#     n_rows: int,
#     n_cols: int,
#     tilt: float,
#     azimuthal: float,
#     sample_tilt: float,
#     quaternions: Tensor,
#     deformation_gradients: Tensor,
#     master_pattern_MSLNH: Tensor,
#     master_pattern_MSLSH: Tensor,
#     signal_mask: Optional[Tensor] = None,
# ) -> Tensor:
#     """

#     This function projects the master pattern onto the detector for each crystalline orientation.
#     It is called "paired" because each orientation is paired with another pattern center triplet of
#     direction cosines. This function would make sense to use in the context of indexing a map of
#     EBSD patterns. Each crystalline orientation would be paired with a pattern center triplet that
#     corresponds to that location on the sample.

#     Args:
#         pcs: Projection centers. Shape (n, 3)
#         n_rows: Number of detector rows.
#         n_cols: Number of detector columns.
#         tilt: Detector tilt from horizontal in degrees.
#         azimuthal: Sample tilt about the sample RD axis in degrees.
#         sample_tilt: Sample tilt from horizontal in degrees.
#         quaternions: Quaternions for each crystalline orientation. Shape (n, 4)
#         deformation_gradients: Deformation gradients for each crystalline orientation. Shape (n, 3, 3)
#         master_pattern_MSLNH: modified Square Lambert projection for the Northern Hemisphere. Shape (H, W)
#         master_pattern_MSLSH: modified Square Lambert projection for the Southern Hemisphere. Shape (H, W)


#     Returns:
#         The projected master pattern. Shape (n, n_det_pixels)

#     """
#     # sanitize inputs
#     if not pcs.ndim == 2 or not pcs.shape[1] == 3:
#         raise ValueError(f"pcs must be shape (1, 3) or (n, 3) but got {pcs.shape}")
#     if pcs.ndim == 1:
#         pcs = pcs[None, :]
#     if not quaternions.ndim == 2 or not quaternions.shape[1] == 4:
#         raise ValueError(
#             f"quaternions must be shape (1, 4) or (n, 4) but got {quaternions.shape}"
#         )
#     if quaternions.ndim == 1:
#         quaternions = quaternions[None, :]
#     if not deformation_gradients.ndim == 3 or not deformation_gradients.shape[1:] == (
#         3,
#         3,
#     ):
#         raise ValueError(
#             f"deformation_gradients must be shape (1, 3, 3) or (n, 3, 3) but got {deformation_gradients.shape}"
#         )
#     if deformation_gradients.ndim == 2:
#         deformation_gradients = deformation_gradients[None, ...]

#     # check that the shapes are broadcastable
#     if (
#         (not pcs.shape[0] == quaternions.shape[0])
#         and pcs.shape[0] != 1
#         and quaternions.shape[0] != 1
#     ):
#         raise ValueError(
#             f"Not broadcastable: pcs shaped {pcs.shape} and quaternions {quaternions.shape}"
#         )
#     if (
#         (not pcs.shape[0] == deformation_gradients.shape[0])
#         and pcs.shape[0] != 1
#         and deformation_gradients.shape[0] != 1
#     ):
#         raise ValueError(
#             f"Not broadcastable: pcs shaped {pcs.shape} and deformation gradient {deformation_gradients.shape}"
#         )
#     if (
#         (not quaternions.shape[0] == deformation_gradients.shape[0])
#         and quaternions.shape[0] != 1
#         and deformation_gradients.shape[0] != 1
#     ):
#         raise ValueError(
#             f"Not broadcastable: quaternions shaped {quaternions.shape} and deformation gradient {deformation_gradients.shape}"
#         )

#     if not master_pattern_MSLNH.ndim == 2:
#         raise ValueError(
#             f"master_pattern_MSLNH must be shape (H, W) but got {master_pattern_MSLNH.shape}"
#         )
#     if not master_pattern_MSLSH.ndim == 2:
#         raise ValueError(
#             f"master_pattern_MSLSH must be shape (H, W) but got {master_pattern_MSLSH.shape}"
#         )

#     # get direction cosines
#     direction_cosines = average_pc(
#         pcs,
#         n_rows,
#         n_cols,
#         tilt,
#         azimuthal,
#         sample_tilt,
#         signal_mask=signal_mask,
#     )

#     n_orientations = quaternions.shape[0]
#     n_det_pixels = direction_cosines.shape[1]

#     output = torch.empty(
#         (n_orientations, n_det_pixels),
#         dtype=master_pattern_MSLNH.dtype,
#         device=master_pattern_MSLNH.device,
#     )

#     # rotate the outgoing vectors on the K-sphere according to the crystal orientations
#     rotated_vectors = qu_apply(quaternions[:, None, :], direction_cosines)

#     # apply the inverse of the deformation gradients to the rotated vectors
#     rotated_vectors = torch.matmul(
#         torch.inverse(deformation_gradients), rotated_vectors[:, :, :, None]
#     ).squeeze(-1)

#     # renormalize the rotated vectors
#     rotated_vectors = rotated_vectors / torch.linalg.norm(
#         rotated_vectors, dim=-1, keepdim=True
#     )

#     # mask for positive z component
#     mask = rotated_vectors[..., 2] > 0

#     # get the coordinates within the image square
#     coords_within_square = rosca_lambert(rotated_vectors)

#     # where the z component is positive, use the Northern Hemisphere projection
#     output[mask] = torch.nn.functional.grid_sample(
#         master_pattern_MSLNH[None, None, ...],
#         coords_within_square[mask][None, None, :],
#         align_corners=True,
#     ).squeeze()

#     # where the z component is negative, use the Southern Hemisphere projection
#     output[~mask] = torch.nn.functional.grid_sample(
#         master_pattern_MSLSH[None, None, ...],
#         coords_within_square[~mask][None, None, :],
#         align_corners=True,
#     ).squeeze()

#     return output


# class HREBSDProjPerPC(Module):
#     """
#     This class contains the functions to project EBSD patterns onto a detector for a given set of
#     crystalline orientations, pattern projection center(s), and deformation gradient matrices.

#     Args:
#         n_rows_per_pattern: Number of rows in each pattern
#         n_cols_per_pattern: Number of columns in each pattern
#         binning_amounts: Binning factors of form (binning_rows, binning_cols)
#         master_pattern_MSLNH: (H, W) modified Square Lambert projection (Northern Hemisphere)
#         master_pattern_MSLSH: (H, W) modified Square Lambert projection (Southern Hemisphere)
#         fit_quaternions: fit orientations for each pattern
#         pattern_center_mode: string either" single" or "multi"
#         fit_F_matrix: fit individual deformation gradient matrices for each pattern
#         quats_init: If None, then identity rotation for all, otherwise (1, 4) or (n, 4) tensor
#         pcs_init: If None, then (0.5, 0.5, 0.5) Bruker convention, otherwise (3,), (1, 3) or (n, 3) tensor
#         F_matrix_init: If None, then identity matrix, otherwise (3, 3), (1, 3, 3) or (n, 3, 3) tensor

#     """

#     def __init__(
#         self,
#         n_rows_per_pattern: int,
#         n_cols_per_pattern: int,
#         n_patterns: int,
#         binning_amounts: Tuple[int, int],
#         master_pattern_MSLNH: Tensor,
#         master_pattern_MSLSH: Tensor,
#         fit_quaternions: bool = False,
#         fit_F_matrix: bool = False,
#         pattern_center_mode: str = "single",
#         quats_init: Optional[Tensor] = None,
#         pcs_init: Optional[Tensor] = None,
#         F_matrix_init: Optional[Tensor] = None,
#     ):
#         super(HREBSDProjPerPC, self).__init__()

#         # check the arguments
#         if not isinstance(n_rows_per_pattern, int):
#             raise ValueError(
#                 "n_rows_per_pattern must be an integer but got {n_rows_per_pattern}"
#             )
#         if not isinstance(n_cols_per_pattern, int):
#             raise ValueError(
#                 "n_cols_per_pattern must be an integer but got {n_cols_per_pattern}"
#             )
#         if not isinstance(n_patterns, int):
#             raise ValueError("n_patterns must be an integer but got {n_patterns}")
#         if not isinstance(binning_amounts, tuple) or len(binning_amounts) != 2:
#             raise ValueError("binning_amounts must be a tuple of two integers")
#         if not isinstance(master_pattern_MSLNH, Tensor):
#             raise ValueError("master_pattern_MSLNH must be a tensor")
#         if not isinstance(master_pattern_MSLSH, Tensor):
#             raise ValueError("master_pattern_MSLSH must be a tensor")

#         # initialize buffer for quats
#         if quats_init is None:
#             if fit_quaternions:
#                 quats = torch.tensor(
#                     [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
#                 ).repeat(n_patterns, 1)
#             else:
#                 quats = None
#         else:
#             # check that the shape of the quaternions is either 1D or 2D
#             if not (
#                 (quats_init.ndim == 1 and quats_init.shape[1] == 4)
#                 or (
#                     quats_init.ndim == 2
#                     and quats_init.shape[1] == 4
#                     and (quats_init.shape[0] == n_patterns or quats_init.shape[0] == 1)
#                 )
#             ):
#                 raise ValueError(
#                     f"quats_init must be a (4,), (1, 4), or (n_patterns, 4) shaped tensor, but got {quats_init.shape}"
#                 )
#             # if 1 quaternion is provided, repeat it for all patterns and inform the user
#             if quats_init.shape[0] == 1:
#                 quats = quats_init.repeat(n_patterns, 1)
#                 print(
#                     "HREBSDProjectPerPC Initialization: 1 quaternion provided for all patterns."
#                 )
#             else:
#                 quats = quats_init

#         if quats is not None:
#             self.register_buffer("quats", quats)

#         # initialize buffer for deformation gradients
#         if F_matrix_init is None:
#             if fit_F_matrix:
#                 F_matrix = torch.eye(3, dtype=torch.float32).repeat(n_patterns, 1, 1)
#             else:
#                 F_matrix = None
#         else:
#             # check that the shape of the F matrices is either 2D or 3D
#             if not (
#                 (F_matrix_init.ndim == 2 and F_matrix_init.shape[1:] == (3, 3))
#                 or (
#                     F_matrix_init.ndim == 3
#                     and F_matrix_init.shape[1:] == (3, 3)
#                     and (
#                         F_matrix_init.shape[0] == n_patterns
#                         or F_matrix_init.shape[0] == 1
#                     )
#                 )
#             ):
#                 raise ValueError(
#                     f"F_matrix_init must be a (3, 3), (1, 3, 3), or (n_patterns, 3, 3) shaped tensor, but got {F_matrix_init.shape}"
#                 )
#             # if 1 F matrix is provided, repeat it for all patterns and inform the user
#             if F_matrix_init.shape[0] == 1:
#                 F_matrix = F_matrix_init.repeat(n_patterns, 1, 1)
#                 print(
#                     "HREBSDProjectPerPC Initialization: 1 F matrix provided for all patterns."
#                 )
#             else:
#                 F_matrix = F_matrix_init

#         if F_matrix is not None:
#             self.register_buffer("F_matrix", F_matrix)

#         # initialize buffer for pattern centers
#         if pcs_init is None:
#             if pattern_center_mode == "single":
#                 pcs = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32).view(1, 3)
#             elif pattern_center_mode == "multi":
#                 pcs = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32).repeat(
#                     n_patterns, 1
#                 )
#             else:
#                 raise ValueError(
#                     f"pattern_center_mode must be either 'single' or 'multi' but got {pattern_center_mode}"
#                 )
#         else:
#             # check that the shape of the pcs is either 1D or 2D
#             if not (
#                 (pcs_init.ndim == 1 and pcs_init.shape[0] == 3)
#                 or (
#                     pcs_init.ndim == 2
#                     and pcs_init.shape[1] == 3
#                     and (pcs_init.shape[0] == n_patterns or pcs_init.shape[0] == 1)
#                 )
#             ):
#                 raise ValueError(
#                     f"pcs_init must be a (3,), (1, 3), or (n_patterns, 3) shaped tensor, but got {pcs_init.shape}"
#                 )
#             # if 1 pcs is provided, repeat it for all patterns depending on the mode and inform the user
#             if pcs_init.shape[0] == 1:
#                 if pattern_center_mode == "single":
#                     pcs = pcs_init.repeat(1, 1)
#                     print(
#                         "HREBSDProjectPerPC Initialization: 1 pcs fitted for all patterns."
#                     )
#                 elif pattern_center_mode == "multi":
#                     pcs = pcs_init.repeat(n_patterns, 1)
#                     print(
#                         "HREBSDProjectPerPC Initialization: 1 pcs provided as initialization for all patterns."
#                     )
#             else:
#                 pcs = pcs_init

#         self.register_buffer("pcs", pcs)

#         # register the master patterns after checking their shapes
#         if not master_pattern_MSLNH.ndim == 2:
#             raise ValueError(
#                 f"master_pattern_MSLNH must be shape (H, W) but got {master_pattern_MSLNH.shape}"
#             )
#         if not master_pattern_MSLSH.ndim == 2:
#             raise ValueError(
#                 f"master_pattern_MSLSH must be shape (H, W) but got {master_pattern_MSLSH.shape}"
#             )

#         self.register_buffer("master_pattern_MSLNH", master_pattern_MSLNH)
#         self.register_buffer("master_pattern_MSLSH", master_pattern_MSLSH)

#     def forward(
#         self,
#     ) -> Tensor:
#         """
#         This function projects the master pattern onto the detector for each crystalline orientation.

#         Returns:
#             The projected master pattern. Shape (n, n_det_pixels)

#         Notes:

#         step 1: use the projection center(s) to get the direction cosines
#         step 2: rotate the outgoing vectors on the K-sphere according to the crystal orientations
#             We use an exponentiated axis-angle vector with a 3x3 rotation matrix for stability.
#         step 3: apply the inverse of the deformation gradients to the rotated vectors
#         step 4: renormalize the rotated ellipsoid vectors back to the K-sphere
#         step 5: do the projection
#         """
