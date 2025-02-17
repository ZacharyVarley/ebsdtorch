"""

This module contains classes for managing the geometry of EBSD experiments. The
geometry is defined by the relationship between the detector and sample
coordinate reference frames. The PointEBSDGeometry class assumes that the sample
surface is a single point in 3D space. The true relationship between the
coordinate frames assuming the sample is perfectly planar, is a 2D homography,
an element of SL(2). If the angle between the unit vectors spanning each frame
are exactly known, then there is neither shear nor anisotropic stretching of the
plane, reducing the transformation by 2 degree of freedom from 8 to 6. For ease
of implementation, I chose to instead use the SE(3) Lie Group and algebra to
model the same transformation.

I made a still unresolved error in naively using Rodrigues' formula and its
extension to 3D Euclidean motions to model the transformation. The error is that
the naive implementation in PyTorch has known extreme numerical instability.

"""

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from ebsdtorch.lie_algebra.se3 import (
    se3_log_map_split,
    se3_exp_map_split,
    se3_exp_map_om,
)


@torch.jit.script
def bruker_geometry_to_SE3(
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
    Convert SE3 transformation back to Bruker geometry (invalid if z-axis rotation was optimized).

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


class EBSDGeometry(Module):
    """
    Args:
        :detector_shape:
            Pattern shape in pixels, H x W with 'ij' indexing. Number of rows of
            pixels then number of columns of pixels.
        :tilts_degrees:
            Tilt of the sample about the x-axis in degrees (default 0). Tilt of
            the sample about the y-axis in degrees (default 70). Declination of
            the detector below the horizontal in degrees (default 0).
        :proj_center:
            The initial guess for the pattern center. This pattern center is in
            the Bruker convention so it is implicitly specifying the pixel size
            in microns (default (0.5, 0.5, 0.5)).
        :se3_vector:
            The initial guess for the SE3 transformation specified as a Lie
            algebra vector. The vector is in the form (rx, ry, rz, tx, ty, tz).
        :with_se3:
            Whether to fit an SE3 matrix on top of the pattern center
            parameterization.
        :opt_rots:
            Which rotations to optimize. Tuple of booleans (yz rot, xz rot, xy
            rot). Default is (False, False, False) - fitting a pattern center
            only.
        :opt_shifts:
            Which shifts to optimize. Tuple of booleans (x shift, y shift, z
            shift).

    Notes:
        The opt_rots and opt_shifts flags allow fine-grained control over which components
        of the SE3 matrix are optimized. A simple pattern center can only be returned if
        none of the rotations for the SE3 matrix are optimized. Optimizing the translation
        components of the SE3 matrix alone is equivalent to optimizing the pattern center.

    The EBSDGeometry class manages the spatial relationship between detector and
    sample coordinate reference frames. This can be represented as a triplet of
    pattern center coordinates or as an SE3 transformation matrix. The SE3 matrix
    does not assume that the relative pose of the detector is perfectly known. It
    introduces a free rotation in addition to the pattern center translation.

    The indended use is to fit a projection center via backpropagation or global
    optimization and then to fine tune the geometry with the SE3 matrix with either
    1, 2, or 3 degrees of freedom for it's pose. The SE3 matrix is applied before
    the pattern center so it is in the detector frame of reference.

    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        tilts_degrees: Optional[Tuple[float, float, float]] = (0.0, 70.0, 0.0),
        proj_center: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        se3_vector: Optional[Tensor] = None,
        opt_rots: Tuple[bool, bool, bool] = (False, False, False),
        opt_shifts: Tuple[bool, bool, bool] = (False, False, False),
    ):

        super(EBSDGeometry, self).__init__()

        self.detector_shape = detector_shape

        # convert the tilts to radians
        tilts_deg = tuple(torch.tensor(tilt) for tilt in tilts_degrees)
        self.register_buffer("sample_x_tilt_deg", tilts_deg[0])
        self.register_buffer("sample_y_tilt_deg", tilts_deg[1])
        self.register_buffer("detector_tilt_deg", tilts_deg[2])

        # convert the projection center to a un-optimized SE3 matrix
        rot, translation = bruker_geometry_to_SE3(
            torch.tensor(proj_center).view(1, 3),
            self.detector_tilt_deg - self.sample_y_tilt_deg,
            self.sample_x_tilt_deg,
            self.detector_shape,
        )
        pc_SE3 = torch.zeros(1, 4, 4)
        pc_SE3[..., :3, :3] = rot
        pc_SE3[..., :3, 3] = translation
        pc_SE3[..., 3, 3] = 1.0
        self.register_buffer("pc_SE3_matrix", pc_SE3)

        # convert the SE3 matrix to a parameter
        if se3_vector is None:
            se3_vector = torch.zeros(1, 6)
            self.se3_vector = torch.nn.Parameter(se3_vector)
        else:
            self.se3_vector = torch.nn.Parameter(se3_vector.view(1, 6))

        # make a mask from the opt_rots and opt_shifts
        rotation_mask = torch.tensor(opt_rots).bool()
        translation_mask = torch.tensor(opt_shifts).bool()
        self.register_buffer("mask", torch.cat([rotation_mask, translation_mask]))
        # if the mask has no True values, set opt_se3 to False
        self.opt_se3 = torch.any(self.mask).item()

    def set_optimizable_se3_params(
        self,
        opt_rots: Optional[Tuple[bool, bool, bool]] = None,
        opt_shifts: Optional[Tuple[bool, bool, bool]] = None,
    ):
        """
        Set which parameters are optimizable.

        Args:
            opt_rots:
                Which rotations to optimize. Tuple of booleans (yz rot, xz rot, xy rot).
            opt_shifts:
                Which shifts to optimize. Tuple of booleans (x shift, y shift, z shift).

        """
        # set the masks
        if opt_rots is not None:
            self.mask[:3] = torch.tensor(opt_rots).bool()
        if opt_shifts is not None:
            self.mask[3:] = torch.tensor(opt_shifts).bool()

        # update the opt_se3 flag
        self.opt_se3 = torch.any(self.mask).item()

    def get_detector2sample(self) -> Tuple[Tensor, Tensor]:
        """
        Get the SE3 transformation matrix from detector to sample.

        Returns:
            Rotation matrix and translation vector. Shapes (3, 3) and (3,)

        """
        # check if using a tunable SE3 matrix
        if self.opt_se3:
            # get the SE3 matrix
            tune_se3 = se3_exp_map_om(self.se3_vector * self.mask).squeeze(0)
        else:
            with torch.no_grad():
                tune_se3 = se3_exp_map_om(self.se3_vector * self.mask).squeeze(0)

        # find the combined transform with the tunable one first
        detector_2_sample = self.pc_SE3_matrix @ tune_se3

        # split the SE3 matrix into rotation and translation
        rotation_matrix = detector_2_sample[..., :3, :3]
        translation_vector = detector_2_sample[..., :3, 3]

        return rotation_matrix, translation_vector

    def project2sample(
        self,
        pixel_coordinates: Tensor,
        scan_coordinates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Project detector coordinates to the sample reference frame.

        Args:
            pixel_coordinates:
                The pixel coordinates in the detector plane. Shape (..., 2).
                Where the z-coordinate is implicitly 0.
            scan_coordinates:
                The offsets of the scan in microns. Shape (..., 2) with (i, j) indexing.

        pixel_coordinates and scan_coordinates are broadcastable:

        If we have 100x100 detector with 10000 separate pixel coordinates with
        placeholder dimensions: (100, 100, 1, 1, 2) and want to find the
        projected coordinates for over each choice of my 100 scan positions: (1,
        1, 10, 10, 2).

        The result would be (100, 100, 10, 10, 2) with the 10000 pixel
        coordinates projected for each of the 100 scan positions.

        Returns:
            The coordinates in the sample frame. Shape (..., 3). If
            scan_coordinates are provided then the coordinates are manually
            shifted by the scan offsets.

        Notes:
            If the scan position was large and in the bottom right away from the
            origin then the relative coordinates would tend negative - "behind"
            the origin - in both X and Y (Kikuchipy convention).

        See Also:
            https://kikuchipy.org/en/stable/tutorials/reference_frames.html

        """

        # get the transformation matrix
        rotation_matrix, translation_vector = self.get_detector2sample()

        # convert to 3D coordinates
        pixel_coordinates = torch.cat(
            [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
        )

        # apply the transformation
        sample_coordinates = (
            ((rotation_matrix @ pixel_coordinates[..., None]))
            + translation_vector[..., None]
        ).squeeze(-1)

        # if scan offsets are provided, apply them
        if scan_coordinates is not None:
            scan_coordinates_3D = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            sample_coordinates = sample_coordinates - scan_coordinates_3D

        return sample_coordinates

    def backproject2detector(
        self,
        ray_directions: Tensor,
        scan_coordinates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Backproject ray directions to the detector plane. By default the origin
        of the sample reference frame is assumed to be the origin for all rays.
        Sample coordinates are given in microns but the results are the same up
        to a scale factor due to the projective nature of the transformation.

        Providing the scan coordinates will define the ray origins a small way
        away from the sample reference frame origin. This is useful for
        backprojecting to the detector plane with a single transformation matrix
        given a firm belief about the relative scan point positions.

        Args:
            ray_directions: sample frame coordinates. Shape (..., 3).
                Technically given in microns but projection is identical up to a scale factor.
            scan_coords: offsets of the scan in microns. Shape (..., 2) with (i, j) indexing.

        Returns:
            Coordinates in the detector plane. Shape (..., 2) in pixels.

        """

        # get the transformation matrix
        rotation_matrix, translation_vector = self.get_detector2sample()

        if scan_coordinates is not None:
            # add the scan coordinates to the ray directions to get the sample coordinates
            ray_origins_sample = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            # apply the inverse transformation
            ray_origins_detector = (
                -translation_vector + ray_origins_sample
            ) @ rotation_matrix
        else:
            # if no scan points given, assume all rays originate from the origin
            ray_origins_detector = -translation_vector @ rotation_matrix

        # situate the rays as outgoing from the actual origins
        ray_tips_sample = ray_directions

        if scan_coordinates is not None:
            scan_coordinates_3D = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            ray_tips_sample = ray_tips_sample + scan_coordinates_3D

        # transform the tip coordinates to the detector frame
        ray_tips_detector = (ray_tips_sample - translation_vector) @ rotation_matrix

        # find the rays from the origins out towards the tips all in the detector frame
        rays_detector = ray_tips_detector - ray_origins_detector

        # find the intersection with the detector plane
        t = -ray_origins_detector[..., 2] / rays_detector[..., 2]
        pixel_coordinates = (
            ray_origins_detector[..., :2] + t[..., None] * rays_detector[..., :2]
        )

        return pixel_coordinates

    def get_coords_sample_frame(
        self,
        binning: Tuple[int, int],
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> Tensor:
        """
        Get the coordinates of each of the detector pixels in the sample reference frame.

        Args:
            binning:
                The binning of the detector (factor along detector H, factor along detector W).
            dtype:
                The data type of the coordinates. Default is torch.float32.

        Returns:
            Detector pixel coordinates in the sample reference frame. Shape (h, w, 3).

        Notes:
            Norming the coordinates will yield rays from the sample reference
            frame origin to each pixel in the detector plane. These are returned
            unnormed in microns so that each scan location can be used as a ray
            origin. This is handled by the ExperimentPatterns class which modifies
            rays by individual orientation, F-matrix, and scan position.

        """

        # check the binning evenly divides the detector shape
        if self.detector_shape[0] % binning[0] != 0:
            raise ValueError(
                f"A height binning of {binning[0]} does not evenly divide the detector height of {self.detector_shape[0]}."
            )
        if self.detector_shape[1] % binning[1] != 0:
            raise ValueError(
                f"A width binning of {binning[1]} does not evenly divide the detector width of {self.detector_shape[1]}."
            )

        # get binned shape
        binned_shape = (
            self.detector_shape[0] // binning[0],
            self.detector_shape[1] // binning[1],
        )

        binning_fp = (float(binning[0]), float(binning[1]))

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
                    self.detector_shape[0] - (0.5 * (binning_fp[0] - 1)),
                    binned_shape[0],
                    dtype=dtype,
                    device=self.pc_SE3_matrix.device,  # is always on model device
                ),
                torch.linspace(
                    0.5 * (binning_fp[1] - 1),
                    self.detector_shape[1] - (0.5 * (binning_fp[1] - 1)),
                    binned_shape[1],
                    dtype=dtype,
                    device=self.pc_SE3_matrix.device,  # is always on model device
                ),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)

        return self.project2sample(pixel_coordinates)


# # test the forward and inverse projection for 3 example pixel coordinates
# # start without any scan offsets

# # create the geometry
# geometry = EBSDGeometry((128, 128))

# # create some pixel coordinates
# pixel_coordinates = torch.tensor(
#     [[64.0, 64.0], [32.0, 32.0], [96.0, 96.0]], dtype=torch.float32
# )

# # forward projection
# sample_coordinates = geometry.project2sample(pixel_coordinates)

# # backprojection
# pixel_coordinates_bp = geometry.backproject2detector(sample_coordinates)

# # print the results
# print("Pixel Coordinates:")
# print(pixel_coordinates)
# print("Sample Coordinates:")
# print(sample_coordinates)
# print("Backprojected Pixel Coordinates:")
# print(pixel_coordinates_bp)

# # now test broadcasting with scan offsets
# # create the scan coordinates of shape (1, 1, 10, 10, 2)
# scan_coordinates = torch.stack(
#     torch.meshgrid(
#         torch.linspace(-5, 5, 10),
#         torch.linspace(-5, 5, 10),
#         indexing="ij",
#     ),
#     dim=-1,
# ).view(1, 100, 2)

# # now create the pixel coordinates of shape (5, 5, 1, 1, 2)
# pixel_coordinates = torch.stack(
#     torch.meshgrid(
#         torch.linspace(0, 128, 5),
#         torch.linspace(0, 128, 5),
#         indexing="ij",
#     ),
#     dim=-1,
# ).view(25, 1, 2)

# # forward projection which does the first broadcast
# sample_coordinates = geometry.project2sample(pixel_coordinates, scan_coordinates)

# # backprojection
# pixel_coordinates_bp = geometry.backproject2detector(
#     sample_coordinates, scan_coordinates
# )

# # reforward projection
# sample_coordinates_bp = geometry.project2sample(pixel_coordinates_bp, scan_coordinates)

# # print the results
# print("Pixel Coordinates:")
# print(pixel_coordinates)
# print("Sample Coordinates:")
# print(sample_coordinates)
# print("Backprojected Pixel Coordinates:")
# print(pixel_coordinates_bp)
# print("Reforwarded Sample Coordinates:")
# print(sample_coordinates_bp)

# # check that the refowarded sample coordinates are the same as the original
# assert torch.allclose(sample_coordinates, sample_coordinates_bp, atol=1e-6)
