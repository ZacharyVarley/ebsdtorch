"""

This module contains classes for managing the geometry of EBSD experiments. The
geometry is defined by the relationship between the detector and sample
coordinate reference frames. The PointEBSDGeometry class assumes that the sample
surface is a single point in 3D space. The true relationship between the
coordinate frames assuming the sample is perfectly planar, with it's step size
between dwell points exactly situated and known, is a 2D homography, an element
of SL(2), without the shear and anisotropic scaling of the plane: an 8-2=6
degree of freedom transformation. For ease of implementation, I chose to instead
use the SE(3) Lie Group and algebra to model the same transformation.

"""

from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from ebsdtorch.lie_algebra.se3 import se3_log_map_R_tvec, se3_exp_map


@torch.jit.script
def bruker_pattern_centers_to_SE3(
    pattern_centers: Tensor,
    sample_y_tilt_radians: Tensor,
    sample_x_tilt_radians: Tensor,
    detector_tilt_radians: Tensor,
    detector_shape: Tuple[int, int],
):
    """
    Convert pattern centers in Bruker coordinates to SE3 transformation matrix.

    Args:
        pattern_centers:
            The projection center(s). Shape (n_pcs, 3)
        sample_y_tilt_degrees:
            Tilts of the sample about the y-axis in degrees.
        sample_x_tilt_degrees:
            Tilts of the sample about the x-axis in degrees.
        detector_tilt_degrees:
            Declination tilts of the detector in degrees.
        detector_shape:
            Pattern shape in pixels, H x W with 'ij' indexing. Number
            of rows of pixels then number of columns of pixels.

    Returns:
        Rotation matrices (n_pcs, 3, 3) and translation vectors (n_pcs, 3).

    """

    pcx, pcy, pcz = torch.unbind(pattern_centers, dim=-1)
    rows, cols = detector_shape
    rows, cols = float(rows), float(cols)
    sy, sx, dt = sample_y_tilt_radians, sample_x_tilt_radians, detector_tilt_radians

    rotation_matrix = torch.stack(
        [
            -torch.sin(dt - sy),
            -torch.sin(sx) * torch.cos(dt - sy),
            torch.cos(sx) * torch.cos(dt - sy),
            torch.zeros_like(sx),
            torch.cos(sx),
            torch.sin(sx),
            -torch.cos(dt - sy),
            torch.sin(sx) * torch.sin(dt - sy),
            -torch.sin(dt - sy) * torch.cos(sx),
        ],
        dim=0,
    ).view(-1, 3, 3)

    translation_vector = torch.stack(
        [
            (
                pcx * cols * torch.sin(sx) * torch.cos(dt - sy)
                + pcy * rows * torch.sin(dt - sy)
                + pcz * rows * torch.cos(sx) * torch.cos(dt - sy)
                - torch.sin(sx) * torch.cos(dt - sy) / 2
                - torch.sin(dt - sy) / 2
            ),
            (
                -cols * pcx * torch.cos(sx)
                + pcz * rows * torch.sin(sx)
                + torch.cos(sx) / 2
            ),
            (
                -cols * pcx * torch.sin(sx) * torch.sin(dt - sy)
                + pcy * rows * torch.cos(dt - sy)
                - pcz * rows * torch.sin(dt - sy) * torch.cos(sx)
                + torch.sin(sx) * torch.sin(dt - sy) / 2
                - torch.cos(dt - sy) / 2
            ),
        ],
        dim=0,
    ).view(-1, 3)
    return rotation_matrix, translation_vector


class PointEBSDGeometry(Module):
    """
    The PointEBSDGeometry class assumes that the sample surface is a single
    point in 3D space. It manages the relationship between detector and sample
    coordinate reference frames with a single transformation matrix in SE3, or
    average projection center in the Bruker convention. It's internal
    representation is differentiable and can be optimized with gradient descent.

    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        geometry_mode: str = "se3",
        sample_x_tilt_deg: float = 0.0,
        sample_y_tilt_deg: float = 70.0,
        detector_tilt_deg: float = 0.0,
        pattern_center_guess: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
    ):
        """
        Args:
            detector_shape:
                Pattern shape in pixels, H x W with 'ij' indexing. Number of
                rows of pixels then number of columns of pixels.
            geometry_mode:
                The mode of the geometry. Options are "bruker_pc", and "se3".
                The bruker_pc assumes that the movement of the detector,
                changing the projection center are the only fittable parameters.
                The se3 mode assumes that the detector is free to rotate and
                translate in 3D space and uses the Lie algebra se(3) to optimize
                the geometry (default "se3") on the manifold.
            sample_x_tilt_degrees:
                Tilt of the sample about the x-axis in degrees (default 0).
            sample_y_tilt_degrees:
                Tilt of the sample about the y-axis in degrees (default 70).
            detector_tilt_degrees:
                Declination of the detector below the horizontal in degrees (default 0).
            pattern_center_guess:
                The initial guess for the pattern center. This pattern center is
                in the Bruker convention so it is implicitly specifying the
                pixel size in microns (default (0.5, 0.5, 0.5)). If you provide
                a pattern center implicitly carrying a certain pixel size and
                use this class with differently binned patterns, you will be
                initializing a very incorrect geometry.

        """

        super(PointEBSDGeometry, self).__init__()

        self.detector_shape = detector_shape
        self.geometry_mode = geometry_mode

        if self.geometry_mode == "se3":
            rotation_matrix, translation_vector = bruker_pattern_centers_to_SE3(
                torch.tensor(pattern_center_guess).view(1, 3),
                torch.tensor(sample_y_tilt_deg * (torch.pi / 180.0)),
                torch.tensor(sample_x_tilt_deg * (torch.pi / 180.0)),
                torch.tensor(detector_tilt_deg * (torch.pi / 180.0)),
                self.detector_shape,
            )
            se3 = se3_log_map_R_tvec(rotation_matrix, translation_vector)
            self.se3 = torch.nn.Parameter(se3.view(1, 6))
        elif self.geometry_mode == "bruker_pc":
            self.projection_center = torch.nn.Parameter(
                torch.tensor(pattern_center_guess).view(1, 3)
            )
            self.register_buffer(
                "sample_x_tilt_deg", torch.deg2rad(torch.tensor(sample_x_tilt_deg))
            )
            self.register_buffer(
                "sample_y_tilt_deg", torch.deg2rad(torch.tensor(sample_y_tilt_deg))
            )
            self.register_buffer(
                "detector_tilt_deg", torch.deg2rad(torch.tensor(detector_tilt_deg))
            )
        else:
            raise ValueError(
                f"Invalid geometry mode, must be 'bruker_pc' or 'se3', but got {geometry_mode}."
            )

    def forward(self) -> Tensor:
        """
        Forward pass for the AverageEBSDGeometry module.

        Returns:
            Rotation matrix and translation vector. Shapes (3, 3) and (3,)

        """
        # only one transformation matrix in this class so we can just return it
        if self.geometry_mode == "bruker_pc":
            rotation_matrix, translation_vector = bruker_pattern_centers_to_SE3(
                self.projection_center,
                self.sample_y_tilt_deg,
                self.sample_x_tilt_deg,
                self.detector_tilt_deg,
                self.detector_shape,
            )
            rotation_matrix = rotation_matrix[0]
            translation_vector = translation_vector[0]
        elif self.geometry_mode == "se3":
            rotation_matrix, translation_vector = se3_exp_map(self.se3.view(1, 6))
            rotation_matrix = rotation_matrix[0, :3, :3]
            translation_vector = translation_vector[0, :3]

        return rotation_matrix, translation_vector

    def transform(self, pixel_coordinates: Tensor) -> Tensor:
        """
        Transform detector coordinates in units of pixels to the sample reference frame.

        Args:
            pixel_coordinates:
                The pixel coordinates in the detector plane. Shape (..., 2).
                Where the z-coordinate is implicitly 0.

        Returns:
            The coordinates in the sample frame. Shape (..., 3)

        """
        # convert to homogeneous coordinates
        pixel_coordinates = torch.cat(
            [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
        )

        # call the forward pass to get the transformation matrix
        rotation_matrix, translation_vector = self()

        # apply the transformation
        sample_coordinates = (
            ((rotation_matrix @ pixel_coordinates[..., None]))
            + translation_vector[..., None]
        ).squeeze(-1)

        return sample_coordinates

    def backproject(self, sample_coordinates: Tensor) -> Tensor:
        """
        Backproject sample coordinates to the detector plane.

        Args:
            sample_coordinates: sample frame coordinates. Shape (..., 3)

        Returns:
            Coordinates in the detector plane. Shape (..., 2) in pixels.

        Notes:
            This class considers all patterns to have been observed at the
            sample reference frame origin (0, 0, 0). To backproject to the
            detector we take rays from the sample reference frame origin to the
            sample coordinates and find where they intersect the detector plane
            (z = 0) all within the detector frame of reference.

        """
        # call the forward pass to get the transformation matrix
        rotation_matrix, translation_vector = self()

        # find the origin in the detector frame
        origin_detector = (
            rotation_matrix.transpose(-1, -2) @ -translation_vector[..., None]
        ).squeeze(-1)

        # transform the sample coordinates to the detector frame
        detector_coords_3d = (
            (sample_coordinates - translation_vector) @ rotation_matrix
        ).squeeze(-1)

        # find the rays from the origin to the sample coordinates
        rays = detector_coords_3d - origin_detector

        # find the intersection with the detector plane
        t = -origin_detector[..., 2] / rays[..., 2]
        pixel_coordinates = origin_detector[..., :2] + t[..., None] * rays[..., :2]

        return pixel_coordinates
