"""
This is a class to represent the relative positioning of the detector plane and
sample reference frame. Normally, this would just be a simple reference change
matrix and its inverse. However, EBSD vendors reinvented the wheel and decided
to each use their own convoluted coordinate systems and conventions. Nice!

"""

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module, Parameter


class RigidPlanar(Module):
    """
    Class to represent the relative positioning of the detector and sample.

    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        pixel_shape: Tuple[float, float],
        sample_y_tilt_degrees: float = 70.0,
        sample_x_tilt_degrees: float = 0.0,
        detector_tilt_degrees: float = 0.0,
        binning_amounts: Union[Tuple[int, int], Tuple[float, float]] = (1, 1),
        pattern_center_guess: Optional[Tuple[float, float, float]] = None,
    ):
        """

        Args:
            detector_shape: Pattern shape in pixels, H x W with 'ij' indexing.
            pixel_shape: Size of the pixels in micrometers, (pixel_height, pixel_width).
            sample_y_tilt_degrees: Tilt of the sample about the y-axis in degrees.
            sample_x_tilt_degrees: Tilt of the sample about the x-axis in degrees.
            detector_tilt_degrees: Declination tilt of the detector in degrees.
            pattern_center_guess: Initial guess for the pattern center, if no pattern center is provided,
                the detector is simply assumed to be 1.5 centimeters from the sample with no lateral shift.
            binning_amounts: Amount of binning already applied to the patterns.
            signal_mask: Mask to apply to the experimental data. If None, no mask is applied.

        """

        super(RigidPlanar, self).__init__()

        # assert pixel_size is a tuple of two floats
        if not isinstance(pixel_shape, tuple) or len(pixel_shape) != 2:
            raise ValueError("pixel_size must be a tuple of two floats")
        # assert sample_y_tilt_degrees is a float
        if not isinstance(sample_y_tilt_degrees, float):
            raise ValueError("sample_y_tilt_degrees must be a float")
        # assert sample_x_tilt_degrees is a float
        if not isinstance(sample_x_tilt_degrees, float):
            raise ValueError("sample_x_tilt_degrees must be a float")
        # assert detector_tilt_degrees is a float
        if not isinstance(detector_tilt_degrees, float):
            raise ValueError("detector_tilt_degrees must be a float")
        # assert binning_amounts is a tuple of two floats
        if not isinstance(binning_amounts, tuple) or len(binning_amounts) != 2:
            raise ValueError("binning_amounts must be a tuple of two floats")

        # Must accept detector pixel locations with usual top left ij indexing
        n_rows, n_cols = detector_shape
        ingest_matrix = torch.eye(4, device="cpu")
        ingest_matrix[0, 0] = -1
        ingest_matrix[0, 3] = n_rows - 1

        # Convert to detector coordinates
        # Extract pcx, pcy, pcz from the pc tensor
        pcx_bruker, pcy_bruker, pcz_bruker = torch.unbind(
            torch.tensor(pattern_center_guess), dim=-1
        )
        pcx_delta = n_cols * (0.5 - pcx_bruker)
        pcy_delta = n_rows * (0.5 - pcy_bruker)
        pcz_delta = n_rows * pcz_bruker

        # find det_x and det_y also using matrix operations
        matrix_to_det = torch.zeros(4, 4, device="cpu")
        matrix_to_det[0, 0] = -1
        matrix_to_det[1, 1] = 1
        matrix_to_det[0, 3] = (n_rows - 1) * 0.5 + pcy_delta
        matrix_to_det[1, 3] = (1 - n_cols) * 0.5 + pcx_delta
        matrix_to_det[2, 3] = pcz_delta
        matrix_to_det[2, 2] = 1
        matrix_to_det[3, 3] = 1

        # now do the azimuthal rotation
        azimuthal_rad = torch.deg2rad(torch.tensor(sample_x_tilt_degrees))
        cos_omega = torch.cos(azimuthal_rad)
        sin_omega = torch.sin(azimuthal_rad)
        omega_rotation = torch.eye(4, device="cpu")
        omega_rotation[1, 1] = cos_omega
        omega_rotation[1, 2] = sin_omega
        omega_rotation[2, 1] = -sin_omega
        omega_rotation[2, 2] = cos_omega

        # now we do the major tilt rotation
        alpha_rad = torch.deg2rad(
            torch.tensor(detector_tilt_degrees - sample_y_tilt_degrees + 90.0)
        )
        cos_alpha = torch.cos(alpha_rad)
        sin_alpha = torch.sin(alpha_rad)

        alpha_rotation = torch.eye(4, device="cpu")
        alpha_rotation[0, 0] = cos_alpha
        alpha_rotation[0, 2] = sin_alpha
        alpha_rotation[2, 0] = -sin_alpha
        alpha_rotation[2, 2] = cos_alpha

        # find overall transformation matrix
        transformation = alpha_rotation @ omega_rotation @ matrix_to_det @ ingest_matrix

        # extract the rotation matrix and translation vector
        rotation_matrix = transformation[:3, :3]
        translation_vector = transformation[:3, 3]

        # register the parameters
        self.register_buffer("rotation_matrix", rotation_matrix)
        self.register_buffer("translation_vector", translation_vector)

    def forward(self, pixel_coordinates: Tensor) -> Tensor:
        """
        Forward pass for the RigidPlanar module.

        Args:
            pixel_coordinates: The pixel coordinates in the detector plane. Shape (n_pixels, 2)

        Returns:
            The sample coordinates in the sample frame. Shape (n_pixels, 3)

        """
        # convert to homogeneous coordinates
        pixel_coordinates = torch.cat(
            [pixel_coordinates, torch.zeros_like(pixel_coordinates[:, 0:1])], dim=-1
        )

        # apply the transformation
        sample_coordinates = (
            self.rotation_matrix @ pixel_coordinates.t()
            + self.translation_vector[:, None]
        ).t()

        return sample_coordinates

    def inverse(self, sample_coordinates: Tensor) -> Tensor:
        """
        Inverse pass for the RigidPlanar module.

        Args:
            sample_coordinates: The sample coordinates in the sample frame. Shape (n_pixels, 3)

        Returns:
            The pixel coordinates in the detector plane. Shape (n_pixels, 2)

        """
        # convert to homogeneous coordinates
        det_coords = (
            self.rotation_matrix.t()
            @ (sample_coordinates.t() - self.translation_vector[:, None])
        ).t()

        return det_coords
