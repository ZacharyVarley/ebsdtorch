import numpy as np
from typing import Optional, Tuple, Union, List
import torch


@torch.jit.script
class EBSDGeometry(torch.nn.Module):
    r"""An EBSD geometry class for storing the detector and sample
    information, and for converting between different coordinate
    reference frames. The sample plane is assumed to be at the center
    of the world coordinate system with the positive x-axis pointing
    towards the right, the positive y-axis pointing upwards, and the
    positive z-axis pointing towards the detector. The detector plane
    has the exact same convention as the sample plane with the positive
    z-axis continuing to point away from the sample plane. Both origins
    are at the center of the sample plane and detector plane. This
    means that the translation between origins does not depend on the
    rotation of the planes relative to each other.

    Parameters
    ----------
    detector_shape
        The shape of the detector in pixels. This is a tuple of the
        form (height, width).
    detector_pixel_size
        The size of the detector pixels in microns. This is either
        a scalar or a tuple of the form (height, width).
    sample_shape
        The shape of the sample in pixels. This is a tuple of the
        form (height, width).
    sample_pixel_size
        The size of the sample pixels in microns. This is either
        a scalar or a tuple of the form (height, width).
    sample_tilt_estimate
        An estimate of the sample tilt in degrees. This is either
        a scalar in degrees.
    sample_distance_estimate
        An estimate of the sample to detector distance in microns.
        This is either a scalar in microns.

    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        detector_pixel_size: Tuple[float, float],
        sample_shape: Tuple[int, int],
        sample_pixel_size: Tuple[float, float],
        sample_tilt_estimate: float = 70.0,
        sample_distance_estimate: float = 15000.0,
    ):
        super().__init__()
        self.detect_shape = detector_shape
        self.detect_ps = detector_pixel_size
        self.sample_shape = sample_shape
        self.sample_ps = sample_pixel_size

        # make detector coordinates (in um)
        d_coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, self.detect_shape[0] - 1, self.detect_shape[0])
                * self.detect_ps[0],
                torch.linspace(0, self.detect_shape[1] - 1, self.detect_shape[1])
                * self.detect_ps[1],
            ),
            dim=-1,
        )
        self.register_buffer("d_coords", d_coords)

        # make sample coordinates (in um)
        s_coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, self.sample_shape[0] - 1, self.sample_shape[0])
                * self.sample_ps[0],
                torch.linspace(0, self.sample_shape[1] - 1, self.sample_shape[1])
                * self.sample_ps[1],
            ),
            dim=-1,
        )
        self.register_buffer("s_coords", s_coords)

        # make detector to sample rotation matrix (+70 degrees in the xz plane)
        self.d2s_rotation = torch.tensor(
            [
                [
                    torch.cos(sample_tilt_estimate * np.pi / 180.0),
                    0,
                    torch.sin(sample_tilt_estimate * np.pi / 180.0),
                ],
                [0, 1, 0],
                [
                    -torch.sin(sample_tilt_estimate * np.pi / 180.0),
                    0,
                    torch.cos(sample_tilt_estimate * np.pi / 180.0),
                ],
            ]
        )

        # make detector to sample translation vector (in um)
        self.d2s_translation = torch.tensor([[0], [0], [sample_distance_estimate]])
