import torch
from torch import Tensor


@torch.jit.script
def theta_phi_to_xyz(theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates to cartesian coordinates.

    Args:
        theta (Tensor): shape (..., ) of polar declination angles
        phi (Tensor): shape (..., ) of azimuthal angles

    Returns:
        Tensor: torch tensor of shape (..., 3) containing the cartesian
        coordinates
    """
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=1,
    )


@torch.jit.script
def xyz_to_theta_phi(xyz: Tensor) -> Tensor:
    """
    Convert cartesian coordinates to latitude and longitude.

    Args:
        xyz (Tensor): torch tensor of shape (..., 3) of cartesian coordinates

    Returns:
        Tensor: torch tensor of shape (..., 2) of declination from z-axis and
        azimuthal angle

    """
    return torch.stack(
        (
            torch.atan2(torch.norm(xyz[:, :2], dim=1), xyz[:, 2]),
            torch.atan2(xyz[:, 1], xyz[:, 0]),
        ),
        dim=1,
    )
