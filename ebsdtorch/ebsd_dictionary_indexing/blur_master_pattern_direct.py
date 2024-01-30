"""

Blur the master pattern given a simple backscatter electron inner and outer
detector angle, and distance from the sample. This is an extremely slow method
computed in direct space. It is only meant for testing purposes.

"""

import torch
from ebsdtorch.patterns.square_hemisphere_bijection import (
    square_lambert,
    inv_square_lambert,
)
from ebsdtorch.s2_and_so3.orientations import (
    quaternion_apply,
    quaternion_rotate_sets_sphere,
)
from ebsdtorch.s2_and_so3.sampling import s2_fibonacci_lattice
from ebsdtorch.s2_and_so3.orientations import xyz_to_theta_phi


@torch.jit.script
def annulus_bse_detector(
    square_lambert_mp: torch.Tensor,
    inner_opening_angle: float,
    outer_opening_angle: float,
    n_s2_points: int = 200000,
    limit_GB: int = 4,
) -> torch.Tensor:
    """Compute the area of a detector annulus in BSE coordinates.

    Parameters
    ----------
    inner_opening_theta : torch.Tensor
        The inner radius of the annulus in radians
    outer_opening_theta : torch.Tensor
        The outer radius of the annulus in radians
    square_lambert_mp : torch.Tensor
        The modified square Lambert projection of electron channeling sphere. Shape (n, n).
        Northern hemisphere is first, southern hemisphere is second.
    n_s2_points : int
        The number of points to sample on the sphere for integration.

    Returns
    -------
    torch.Tensor
        Shape (n, n). The detector annulus convolved with the incoming signal on the sphere.

    """
    # sample points on the sphere used for integration - returned as (n, 3) xyz coordinates
    s2_points = s2_fibonacci_lattice(n_s2_points).double().to(square_lambert_mp.device)

    # convert xyz coordinates to latitude and longitude
    theta_phi_pts = xyz_to_theta_phi(s2_points).double()

    # make an indicator for the points in the annulus
    annulus_mask = (theta_phi_pts[:, 0] > inner_opening_angle) & (
        theta_phi_pts[:, 0] < outer_opening_angle
    )

    annulus_points = s2_points[annulus_mask]

    # now for each point in the master pattern, we move the north pole to that point
    # then we apply this rotation to our annulus points and interpole them onto the master pattern
    # finally we mean the values of the annulus points... for all points in the master pattern

    # calculate the rotations that bring each point on the master pattern to the north pole
    grid_points_xy = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, square_lambert_mp.shape[0]),
                torch.linspace(-1, 1, square_lambert_mp.shape[1]),
                indexing="ij",
            ),
            dim=-1,
        )
        .float()
        .reshape(-1, 2)
        .to(square_lambert_mp.device)
    )
    grid_points_xyz = inv_square_lambert(grid_points_xy.reshape(-1, 2))

    mp_blurred = torch.zeros(
        (len(grid_points_xyz),), dtype=torch.float32, device=square_lambert_mp.device
    )

    # find quaternion rotations that bring the north pole to each point on the master pattern
    north_pole = torch.tensor(
        [0, 0, 1], dtype=torch.float32, device=square_lambert_mp.device
    )
    grid_points_quat = quaternion_rotate_sets_sphere(
        north_pole.repeat(grid_points_xyz.shape[0], 1), grid_points_xyz
    )

    # calculate the batch size to keep the GPU memory usage under 1 GB
    batch_size = int(limit_GB * 1e9 / (annulus_points.shape[0] * 32))

    # for each point on the master pattern, we rotate the annulus points to that point
    for i in range(0, grid_points_xyz.shape[0], batch_size):
        # rotate the annulus points to the point on the master pattern
        annulus_points_rotated = quaternion_apply(
            grid_points_quat[i : i + batch_size, None, :], annulus_points[None, :, :]
        )

        # convert the latitude and longitude to planar coordinates
        planar_coords = square_lambert(annulus_points_rotated)

        # interpolate the rotated points onto the master pattern
        mp_blurred[i : i + batch_size] = (
            torch.nn.functional.grid_sample(
                square_lambert_mp[None, None, :, :],
                planar_coords[None, :, :, :].float(),
                align_corners=True,
                mode="bilinear",
                padding_mode="reflection",
            )
            .squeeze()
            .mean(dim=-1)
        )

    return mp_blurred.reshape(square_lambert_mp.shape[0], square_lambert_mp.shape[1])
