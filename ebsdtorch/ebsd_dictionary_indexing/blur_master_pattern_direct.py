import torch
import numpy as np
from torch_harmonics.quadrature import legendre_gauss_weights
from torch_harmonics import RealSHT, InverseRealSHT
from ebsdtorch.patterns.square_hemisphere_bijection import (
    square_lambert,
    inv_square_lambert,
)
from ebsdtorch.s2_and_so3.orientations import (
    quaternion_apply,
    quaternion_rotate_sets_sphere,
)
from ebsdtorch.s2_and_so3.sampling import (
    s2_fibonacci_lattice,
    theta_phi_to_xyz,
    xyz_to_theta_phi,
)


# def annulus_bse_detector_fft(
#     square_lambert_mp: torch.Tensor,
#     inner_opening_angle: float,
#     outer_opening_angle: float,
# ) -> torch.Tensor:
#     """Compute the area of a detector annulus in BSE coordinates.

#     Parameters
#     ----------
#     inner_opening_theta : torch.Tensor
#         The inner radius of the annulus in radians
#     outer_opening_theta : torch.Tensor
#         The outer radius of the annulus in radians
#     square_lambert_mp : torch.Tensor
#         The modified square Lambert projection of electron channeling sphere. Shape (2, n, n).
#         Northern hemisphere is first, southern hemisphere is second.

#     Returns
#     -------
#     torch.Tensor
#         Shape (2, n, n). The detector annulus convolved with the incoming signal on the sphere.

#     """
#     # sample points on the sphere
#     nlat = 512
#     nlon = 2*nlat
#     lmax = mmax = nlat

#     device = square_lambert_mp.device

#     sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss").to(device)
#     isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="legendre-gauss").to(device)

#     # get the latitude positions according to Gauss-Legendre quadrature
#     costheta, _ = legendre_gauss_weights(nlat)
#     thetas = np.flip(np.arccos(costheta))
#     # get the longitude positions
#     phi = np.linspace(-np.pi, np.pi, nlon, endpoint=False)

#     # make a list of all the points (..., 2)
#     theta_phi_pts = torch.stack(
#         torch.meshgrid(
#             torch.from_numpy(np.ascontiguousarray(thetas)),
#             torch.from_numpy(phi),
#             indexing="ij",
#         ),
#         dim=-1,
#     ).to(square_lambert_mp.dtype).to(device)

#     # convert to cartesian coordinates
#     xyz_pts = theta_phi_to_xyz(theta_phi_pts[..., 0], theta_phi_pts[..., 1]).reshape(-1, 3)

#     # interpolate the square Lambert projection onto the sphere
#     planar_coords = square_lambert(xyz_pts).to(square_lambert_mp.dtype).reshape(1, nlat, nlon, 2)

#     mp_values_GL_grid = torch.nn.functional.grid_sample(
#         square_lambert_mp[None, :, :, :],
#         planar_coords,
#         align_corners=True,
#     ).squeeze().double()

#     print("mp_values_GL_grid.shape", mp_values_GL_grid.shape)

#     # make an indicator for the points in the annulus
#     annulus_mask = ((theta_phi_pts[..., 0] > inner_opening_angle)
#                     & (theta_phi_pts[..., 0] < outer_opening_angle))
#     annulus_mask_GL_grid = annulus_mask.reshape(1, nlat, nlon).double()

#     # perform convolution
#     mp_values_GL_grid_sht = sht(mp_values_GL_grid)
#     annulus_mask_GL_grid_sht = sht(annulus_mask_GL_grid)
#     mp_blurred_GL_grid_sht = mp_values_GL_grid_sht * annulus_mask_GL_grid_sht.conj()

#     # convert back to grid
#     mp_blurred_GL_grid = isht(mp_blurred_GL_grid_sht)

#     # take the planar fractional coordinates in range (-1, 1) and do inverse square Lambert
#     image_grid = torch.stack(
#         torch.meshgrid(
#             torch.linspace(-1, 1, square_lambert_mp.shape[1]),
#             torch.linspace(-1, 1, square_lambert_mp.shape[2]),
#             indexing="ij",
#         ),
#         dim=-1,
#     ).to(square_lambert_mp.dtype).to(device)

#     # convert to Cartesian coordinates on sphere
#     xyz_grid = inv_square_lambert(image_grid.reshape(-1, 2))

#     # convert to latitude and longitude
#     theta_phi_grid = xyz_to_theta_phi(xyz_grid).reshape(1, square_lambert_mp.shape[1], square_lambert_mp.shape[2], 2)

#     print("mp_blurred_GL_grid.shape", mp_blurred_GL_grid.shape)

#     print("theta_minmax", theta_phi_grid[..., 0].min(), theta_phi_grid[..., 0].max())
#     print("phi_minmax", theta_phi_grid[..., 1].min(), theta_phi_grid[..., 1].max())

#     # rescale theta from [0, pi/2] to [-1, 0]
#     theta_phi_grid[..., 0] = (2 * theta_phi_grid[..., 0] / np.pi) - 1
#     # rescale phi from [-pi, pi] to [-1, 1]
#     theta_phi_grid[..., 1] = (theta_phi_grid[..., 1] / np.pi)

#     print("post theta_minmax", theta_phi_grid[..., 0].min(), theta_phi_grid[..., 0].max())
#     print("post phi_minmax", theta_phi_grid[..., 1].min(), theta_phi_grid[..., 1].max())

#     print("mp_blurred_GL_grid.shape", mp_blurred_GL_grid.shape)
#     print("theta_phi_grid.shape", theta_phi_grid.shape)

#     # interpolate the theta_phi_grid points with the blurred mp values using grid_sample
#     mp_blurred = torch.nn.functional.grid_sample(
#         mp_blurred_GL_grid[None, :, :, :],
#         theta_phi_grid,
#         align_corners=True,
#     ).squeeze().reshape(2, square_lambert_mp.shape[1], square_lambert_mp.shape[2])

#     return mp_blurred


# # test it out
# import h5py as hf

# def load_master_pattern(master_pattern_path):
#     """
#     Load the master patterns from the hdf5 file
#     """
#     with hf.File(master_pattern_path, 'r') as f:
#         master_pattern_north = f['EMData']['ECPmaster']['mLPNH'][...]
#         master_pattern_south = f['EMData']['ECPmaster']['mLPSH'][...]
#     return master_pattern_south, master_pattern_north

# device = torch.device('cuda:0')

# mn, ms = load_master_pattern('./scratch/Ni_20kV_EC_MP.h5')
# mn = mn[0, :, :]
# mn = (mn - mn.min()) / (mn.max() - mn.min())
# mn = torch.from_numpy(mn).to(device).to(torch.float32)
# ms = ms[0, :, :]
# ms = (ms - ms.min()) / (ms.max() - ms.min())
# ms = torch.from_numpy(ms).to(device).to(torch.float32)
# mp = torch.stack((mn, ms), dim=0).double()

# inner_opening_angle = 0.244978 # 0.5cm hole in the center of the detector (1 cm working distance)
# outer_opening_angle = 0.785398 # 2 cm detector diameter (1 cm working distance)

# mp_blurred = annulus_bse_detector_fft(mp, inner_opening_angle, outer_opening_angle)

# print("mp_blurred.shape", mp_blurred.shape)

# # renormalize the blurred master pattern
# mp_blurred = (mp_blurred - mp_blurred.min()) / (mp_blurred.max() - mp_blurred.min())

# # save png image of the original and blurred master patterns
# from PIL import Image
# mp_original_pil = Image.fromarray((mp[0, :, :] * 255).byte().cpu().numpy())
# mp_original_pil.save('./scratch/Ni_20kV_EC_MP.png')
# mp_blurred_pil = Image.fromarray((mp_blurred[0, :, :] * 255).byte().cpu().numpy())
# mp_blurred_pil.save('./scratch/Ni_20kV_EC_MP_blurred.png')

# def save_master_pattern(master_pattern_path, master_pattern, scaffold_path):
#     # use os to copy the file at the scaffold path to the master pattern path
#     import os
#     import shutil
#     shutil.copy(scaffold_path, master_pattern_path)

#     # open the file at the master pattern path
#     with hf.File(master_pattern_path, 'r+') as f:
#         # write the master pattern to the file
#         f['EMData']['ECPmaster']['mLPNH'][...] = master_pattern[0, :, :].cpu().numpy()
#         f['EMData']['ECPmaster']['mLPSH'][...] = master_pattern[1, :, :].cpu().numpy()

# save_master_pattern('./scratch/Ni_20kV_EC_MP_blurred.h5', mp_blurred, './scratch/Ni_20kV_EC_MP.h5')


# it didn't work wtih FFT, so I will just do a direct space convolution
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

    # confirm that the rotations are correct
    rotated_north_poles = quaternion_apply(
        grid_points_quat, north_pole.repeat(grid_points_xyz.shape[0], 1)
    )
    max_error = (rotated_north_poles - grid_points_xyz).abs().max()
    print("max_error:", max_error)
    print("mean_error:", (rotated_north_poles - grid_points_xyz).abs().mean())

    # calculate the batch size to keep the GPU memory usage under 1 GB
    batch_size = int(limit_GB * 1e9 / (annulus_points.shape[0] * 32))

    print("batch_size:", batch_size)

    # for each point on the master pattern, we rotate the annulus points to that point
    for i in range(0, grid_points_xyz.shape[0], batch_size):
        # rotate the annulus points to the point on the master pattern
        annulus_points_rotated = quaternion_apply(
            grid_points_quat[i : i + batch_size, None, :], annulus_points[None, :, :]
        )

        print("annulus_points_rotated.shape", annulus_points_rotated.shape)

        # convert the latitude and longitude to planar coordinates
        planar_coords = square_lambert(annulus_points_rotated)

        print("planar_coords.shape", planar_coords.shape)

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


# test it out
import h5py as hf


def load_master_pattern(master_pattern_path):
    """
    Load the master patterns from the hdf5 file
    """
    with hf.File(master_pattern_path, "r") as f:
        master_pattern_north = f["EMData"]["ECPmaster"]["mLPNH"][...]
        master_pattern_south = f["EMData"]["ECPmaster"]["mLPSH"][...]
    return master_pattern_south, master_pattern_north


device = torch.device("cuda:0")

mn, ms = load_master_pattern("./scratch/Ni_20kV_EC_MP.h5")
mn = mn[0, :, :]
mn = (mn - mn.min()) / (mn.max() - mn.min())
mp = torch.from_numpy(mn).to(device).to(torch.float32)

inner_opening_angle = (
    0.244978  # 0.5cm hole in the center of the detector (1 cm working distance)
)
outer_opening_angle = 0.785398  # 2 cm detector diameter (1 cm working distance)

mp_blurred = annulus_bse_detector(mp, inner_opening_angle, outer_opening_angle)

# # renormalize the blurred master pattern
# mp_blurred = (mp_blurred - mp_blurred.min()) / (mp_blurred.max() - mp_blurred.min())

# save png image of the original and blurred master patterns
from PIL import Image

mp_original_pil = Image.fromarray((mp[:, :] * 255).byte().cpu().numpy())
mp_original_pil.save("./scratch/Ni_20kV_EC_MP.png")

mp_blurred_pil = Image.fromarray((mp_blurred[:, :] * 255).byte().cpu().numpy())
mp_blurred_pil.save("./scratch/Ni_20kV_EC_MP_blurred.png")

mp_blurred_normed = (mp_blurred - mp_blurred.min()) / (
    mp_blurred.max() - mp_blurred.min()
)
mp_blurred_normed_pil = Image.fromarray(
    (mp_blurred_normed[:, :] * 255).byte().cpu().numpy()
)
mp_blurred_normed_pil.save("./scratch/Ni_20kV_EC_MP_blurred_normed.png")


def save_master_pattern(master_pattern_path, master_pattern, scaffold_path):
    # use os to copy the file at the scaffold path to the master pattern path
    import os
    import shutil

    shutil.copy(scaffold_path, master_pattern_path)

    # open the file at the master pattern path
    with hf.File(master_pattern_path, "r+") as f:
        # write the master pattern to the file
        f["EMData"]["ECPmaster"]["mLPNH"][...] = master_pattern.cpu().numpy()
        f["EMData"]["ECPmaster"]["mLPSH"][...] = master_pattern.cpu().numpy()


save_master_pattern(
    "./scratch/Ni_20kV_EC_MP_blurred.h5", mp_blurred, "./scratch/Ni_20kV_EC_MP.h5"
)
