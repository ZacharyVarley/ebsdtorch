"""
This file contains the covariance master class, which is used to
calculate the covariance matrix of master pattern when acted upon
by SO(3) fundamental zone for a given Laue group.

Downstream applications can project detector pixel locations onto 
the sphere and then use the covariance matrix calculated in the 
fundamental sector to interpolate the covariance matrix for the
detector pixels. Then eigenvalues and eigenvectors can be calculated
without having to do the covariance matrix calculation for each
separate detector geometry.

One important note is that the covariance should be different if the
master pattern is enhanced with CLAHE or not, etc. So filtering of the
master pattern should be done before calculating the covariance matrix.

The class can be used in two modes: 'online' and 'exact'. The 'online'
mode uses Welford's online algorithm to calculate the covariance matrix.
This mode is the best one and the 'exact' mode is only included for 
testing purposes.

SO3 FZ can be sampled with cubochoric uniform grid or Halton sequence grid.
The reason for a Halton sequence would be to avoid potential grid artifacts 
and/or "aliasing" at the fundamental zone boundaries. 

S2 FZ is rejection sampled with a Fibonacci lattice.

The main idea here is to update the covariance matrix batch by batch
because we cannot fit the fundamental sector evaluated anywhere from 
100,000 to 100 million times into memory. So we sample the rotations 
in batches and update the covariance matrix batch by batch. A 
tolerance is set so that the algorithm can stop early if the absolute 
mean change in the covariance matrix is below the threshold. 

"""

import torch
from ebsdtorch.geometries.interpolation import square_lambert
from ebsdtorch.laue.fundamental_zone import (
    oris_are_in_so3_fz,
    _points_are_in_s2_fz,
    LAUE_MULTS,
    LAUE_GROUPS
)
from ebsdtorch.laue.orientations import quaternion_apply
from ebsdtorch.laue.sampling import s2_fibonacci_lattice, so3_halton_cubochoric, so3_cubochoric_grid

# get LAPACK for the eigenvalue decomposition


# @torch.jit.script
def s2_over_rotations(
    s2_fz: torch.Tensor,
    so3_fz: torch.Tensor,
    master_pattern: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate the points on the sphere by the given rotations.
    :param s2_points: torch tensor of shape (N, 3) containing the points
    :param so3_fz: torch tensor of shape (N, 4) containing the rotations
    :param master_pattern: torch tensor of shape (H, W) containing the master pattern
    :return: torch tensor of shape (N, 3) containing the rotated points
    """
    # rotate the points according to the sampled orientations
    # this yields shape (N_so3, N_s2, 3)
    s2_rotated_coords = quaternion_apply(so3_fz[:, None], s2_fz[None])

    # move the points to the northern hemisphere via inversion
    s2_rotated_coords[s2_rotated_coords[:, :, 2] < 0] *= -1

    # shape (N_so3, N_s2, 2)
    s2_projected_coords = square_lambert(s2_rotated_coords.view(-1, 3)).view(
        so3_fz.shape[0], s2_fz.shape[0], 2
    )

    s2_projected_coords = s2_projected_coords.to(master_pattern.dtype)

    s2_values = torch.nn.functional.grid_sample(
        master_pattern[None, None],
        s2_projected_coords[None, :, :, :],
        align_corners=True,
    ).squeeze()

    return s2_values


@torch.jit.script
def _covmat_interpolate(
    query_pts: torch.Tensor, 
    s2_fz_pts: torch.Tensor,
    laue_group: torch.Tensor,
    covmat: torch.Tensor,
    ) -> torch.Tensor:

    # enumerate the equivalent points of the query points (N, |LaueGroup|, 3)
    # get the important shapes
    data_shape = query_pts.shape
    N = int(torch.prod(torch.tensor(data_shape[:-1])))
    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    eqiv_pts = quaternion_apply(
        laue_group.reshape(-1, 4), query_pts.view(N, 1, 3)
    )
    # concatenate all of the points with their inverted coordinates
    eqiv_pts = torch.cat([eqiv_pts, -eqiv_pts], dim=1)

    # for each point, find the smallest distance between any equivalent point and a point in the FZ
    # this gives us the index of the closest point in the FZ for each point
    dist_min_ids = torch.empty(N, dtype=torch.int64, device=query_pts.device)

    for i in range(N):
        # calculate the distance between the query points and the FZ points
        dists = torch.norm(
            eqiv_pts[i, :, None, :] - s2_fz_pts[None, :, :],
            dim=-1,
        )
        # get the column index of the minimum distance of the entire (2 * |LaueGroup|, N_s2) matrix
        # this is the index of the closest point in the FZ
        _, j_index = torch.where(dists == torch.min(dists))
        dist_min_ids[i] = j_index[0]

    # fill in the covariance matrix
    covmat_query = covmat[dist_min_ids][:, dist_min_ids]

    return covmat_query


@torch.jit.script
def update_covmat(
    current_covmat: torch.Tensor,
    current_obs: torch.Tensor,
    current_mean: torch.Tensor,
    data_new: torch.Tensor) -> None:
    """
    Update the covariance matrix and mean using Welford's online algorithm.
    :param current_covmat: torch tensor of shape (n_features, n_features)
    :param current_mean: torch tensor of shape (n_features)
    :param data_new: torch tensor of shape (batch_size, n_features)
    :return: None
    """
    # compute the batch mean
    N = data_new.shape[0]
    batch_mean = torch.mean(data_new, dim=0, keepdim=True)

    # update the global mean
    new_mean = (current_mean * current_obs + batch_mean * N) / (current_obs + N)

    # compute the deltas
    delta = (data_new - current_mean).float()
    delta_prime = (data_new - new_mean).float()

    # update the running covariance matrix
    current_covmat += torch.einsum('ij,ik->jk', delta, delta_prime).to(current_covmat.dtype)

    # update the number of observations and mean
    current_obs += N
    current_mean.copy_(new_mean)

    
class OnlineCovMatrix(torch.nn.Module):
    """
    Online covariance matrix calculator
    """

    def __init__(self, n_features: int, dtype: torch.dtype = torch.float64):
        super(OnlineCovMatrix, self).__init__()
        self.n_features = n_features
        self.dtype = dtype

        # Initialize the running mean
        self.register_buffer("mean", torch.zeros(1, n_features, dtype=dtype))

        # Initialize the running covariance matrix
        self.register_buffer(
            "covmat_aggregate", torch.zeros((n_features, n_features), dtype=dtype)
        )

        # Initialize the number of observations
        self.register_buffer("obs", torch.tensor(0, dtype=dtype))


    def forward(self, x: torch.Tensor):
        """
        :param x: torch tensor of shape (batch_size, n_features)
        :return: torch tensor of shape (n_features, n_features)
        """
        # update the covariance matrix
        update_covmat(self.covmat_aggregate, self.obs, self.mean, x)
    
    def get_covmat(self):
        """
        :return: torch tensor of shape (n_features, n_features)
        """
        return (self.covmat_aggregate / (self.obs - 1))


class EBSDCovmatSphere(torch.nn.Module):
    """
    Class to calculate the covariance matrix of a master pattern
    when acted upon by SO(3) fundamental zone for a given Laue group.

    covmat_mode can be 'online' or 'exact'
    'online' uses Welford's online algorithm to calculate the covariance matrix
    'exact' uses a double loop over the sampled points on the sphere

    so3_sample_mode can be 'grid' or 'halton'
    'grid' uses a grid of points in the unit cube to sample SO(3)
    'halton' uses a Halton sequence in the unit cube to sample SO(3)

    """

    def __init__(
        self,
        laue_group: int,
        s2_n_samples: int = 10000,
        so3_n_samples: int = 300000,
        so3_batch_size: int = 256,
        covmat_mode: str = 'online',
        so3_sample_mode: str = 'grid',
    ):
        """
        :param laue_group: Laue group number
        :param s2_n_samples: number sphere sample points (rejection sampling changes final value)
        :param so3_n_samples: number of points to sample on SO(3)
        :param so3_batch_size: number of SO(3) rotations to sample at a time (for 'online' mode)
        :param tolerance: tolerance for the running covariance matrix (for 'online' mode)
        :param covmat_mode: 'online' or 'exact'
        :param so3_sample_mode: 'grid' or 'halton'

        """
        super().__init__()

        # assert LAUE_GROUP is an int between 1 and 11 inclusive
        if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
            raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
        # set the laue group

        self.laue_group = laue_group
        self.so3_n_samples = so3_n_samples
        self.so3_batch_size = so3_batch_size
        self.covmat_mode = covmat_mode
        self.so3_sample_mode = so3_sample_mode

        # get the sampling locations on the fundamental sector of S2
        s2_samples = s2_fibonacci_lattice(s2_n_samples * LAUE_MULTS[laue_group - 1])

        # filter out all but the S2 fundamental sector of the laue group
        s2_samples_fz = s2_samples[_points_are_in_s2_fz(s2_samples, laue_group)]

        # this gives the actual number of samples in the fundamental zone aquired
        self.num_s2_samples = len(s2_samples_fz)
        self.register_buffer("s2_samples_fz", s2_samples_fz)

    def forward(self, master_pattern: torch.Tensor) -> torch.Tensor:

        # sample SO(3) with uniform distribution (two extra batches)
        if self.so3_sample_mode == 'grid':
            # estimate the edge length needed to yield the desired number of samples
            required_oversampling = torch.tensor([self.so3_n_samples + 2 * self.so3_batch_size])
            # multiply by half the Laue multiplicity (inversion is not included in the operators)
            required_oversampling = required_oversampling * 0.5 * LAUE_MULTS[self.laue_group - 1]
            # take the cube root to get the edge length
            edge_length = int(torch.ceil(torch.pow(required_oversampling, 1/3)))
            so3_samples = so3_cubochoric_grid(edge_length, 
                                              master_pattern.device)
        elif self.so3_sample_mode == 'halton':
            halton_id_start = 0
            halton_id_stop = self.so3_n_samples + 2 * self.so3_batch_size
            so3_samples = so3_halton_cubochoric(
                halton_id_start, halton_id_stop, master_pattern.device
            )
        else:
            raise ValueError(f"SO(3) sampling mode {self.so3_sample_mode} not in ['grid', 'halton']")

        # # reject the points that are not in the fundamental zone
        so3_samples_fz = so3_samples[oris_are_in_so3_fz(so3_samples, self.laue_group)]

        if self.covmat_mode == 'online':
            running_covmat = OnlineCovMatrix(self.num_s2_samples).to(master_pattern.device)

            # randomly permute the samples
            so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

            # calculate the number of batches   
            n_batches = (len(so3_samples_fz) // self.so3_batch_size) + 1
            start_id = 0
            
            for i in range(n_batches):
                # sample SO(3) with uniform distribution
                so3_samples_fz_batch = so3_samples_fz[start_id: (start_id + self.so3_batch_size)]
                start_id += self.so3_batch_size

                # get the values of the master pattern at the rotated points over FZ
                # this is a (N_so3, N_s2) tensor, our "data matrix" 
                mat = s2_over_rotations(self.s2_samples_fz, so3_samples_fz_batch, master_pattern)
                running_covmat(mat)

                if i % 10 == 0:
                    print(f"batch: {i:06d} out of {n_batches:06d}")
            
            covmat = running_covmat.get_covmat()
            
        elif self.covmat_mode == 'exact':
            # do the exact calculation
            covmat = torch.empty(
                (self.num_s2_samples, self.num_s2_samples),
                dtype=torch.float64,
                device=master_pattern.device,
            )

            # double loop over the s2 points and use all the so3 points to rotate them 
            # take those columns of the data matrix and fill in one covmat entry at a time
            for i in range(self.num_s2_samples):
                for j in range(self.num_s2_samples):
                    # get the values of the master pattern at the rotated points over FZ
                    # this is a (N_so3, N_s2) tensor, our data matrix
                    mat = s2_over_rotations(
                        self.s2_samples_fz[[i, j]], so3_samples, master_pattern
                    )

                    # normalize the data matrix to have zero mean and unit variance
                    mat = (mat - mat.mean(dim=0, keepdim=True))
                    mat = mat / mat.std(dim=0, keepdim=True)

                    # calculate the covariance matrix of the data matrix
                    covmat[i, j] = mat[:, 0] @ mat[:, 1] / (so3_samples.shape[0] - 1)
                
                print(f"finished row {i:05d} of {self.num_s2_samples}")
        else:
            raise ValueError(f"Covariance calculation mode {self.mode} not recognized")
        
        self.register_buffer("covmat", covmat)
        
        # return the covariance matrix
        return covmat
    
    def get_covmat(self):
        """
        :return: torch tensor of shape (n_features, n_features)
        """
        return self.covmat
    
    def get_corr_mat(self):
        """
        :return: torch tensor of shape (n_features, n_features)
        """
        d_sqrt_inv = 1.0 / torch.sqrt(torch.diag(self.covmat))
        corr_mat = torch.einsum('ij,i,j->ij', self.covmat, d_sqrt_inv, d_sqrt_inv)
        return corr_mat

    def covmat_interpolate(self, s2_query_points: torch.Tensor) -> torch.Tensor:
        """
        Interpolate the covariance matrix for the given points on the sphere.
        :param s2_query_points: torch tensor of shape (N, 3) containing the points
            on the sphere. Need not be in the fundamental sector.
        :return: torch tensor of shape (N, N) containing the interpolated covmat
        """
        covmat_query = _covmat_interpolate(
            s2_query_points,
            self.s2_samples_fz, 
            LAUE_GROUPS[self.laue_group - 1].to(s2_query_points.device).to(s2_query_points.dtype),
            self.covmat,
        )

        return covmat_query
        
    

# # test it out
# import h5py as hf
# import plotly.graph_objects as go

# def load_master_pattern(master_pattern_path):
#     """
#     Load the master patterns from the hdf5 file
#     """
#     with hf.File(master_pattern_path, 'r') as f:
#         master_pattern_north = f['EMData']['ECPmaster']['mLPNH'][...]
#         master_pattern_south = f['EMData']['ECPmaster']['mLPSH'][...]
#     return master_pattern_south, master_pattern_north

# device = torch.device('cuda:0')

# mn, ms = load_master_pattern('Ti_HCP_20kV_tilt_10_EC_MP.h5')
# mn = mn[0, :, :]
# mn = (mn - mn.min()) / (mn.max() - mn.min())
# mn = torch.from_numpy(mn).to(device).to(torch.float32)

# # initialize the covariance matrix calculator
# covmat_calc = EBSDCovmatSphere(9).to(device)
# # calculate the covariance matrix
# covmat = covmat_calc(mn)

# # make an image of the covariance matrix and plot it and save it
# import matplotlib.pyplot as plt

# plt.imshow(covmat.cpu().numpy())
# plt.colorbar()
# plt.savefig("covmat_sphere.png")
# plt.close()

# # make the covariance matrix symmetric
# covmat = (covmat + covmat.t()) / 2

# # take the covaranice matrix and calculate the eigenvectors
# # use torch.linalg
# eigvals, eigvecs = torch.linalg.eigh(covmat)

# # plot the first two eigenvectors as greyscale color of points on the sphere
# # use the points s2_samples_fz as the points on the sphere and use plotly

# for i in [1, 10, 100, 1000]:

#     eigvec = eigvecs[:, -i]
#     eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min())

#     # make a plot of the first two eigenvectors
#     fig = go.Figure(
#         data=[
#             go.Scatter3d(
#                 x=covmat_calc.s2_samples_fz[:, 0].cpu().numpy(),
#                 y=covmat_calc.s2_samples_fz[:, 1].cpu().numpy(),
#                 z=covmat_calc.s2_samples_fz[:, 2].cpu().numpy(),
#                 mode="markers",
#                 marker=dict(
#                     size=2,
#                     color=eigvec.cpu().numpy(),
#                     colorscale="viridis",
#                     cmin=0,
#                     cmax=1,
#                 ),
#             ),
#         ]
#     )
#     # add title
#     fig.update_layout(title_text=f"eigvec_{i}")

#     fig.show()
#     # fig.write_html(f"eigvec_{i}.html")


# # project a square of edge 1.0 x 1.0 (100, 100) centered at (0,0) onto the unit sphere 
# detector_height = 100
# detector_width = 100
# square_indices = torch.stack(torch.meshgrid(torch.linspace(-0.5, 0.5, detector_height),
#                                             torch.linspace(-0.5, 0.5, detector_width),
#                                             indexing='ij'), dim=-1)

# # offset it by 0.1, 0.1 so that it is not centered at the origin
# square_indices += 0.1
# square_indices = square_indices.reshape(-1, 2).to(device)

# # project the square onto the sphere using gnomonic projection
# sphere_points_z = 0.5
# sphere_points = torch.cat((square_indices, torch.ones_like(square_indices[:, 0:1]) * sphere_points_z), dim=1)
# sphere_points_norm = sphere_points / torch.norm(sphere_points, dim=1, keepdim=True)

# # retrieve the covariance matrix for the sphere points
# covmat_sphere = covmat_calc.covmat_interpolate(sphere_points_norm)

# # make it symmetric and add epsilon to the diagonal so it is positive semi-definite
# covmat_sphere = (covmat_sphere + covmat_sphere.t()) / 2 #+ 1e-6 * torch.eye(covmat_sphere.shape[0], device=covmat_sphere.device)

# # weight the covariance matrix by the inverse of the distance from the detector plane
# # this is a crude approximation of the detector geometry
# covmat_sphere = covmat_sphere / torch.norm(sphere_points, dim=1, keepdim=True)**2

# # save an image of the spherical covariance matrix
# plt.imshow(covmat_sphere.cpu().numpy())
# plt.colorbar()
# plt.savefig("covmat_detector_from_sphere.png")
# plt.close()

# print("starting eigendecomposition")

# eigvals_sphere, eigvecs_sphere = torch.linalg.eigh(covmat_sphere)

# # plot images of the 1st, 10th, 100th, and 1000th eigenvectors
# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
#     eigvec = eigvecs_sphere[:, -i]
#     # eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min())
#     eigvec = eigvec.reshape(detector_height, detector_width).cpu().numpy()

#     plt.imshow(eigvec)
#     plt.colorbar()
#     plt.savefig(f"eigvec_{i}_sphere.png")
#     plt.close()


# # now project 300000 128x128 patterns onto the detector plane
# # and calculate the covariance matrix at the detector plane

# so3_n_samples = 300000
# laue_group = 9
# master_pattern = mn
# batch_size = 256

# running_covmat_detector = OnlineCovMatrix(detector_height * detector_width).to(device)

# # sample SO(3) with uniform distribution (two extra batches)
# # estimate the edge length needed to yield the desired number of samples
# required_oversampling = torch.tensor([so3_n_samples])
# # multiply by half the Laue multiplicity (inversion is not included in the operators)
# required_oversampling = required_oversampling * 0.5 * LAUE_MULTS[laue_group - 1]
# # take the cube root to get the edge length
# edge_length = int(torch.ceil(torch.pow(required_oversampling, 1/3)))
# so3_samples = so3_cubochoric_grid(edge_length, master_pattern.device)

# # reject the points that are not in the fundamental zone
# so3_samples_fz = so3_samples[oris_are_in_so3_fz(so3_samples, laue_group)]

# # randomly permute the samples
# so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

# # calculate the number of batches
# n_batches = (len(so3_samples_fz) // batch_size) + 1

# start_id = 0

# for i in range(n_batches):
#     # sample SO(3) with uniform distribution
#     so3_samples_fz_batch = so3_samples_fz[start_id: (start_id + batch_size)]
#     start_id += batch_size

#     # get the values of the master pattern at the rotated points over FZ
#     # this is a (N_so3, N_s2) tensor, our "data matrix" 
#     mat = s2_over_rotations(sphere_points_norm, so3_samples_fz_batch, master_pattern)
#     running_covmat_detector(mat)

#     if i % 10 == 0:
#         print(f"batch: {i:06d} out of {n_batches:06d}")
    
# covmat_detector = running_covmat_detector.get_covmat()

# # make it symmetric and add epsilon to the diagonal so it is positive semi-definite
# covmat_detector = (covmat_detector + covmat_detector.t()) / 2 #+ 1e-6 * torch.eye(covmat_detector.shape[0], device=covmat_detector.device)

# # save an image of the planar covariance matrix
# plt.imshow(covmat_detector.cpu().numpy())
# plt.colorbar()
# plt.savefig("covmat_detector_from_detector.png")
# plt.close()

# print("starting eigendecomposition")

# eigvals_detector, eigvecs_detector = torch.linalg.eigh(covmat_detector)

# # plot images of the 1st, 10th, 100th, and 1000th eigenvectors
# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
#     eigvec = eigvecs_detector[:, -i]
#     # eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min())
#     eigvec = eigvec.reshape(detector_height, detector_width).cpu().numpy()

#     plt.imshow(eigvec)
#     plt.colorbar()
#     plt.savefig(f"eigvec_{i}_detector.png")
#     plt.close()


# # use each set of eigenvectors to transform an example projected pattern

# example_pattern = mat = s2_over_rotations(sphere_points_norm, so3_samples_fz[[100]], master_pattern)
# example_pattern = example_pattern.reshape(-1, 1)

# # zero mean
# example_pattern = example_pattern - example_pattern.mean()

# # cast to same dtype as eigvecs
# example_pattern = example_pattern.to(eigvecs_detector.dtype)


# plt.imshow(example_pattern.cpu().numpy().reshape(detector_height, detector_width))
# plt.colorbar()
# plt.savefig(f"example_pattern.png")
# plt.close()

# # the transform and inverse transform from eigenvectors and pseudo-inverse
# t_detector = eigvecs_detector[:, -1000:].T
# t_sphere = eigvecs_sphere[:, -1000:].T

# t_inv_detector = torch.pinverse(t_detector) # shape (16384, 1000)
# t_inv_sphere = torch.pinverse(t_sphere) # shape (16384, 1000)

# mse_sphere = []
# mse_detector = []

# recon_sphere = []
# recon_detector = []

# # transform the example pattern with the eigenvectors
# for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
#     # transform the pattern with the forward transform
#     pattern_transformed_sphere = t_sphere[-i:] @ example_pattern.reshape(-1, 1)
#     pattern_transformed_detector = t_detector[-i:] @ example_pattern.reshape(-1, 1)

#     # transform the pattern back with the inverse transform
#     pattern_recon_sphere = t_inv_sphere[:, -i:] @ pattern_transformed_sphere
#     pattern_recon_detector = t_inv_detector[:, -i:] @ pattern_transformed_detector

#     # calculate the MSE
#     mse_sphere.append(torch.mean((example_pattern.reshape(-1, 1) - pattern_recon_sphere) ** 2))
#     mse_detector.append(torch.mean((example_pattern.reshape(-1, 1) - pattern_recon_detector) ** 2))

#     # save the reconstructed patterns
#     recon_sphere.append(pattern_recon_sphere.reshape(detector_height, detector_width).cpu().numpy())
#     recon_detector.append(pattern_recon_detector.reshape(detector_height, detector_width).cpu().numpy())

# # plot the MSE
# plt.plot(torch.arange(1, 11).numpy(), torch.tensor(mse_sphere).numpy(), label='sphere')
# plt.plot(torch.arange(1, 11).numpy(), torch.tensor(mse_detector).numpy(), label='detector')
# plt.legend()
# plt.savefig("mse.png")
# plt.close()

# # plot the reconstructed patterns in a grid
# fig, axs = plt.subplots(2, 10, figsize=(20, 4))
# for i in range(10):
#     axs[0, i].imshow(recon_sphere[i])
#     axs[0, i].axis('off')
#     axs[1, i].imshow(recon_detector[i])
#     axs[1, i].axis('off')
# plt.savefig("recon.png")
# plt.close()
