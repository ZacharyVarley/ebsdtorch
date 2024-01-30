"""
This module contains functions to calculate the covariance matrix of a master
pattern when acted upon by SO(3) fundamental zone for a given Laue group. This
is done only within the fundamental sector of S2. Downstream applications can
project detector pixel locations onto the sphere and then use the covariance
matrix calculated in the fundamental sector to interpolate the covariance matrix
for the detector pixels. Then eigenvalues and eigenvectors can be calculated
without having to do the covariance matrix calculation for each separate
detector geometry.

"""

from typing import Tuple
import torch
from torch import Tensor

from ebsdtorch.patterns.pattern_projection import project_patterns
from ebsdtorch.ebsd_dictionary_indexing.utils_covariance_matrix import OnlineCovMatrix
from ebsdtorch.s2_and_so3.orientations import quaternion_apply
from ebsdtorch.s2_and_so3.laue import (
    s2_sample_fz_laue,
    so3_sample_fz_laue,
    laue_elements,
)


@torch.jit.script
def _covmat_interpolate(
    query_pts: Tensor,
    s2_fz_pts: Tensor,
    laue_group: Tensor,
    covmat: Tensor,
) -> torch.Tensor:
    # enumerate the equivalent points of the query points (N, |LaueGroup|, 3)
    # get the important shapes
    data_shape = query_pts.shape
    N = int(torch.prod(torch.tensor(data_shape[:-1])))
    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    eqiv_pts = quaternion_apply(laue_group.reshape(-1, 4), query_pts.view(N, 1, 3))
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


def _spherical_covmat(
    laue_group: int,
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    s2_n_fz_pts: int,
    so3_n_samples: int,
    so3_batch_size: int,
    correlation: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """

    A function to calculate the covariance matrix of a master pattern
    when acted upon by SO(3) fundamental zone for a given Laue group.

    Args:
        laue_group: int between 1 and 11 inclusive
        master_pattern_MSLNH: torch tensor of shape (n_pixels, n_pixels) containing
            the modified square Lambert projected master pattern in the northern hemisphere
        master_pattern_MSLSH: torch tensor of shape (n_pixels, n_pixels) containing
            the modified square Lambert projected master pattern in the southern hemisphere
        s2_n_fz_pts: target number of points to use on the fundamental sector of S2
        so3_n_samples: number of samples to use on the fundamental zone of SO3
        so3_batch_size: number of samples to use per batch when calculating the covariance matrix
        correlation: if True, return the correlation matrix instead of the covariance matrix. We
            expect that the pixels vary with comparable variance so the covariance matrix is sufficient.

    Returns:
        covmat: torch tensor of shape (~s2_n_fz_pts, ~s2_n_fz_pts) containing the covariance matrix
            rejection sampling means that the number of points in the fundamental sector is not exact


    """

    # filter out all but the S2 fundamental sector of the laue group
    s2_samples_fz = s2_sample_fz_laue(laue_id=laue_group, n_samples=s2_n_fz_pts)

    # find actual number of points in the fundamental sector
    num_s2_samples = s2_samples_fz.shape[0]

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_sample_fz_laue(laue_id=laue_group, n_samples=so3_n_samples)

    running_covmat = OnlineCovMatrix(num_s2_samples).to(master_pattern_MSLNH.device)

    # randomly permute the samples
    so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    # calculate the number of batches
    n_batches = (len(so3_samples_fz) // so3_batch_size) + 1
    start_id = 0

    for _ in range(n_batches):
        # sample SO(3) with uniform distribution
        so3_samples_fz_batch = so3_samples_fz[start_id : (start_id + so3_batch_size)]
        start_id += so3_batch_size

        # get the values of the master pattern at the rotated points over FZ
        # this is a (N_so3, N_s2) tensor, our "data matrix"
        mat = project_patterns(
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            quaternions=so3_samples_fz_batch,
            direction_cosines=s2_samples_fz,
        )

        running_covmat(mat)

    covmat = running_covmat.get_covmat()

    if correlation:
        # calculate the correlation matrix
        d_sqrt_inv = 1.0 / torch.sqrt(torch.diag(covmat))
        corr_mat = torch.einsum("ij,i,j->ij", covmat, d_sqrt_inv, d_sqrt_inv)
        return corr_mat, num_s2_samples, s2_samples_fz
    else:
        return covmat, num_s2_samples, s2_samples_fz


class EBSDCovmatKSphere(torch.nn.Module):
    """
    Class to calculate the covariance matrix of a master pattern
    when acted upon by SO(3) fundamental zone for a given Laue group.

    covmat_mode can be 'online' or 'exact'
    'online' uses Welford's online algorithm to calculate the covariance matrix

    so3_sample_mode can be 'grid' or 'halton'
    'grid' uses a grid of points in the unit cube to sample SO(3)
    'halton' uses a Halton sequence in the unit cube to sample SO(3)

    """

    def __init__(
        self,
        laue_group: int,
        master_pattern_MSLNH: Tensor,
        master_pattern_MSLSH: Tensor,
        s2_n_samples: int = 10000,
        so3_n_samples: int = 300000,
        so3_batch_size: int = 512,
        correlation: bool = False,
    ):
        """

        Args:
            laue_group: int between 1 and 11 inclusive
            master_pattern_MSLNH: torch tensor of shape (n_pixels, n_pixels) containing
                the modified square Lambert projected master pattern in the northern hemisphere
            master_pattern_MSLSH: torch tensor of shape (n_pixels, n_pixels) containing
                the modified square Lambert projected master pattern in the southern hemisphere
            s2_n_samples: number of points to use on the fundamental sector of S2
            so3_n_samples: number of samples to use on the fundamental zone of SO3
            so3_batch_size: number of samples to use per batch when calculating the covariance matrix
            correlation: if True, return the correlation matrix instead of the covariance matrix. We
                expect that the pixels vary with comparable variance so the covariance matrix is sufficient.

        """
        super().__init__()

        # assert LAUE_GROUP is an int between 1 and 11 inclusive
        if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
            raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
        # set the laue group

        self.laue_group = laue_group

        # do the covariance matrix calculation
        covmat, num_s2_samples, s2_samples_fz = _spherical_covmat(
            laue_group=laue_group,
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            s2_n_fz_pts=s2_n_samples,
            so3_n_samples=so3_n_samples,
            so3_batch_size=so3_batch_size,
            correlation=correlation,
        )

        self.register_buffer("covmat", covmat)
        self.register_buffer("num_s2_samples", num_s2_samples)
        self.register_buffer("s2_samples_fz", s2_samples_fz)

    def get_covmat(self):
        """
        :return: torch tensor of shape (n_features, n_features)
        """
        return self.covmat

    def covmat_interpolate(self, s2_query_points: Tensor) -> torch.Tensor:
        """
        Interpolate the covariance matrix for the given points on the sphere.
        :param s2_query_points: torch tensor of shape (N, 3) containing the points
            on the sphere. Need not be in the fundamental sector.
        :return: torch tensor of shape (N, N) containing the interpolated covmat
        """
        covmat_query = _covmat_interpolate(
            s2_query_points,
            self.s2_samples_fz,
            laue_elements(self.laue_group)
            .to(s2_query_points.device)
            .to(s2_query_points.dtype),
            self.covmat,
        )

        return covmat_query
