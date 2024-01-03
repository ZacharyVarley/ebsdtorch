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

S2 FZ is rejection sampled with a Fibonacci lattice.

The main idea here is to update the covariance matrix batch by batch
because we cannot fit the fundamental sector evaluated anywhere from 
100,000 to 100 million times into memory. So we sample the rotations 
in batches and update the covariance matrix batch by batch. A 
tolerance is set so that the algorithm can stop early if the absolute 
mean change in the covariance matrix is below the threshold. 

"""

from typing import Tuple
import torch
from torch import Tensor
from ebsdtorch.patterns.pattern_projection import (
    project_pattern_single_geometry,
    detector_coords_to_ksphere_via_pc,
)
from ebsdtorch.ebsd_dictionary.welford_covariance_matrix import OnlineCovMatrix
from ebsdtorch.laue.laue_orientation_fz import (
    oris_are_in_so3_fz,
    _points_are_in_s2_fz,
    LAUE_MULTS,
    LAUE_GROUPS,
)
from ebsdtorch.laue.orientations import quaternion_apply
from ebsdtorch.laue.sampling import (
    s2_fibonacci_lattice,
    so3_halton_cubochoric,
    so3_cubochoric_grid,
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


# @torch.jit.script
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
        s2_n_fz_pts: number of points to use on the fundamental sector of S2
        so3_n_samples: number of samples to use on the fundamental zone of SO3
        so3_batch_size: number of samples to use per batch when calculating the covariance matrix
        correlation: if True, return the correlation matrix instead of the covariance matrix. We
            expect that the pixels vary with comparable variance so the covariance matrix is sufficient.

    Returns:
        covmat: torch tensor of shape (~s2_n_fz_pts, ~s2_n_fz_pts) containing the covariance matrix
            rejection sampling means that the number of points in the fundamental sector is not exact


    """

    # get the sampling locations on the fundamental sector of S2
    s2_samples = s2_fibonacci_lattice(s2_n_fz_pts * LAUE_MULTS[laue_group - 1])

    # filter out all but the S2 fundamental sector of the laue group
    s2_samples_fz = s2_samples[_points_are_in_s2_fz(s2_samples, laue_group)]

    # this gives the actual number of samples in the fundamental zone aquired
    num_s2_samples = len(s2_samples_fz)

    # estimate the edge length needed to yield the desired number of samples
    required_oversampling = torch.tensor([so3_n_samples + 2 * so3_batch_size])
    # multiply by half the Laue multiplicity (inversion is not included in the operators)
    required_oversampling = required_oversampling * 0.5 * LAUE_MULTS[laue_group - 1]
    # take the cube root to get the edge length
    edge_length = int(torch.ceil(torch.pow(required_oversampling, 1 / 3)))
    so3_samples = so3_cubochoric_grid(edge_length, master_pattern_MSLNH.device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[oris_are_in_so3_fz(so3_samples, laue_group)]

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
        mat = project_pattern_single_geometry(
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


def _sample_orientations_fz(
    laue_group: int,
    so3_n_samples: int,
    device: torch.device,
) -> Tensor:
    # find so3 samples in fundamental zone (same as spherical case)
    # estimate the edge length needed to yield the desired number of samples
    required_oversampling = torch.tensor([int(so3_n_samples * 1.018)])

    # multiply by half the Laue multiplicity (inversion is not included in the operators)
    required_oversampling = required_oversampling * 0.5 * LAUE_MULTS[laue_group - 1]

    # take the cube root to get the edge length
    edge_length = int(torch.ceil(torch.pow(required_oversampling, 1 / 3)))
    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[oris_are_in_so3_fz(so3_samples, laue_group)]

    # randomly permute the samples
    so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


# @torch.jit.script
def _detector_covmat(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    detector_cosines: Tensor,
    so3_samples_fz: Tensor,
    so3_batch_size: int,
    correlation: bool = False,
) -> Tensor:
    # calculate the number of batches
    n_batches = (len(so3_samples_fz) // so3_batch_size) + 1

    # n_pixels is same as the direction cosines -2 dimension
    n_pixels = detector_cosines.shape[-2]

    # initialize the running covariance matrix
    running_covmat = OnlineCovMatrix(n_pixels).to(master_pattern_MSLNH.device)

    # loop over the batches of orientations and project the patterns
    start_id = 0

    for _ in range(n_batches):
        # sample SO(3) with uniform distribution
        so3_samples_fz_batch = so3_samples_fz[start_id : (start_id + so3_batch_size)]
        start_id += so3_batch_size

        # get the values of the master pattern at the rotated points over FZ
        # this is a (N_so3, N_s2) tensor, our "data matrix"
        mat = project_pattern_single_geometry(
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            quaternions=so3_samples_fz_batch,
            direction_cosines=detector_cosines.squeeze(),
        )

        running_covmat(mat)

    covmat = running_covmat.get_covmat()

    if correlation:
        # calculate the correlation matrix
        d_sqrt_inv = 1.0 / torch.sqrt(torch.diag(covmat))
        corr_mat = torch.einsum("ij,i,j->ij", covmat, d_sqrt_inv, d_sqrt_inv)
        return corr_mat
    else:
        return covmat


@torch.jit.script
def _dictionary_pca_loadings(
    pca_matrix: Tensor,
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    detector_cosines: Tensor,
    so3_samples_fz: Tensor,
    batch_size: int,
) -> Tensor:
    # set the output
    output_pca_loadings = torch.empty(
        (len(so3_samples_fz), pca_matrix.shape[1]),
        dtype=torch.float32,
        device=master_pattern_MSLNH.device,
    )

    # patterns
    output_pca_loadings = []

    for so3_samples_fz_batch in torch.split(so3_samples_fz, batch_size):
        # this is a (N_so3_batch, N_pixels) tensor, our "data matrix"
        patterns = project_pattern_single_geometry(
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            quaternions=so3_samples_fz_batch,
            direction_cosines=detector_cosines.squeeze(),
        )

        # subtract the mean from each pattern
        patterns = patterns - torch.mean(patterns, dim=1)[:, None]

        # project the patterns onto the PCA basis
        output_pca_loadings.append(torch.matmul(patterns, pca_matrix))

    return torch.cat(output_pca_loadings, dim=0)


"""

Taken directly from https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html

Feydy, Jean, Alexis Glaunès, Benjamin Charlier, and Michael Bronstein. "Fast geometric learning 
with symbolic matrices." Advances in Neural Information Processing Systems 33 (2020): 14448-14462.

WARNING: I have explored many approximation methods such as HNSW, IVF-Flat, etc (on the GPU too).

They are not worth running on the raw EBSD images because the time invested in building the index is
large. For cubic materials, a dictionary around 100,000 images is needed, and most methods will take
10s of seconds if the patterns are larger than 60x60 = 3600 dimensions. PCA requires <10 seconds
investment, even using Online covariance matrix estimation approaches. 

"""


@torch.jit.script
def knn_batch(
    x_train: Tensor,
    x_train_norm: Tensor,
    query: Tensor,
    K: int,
    metric: str,
):
    largest = False  # Default behaviour is to look for the smallest values

    if metric == "euclidean":
        query_norm = (query**2).sum(-1)
        diss = (
            query_norm.view(-1, 1)
            + x_train_norm.view(1, -1)
            - 2 * query @ x_train.t()  # Rely on cuBLAS for better performance!
        )

    elif metric == "manhattan":
        diss = (query[:, None, :] - x_train[None, :, :]).abs().sum(dim=2)

    elif metric == "angular":
        diss = query @ x_train.t()
        largest = True

    elif metric == "hyperbolic":
        query_norm = (query**2).sum(-1)
        diss = (
            query_norm.view(-1, 1) + x_train_norm.view(1, -1) - 2 * query @ x_train.t()
        )
        diss /= query[:, 0].view(-1, 1) * x_train[:, 0].view(1, -1)
    else:
        raise NotImplementedError(f"The '{metric}' distance is not supported.")

    return diss.topk(K, dim=1, largest=largest).indices


@torch.jit.script
def knn_brute_force(
    x_train_in: Tensor,
    query_in: Tensor,
    topk: int,
    av_mem: float,
    metric: str = "euclidean",
    match_dtype: torch.dtype = torch.float16,
):
    """
    Batched exact K-NN query on a GPU.

    """
    # cast to matching dtype
    x_train = x_train_in.to(match_dtype)
    query = query_in.to(match_dtype)

    # Setup the K-NN estimator:
    Ntrain, D = x_train.shape
    # The "training" time here should be negligible:
    x_train_norm = (x_train**2).sum(-1)

    # Estimate the largest reasonable batch size:
    Ntest = query.shape[0]
    # Remember that a vector of D float32 number takes up 4*D bytes:

    Ntest_loop = int(min(max(1, av_mem * 1e9 // (4 * D * Ntrain)), Ntest))
    knn_indices = []

    print(f"{Ntrain} dictionary entries against {Ntest_loop} patterns per batch.")

    for query_batch in torch.split(query, Ntest_loop):
        knn_indices.append(knn_batch(x_train, x_train_norm, query_batch, topk, metric))

    return torch.cat(knn_indices, dim=0)


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
            LAUE_GROUPS[self.laue_group - 1]
            .to(s2_query_points.device)
            .to(s2_query_points.dtype),
            self.covmat,
        )

        return covmat_query


class EBSDExperiment(torch.nn.Module):
    """
    Class to calculate the covariance matrix of a master pattern
    when projected onto a virtual detector plane over SO(3) fundamental
    zone for a given Laue group.

    This code generally operates under the assumption that the number of
    sampled FZ orientations fits into memory, their associated condensed
    PCA patterns fit into memory, but the actual projected patterns do not
    all fit into memory. Under this assumption we have to project the
    patterns twice. Once to find the PCA decomposition on the first pass
    and a second pass to actually project each projected pattern onto the
    discovered decomposition.

    """

    def __init__(
        self,
        laue_group: int,
        master_pattern_MSLNH: Tensor,
        master_pattern_MSLSH: Tensor,
        pattern_center: Tuple[float, float, float],
        detector_height: int,
        detector_width: int,
        detector_tilt_deg: float,
        azimuthal_deg: float,
        sample_tilt_deg: float,
        signal_mask=None,
    ):
        """

        Args:
            laue_group: int between 1 and 11 inclusive
            master_pattern_MSLNH: torch tensor of shape (n_pixels, n_pixels) containing
                the modified square Lambert projected master pattern in the northern hemisphere
            master_pattern_MSLSH: torch tensor of shape (n_pixels, n_pixels) containing
                the modified square Lambert projected master pattern in the southern hemisphere
            detector_height: height of the detector in pixels
            detector_width: width of the detector in pixels
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
        self.register_buffer("master_pattern_MSLNH", master_pattern_MSLNH)
        self.register_buffer("master_pattern_MSLSH", master_pattern_MSLSH)

        # set the detector geometry
        pattern_center_tensor = torch.tensor(pattern_center, dtype=torch.float32)[None]
        self.register_buffer("pattern_center", pattern_center_tensor)
        self.detector_height = detector_height
        self.detector_width = detector_width
        self.detector_tilt_deg = detector_tilt_deg
        self.azimuthal_deg = azimuthal_deg
        self.sample_tilt_deg = sample_tilt_deg
        self.signal_mask = signal_mask

    def compute_PCA_detector_plane(
        self,
        so3_n_samples: int = 300000,
        so3_batch_size: int = 512,
        correlation: bool = False,
    ) -> None:
        """
        Compute the PCA decomposition of the detector plane.

        Args:
            so3_n_samples: number of samples to use on the fundamental zone of SO3
            so3_batch_size: number of samples to use per batch when calculating the covariance matrix
            correlation: if True, return the correlation matrix instead of the covariance matrix. We
                expect that the pixels vary with comparable variance so the covariance matrix is sufficient.

        """

        # get the direction cosines for each detector pixel
        detector_cosines = detector_coords_to_ksphere_via_pc(
            pcs=self.pattern_center,
            n_rows=self.detector_height,
            n_cols=self.detector_width,
            tilt=self.detector_tilt_deg,
            azimuthal=self.azimuthal_deg,
            sample_tilt=self.sample_tilt_deg,
            signal_mask=self.signal_mask,
        )

        # sample orientation space
        so3_samples_fz = _sample_orientations_fz(
            laue_group=self.laue_group,
            so3_n_samples=so3_n_samples,
            device=detector_cosines.device,
        )

        # do the covariance matrix calculation
        covmat = _detector_covmat(
            master_pattern_MSLNH=self.master_pattern_MSLNH,
            master_pattern_MSLSH=self.master_pattern_MSLSH,
            detector_cosines=detector_cosines,
            so3_samples_fz=so3_samples_fz,
            so3_batch_size=so3_batch_size,
            correlation=correlation,
        )

        # find the eigenvalues and eigenvectors of the symmetric covariance matrix
        eigvals, eigvecs = torch.linalg.eigh(covmat)

        # save the eigenvectors and eigenvalues
        self.register_buffer("covmat", covmat)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)
        self.register_buffer("detector_cosines", detector_cosines)

    def project_dictionary_pca(
        self,
        so3_n_samples: int,
        so3_batch_size: int,
        pca_n_max_components: int,
    ):
        """
        Project the dictionary onto the PCA basis.

        Args:
            so3_n_samples: number of samples to use on the fundamental zone of SO3
            so3_batch_size: number of samples to use per batch when calculating the covariance matrix
            pca_n_max_components: number of PCA components to use

        """

        # sample orientation space (can be different than the one used for the covariance matrix)
        so3_samples_fz = _sample_orientations_fz(
            laue_group=self.laue_group,
            so3_n_samples=so3_n_samples,
            device=self.detector_cosines.device,
        )

        # project the patterns using helper function that does the projection in batches
        pca_loadings = _dictionary_pca_loadings(
            pca_matrix=self.eigvecs[:, -pca_n_max_components:].float(),
            master_pattern_MSLNH=self.master_pattern_MSLNH,
            master_pattern_MSLSH=self.master_pattern_MSLSH,
            detector_cosines=self.detector_cosines,
            so3_samples_fz=so3_samples_fz,
            batch_size=so3_batch_size,
        )

        # save the PCA loadings
        self.n_pca_components_max = pca_n_max_components
        self.register_buffer("so3_samples_fz", so3_samples_fz)
        self.register_buffer("dictionary_pca_loadings", pca_loadings)

    def pca_index_experimental_data(
        self,
        experimental_data: Tensor,
        n_pca_components: int,
        topk: int,
        target_ram_allocation_GB: float,
        metric: str = "euclidean",
        match_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Index the experimental data.

        Args:
            experimental_data: experimental dataset of shape (n_patterns, n_pixels)
            n_pca_components: number of PCA components to use
            topk: number of top matches to return for each experimental pattern
            target_ram_allocation: target VRAM that batch distance matrix should occupy
            zero_mean_each_pattern: if True, zero mean each pattern before projecting onto the PCA basis

        Returns:
            indexed dataset of shape (n_patterns, k) with k as the index into self.so3_samples_fz

        """

        # input sanity checks
        if (
            not isinstance(n_pca_components, int)
            or n_pca_components < 1
            or n_pca_components > self.n_pca_components_max
        ):
            raise ValueError(
                f"n_pca_components{n_pca_components} must be an int between 1 and {self.n_pca_components_max}"
            )
        if (
            not isinstance(topk, int)
            or topk < 1
            or topk > self.dictionary_pca_loadings.shape[0]
        ):
            raise ValueError(
                f"topk{topk} must be an int between 1 and {self.dictionary_pca_loadings.shape[0]}"
            )

        # subtract the per pattern mean
        experimental_data = experimental_data - torch.mean(
            experimental_data, dim=1, keepdims=True
        )

        # project the experimental dataset onto the PCA components
        projected_dataset = torch.matmul(
            experimental_data, self.eigvecs[:, -n_pca_components:].float()
        )

        # compute the indices of the topk matches
        indices = knn_brute_force(
            x_train_in=self.dictionary_pca_loadings[:, -n_pca_components:].float(),
            query_in=projected_dataset,
            topk=topk,
            av_mem=target_ram_allocation_GB,
            metric=metric,
            match_dtype=match_dtype,
        )
        return indices


# # test it out
# import kikuchipy as kp
# import matplotlib.pyplot as plt


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# # get the example dataset
# s = kp.load("../datafolder/Hakon_Ni/Pattern.dat")

# print(f"s raw data shape: {s.data.shape}")

# # find the covariance matrix for the example dataset in KikuchiPy
# patterns = s.data.reshape(-1, 60 * 60)
# patterns = torch.from_numpy(patterns).to(torch.float32).to(device)

# # subtract static background
# patterns = patterns - torch.mean(patterns, dim=0)[None, :]

# # get the master pattern
# mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
# ni = mp.phase
# mLPNH = mp.data[0, :, :]
# mLPSH = mp.data[1, :, :]

# # to torch tensor
# mLPNH = torch.from_numpy(mLPNH).to(torch.float32).to(device)
# mLPSH = torch.from_numpy(mLPSH).to(torch.float32).to(device)

# # normalize each master pattern to 0 to 1
# mLPNH = (mLPNH - torch.min(mLPNH)) / (torch.max(mLPNH) - torch.min(mLPNH))
# mLPSH = (mLPSH - torch.min(mLPSH)) / (torch.max(mLPSH) - torch.min(mLPSH))

# ebsd = EBSDExperiment(
#     laue_group=11,
#     master_pattern_MSLNH=mLPNH,
#     master_pattern_MSLSH=mLPSH,
#     pattern_center=(0.4221, 0.2179, 0.4954),
#     detector_height=60,
#     detector_width=60,
#     detector_tilt_deg=0.0,
#     azimuthal_deg=0.0,
#     sample_tilt_deg=70.0,
# ).to(device)

# ebsd.compute_PCA_detector_plane(
#     so3_n_samples=100000,
#     so3_batch_size=10000,
#     correlation=False,
# )

# ebsd.project_dictionary_pca(
#     pca_n_max_components=2000,
#     so3_n_samples=200000,
#     so3_batch_size=10000,
# )

# # time it
# import time

# start = time.time()

# indices = ebsd.pca_index_experimental_data(
#     experimental_data=patterns,
#     n_pca_components=1000,
#     topk=1,
#     target_ram_allocation_GB=12.0,
#     metric="angular",
#     match_dtype=torch.float16,
# )

# duration = time.time() - start

# print(f"Patterns per second: {len(patterns) / duration}")

# fz_quats_indexed = ebsd.so3_samples_fz[indices.cpu().numpy().squeeze()]

# from orix import plot
# from orix.vector import Vector3d
# from orix.quaternion import Orientation

# pg_m3m = ni.point_group.laue

# # Orientation colors
# ckey_m3m = plot.IPFColorKeyTSL(ni.point_group, direction=Vector3d.zvector())

# orientations = Orientation(fz_quats_indexed.cpu().numpy())
# rgb = ckey_m3m.orientation2color(orientations)

# plt.imshow(rgb.reshape(s.data.shape[0], s.data.shape[1], 3))
# plt.savefig("fz_quat_colors.png")


# cov_detector = ebsd.covmat

# # plot the first 3 eigenvectors
# eigvals, eigvecs = torch.linalg.eigh(cov_detector)

# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
#     eigvec = eigvecs[:, -i]
#     eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min())
#     eigvec = eigvec.reshape(60, 60).cpu().numpy()

#     plt.imshow(eigvec)
#     plt.colorbar()
#     plt.savefig(f"eigvec_{i}.png")
#     plt.close()

# # # unit variance
# # patterns = patterns / patterns.std(axis=0, keepdims=True)

# # find the covariance matrix
# cov_experiment = torch.cov(torch.from_numpy(patterns).to(torch.float32).to(device).T)

# # use the mean of each row as weighting in the covariance matrix of the detector covariance matrix
# # this will act as an automatic masking
# # set diagonal to zero
# cov_exper_weights_zero_diag = cov_experiment - torch.diag(torch.diag(cov_experiment))

# # find the mean of each row
# cov_exper_weights = (
#     torch.mean(torch.abs(cov_exper_weights_zero_diag), dim=1)
#     * torch.diag(cov_experiment)
# ) ** 0.25

# # save an image of the weighting matrix
# plt.imshow(cov_exper_weights.cpu().numpy().reshape(60, 60))
# plt.colorbar()
# plt.savefig("cov_exper_weights.png")
# plt.close()

# # take outer product
# cov_exper_weights = torch.einsum("i,j->ij", cov_exper_weights, cov_exper_weights) ** 0.5

# # make symmetric
# cov_exper_weights = (cov_exper_weights + cov_exper_weights.t()) / 2

# cov_exper_weights = (cov_exper_weights - cov_exper_weights.min()) / (
#     cov_exper_weights.max() - cov_exper_weights.min()
# )

# # multiply the covariance matrix by the weights
# _, eigvecs_experiment = torch.linalg.eigh(cov_detector * cov_exper_weights)

# # plot the first 3 eigenvectors
# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
#     eigvec = eigvecs_experiment[:, -i]
#     eigvec = (eigvec - eigvec.min()) / (eigvec.max() - eigvec.min())
#     eigvec = eigvec.reshape(60, 60).cpu().numpy()

#     plt.imshow(eigvec)
#     plt.colorbar()
#     plt.savefig(f"eigvec_weighted_{i}.png")
#     plt.close()


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
