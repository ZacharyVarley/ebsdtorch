"""
This implements an EBSDDIwithPCA class that can be used to perform EBSD
dictionary indexing of patterns using PCA. The class is designed to be used with
a single pattern center and a single detector geometry, which is the
conventional approach for EBSD. In the future, I will add support for fitting
the actual detector geometry, as has been done for LabDCT. This should be
straightforward to implement but requires me to port Spherical Indexing and
generalized spherical harmonics into PyTorch. I need to do the indexing on the
sphere becuase I am face with two choices:

1. Project the patterns onto the detector plane and then index them. This
requires a reprojection of the dictionary for each pattern on the sample
surface. This is slow because the dictionary is large and the reprojection is
expensive.

2. Index the patterns on the sphere. This requires a reprojection of the
experimental patterns onto the sphere, followed by an expensive evaluation of
the quality of the best possible match over orientation space. This is still
cheaper than dictionary reprojection.


"""

from typing import Tuple
import torch
from torch import Tensor

from ebsdtorch.patterns.pattern_projection import (
    project_pattern_single_geometry,
    detector_coords_to_ksphere_via_pc,
)

from ebsdtorch.ebsd_dictionary_indexing.utils_covariance_matrix import OnlineCovMatrix
from ebsdtorch.ebsd_dictionary_indexing.utils_nearest_neighbors import knn
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar

from ebsdtorch.s2_and_so3.laue import so3_sample_fz_laue


def _detector_covmat(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    detector_cosines: Tensor,
    so3_samples_fz: Tensor,
    batch_size: int,
    correlation: bool = False,
) -> Tensor:
    # n_pixels is same as the direction cosines -2 dimension
    n_pixels = detector_cosines.shape[-2]

    running_covmat = OnlineCovMatrix(n_pixels, correlation=correlation).to(
        master_pattern_MSLNH.device
    )

    # loop over the batches of orientations and project the patterns
    pb = progressbar(
        list(torch.split(so3_samples_fz, batch_size)),
        prefix="PCA Calculations",
    )

    for so3_samples_fz_batch in pb:
        # get the values of the master pattern at the rotated points over FZ
        # this is a (N_so3, N_s2) tensor, our "data matrix"
        patterns = project_pattern_single_geometry(
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            quaternions=so3_samples_fz_batch,
            direction_cosines=detector_cosines.squeeze(),
        )

        running_covmat(patterns)

    covmat = running_covmat.get_covmat()

    return covmat


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

    # loop over the batches of orientations and project the patterns
    pb = progressbar(
        list(torch.split(so3_samples_fz, batch_size)),
        prefix="PCA Projections",
    )

    for so3_samples_fz_batch in pb:
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


class EBSDDIwithPCA(torch.nn.Module):
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
        so3_batch_size: int = 10000,
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
        so3_samples_fz = so3_sample_fz_laue(
            laue_id=self.laue_group,
            target_n_samples=so3_n_samples,
            device=detector_cosines.device,
        )

        # do the covariance matrix calculation
        covmat = _detector_covmat(
            master_pattern_MSLNH=self.master_pattern_MSLNH,
            master_pattern_MSLSH=self.master_pattern_MSLSH,
            detector_cosines=detector_cosines,
            so3_samples_fz=so3_samples_fz,
            batch_size=so3_batch_size,
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
        so3_samples_fz = so3_sample_fz_laue(
            laue_id=self.laue_group,
            target_n_samples=so3_n_samples,
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

    def pca_di_patterns(
        self,
        experimental_data: Tensor,
        n_pca_components: int,
        topk: int,
        match_device: torch.device,
        metric: str = "angular",
        data_chunk_size: int = 32768,
        query_chunk_size: int = 4096,
        match_dtype: torch.dtype = torch.float16,
        override_quantization: bool = True,
        subtract_mean: bool = True,
    ) -> Tensor:
        """
        Index the experimental data. The experimental data is first projected onto the PCA
        decomposition of the dictionary. The PCA decomposition is truncated to the number
        of components specified by n_pca_components.

        Args:
            experimental_data: experimental dataset of shape (n_patterns, n_pixels)
            n_pca_components: number of PCA components to use for this indexing
            topk: number of nearest neighbors to return
            match_device: device to use for the distance calculations
            metric: distance metric to use. Can be "angular" or "euclidean" or "manhattan"
            data_chunk_size: number of dictionary patterns to use per chunk
            query_chunk_size: number of experimental patterns to use per chunk
            match_dtype: dtype to use for the distance calculations
            override_quantization: if True, force quantized when on the CPU
            subtract_mean: if True, subtract the mean from each **masked** pattern (if not already done)

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

        # mask the experimental data
        if self.signal_mask is not None:
            experimental_data = experimental_data[:, self.signal_mask]

        # subtract the per pattern mean
        if subtract_mean:
            experimental_data = experimental_data - torch.mean(
                experimental_data, dim=1, keepdims=True
            )

        # project the experimental dataset onto the PCA components
        projected_dataset = torch.matmul(
            experimental_data, self.eigvecs[:, -n_pca_components:].float()
        )

        # if the projected dataset is on the CPU, use the quantized version
        # and force the usage of angular distance
        indices = knn(
            data=self.dictionary_pca_loadings[:, -n_pca_components:],
            query=projected_dataset,
            data_chunk_size=data_chunk_size,
            query_chunk_size=query_chunk_size,
            topk=topk,
            distance_metric=metric,
            match_dtype=match_dtype,
            match_device=match_device,
            quantized=(override_quantization and match_device == torch.device("cpu")),
        )
        return indices

    def lookup_orientations(
        self,
        indices: Tensor,
    ) -> Tensor:
        """
        Lookup the orientations associated with the indices.

        Args:
            indices: indices into the fundamental zone of SO3

        Returns:
            orientations: orientations associated with the indices

        """
        return self.so3_samples_fz[indices]
