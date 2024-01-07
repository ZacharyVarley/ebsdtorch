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

from ebsdtorch.ebsd_dictionary_indexing.utils_covariance_matrix import (
    OnlineCovMatrix,
    ApproxOnlineCovMatrix,
)
from ebsdtorch.ebsd_dictionary_indexing.utils_nearest_neighbors import (
    knn,
    knn_quantized,
)
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar

from ebsdtorch.s2_and_so3.laue import (
    so3_sample_fz_laue,
)


def _detector_covmat(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    detector_cosines: Tensor,
    so3_samples_fz: Tensor,
    batch_size: int,
    correlation: bool = False,
    approx: bool = False,
) -> Tensor:
    # n_pixels is same as the direction cosines -2 dimension
    n_pixels = detector_cosines.shape[-2]

    # initialize the running covariance matrix
    if approx:
        running_covmat = ApproxOnlineCovMatrix(n_pixels, correlation=correlation).to(
            master_pattern_MSLNH.device
        )
    else:
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
        target_VRAM_GB: float = 0.25,
        target_RAM_GB: float = 8.0,
        metric: str = "euclidean",
        match_dtype: torch.dtype = torch.float16,
        override_quantization: bool = True,
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
            target_ram_allocation_GB: target RAM allocation in GB
            metric: distance metric to use. One of "euclidean", "manhattan", "angular", "hyperbolic"
            match_dtype: dtype to use for the distance calculations
            override_quantization: if True, force quantized calculations for dictionaries on the CPU

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
        experimental_data = experimental_data - torch.mean(
            experimental_data, dim=1, keepdims=True
        )

        # project the experimental dataset onto the PCA components
        projected_dataset = torch.matmul(
            experimental_data, self.eigvecs[:, -n_pca_components:].float()
        )

        # if the projected dataset is on the CPU, use the quantized version
        # and force the usage of angular distance
        if override_quantization and match_device == torch.device("cpu"):
            indices = knn_quantized(
                data=self.dictionary_pca_loadings[:, -n_pca_components:].float().cpu(),
                query=projected_dataset.float().cpu(),
                topk=topk,
                available_ram_GB=target_RAM_GB,
            )
        else:
            indices = knn(
                data=self.dictionary_pca_loadings[:, -n_pca_components:],
                query=projected_dataset,
                topk=topk,
                available_ram_GB=target_VRAM_GB,
                distance_metric=metric,
                match_dtype=match_dtype,
                match_device=match_device,
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


# # test it out
# import kikuchipy as kp
# import matplotlib.pyplot as plt


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# print(f"Using device: {device}")
# # device = torch.device("cpu")

# # get the example dataset
# s = kp.load("../datafolder/EBSD_Hakon_Ni/scan10/Pattern.dat")

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

# # get the signal mask
# signal_mask = kp.filters.Window("circular", (60, 60)).astype(bool).reshape(-1)
# signal_mask = torch.from_numpy(signal_mask).to(torch.bool).to(device)

# # to torch tensor
# mLPNH = torch.from_numpy(mLPNH).to(torch.float32).to(device)
# mLPSH = torch.from_numpy(mLPSH).to(torch.float32).to(device)

# # normalize each master pattern to 0 to 1
# mLPNH = (mLPNH - torch.min(mLPNH)) / (torch.max(mLPNH) - torch.min(mLPNH))
# mLPSH = (mLPSH - torch.min(mLPSH)) / (torch.max(mLPSH) - torch.min(mLPSH))

# with torch.no_grad():
#     ebsd = EBSDDIwithPCA(
#         laue_group=11,
#         master_pattern_MSLNH=mLPNH,
#         master_pattern_MSLSH=mLPSH,
#         pattern_center=(0.4221, 0.2179, 0.4954),
#         detector_height=60,
#         detector_width=60,
#         detector_tilt_deg=0.0,
#         azimuthal_deg=0.0,
#         sample_tilt_deg=70.0,
#         signal_mask=signal_mask,
#     ).to(device)

#     ebsd.compute_PCA_detector_plane(
#         so3_n_samples=100000,
#         so3_batch_size=10000,
#         correlation=False,
#     )

#     ebsd.project_dictionary_pca(
#         pca_n_max_components=2000,
#         so3_n_samples=300000,
#         so3_batch_size=10000,
#     )

# # time it
# import time

# start = time.time()

# indices = ebsd.pca_di_patterns(
#     experimental_data=patterns,
#     n_pca_components=1000,
#     match_device=device,
#     topk=1,
#     target_VRAM_GB=0.25,
#     target_RAM_GB=4.0,
#     metric="angular",
#     match_dtype=torch.float16,
# )

# duration = time.time() - start

# print(f"Patterns per second: {len(patterns) / duration}")

# fz_quats_indexed = ebsd.lookup_orientations(indices)

# from orix import plot
# from orix.vector import Vector3d
# from orix.quaternion import Orientation

# pg_m3m = ni.point_group.laue

# # Orientation colors
# ckey_m3m = plot.IPFColorKeyTSL(ni.point_group, direction=Vector3d.zvector())

# orientations = Orientation(fz_quats_indexed.cpu().numpy())
# rgb = ckey_m3m.orientation2color(orientations)

# plt.imshow(rgb.reshape(s.data.shape[0], s.data.shape[1], 3))
# plt.savefig("scratch/fz_quat_colors.png")
