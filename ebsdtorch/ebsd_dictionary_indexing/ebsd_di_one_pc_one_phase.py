"""
This implements an EBSDDI class that can be used to perform EBSD dictionary
indexing of patterns. The class is designed to be used with a single pattern
center and a single detector geometry, which is the conventional approach for
EBSD. In the future, I will add support for fitting the actual detector
geometry, as has been done for LabDCT. This should be straightforward to
implement but requires me to port Spherical Indexing and generalized spherical
harmonics into PyTorch. I need to do the indexing on the sphere becuase I am
face with two choices:

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

from ebsdtorch.patterns.pattern_projection import project_patterns, average_pc_geometry
from ebsdtorch.ebsd_dictionary_indexing.utils_nearest_neighbors import knn
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar
from ebsdtorch.s2_and_so3.laue import so3_sample_fz_laue


def _project_dictionary(
    master_pattern_MSLNH: Tensor,
    master_pattern_MSLSH: Tensor,
    detector_cosines: Tensor,
    so3_samples_fz: Tensor,
    batch_size: int,
    subtract_mean: bool = True,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    # list of patterns batches
    patterns = []

    # loop over the batches of orientations and project the patterns
    pb = progressbar(
        list(torch.split(so3_samples_fz, batch_size)),
        prefix="Dictionary Projection",
    )

    for so3_samples_fz_batch in pb:
        # get the values of the master pattern at the rotated points over FZ
        # this is a (N_so3, N_s2) tensor, our "data matrix"
        patterns_batch = project_patterns(
            master_pattern_MSLNH=master_pattern_MSLNH,
            master_pattern_MSLSH=master_pattern_MSLSH,
            quaternions=so3_samples_fz_batch,
            direction_cosines=detector_cosines.squeeze(),
        )

        # subtract the mean
        if subtract_mean:
            patterns_batch = patterns_batch - torch.mean(patterns_batch, dim=1)[:, None]

        patterns.append(patterns_batch.to(out_dtype))

    return torch.cat(patterns, dim=0)


class EBSDDI(torch.nn.Module):
    """
    Class to calculate the covariance matrix of a master pattern
    when projected onto a virtual detector plane over SO(3) fundamental
    zone for a given Laue group.

    This code currently operates under the assumption that the number of
    sampled FZ orientations fits into memory, the dictionary fits into
    memory, but the experimental dataset does not need to fit into memory.

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
            pattern_center: pattern center in pixels given in Kikuchipy convention
            detector_height: height of the detector in pixels
            detector_width: width of the detector in pixels
            detector_tilt_deg: detector tilt from horizontal in degrees
            azimuthal_deg: sample tilt about the sample RD axis in degrees
            sample_tilt_deg: sample tilt from horizontal in degrees
            signal_mask: mask to apply to the experimental data. If None, no mask is applied.

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

    def project_dictionary(
        self,
        so3_n_samples: int = 300000,
        so3_batch_size: int = 512,
        subtract_mean: bool = True,
        storage_dtype: torch.dtype = torch.float16,
    ):
        """
        Compute the PCA decomposition of the detector plane.

        Args:
            so3_n_samples: number of samples to use on the fundamental zone of SO3
            so3_batch_size: number of samples to use per batch when calculating the covariance matrix
            subtract_mean: if True, subtract the mean from each pattern
            storage_dtype: datatype to use for the storage of the patterns. Default is float16.

        """

        # get the direction cosines for each detector pixel
        detector_cosines = average_pc_geometry(
            pcs=self.pattern_center,
            n_rows=self.detector_height,
            n_cols=self.detector_width,
            tilt=self.detector_tilt_deg,
            azimuthal=self.azimuthal_deg,
            sample_tilt=self.sample_tilt_deg,
            signal_mask=self.signal_mask,
        )

        # save the detector cosines in case we want to reproject a different sized dictionary
        self.register_buffer("detector_cosines", detector_cosines)

        # sample orientation space
        so3_samples_fz = so3_sample_fz_laue(
            laue_id=self.laue_group,
            target_n_samples=so3_n_samples,
            device=detector_cosines.device,
        )

        print(f"Targeted {so3_n_samples} samples, and received {len(so3_samples_fz)}")

        # save the orientations for lookup
        self.register_buffer("so3_samples_fz", so3_samples_fz)

        # project the dictionary onto the detector plane
        patterns = _project_dictionary(
            master_pattern_MSLNH=self.master_pattern_MSLNH,
            master_pattern_MSLSH=self.master_pattern_MSLSH,
            detector_cosines=detector_cosines,
            so3_samples_fz=so3_samples_fz,
            batch_size=so3_batch_size,
            subtract_mean=subtract_mean,
            out_dtype=storage_dtype,
        )

        # save the patterns
        self.register_buffer("patterns", patterns)

    def di_patterns(
        self,
        experimental_data: Tensor,
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
        Index the experimental data. target_RAM_GB is the amount of RAM to use
        in the case that calculations are done on the CPU.

        Args:
            experimental_data: experimental dataset of shape (n_patterns, n_pixels)
            topk: number of nearest neighbors to return
            match_device: device to use for the matching
            metric: distance metric to use for the nearest neighbor search
            data_chunk_size: number of dictionary patterns to process per batch
            query_chunk_size: number of experimental patterns to process per batch
            match_dtype: datatype to use for the matching
            override_quantization: if True, force the usage of quantized distance
                calculations on CPUs. This is useful if you don't have a GPU.
                If you are using Apple Silicon, use torch.device("mps") instead.

        Returns:
            indexed dataset of shape (n_patterns, k) with k as the index into self.so3_samples_fz

        """

        # mask the experimental data
        if self.signal_mask is not None:
            experimental_data = experimental_data[:, self.signal_mask]

        # subtract the per pattern mean
        if subtract_mean:
            experimental_data = experimental_data - torch.mean(
                experimental_data, dim=1, keepdims=True
            )

        indices = knn(
            data=self.patterns,
            query=experimental_data,
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
