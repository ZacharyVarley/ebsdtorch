"""

This module has several purposes: 
1) Store the EBSD pattern data and metadata
2) Preprocessing and cleaning of experimental EBSD patterns
3) Device management: keep patterns in RAM until processing


"""

from typing import List, Optional
import torch
from torch import Tensor
from torch.nn import Module
from ebsdtorch.preprocessing.clahe import clahe_grayscale
from ebsdtorch.preprocessing.nlpar import nlpar


class ExperimentPatterns(Module):
    """

    Class for processing and sampling EBSD patterns from a scan.

    Args:
        :patterns (Tensor): Pattern data tensor shaped (SCAN_H, SCAN_W, H, W),
            (N_PATS, H, W), or (H, W).

    """

    def __init__(
        self,
        patterns: Tensor,
    ):
        super(ExperimentPatterns, self).__init__()

        if len(patterns.shape) < 2:
            raise ValueError("'patterns' requires >= 2 dimensions (B, H, W) or (H, W)")
        if len(patterns.shape) < 3:
            patterns = patterns.unsqueeze(0)
        if len(patterns.shape) > 4:
            raise ValueError(
                "3D Volume not yet supported... "
                + f"'patterns' requires dimensions: "
                + "(SCAN_H, SCAN_W, H, W), (N_PATS, H, W), or (H, W)"
            )

        self.pattern_shape = patterns.shape[-2:]
        self.spatial_shape = patterns.shape[:-2] if len(patterns.shape) > 2 else (1, 1)
        self.patterns = patterns.view(-1, *patterns.shape[-2:])

        # set number of patterns and pixels
        self.n_patterns = self.patterns.shape[0]
        self.n_pixels = self.pattern_shape[0] * self.pattern_shape[1]

        self.phases = None
        self.orientations = None
        self.inv_f_matrix = None

    def set_spatial_coords(
        self,
        spatial_coords: Tensor,
        indices: Optional[Tensor] = None,
    ):
        """
        Set the spatial coordinates for the ExperimentPatterns object.

        Args:
            :spatial_coords (Tensor): Spatial coordinates tensor shaped (N_PATS, N_Spatial_Dims).

        """
        if indices is None:
            if spatial_coords.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Spatial coordinates must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {spatial_coords.shape[0]} spatial coordinates and {self.n_patterns} patterns."
                )
            self.spatial_coords = spatial_coords
        else:
            if spatial_coords.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"Spatial coordinates must have the same number of patterns as the indices. "
                    + f"Got {spatial_coords.shape[0]} spatial coordinates and {indices.shape[0]} indices."
                )
            self.spatial_coords[indices] = spatial_coords

    def get_spatial_coords(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the spatial coordinates for the ExperimentPatterns object.

        Args:
            :indices (Tensor): Indices of the patterns to retrieve.

        Returns:
            Tensor: Retrieved spatial coordinates.

        """
        if self.spatial_coords is None:
            raise ValueError("Spatial coordinates must be set before retrieving them.")
        else:
            if indices is None:
                return self.spatial_coords
            return self.spatial_coords[indices]

    def set_orientations(
        self,
        orientations: Tensor,
        indices: Optional[Tensor] = None,
    ):
        """
        Set the orientations for the ExperimentPatterns object.

        Args:
            :orientations (Tensor): Orientations tensor shaped (N_PATS, 4).

        """
        # check the shape
        if len(orientations.shape) != 2:
            raise ValueError(
                f"Orientations must be quaternions (N_ORI, 4). Got {orientations.shape}."
            )
        if indices is None:
            if orientations.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Orientations must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {orientations.shape[0]} orientations and {self.n_patterns} patterns."
                )
            if orientations.shape[1] != 4:
                raise ValueError(
                    f"Orientations must be quaternions (w, x, y, z). Got {orientations.shape[1]}."
                )
            self.orientations = orientations
        else:
            # check dim of indices
            if len(indices.shape) != 1:
                raise ValueError(f"Indices must be (N_ORI,). Got {indices.shape}.")
            if orientations.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"Orientations must have the same number of patterns as the indices. "
                    + f"Got {orientations.shape[0]} orientations and {indices.shape[0]} indices."
                )
            if orientations.shape[1] != 4:
                raise ValueError(
                    f"Orientations must be quaternions (w, x, y, z). Got {orientations.shape[1]} components."
                )
            self.orientations[indices] = orientations

    def get_orientations(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the orientations for the ExperimentPatterns object.

        Returns:
            Tensor: Rotations tensor shaped (N_PATS, 4).

        """
        if indices is None:
            ori = self.orientations
        else:
            ori = self.orientations[indices]
        return ori

    def set_inv_f_matrix(
        self,
        f_matrix: Optional[Tensor] = None,
    ):
        """
        Set the deformation gradient for the ExperimentPatterns object.

        Args:
            :deformation_gradient (Tensor): Deformation gradient tensor shaped
                (N_PATS, 3, 3) or (N_PATS, 9) or (1, 3, 3) or (1, 9). If None,
                the deformation gradient is set to the identity for each pattern.

        """
        if f_matrix is None:
            f_matrix = torch.zeros(
                self.n_patterns,
                9,
                device=self.patterns.device,
            )
            f_matrix[:, 0] = 1.0
            f_matrix[:, 4] = 1.0
            f_matrix[:, 8] = 1.0
            f_matrix = f_matrix.reshape(self.n_patterns, 3, 3)
        else:
            if f_matrix.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Deformation gradients must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {f_matrix.shape[0]} gradients and {self.n_patterns} patterns."
                )
            if (f_matrix.shape[-2] != 3 or f_matrix.shape[-1] != 3) and f_matrix.shape[
                -1
            ] != 9:
                raise ValueError(
                    f"Deformation gradients must be 3x3 matrices. Got {f_matrix.shape[-2:]} components."
                )
        self.inv_f_matrix = f_matrix.view(self.n_patterns, 3, 3)

    def get_inv_f_matrix(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the deformation gradient tensor inverse for the ExperimentPatterns object.

        Returns:
            Tensor: Deformation gradient tensor shaped (N_PATS, 3, 3).

        """
        if self.inv_f_matrix is None:
            raise ValueError(
                "Deformation gradients must be set before retrieving them."
            )
        if indices is None:
            return self.inv_f_matrix
        return self.inv_f_matrix[indices]

    def subtract_overall_background(
        self,
    ):
        """
        Subtract the overall background from the patterns.

        """

        # subtract the mean of each pixel from all patterns
        self.patterns -= torch.mean(self.patterns, dim=0, keepdim=True)

    def contrast_enhance_clahe(
        self,
        clip_limit: float = 40.0,
        tile_grid_size: int = 4,
        n_bins: int = 64,
    ):
        """
        Contrast enhance the patterns using CLAHE.

        Args:
            :clip_limit (float): Clip limit for CLAHE.
            :tile_grid_size (int): Tile grid size for CLAHE.

        """

        self.patterns = clahe_grayscale(
            self.patterns[:, None],
            clip_limit=clip_limit,
            n_bins=n_bins,
            grid_shape=(tile_grid_size, tile_grid_size),
        ).squeeze(1)

    def normalize_per_pattern(
        self,
        norm_type: str = "minmax",
    ):
        """
        Normalize the patterns.

        Args:
            :method (str): Normalization method: "minmax", "zeromean", "standard"

        """

        self.patterns = self.patterns.view(self.n_patterns, -1)
        if norm_type == "minmax":
            pat_mins = torch.min(self.patterns, dim=-1).values
            pat_maxs = torch.max(self.patterns, dim=-1).values
            self.patterns -= pat_mins[..., None]
            self.patterns /= 1e-4 + pat_maxs[..., None] - pat_mins[..., None]
        elif norm_type == "zeromean":
            self.patterns -= torch.mean(self.patterns, dim=-1, keepdim=True)
        elif norm_type == "standard":
            self.patterns -= torch.mean(self.patterns, dim=-1, keepdim=True)
            self.patterns /= torch.std(self.patterns, dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Invalid normalization method. Got {norm_type} but expected 'minmax', 'zeromean', or 'standard'."
            )
        self.patterns = self.patterns.view(-1, *self.pattern_shape)

    def standard_clean(
        self,
    ):
        """
        Standard cleaning of the patterns.

        """
        self.subtract_overall_background()
        self.normalize_per_pattern(norm_type="minmax")
        self.contrast_enhance_clahe()
        self.normalize_per_pattern(norm_type="minmax")

    def do_nlpar(self, k_rad: int = 3, coeff: float = 0.375):
        """
        Apply non-local patch average regression to the patterns.

        Args:
            :k_rad (int): Radius of the search window.
            :coeff (float): Coefficient for the NL-PAR.

        """
        if len(self.spatial_shape) != 2:
            raise ValueError(
                f"NLPAR only supported for 2D EBSD scans, got {self.spatial_shape}"
            )

        # temporarily view the patterns as a 4D tensor then shape back
        self.patterns = nlpar(
            self.patterns.view(self.spatial_shape + self.pattern_shape),
            k_rad=k_rad,
            coeff=coeff,
        ).view(-1, *self.pattern_shape)

    def get_patterns(
        self,
        indices: Tensor,
        binning: int,
    ):
        """
        Retrieve patterns from the ExperimentPatterns object.

        Args:
            :indices (Tensor): Indices of the patterns to retrieve.
            :binning (Union[float, int]): Binning factor can be non-integer for pseudo-binning.

        Returns:
            Tensor: Retrieved patterns.

        """

        # binning is always a factor of both H and W
        # so downscaling is easy
        pats = self.patterns[indices]

        # use avg pooling to bin the patterns if integer binning factor
        if binning > 1:
            # if isinstance(binning, int) or (binning % 1 == 0):
            pats = torch.nn.functional.avg_pool2d(pats, binning)
            # else:
            #     blurrer = BlurAndDownsample(scale_factor=binning).to(pats.device)
            #     pats = blurrer(pats.view(-1, 1, *self.pattern_shape)).squeeze(1)
        return pats

    def get_indices_per_phase(
        self,
    ) -> List[Tensor]:
        """
        Retrieve the indices of the patterns for each phase.

        Returns:
            List[Tensor]: List of indices for each phase.

        """
        if self.phases is None:
            raise ValueError(
                "Phases must be indexed before retrieving indices per phase."
            )
        return [
            torch.nonzero(self.phases == i, as_tuple=False).squeeze()
            for i in range(self.phases.max().item() + 1)
        ]

    def set_raw_indexing_results(
        self,
        quaternions: Tensor,
        metrics: Tensor,
        phase_id: int,
    ):
        """
        Set the dictionary matches for the ExperimentPatterns object for a single phase.

        Args:
            :dictionary_matches (Tensor): Dictionary matches tensor.

        """
        if not hasattr(self, "raw_indexing_results"):
            self.raw_indexing_results = {
                phase_id: (quaternions, metrics),
            }
        else:
            self.raw_indexing_results[phase_id] = (quaternions, metrics)

    def combine_indexing_results(
        self,
        higher_is_better: bool,
    ) -> None:
        """
        Collapse the raw indexing results along the phase and only take the
        top match for each pattern.
        """

        if not hasattr(self, "raw_indexing_results"):
            raise ValueError("Raw indexing results must be set before combining.")

        if len(self.raw_indexing_results) > 1:
            # loop over the phases, concatenate the metrics and only store the phase
            # and quaternion with the best metric
            if higher_is_better:
                # fill metric with -inf
                best_metrics = torch.full(
                    (self.n_patterns,),
                    float("-inf"),
                    device=self.patterns.device,
                )
            else:
                # fill metric with +inf
                best_metrics = torch.full(
                    (self.n_patterns,),
                    float("inf"),
                    device=self.patterns.device,
                )

            # make a phase_id tensor
            phase_ids = torch.full(
                self.n_patterns,
                fill_value=-1,
                dtype=torch.uint8,
                device=self.patterns.device,
            )

            # make a quaternion tensor
            orientations = torch.zeros(
                self.n_patterns,
                4,
                device=self.patterns.device,
            )

            for phase_id, (quaternions, metrics) in self.raw_indexing_results.items():
                if higher_is_better:
                    mask = metrics > best_metrics
                else:
                    mask = metrics < best_metrics
                best_metrics[mask] = metrics[mask]
                phase_ids[mask] = phase_id
                orientations[mask] = quaternions[mask]
            self.phases = phase_ids
            self.orientations = orientations
            self.best_metrics = best_metrics
        else:
            self.phases = torch.full(
                (self.n_patterns,),
                fill_value=list(self.raw_indexing_results.keys())[0],
                dtype=torch.uint8,
                device=self.patterns.device,
            )
            self.orientations = self.raw_indexing_results[0][0][:, 0]
            self.best_metrics = self.raw_indexing_results[0][1][:, 0]
