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
from ebsdtorch.s2_and_so3.orientations import qu2zh


class ExperimentPatterns(Module):
    """

    Class for processing and sampling EBSD patterns from a scan.

    Args:
        :patterns (Tensor): Pattern data tensor shaped (SCAN_H, SCAN_W, H, W),
            (N_PATS, H, W), or (H, W).
        :spatial_coords (Tensor): Spatial coordinates tensor shaped (SCAN_H,
            SCAN_W, N_Spatial_Dims), (N_PATS, N_Spatial_Dims), or
            (N_Spatial_Dims,).
        :consider_rotation (bool): Whether to consider rotations
        :consider_phases (bool): Whether to consider phases.
        :consider_strains (bool): Whether to consider strains.

    """

    def __init__(
        self,
        patterns: Tensor,
        spatial_coords: Tensor,
        grad_for_rotation: bool = False,
        consider_strains: bool = False,
        grad_for_strains: bool = False,
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

        # check that the spatial coordinates have the same shape as the patterns
        # except for the last two dimensions
        if spatial_coords.shape[:-1] != patterns.shape[:-2]:
            raise ValueError(
                f"'patterns' shape {tuple(patterns.shape)}, needs 'spatial_coords' "
                + f"shape {str(tuple(patterns.shape[:-2]))[:-1]} N_Spatial_Dims)"
            )

        self.pattern_shape = patterns.shape[-2:]
        self.spatial_shape = spatial_coords.shape[:-1]
        self.patterns = patterns.view(-1, *patterns.shape[-2:])

        # set number of patterns and pixels
        self.n_patterns = self.patterns.shape[0]
        self.n_pixels = self.pattern_shape[0] * self.pattern_shape[1]

        self.phases = None
        self.orientations = None
        self.f_matrix = None
        self.grad_for_strains = grad_for_strains
        self.consider_strains = consider_strains

    def set_rotations(
        self,
        rotations: Optional[Tensor] = None,
    ):
        """
        Set the rotations for the ExperimentPatterns object.

        Args:
            :rotations (Tensor): Rotations tensor shaped (N_PATS, 4) or (1, 4). If
                None, the rotations are set to the identity.

        """
        if rotations is None:
            rotations = torch.zeros(
                self.n_patterns,
                4,
                device=self.patterns.device,
            )
            rotations[:, 0] = 1.0
        else:
            if rotations.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Rotations must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {rotations.shape[0]} rotations and {self.n_patterns} patterns."
                )

            if rotations.shape[1] != 4:
                raise ValueError(
                    f"Rotations must be quaternions (w, x, y, z). Got {rotations.shape[1]} components."
                )
        self.orientations = rotations

    def set_rot_grad(
        self,
        rot_requires_grad: bool,
    ):
        """
        Set the rotations to require gradients or remove requirement. Diffable
        rotations are stored a Zhou et al 6D vectors with no option to use
        quaternions because they're strictly better for optimization. The method
        get_rotations will convert the rotations to quaternions (with
        differentiable gradients).

        Args:
            :rot_requires_grad (bool): Whether the rotations require gradients.

        """
        # if rotations are required to have gradients
        if rot_requires_grad:
            if self.orientations is not None:
                if self.orientations.shape[-1] == 4:
                    self.orientations = torch.nn.Parameter(qu2zh(self.orientations))
            else:
                raise ValueError(
                    "Rotations must be set via indexing or manually before requiring gradients."
                )
        else:
            if self.orientations is not None:
                if self.orientations.shape[-1] == 6:
                    self.orientations = qu2zh(self.orientations)
            else:
                raise ValueError(
                    "Rotations must be set via indexing or manually before removing gradients."
                )

    def set_deformation_gradient(
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
            f_matrix = f_matrix.view(self.n_patterns, 3, 3)
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
        self.f_matrix = f_matrix.view(self.n_patterns, 3, 3)

    def set_f_matrix_grad(
        self,
        grad_for_strains: bool,
    ):
        """
        Set the deformation gradient to require gradients or remove requirement.

        Args:
            :grad_for_strains (bool): Whether the deformation gradients require gradients.

        """
        if grad_for_strains:
            if self.f_matrix is not None:
                self.f_matrix = torch.nn.Parameter(self.f_matrix)
            else:
                raise ValueError(
                    "Deformation gradients must be set via indexing or manually before requiring gradients."
                )
        else:
            if self.f_matrix is not None:
                f_matrix_tmp = self.f_matrix
                del self.f_matrix
                self.register_buffer("f_matrix", f_matrix_tmp)
            else:
                raise ValueError(
                    "Deformation gradients must be set via indexing or manually before removing gradients."
                )

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
        n_bins: int = 256,
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
        self.normalize_per_pattern(norm_type="zeromean")

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
            return self.orientations
        return self.orientations[indices]

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
        if indices is None:
            if orientations.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Orientations must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {orientations.shape[0]} orientations and {self.n_patterns} patterns."
                )
            if orientations.shape[1] != 4:
                raise ValueError(
                    f"Orientations must be quaternions (w, x, y, z). Got {orientations.shape[1]} components."
                )
            self.orientations = orientations
        else:
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
