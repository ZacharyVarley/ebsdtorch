"""

This module contains the MasterPattern class, which is used to handle interpolation
of master patterns and apply CLAHE to them.

"""

from typing import Optional, Tuple, Union
from math import prod
import torch
from torch import Tensor
from torch.nn import Module
from ebsdtorch.utils.symmetry_classes import (
    space_group_to_laue,
    point_group_to_laue,
)
from ebsdtorch.s2_and_so3.sphere import rosca_lambert_side_by_side
from ebsdtorch.utils.factorize import nearly_square_factors
from ebsdtorch.preprocessing.clahe import clahe_grayscale


class MasterPattern(Module):
    """
    Master pattern class for storing and interpolating master patterns.

    Args:
        master_pattern (Tensor): The master pattern as a 2D tensor. The Northern hemisphere
            is on top and the Southern hemisphere is on the bottom. They are concatenated
            along the first dimension. If a tuple is provided, the first tensor is the
            Northern hemisphere and the second tensor is the Southern hemisphere.
        laue_group (int, str, optional):
            The Laue group number or symbol in Schönflies notation.
        point_group (str, optional):
            The point group symbol in Hermann-Mauguin notation.
        space_group (int, optional):
            The space group number. Must be between 1 and 230 inclusive.

    """

    def __init__(
        self,
        master_pattern: Union[Tensor, Tuple[Tensor, Tensor]],
        laue_group: Optional[int] = None,
        point_group: Optional[str] = None,
        space_group: Optional[int] = None,
    ):
        super(MasterPattern, self).__init__()

        # check that at least one of laue_groups, point_groups, or space_groups is not None
        if laue_group is None and point_group is None and space_group is None:
            raise ValueError(
                "At least one of laue_groups, point_groups, or space_groups must be provided"
            )

        # check that only one of laue_groups, point_groups, or space_groups is not None
        if (
            (laue_group is not None and point_group is not None)
            or (laue_group is not None and space_group is not None)
            or (point_group is not None and space_group is not None)
        ):
            raise ValueError(
                "Only one of laue_groups, point_groups, or space_groups should be provided"
            )

        # if master_pattern is a tuple, concatenate the two tensors
        if isinstance(master_pattern, tuple):
            # check they are both 2D tensors with the same shape
            if len(master_pattern[0].shape) != 2 or len(master_pattern[1].shape) != 2:
                raise ValueError(
                    "Both tensors in 'master_pattern' must be 2D tensors when provided as a tuple"
                )
            master_pattern = torch.cat(master_pattern, dim=-1)

        # check that the master pattern is a 2D tensor
        if len(master_pattern.shape) != 2:
            raise ValueError(
                f"'master_pattern' must be a 2D tensor, got shape {master_pattern.shape}"
            )

        self.register_buffer("master_pattern", master_pattern)

        # set the laue group
        if laue_group is not None:
            self.laue_group = laue_group
        elif point_group is not None:
            self.laue_group = point_group_to_laue(point_group)
        else:
            self.laue_group = space_group_to_laue(space_group)

        self.master_pattern = master_pattern
        self.master_pattern_binned = None
        self.factor_dict = {}

    def bin(
        self,
        binning_factor: int,
    ) -> None:
        """
        Set the binning factor for the master pattern under a separate attribute
        called `master_pattern_binned`.

        Args:
            :binning (Union[float, int]): Binning factor can be non-integer for pseudo-binning.

        """
        # blurrer = BlurAndDownsample(scale_factor=binning_factor).to(
        #     self.master_pattern.device
        # )
        # self.master_pattern_binned = blurrer(self.master_pattern[None, None, ...])[0, 0]

        self.master_pattern_binned = torch.nn.functional.avg_pool2d(
            self.master_pattern[None, None, ...], binning_factor, stride=binning_factor
        )[0, 0]

    def normalize(
        self,
        norm_type: str,
    ) -> None:
        """
        Normalize the master pattern.

        Args:
            :norm_type (str): Normalization type: "minmax", "zeromean", "standard"

        """
        if norm_type == "minmax":
            pat_mins = torch.min(self.master_pattern)
            pat_maxs = torch.max(self.master_pattern)
            self.master_pattern -= pat_mins
            self.master_pattern /= 1e-4 + pat_maxs - pat_mins
        elif norm_type == "zeromean":
            self.master_pattern -= torch.mean(self.master_pattern)
        elif norm_type == "standard":
            self.master_pattern -= torch.mean(self.master_pattern)
            self.master_pattern /= torch.std(self.master_pattern)
        else:
            raise ValueError(
                f"Invalid normalization method. Got {norm_type} but expected 'minmax', 'zeromean', or 'standard'."
            )

    def interpolate(
        self,
        coords_3D: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
        normalize_coords: bool = True,
        virtual_binning: int = 1,
    ) -> Tensor:
        """
        Interpolate the master pattern at the given angles.

        Args:
            :coords_3D (Tensor): (..., 3) Cartesian points to interpolate on the sphere.
            :mode (str): Interpolation mode. Default: "bilinear".
            :padding_mode (str): Padding mode. Default: "zeros".
            :align_corners (bool): Align corners. Default: True.
            :normalize_coords (bool): Normalize the coordinates. Default: True.
            :virtual_binning (int): Virtual binning factor on the passed coordinates. Default: 1.

        Returns:
            The interpolated master pattern pixel values.

        """
        # norm
        if normalize_coords:
            coords_3D = coords_3D / torch.norm(coords_3D, dim=-1, keepdim=True)

        # blur the master pattern if virtual binning is used
        if virtual_binning > 1:
            if self.master_pattern_binned is None:
                self.bin(virtual_binning)
            master_pattern_prep = self.master_pattern_binned
        else:
            master_pattern_prep = self.master_pattern

        # project to the square [-1, 1] x [-1, 1]
        projected_coords_2D = rosca_lambert_side_by_side(coords_3D)

        # we want to accept a generic number of dimensions (..., 2)
        # grid_sample will perform extremely poorly when given grid shaped:
        # (1, 1000000000, 1, 2) or (1, 1, 1000000000, 2) etc.
        # instead we find the pseudo height and width closest to that of a square
        # (..., 3) -> (1, H*, W*, 2) -> (...,) where H* x W* = n_coords
        #  3rd fastest integer factorization algorithm is good for small numbers
        n_elem = prod(projected_coords_2D.shape[:-1])
        if n_elem in self.factor_dict:
            H_pseudo, W_pseudo = self.factor_dict[n_elem]
        else:
            H_pseudo, W_pseudo = nearly_square_factors(n_elem)
            if len(self.factor_dict) > 100:
                self.factor_dict = {}
            self.factor_dict[n_elem] = H_pseudo, W_pseudo

        projected_coords_2D = projected_coords_2D.view(H_pseudo, W_pseudo, 2)

        output = torch.nn.functional.grid_sample(
            # master patterns are viewed as (1, 1, H_master, 2*W_master)
            master_pattern_prep[None, None, ...],
            # coordinates should usually be shaped:
            # (1, n_ori, H_pat*W_pat, 2) or (1, H_pat, W_pat, 2)
            projected_coords_2D[None, ...],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # reshape back to the original shape using coords_3D as a template
        return output.view(coords_3D.shape[:-1])

    def apply_clahe(
        self,
        clip_limit: float = 40.0,
        n_bins: int = 64,
        grid_shape: Tuple[int, int] = (8, 8),
    ):
        """
        Apply CLAHE to the master pattern.

        Args:
            :clip_limit (float): The clip limit for the histogram equalization.
            :n_bins (int): The number of bins to use for the histogram equalization.
            :grid_shape (Tuple[int, int]): The shape of the grid to divide the image into.

        """
        self.master_pattern = clahe_grayscale(
            self.master_pattern[None, None, ...],
            clip_limit=clip_limit,
            n_bins=n_bins,
            grid_shape=grid_shape,
        )[0, 0]
