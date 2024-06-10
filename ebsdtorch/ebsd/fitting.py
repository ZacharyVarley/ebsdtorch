"""
This module implements geometry fitters for EBSD data. The geometry fitters
are used to fit the transformation between the detector and sample reference
frames. The following geometry fitters are implemented:

FitPC: Fits one projection center
FitSE3: Fits the full SE(3) transformation

"""

import torch
from torch import Tensor
from ebsdtorch.ebsd.ebsd_master_patterns import MasterPattern
from ebsdtorch.ebsd.ebsd_experiment_pats import ExperimentPatterns
from ebsdtorch.ebsd.geometry import EBSDGeometry


# def fit_pc(
#     data: Tensor,
#     n_iter: int,
#     master_patterns: MasterPattern,
#     geometry: EBSDGeometry,
#     experiment_patterns: ExperimentPatterns,
# ) -> Tensor:
#     """
#     Fit the pattern center.

#     Args:
#         data (Tensor): The EBSD pattern data tensor.

#     Returns:
#         Tensor: The denoised EBSD pattern data tensor.

#     """
#     assert len(data.shape) == 4, "The EBSD pattern data tensor must have 4 dimensions."

#     # get the dimensions of the data tensor
#     h_scan, w_scan = data.shape[:-2]
#     h_pat, w_pat = data.shape[-2:]

#     return data_padded
