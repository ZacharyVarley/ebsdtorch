import torch
from torch import Tensor


class EBSDPatterns:
    """

    Class for processing and sampling EBSD patterns from a scan.

    Args:
        data (torch.Tensor): The EBSD pattern data tensor.
        signal_dimensions (int): The number of signal dimensions in the data tensor.

    Notes:
        acceptable shapes are (*, H, W) where * is any number of dimensions
        acceptable data types are bool, uint8, int8/16/32/64, bfloat16, float16/32/64

    """

    def __init__(
        self,
        data: torch.Tensor,
    ):
        if len(data.shape) < 4:
            raise ValueError(
                "The EBSD pattern data tensor must have at least 4 dimensions."
            )
        # check data type
        self.map_shape = data.shape[:-2]
        self.pattern_shape = data.shape[-2:]

    def band_pass_filter(self, low_cutoff: float, high_cutoff: float, order: int = 2):
        raise NotImplementedError

    def contrast_limited_AHE(self, grid_size: tuple, clip_limit: float):
        raise NotImplementedError

    def nonlocal_pattern_average(self, kernel_radius: int, coeff: float):
        raise NotImplementedError

    def gaussian_downsample(self, factor: int, shape: tuple):
        raise NotImplementedError
