import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F


class GaussianBlur2d(Module):
    def __init__(
        self,
        sigma: float,
        z_score_cutoff: int = 3,
        reflect_padding: bool = True,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.z_score_cutoff = z_score_cutoff
        self.reflect_padding = reflect_padding

        # compute the Gaussian kernel
        kernel = self._get_gaussian_kernel()
        self.register_buffer("kernel", kernel)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(sigma={self.sigma}, "
            f"z_score_cutoff={self.z_score_cutoff}, "
            f"reflect_padding={self.reflect_padding})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply padding
        pad = self._compute_padding(x)
        if self.reflect_padding:
            x = F.pad(x, pad, mode="reflect")
        else:
            x = F.pad(x, pad, mode="constant", value=0)

        # apply convolution
        x = F.conv2d(x, self.kernel, padding=0)

        # crop off the padding
        h_even = x.shape[-2] % 2 == 0
        w_even = x.shape[-1] % 2 == 0
        x = x[
            :,
            :,
            : -2 if h_even else -1,
            : -2 if w_even else -1,
        ]
        return x

    def _get_gaussian_kernel(self) -> torch.Tensor:
        # establish the kernel size
        kernel_size = 2 * self.z_score_cutoff * self.sigma + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # create the kernel coordinates
        x = torch.arange(
            -kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32
        )
        y = torch.arange(
            -kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32
        )
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        # compute the Gaussian kernel
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel.shape[-1], kernel.shape[-2])

    def _compute_padding(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        kernel_size = self.kernel.shape[-1]
        w_even = x.shape[-1] % 2 == 0
        h_even = x.shape[-2] % 2 == 0
        return (
            kernel_size // 2,
            kernel_size // 2 + int(w_even),
            kernel_size // 2,
            kernel_size // 2 + int(h_even),
        )

    @staticmethod
    def _crop_padding(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
        return x[..., pad[2] : -pad[3], pad[0] : -pad[1]]


class BlurAndDownsample(Module):
    """

    This module uses standard blurring level of 2 * downscale / 6 for a Gaussian blur
    prior to downscaling by the scale factor. This is a common practice to avoid aliasing

    """

    def __init__(
        self,
        scale_factor: float = 2.0,
        mode: str = "bicubic",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.downscale = 1.0 / self.scale_factor
        sigma = 2.0 * self.scale_factor / 6.0
        self.gb = GaussianBlur2d(sigma=sigma)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        blurred = self.gb(x)
        return F.interpolate(
            blurred,
            size=(int(H * self.downscale), int(W * self.downscale)),
            mode=self.mode,
            align_corners=self.align_corners,
        )
