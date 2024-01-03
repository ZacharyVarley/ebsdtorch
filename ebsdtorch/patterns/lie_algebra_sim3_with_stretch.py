import torch
from torch import Tensor
from torch.nn import Module


class LieAlgebraPositionedRectangles(Module):
    """

    Module for differentiable homography estimation using Lie algebra vectors. The additive Lie
    algebra basis vectors are linearly combined according to the the internal parameter weighted
    by the weights parameter. The resulting matrix exponential is the homography. This module is
    meant to be used with a gradient-free optimizer.

    """

    def __init__(
        self,
        dtype_cast_to: torch.dtype = torch.float64,
        dtype_out: torch.dtype = torch.float32,
        x_translation_weight: float = 1.0,
        y_translation_weight: float = 1.0,
        z_translation_weight: float = 1.0,
        xy_rotation_weight: float = 1.0,
        xz_rotation_weight: float = 1.0,
        yz_rotation_weight: float = 1.0,
        xy_stretch_weight: float = 1.0,
    ):
        super().__init__()

        self.dtype_cast_to = dtype_cast_to
        self.dtype_out = dtype_out

        weights = torch.zeros((7,), dtype=dtype_cast_to)
        weights[0] = x_translation_weight
        weights[1] = y_translation_weight
        weights[2] = z_translation_weight
        weights[3] = xy_rotation_weight
        weights[4] = xz_rotation_weight
        weights[5] = yz_rotation_weight
        weights[6] = xy_stretch_weight

        self.register_buffer("weights", weights[None, :, None, None])

        elements = torch.zeros((7, 4, 4), dtype=dtype_cast_to)
        # --- translation ---
        elements[0, 0, 3] = 1.0  # translation in x
        elements[1, 1, 3] = 1.0  # translation in y
        elements[2, 2, 3] = 1.0  # translation in z
        # --- rotation yz ---
        elements[3, 1, 2] = -1.0
        elements[3, 2, 1] = 1.0
        # --- rotation xz ---
        elements[4, 0, 2] = 1.0
        elements[4, 2, 0] = -1.0
        # --- rotation xy ---
        elements[5, 0, 1] = -1.0
        elements[5, 1, 0] = 1.0
        # --- stretch  xy ---
        elements[6, 0, 0] = 1.0
        elements[6, 1, 1] = -1.0

        self.register_buffer("elements", elements)

    def forward(self, lie_vectors) -> Tensor:
        """
        Convert a batch of Lie algebra vectors to Lie group elements.

        Returns:
            The transforms shape (B, 4, 4).

        """
        # lie_vectors is shape (B, 7)
        # elements is shape (7, 4, 4)
        # weights is shape (1, 7, 1, 1)
        transforms = torch.linalg.matrix_exp(
            (
                lie_vectors[:, :, None, None].to(self.dtype_cast_to)
                * self.elements
                * self.weights
            ).sum(dim=1)
        )
        # make sure the transforms are normalized (bottom right element is 1.0)
        transforms = transforms / transforms[:, 3:4, 3:4]
        return transforms.to(self.dtype_out)

    def backwards(self, transforms: Tensor) -> Tensor:
        """

        Args:
            transforms: torch tensor of shape (B, 4, 4) containing the transforms

        Returns:
            The Lie algebra vectors shape (B, 7).

        """
        u, s, v = torch.linalg.svd(transforms)
        logm = u @ torch.diag_embed(torch.log(s)) @ v
        return logm
