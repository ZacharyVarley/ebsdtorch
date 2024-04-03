"""

This module contains the functions cross correlate two functions on the 2-sphere
over the group SO(3), using Spherical Harmonics. 

"""

import torch
from torch.fft import ifftn
from ebsdtorch.wigner.wigner_d_half_pi import wigner_d_SOFT_weights


class CCSOFT(torch.nn.Module):
    """
    This class contains the functions to compute the cross correlation of two
    functions on the 2-sphere over the group SO(3), using Spherical Harmonics.
    """

    def __init__(self, L):
        """
        Args:
            L: int, the maximum degree of the spherical harmonics.
            device: str, the device to run the computations on.
        """
        super(CCSOFT, self).__init__()
        self.L = L
        # get the L x 2L - 1 x 2L - 1 precomputed Wigner d-matrices
        wigner_d_l_m_mprime = wigner_d_SOFT_weights(L)
        # use einsum to precompute d_l_m_k(pi/2) * d_l_k_n(pi/2)
        self.wigner_d_precompute = torch.einsum(
            "lmk,lkn->lmkn", wigner_d_l_m_mprime, wigner_d_l_m_mprime
        ).to(torch.complex64)

    def forward(self, f, g):
        """
        This function computes the cross correlation of two functions on the
        2-sphere over the group SO(3), using Spherical Harmonics.

        Args:
            f: complex Tensor of shape (b, L, 2*L - 1) g: complex Tensor of
            shape (b, L, 2*L - 1)

        Returns:
            cc: Real Tensor of shape (b, 2L - 1, 2L - 1, 2L - 1)

        """
        # check that the input tensors have the correct shape
        if f.shape[1] != self.L or f.shape[2] != 2 * self.L - 1:
            raise ValueError(
                f"Expected f to have shape (b, {self.L}, {2*self.L - 1}), got {f.shape}"
            )
        if g.shape[1] != self.L or g.shape[2] != 2 * self.L - 1:
            raise ValueError(
                f"Expected g to have shape (b, {self.L}, {2*self.L - 1}), got {g.shape}"
            )

        # Compute the outer product of f and g
        f_outer_g = torch.einsum("blm,bln->blmn", f, g.conj())

        # Multiply with the precomputed Wigner d-matrices
        cc_in_SO3_harmonics = torch.einsum(
            "lmkn,blmn->bmkn", self.wigner_d_precompute, f_outer_g
        )

        # Inverse 3D FFT
        cc_real_space = ifftn(cc_in_SO3_harmonics, dim=(-3, -2, -1))

        return cc_real_space
