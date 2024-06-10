"""

Real Valued Spherical Harmonic Transformations from TS2Kit:

https://github.com/twmitchel/TS2Kit/blob/main/TS2Kit.pdf

I had to unexpectedly swap subscripts k and m of the Wigner little d entries
when comparing the convention used by Fukushima / William Lenthe and that used
by Tommy Mitchel in TS2Kit. Spherical harmonic conventions are an absolute mess.

Side note thoughts on EMsoft's EMSphInx:

https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/square_sht.hpp

EMSphInx manually computes quadrature weights for latitude rings that make
pixels relatively equal area. I am not doing that because the quadrature weight
calculation is O(B^4) as far as I understand. Might add this later.

Side note thoughts on Euler angles and SO(3) FFT:

The author of spherical package (https://github.com/moble/spherical), Michael
Boyle, absolutely detests Euler angles, but as far as I understand, the SO(3)
FFT uses ZYZ angles because it must. Cross correlation over SO(3) is simply a
few inverse 3D FFTs with real valued (in ZYZ case) Wignder d-matrix coefficients
prefactors. This is because the volume is periodic about each axis, which is
coincident with choosing to use Euler angles from the beginning... I think. I
believe that no other orientation representations have 1D rings along a given
dimension. Looking for more resources on this.

Side note on torch-harmonics (https://github.com/NVIDIA/torch-harmonics/)

I would have liked to use torch-harmonics, but the round trip mean absolute
error of going from a tensor of complex 128 bit SH coefficients to a tensor of
real spherical signals and back to spectral domain is around 0.005 which is
crazy high. I am almost certain that I am doing something wrong because the
error should be around 1e-15 for double precision. Either I did something wrong
or there is a fatal bug in the library or precision is not a consideration for
that context. They have Github issues disabled so I cannot ask there... O.o

"""

import torch
import torch.nn as nn
from torch import Tensor
from ebsdtorch.wigner.wigner_d_half_pi import wigner_d_SHT_weights_half_pi


@torch.jit.script
def grid_DriscollHealy(
    bandlimit: int,
) -> Tensor:
    """
    Generate Driscoll-Healy grid for spherical harmonics.

    Args: bandlimit: Bandlimit of the grid.

    Returns: (Tensor): shape (2*B, 2*B, 2) of theta then phi coords.

    """
    k = torch.arange(0, 2 * bandlimit).double()
    theta = 2 * torch.pi * k / (2 * bandlimit)
    phi = torch.pi * (2 * k + 1) / (4 * bandlimit)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([theta, phi], dim=-1)


@torch.jit.script
def normCm(
    B: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """

    Args: B: Bandlimit of the grid.

    Returns: (Tensor): shape (2*bandlimit-1) with normalization coefficients.

    """
    m = torch.arange(-(B - 1), B, dtype=torch.double, device=device)
    Cm = torch.pow(-1.0, m) * (2.0 * torch.pi) ** 0.5
    return Cm


# Discrete Legendre Transform weights
@torch.jit.script
def dltWeightsDH(
    B: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    k = torch.arange(0, 2 * B, dtype=torch.double, device=device)
    C = (2.0 / B) * torch.sin(torch.pi * (2 * k + 1) / (4.0 * B))
    p = torch.arange(0, B, dtype=torch.double, device=device).repeat(2 * B, 1)
    wk = torch.sum(
        (1.0 / (2 * p + 1))
        * torch.sin((2 * k[:, None] + 1) * (2 * p + 1) * torch.pi / (4.0 * B)),
        dim=1,
    )
    W = C * wk
    return W


## Inverse (orthogonal) DCT Matrix of dimension N x N
@torch.jit.script
def idctMatrix(
    N: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    kk, nn = torch.meshgrid(
        torch.arange(0, N, dtype=torch.double, device=device),
        torch.arange(0, N, dtype=torch.double, device=device),
        indexing="ij",
    )
    DI = torch.cos(torch.pi * nn * (kk + 0.5) / N)
    DI[:, 0] = DI[:, 0] * (1.0 / N) ** 0.5
    DI[:, 1:] = DI[:, 1:] * (2.0 / N) ** 0.5
    return DI


## Inverse (orthogonal) DST Matrix of dimension N x N
@torch.jit.script
def idstMatrix(
    N: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    # vectorized version:
    kk, nn = torch.meshgrid(
        torch.arange(0, N, dtype=torch.double, device=device),
        torch.arange(0, N, dtype=torch.double, device=device),
        indexing="ij",
    )
    DI = torch.sin(torch.pi * (nn + 1) * (kk + 0.5) / N)
    DI[:, N - 1] = DI[:, N - 1] * (1.0 / N) ** 0.5
    DI[:, : (N - 1)] = DI[:, : (N - 1)] * (2.0 / N) ** 0.5
    return DI


# Weighted DCT and DST implemented as linear layers
# Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
class weightedDCST(nn.Linear):
    """Discrete Cosine Transform and Discrete Sine Transform implemented as linear layers

    Args:
        L: (int) Transform bandlimit
        xform: (str) "c" for DCT, "ic" for inverse DCT, "s" for DST, "is" for inverse DST

    """

    def __init__(self, L, xform):
        self.xform = xform
        self.L = L
        super(weightedDCST, self).__init__(2 * L, 2 * L, bias=False)

    def reset_parameters(self):
        L = self.L

        if self.xform == "c":
            W = torch.diag(dltWeightsDH(L))
            XF = torch.matmul(W, idctMatrix(2 * L))

        elif self.xform == "ic":
            XF = idctMatrix(2 * L).t()

        elif self.xform == "s":
            W = torch.diag(dltWeightsDH(L))
            XF = torch.matmul(W, idstMatrix(2 * L))

        elif self.xform == "is":
            XF = idstMatrix(2 * L).t()

        self.weight.data = XF.t().data
        self.weight.requires_grad = False  # don't learn this!


# Forward Discrete Legendre Transform
class FDLT(nn.Module):

    def __init__(
        self,
        L: int,
    ):
        super(FDLT, self).__init__()
        self.L = L
        self.dct = weightedDCST(L, "c")
        self.dst = weightedDCST(L, "s")

        if ((L - 1) % 2) == 1:
            cInd = torch.arange(1, 2 * L - 1, 2)
            sInd = torch.arange(0, 2 * L - 1, 2)

        else:
            sInd = torch.arange(1, 2 * L - 1, 2)
            cInd = torch.arange(0, 2 * L - 1, 2)

        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("Cm", normCm(L))
        self.register_buffer("D", wigner_d_SHT_weights_half_pi(L))

    def forward(self, psiHat):
        # psiHat = b x M x phi
        L, b = self.L, psiHat.size()[0]

        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, :, None], psiHat)

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])

        # Reshape for sparse matrix multiplication
        psiHat = torch.transpose(torch.reshape(psiHat, (b, 2 * L * (2 * L - 1))), 0, 1)
        # Psi =  b x M x L
        return torch.permute(
            torch.reshape(torch.mm(self.D, psiHat), (2 * L - 1, L, b)), (2, 0, 1)
        )


# Inverse Discrete Legendre Transform
class IDLT(nn.Module):
    """

    Inverse Discrete Legendre Transform

    Args:
        L: (int) Transform bandlimit

    """

    def __init__(
        self,
        L: int,
    ):
        super(IDLT, self).__init__()
        self.L = L
        self.dct = weightedDCST(L, "ic")
        self.dst = weightedDCST(L, "is")
        if ((L - 1) % 2) == 1:
            cInd = torch.arange(1, 2 * L - 1, 2)
            sInd = torch.arange(0, 2 * L - 1, 2)

        else:
            sInd = torch.arange(1, 2 * L - 1, 2)
            cInd = torch.arange(0, 2 * L - 1, 2)
        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("iCm", torch.reciprocal(normCm(L)))
        self.register_buffer(
            "DT", torch.transpose(wigner_d_SHT_weights_half_pi(L), 0, 1)
        )

    def forward(
        self,
        Psi: Tensor,
    ):
        # Psi: b x M x L
        L, b = self.L, Psi.size()[0]
        psiHat = torch.reshape(
            torch.transpose(
                torch.mm(
                    self.DT,
                    torch.transpose(torch.reshape(Psi, (b, (2 * L - 1) * L)), 0, 1),
                ),
                0,
                1,
            ),
            (b, 2 * L - 1, 2 * L),
        )
        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])
        # f: b x theta x phi
        return torch.mul(self.iCm[None, :, None], psiHat)


class FTSHT(nn.Module):
    """
    The Forward "Tensorized" Discrete Spherical Harmonic Transform

    Args:
        L: (int) Transform bandlimit
    """

    def __init__(
        self,
        L: int,
    ):
        super(FTSHT, self).__init__()
        self.L = L
        self.FDL = FDLT(L)

    def forward(self, psi):
        """
        Input:

        psi: (b x 2L x 2L torch.double or torch.cdouble tensor )
             Real or complex spherical signal sampled on the 2L X 2L DH grid with b batch dimensions

        Output:

        Psi: (b x (2L - 1) x L torch.cdouble tensor) Complex tensor of SH coefficients over b batch dimensions

        """

        # psi: b x theta x phi (real or complex)
        L, b = self.L, psi.size()[0]

        ## FFT in polar component
        # psiHat: b x  M x Phi

        psiHat = torch.fft.fftshift(torch.fft.fft(psi, dim=1, norm="forward"), dim=1)[
            :, 1:, :
        ]
        ## Convert to real representation
        psiHat = torch.reshape(
            torch.permute(torch.view_as_real(psiHat), (0, 3, 1, 2)),
            (2 * b, 2 * L - 1, 2 * L),
        )
        # Forward DLT
        Psi = self.FDL(psiHat)
        # Convert back to complex and return
        # Psi: b x M x L (complex)
        return torch.view_as_complex(
            torch.permute(torch.reshape(Psi, (b, 2, 2 * L - 1, L)), (0, 2, 3, 1))
        )


class ITSHT(nn.Module):
    """
    The Inverse "Tensorized" Discrete Spherical Harmonic Transform

    Input:

    B: (int) Transform bandlimit

    """

    def __init__(
        self,
        L: int,
    ):
        super(ITSHT, self).__init__()
        self.L = L
        self.IDL = IDLT(L)

    def forward(
        self,
        Psi: Tensor,
    ):
        """
        Input:

        Psi: (b x (2L - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions

        Output:

        psi: ( b x 2L x 2L torch.cdouble tensor )
             Complex spherical signal sampled on the 2L X 2L DH grid with b batch dimensions

        """

        # Psi: b x  M x L (complex)
        L, b = self.L, Psi.size()[0]

        # Convert to real
        Psi = torch.reshape(
            torch.permute(torch.view_as_real(Psi), (0, 3, 1, 2)), (2 * b, 2 * L - 1, L)
        )

        # Inverse DLT
        psiHat = self.IDL(Psi)

        # Convert back to complex
        psiHat = torch.view_as_complex(
            torch.permute(torch.reshape(psiHat, (b, 2, 2 * L - 1, 2 * L)), (0, 2, 3, 1))
        )

        ## Set up for iFFT
        psiHat = torch.cat(
            (torch.empty(b, 1, 2 * L, device=psiHat.device).float().fill_(0), psiHat),
            dim=1,
        )

        # Inverse FFT and return
        # psi: b x theta x phi (complex)
        return torch.fft.ifft(torch.fft.ifftshift(psiHat, dim=1), dim=1, norm="forward")
