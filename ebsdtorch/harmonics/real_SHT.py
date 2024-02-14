"""

Real Valued Spherical Harmonic Transformations from TS2Kit:

https://github.com/twmitchel/TS2Kit/blob/main/TS2Kit.pdf

Side note thoughts on EMsoft's EMSphInx:

https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/square_sht.hpp

EMSphInx computes quadrature weights for latitude rings that make pixels
relatively equal area. I am not doing that because the quadrature weight
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


# Normalization coeffs for m-th frequency (C_m)
@torch.jit.script
def normCm(
    B: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """

    Args: B: Bandlimit of the grid.

    Returns: (Tensor): shape (2*bandlimit-1) with normalization coefficients.

    """
    # Cm = torch.empty(2 * B - 1, dtype=torch.double)
    # for m in range(-(B - 1), B):
    #     Cm[m + (B - 1)] = np.power(-1.0, m) * np.sqrt(2.0 * PI)

    # vectorized version:
    m = torch.arange(-(B - 1), B, dtype=torch.double, device=device)
    Cm = torch.pow(-1.0, m) * (2.0 * torch.pi) ** 0.5
    return Cm


# Discrete Legendre Transform weights
@torch.jit.script
def dltWeightsDH(
    B: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    # W = torch.empty(2 * B, dtype=torch.double)
    # for k in range(0, 2 * B):
    #     C = (2.0 / B) * np.sin(PI * (2 * k + 1) / (4.0 * B))
    #     wk = 0.0
    #     for p in range(0, B):
    #         wk += (1.0 / (2 * p + 1)) * np.sin(
    #             (2 * k + 1) * (2 * p + 1) * PI / (4.0 * B)
    #         )
    #     W[k] = C * wk

    # vectorized version:
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
    # DI = torch.empty(N, N).double().fill_(0)
    # for k in range(0, N):
    #     for n in range(0, N):
    #         DI[k, n] = np.cos(PI * n * (k + 0.5) / N)
    # DI[:, 0] = DI[:, 0] * np.sqrt(1.0 / N)
    # DI[:, 1:] = DI[:, 1:] * np.sqrt(2.0 / N)

    # vectorized version:
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
    # DI = torch.empty(N, N, dtype=torch.double).fill_(0)
    # for k in range(0, N):
    #     for n in range(0, N):
    #         if n == (N - 1):
    #             DI[k, n] = np.power(-1.0, k)
    #         else:
    #             DI[k, n] = np.sin(PI * (n + 1) * (k + 0.5) / N)
    # DI[:, N - 1] = DI[:, N - 1] * np.sqrt(1.0 / N)
    # DI[:, : (N - 1)] = DI[:, : (N - 1)] * np.sqrt(2.0 / N)

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
    """DCT or DST as a linear layer"""

    def __init__(self, B, xform):
        self.xform = xform
        self.B = B
        super(weightedDCST, self).__init__(2 * B, 2 * B, bias=False)

    def reset_parameters(self):
        B = self.B

        if self.xform == "c":
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idctMatrix(2 * B))

        elif self.xform == "ic":
            XF = idctMatrix(2 * B).t()

        elif self.xform == "s":
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idstMatrix(2 * B))

        elif self.xform == "is":
            XF = idstMatrix(2 * B).t()

        self.weight.data = XF.t().data
        self.weight.requires_grad = False  # don't learn this!


# Forward Discrete Legendre Transform
class FDLT(nn.Module):

    def __init__(self, B):
        super(FDLT, self).__init__()
        self.B = B
        self.dct = weightedDCST(B, "c")
        self.dst = weightedDCST(B, "s")

        if ((B - 1) % 2) == 1:
            cInd = torch.arange(1, 2 * B - 1, 2)
            sInd = torch.arange(0, 2 * B - 1, 2)

        else:
            sInd = torch.arange(1, 2 * B - 1, 2)
            cInd = torch.arange(0, 2 * B - 1, 2)

        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("Cm", normCm(B))

        print("Calling wigner_d_SHT_weights_half_pi")
        self.register_buffer("D", wigner_d_SHT_weights_half_pi(B))
        torch.cuda.synchronize()
        print("Computed d matrix")

    def forward(self, psiHat):
        # psiHat = b x M x phi
        B, b = self.B, psiHat.size()[0]

        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, :, None], psiHat)

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])

        # Reshape for sparse matrix multiplication
        psiHat = torch.transpose(torch.reshape(psiHat, (b, 2 * B * (2 * B - 1))), 0, 1)
        # Psi =  b x M x L
        return torch.permute(
            torch.reshape(torch.mm(self.D, psiHat), (2 * B - 1, B, b)), (2, 0, 1)
        )


# Inverse Discrete Legendre Transform
class IDLT(nn.Module):
    def __init__(self, B):
        super(IDLT, self).__init__()
        self.B = B
        self.dct = weightedDCST(B, "ic")
        self.dst = weightedDCST(B, "is")
        if ((B - 1) % 2) == 1:
            cInd = torch.arange(1, 2 * B - 1, 2)
            sInd = torch.arange(0, 2 * B - 1, 2)

        else:
            sInd = torch.arange(1, 2 * B - 1, 2)
            cInd = torch.arange(0, 2 * B - 1, 2)

        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("iCm", torch.reciprocal(normCm(B)))
        self.register_buffer(
            "DT", torch.transpose(wigner_d_SHT_weights_half_pi(B), 0, 1)
        )

    def forward(self, Psi):
        # Psi: b x M x L
        B, b = self.B, Psi.size()[0]
        psiHat = torch.reshape(
            torch.transpose(
                torch.mm(
                    self.DT,
                    torch.transpose(torch.reshape(Psi, (b, (2 * B - 1) * B)), 0, 1),
                ),
                0,
                1,
            ),
            (b, 2 * B - 1, 2 * B),
        )

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])

        # f: b x theta x phi
        return torch.mul(self.iCm[None, :, None], psiHat)


class FTSHT(nn.Module):
    """
    The Forward "Tensorized" Discrete Spherical Harmonic Transform

    Input:

    B: (int) Transform bandlimit

    """

    def __init__(self, B):
        super(FTSHT, self).__init__()
        self.B = B
        self.FDL = FDLT(B)

    def forward(self, psi):
        """
        Input:

        psi: ( b x 2B x 2B torch.double or torch.cdouble tensor )
             Real or complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions

        Output:

        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions

        """

        # psi: b x theta x phi (real or complex)
        B, b = self.B, psi.size()[0]

        ## FFT in polar component
        # psiHat: b x  M x Phi

        psiHat = torch.fft.fftshift(torch.fft.fft(psi, dim=1, norm="forward"), dim=1)[
            :, 1:, :
        ]

        ## Convert to real representation
        psiHat = torch.reshape(
            torch.permute(torch.view_as_real(psiHat), (0, 3, 1, 2)),
            (2 * b, 2 * B - 1, 2 * B),
        )

        # Forward DLT
        Psi = self.FDL(psiHat)

        # Convert back to complex and return
        # Psi: b x M x L (complex)

        return torch.view_as_complex(
            torch.permute(torch.reshape(Psi, (b, 2, 2 * B - 1, B)), (0, 2, 3, 1))
        )


class ITSHT(nn.Module):
    """
    The Inverse "Tensorized" Discrete Spherical Harmonic Transform

    Input:

    B: (int) Transform bandlimit

    """

    def __init__(self, B):
        super(ITSHT, self).__init__()
        self.B = B
        self.IDL = IDLT(B)

    def forward(self, Psi):
        """
        Input:

        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions

        Output:

        psi: ( b x 2B x 2B torch.cdouble tensor )
             Complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions

        """

        # Psi: b x  M x L (complex)
        B, b = self.B, Psi.size()[0]

        # Convert to real
        Psi = torch.reshape(
            torch.permute(torch.view_as_real(Psi), (0, 3, 1, 2)), (2 * b, 2 * B - 1, B)
        )

        # Inverse DLT
        psiHat = self.IDL(Psi)

        # Convert back to complex
        psiHat = torch.view_as_complex(
            torch.permute(torch.reshape(psiHat, (b, 2, 2 * B - 1, 2 * B)), (0, 2, 3, 1))
        )

        ## Set up for iFFT
        psiHat = torch.cat(
            (torch.empty(b, 1, 2 * B, device=psiHat.device).float().fill_(0), psiHat),
            dim=1,
        )

        # Inverse FFT and return
        # psi: b x theta x phi (complex)

        return torch.fft.ifft(torch.fft.ifftshift(psiHat, dim=1), dim=1, norm="forward")
