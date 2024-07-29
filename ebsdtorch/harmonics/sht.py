"""

Originally based on TS2Kit (it only had complex valued SHTs):

https://github.com/twmitchel/TS2Kit/blob/main/TS2Kit.pdf

I had to unexpectedly swap subscripts k and m of the Wigner little d entries
when comparing the convention used by Fukushima / William Lenthe and that used
by Tommy Mitchel in TS2Kit.

Side note thoughts on EMsoft's EMSphInx:

https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/square_sht.hpp

EMSphInx manually computes quadrature weights for latitude rings that make
pixels relatively equal area. I ended up just using Driscoll-Healy for now.

"""

import torch
from torch import Tensor
from torch.nn import Module, Linear
from ebsdtorch.harmonics.wigner_d_logspace import (
    csht_weights_half_pi,
    rsht_weights_half_pi,
)


@torch.jit.script
def grid_DriscollHealy(
    bandlimit: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate Driscoll-Healy grid for spherical harmonics.

    Args: bandlimit: Bandlimit of the grid.

    Returns: (Tensor): shape (2*B, 2*B, 2) of theta then phi coords.

    """
    k = torch.arange(0, 2 * bandlimit, dtype=dtype)
    theta = torch.pi * (2 * k) / (2 * bandlimit)
    phi = torch.pi * (2 * k + 1) / (4 * bandlimit)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([theta, phi], dim=-1)


# Discrete Legendre Transform normalization coefficients
@torch.jit.script
def normCm(
    B: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """

    Args: B: Bandlimit of the grid.

    Returns: (Tensor): shape (2*bandlimit-1) with normalization coefficients.

    """
    m = torch.arange(-(B - 1), B, dtype=dtype, device=device)
    Cm = torch.pow(-1.0, m) * (2.0 * torch.pi) ** 0.5
    return Cm


# Discrete Legendre Transform normalization coefficients
@torch.jit.script
def rnormCm(
    B: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """

    Args: B: Bandlimit of the grid.

    Returns: (Tensor): shape (2*bandlimit-1) with normalization coefficients.

    """
    m = torch.arange(B, dtype=dtype, device=device)
    Cm = torch.pow(-1.0, m) * (2.0 * torch.pi) ** 0.5
    return Cm


# Discrete Legendre Transform weights for Driscoll-Healy grid
@torch.jit.script
def dltWeightsDH(
    B: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    k = torch.arange(0, 2 * B, dtype=dtype, device=device)
    C = (2.0 / B) * torch.sin(torch.pi * (2 * k + 1) / (4.0 * B))
    p = torch.arange(0, B, dtype=dtype, device=device).repeat(2 * B, 1)
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
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    kk, nn = torch.meshgrid(
        torch.arange(0, N, dtype=dtype, device=device),
        torch.arange(0, N, dtype=dtype, device=device),
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
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    kk, nn = torch.meshgrid(
        torch.arange(0, N, dtype=dtype, device=device),
        torch.arange(0, N, dtype=dtype, device=device),
        indexing="ij",
    )
    DI = torch.sin(torch.pi * (nn + 1) * (kk + 0.5) / N)
    DI[:, N - 1] = DI[:, N - 1] * (1.0 / N) ** 0.5
    DI[:, : (N - 1)] = DI[:, : (N - 1)] * (2.0 / N) ** 0.5
    return DI


# Weighted DCT and DST implemented as linear layers
# Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
class weightedDCST(Linear):
    """Discrete Cosine Transform and Discrete Sine Transform implemented as linear layers

    Args:
        L: (int) Transform bandlimit
        transform_type: (str) 'c' for DCT, 'ic' for inverse DCT, 's' for DST, 'is' for inverse DST
        device: (torch.device) Device to run the transform on
        dtype: (torch.dtype) Data type of the transform

    """

    def __init__(
        self,
        bandlimit: int,
        transform_type: str,
        device: torch.device,
        dtype=torch.float32,
    ):
        self.xform = transform_type
        self.L = bandlimit
        self.device = device
        self.dtype = dtype

        super(weightedDCST, self).__init__(
            in_features=2 * self.L,
            out_features=2 * self.L,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self):
        L = self.L
        if self.xform == "c":
            W = torch.diag(
                dltWeightsDH(
                    L,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            XF = torch.matmul(
                W,
                idctMatrix(
                    2 * L,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
        elif self.xform == "ic":
            XF = idctMatrix(
                2 * L,
                device=self.device,
                dtype=self.dtype,
            ).t()
        elif self.xform == "s":
            W = torch.diag(
                dltWeightsDH(
                    L,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            XF = torch.matmul(
                W,
                idstMatrix(
                    2 * L,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
        elif self.xform == "is":
            XF = idstMatrix(2 * L, device=self.device, dtype=self.dtype).t()
        self.weight.data = XF.t().data
        self.weight.requires_grad = False  # don't learn this!


class DLT(Module):
    def __init__(
        self,
        L: int,
        device: torch.device,
        precision: str = "single",
    ):
        super(DLT, self).__init__()
        self.L = L
        self.device = device
        if precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")
        self.dct = weightedDCST(L, "c", device, dtype)
        self.dst = weightedDCST(L, "s", device, dtype)
        self.idct = weightedDCST(L, "ic", device, dtype)
        self.idst = weightedDCST(L, "is", device, dtype)

        if ((L - 1) % 2) == 1:
            cInd = torch.arange(1, 2 * L - 1, 2, device=device)
            sInd = torch.arange(0, 2 * L - 1, 2, device=device)
        else:
            sInd = torch.arange(1, 2 * L - 1, 2, device=device)
            cInd = torch.arange(0, 2 * L - 1, 2, device=device)

        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("Cm", normCm(L, device=device, dtype=dtype))
        self.register_buffer(
            "iCm", torch.reciprocal(normCm(L, device=device, dtype=dtype))
        )
        wigner_d = csht_weights_half_pi(L, device=device, precision=precision)
        self.register_buffer("D", wigner_d)
        self.register_buffer("DT", wigner_d.transpose(0, 1))

    def fdlt(self, psiHat):
        # psiHat = b x M x phi
        L, b = self.L, psiHat.size()[0]

        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, :, None], psiHat)

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])

        # Compute the matrix multiplication
        Psi = torch.mm(
            torch.reshape(psiHat, (b, -1)),
            self.DT,
        ).reshape(b, 2 * L - 1, L)

        return Psi

    def idlt(self, Psi):
        # Psi: b x M x L
        L, b = self.L, Psi.size()[0]

        psiHat = torch.mm(
            Psi.view(b, -1),
            self.D,
        ).reshape(b, 2 * L - 1, 2 * L)

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.idct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.idst(psiHat[:, self.sInd, :])

        # f: b x theta x phi
        return torch.mul(self.iCm[None, :, None], psiHat)


class CSHT(Module):
    def __init__(self, L: int, device: torch.device, precision: str = "single"):
        super(CSHT, self).__init__()
        self.L = L
        self.device = device
        self.dlt = DLT(L, device, precision)

    def fsht(self, psi):
        """

        Forward spherical harmonic transform (fsht) of complex valued signal on S2.

        Input:

        psi: (b x 2L x 2L torch.double or torch.cdouble tensor)
            Spherical signal sampled on the 2L X 2L DH grid with b batch dimensions

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

        # Separate out a + bi  into (a, b) and move to the 2nd dimension
        # psiHat -> b x 2 x 2(L - 1) x 2L -> 2b x 2(L - 1) x 2L
        psiHat = torch.reshape(
            torch.permute(torch.view_as_real(psiHat), (0, 3, 1, 2)),
            (2 * b, 2 * L - 1, 2 * L),
        )

        # Forward DLT
        Psi = self.dlt.fdlt(psiHat)

        # Convert back to complex and return
        # Psi: b x M x L (complex)
        return torch.view_as_complex(
            torch.permute(torch.reshape(Psi, (b, 2, 2 * L - 1, L)), (0, 2, 3, 1))
        )

    def isht(self, Psi):
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

        Psi = torch.reshape(
            torch.permute(torch.view_as_real(Psi), (0, 3, 1, 2)), (2 * b, 2 * L - 1, L)
        )

        # Inverse DLT
        psiHat = self.dlt.idlt(Psi)

        # Convert back to complex
        psiHat = torch.view_as_complex(
            torch.permute(torch.reshape(psiHat, (b, 2, 2 * L - 1, 2 * L)), (0, 2, 3, 1))
        )

        psiHat = torch.cat(
            [
                torch.full(
                    (b, 1, 2 * L), 0.0, device=psiHat.device, dtype=psiHat.dtype
                ),
                psiHat,
            ],
            dim=1,
        )

        out = torch.fft.ifft(torch.fft.ifftshift(psiHat, dim=1), dim=1, norm="forward")

        # Inverse FFT and return
        # psi: b x theta x phi (complex)
        return out


class RDLT(Module):
    def __init__(
        self,
        L: int,
        device: torch.device,
        precision: str = "single",
    ):
        super(RDLT, self).__init__()
        self.L = L
        self.device = device
        if precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")
        self.dct = weightedDCST(L, "c", device, dtype)
        self.dst = weightedDCST(L, "s", device, dtype)
        self.idct = weightedDCST(L, "ic", device, dtype)
        self.idst = weightedDCST(L, "is", device, dtype)

        # DCT/DST indices for real SHT doesn't depend on parity of L
        sInd = torch.arange(1, L, 2, device=device)
        cInd = torch.arange(0, L, 2, device=device)

        self.register_buffer("cInd", cInd)
        self.register_buffer("sInd", sInd)
        self.register_buffer("Cm", rnormCm(L, device=device, dtype=dtype))
        self.register_buffer("iCm", torch.reciprocal(self.Cm))

        wigner_d = rsht_weights_half_pi(L, device, precision)
        self.register_buffer("D", wigner_d)
        self.register_buffer("DT", self.D.transpose(0, 1))

    def fdlt(self, psiHat):
        # psiHat = b x M x phi
        L, b = self.L, psiHat.size()[0]

        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, :, None], psiHat)

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :])

        # Compute the matrix multiplication
        Psi = torch.mm(
            torch.reshape(psiHat, (b, -1)),
            self.DT,
        ).reshape(b, L, L)

        return Psi

    def idlt(self, Psi):
        # Psi: b x M x L
        L, b = self.L, Psi.size()[0]

        psiHat = torch.mm(
            Psi.view(b, -1),
            self.D,
        ).reshape(b, L, 2 * L)

        # Apply DCT + DST to even + odd
        psiHat[:, self.cInd, :] = self.idct(psiHat[:, self.cInd, :])
        psiHat[:, self.sInd, :] = self.idst(psiHat[:, self.sInd, :])

        # f: b x theta x phi
        return torch.mul(self.iCm[None, :, None], psiHat)


class RSHT(Module):
    def __init__(self, L: int, device: torch.device, precision: str = "single"):
        super(RSHT, self).__init__()
        self.L = L
        self.device = device
        self.dlt = RDLT(L, device, precision)

    def fsht(self, psi):
        """
        Input:

        psi: (b x 2L x 2L tensor )
             Real spherical signal sampled on the 2L X 2L DH grid with b batch dimensions

        Output:

        Psi: (b x L x L tensor) Complex tensor of SH coefficients over b batch dimensions

        """

        # psi: b x theta x phi (real or complex)
        L, b = self.L, psi.size()[0]

        ## FFT in polar component
        # psiHat: b x M x Phi
        # complex fft just for debugging
        # psiHat = torch.fft.fftshift(torch.fft.fft(psi, dim=1, norm="forward"), dim=1)[
        #     :, -L:, :
        # ]
        psiHat = torch.fft.rfft(psi, dim=1, norm="forward")[:, :L, :]

        # Separate out a + bi  into (a, b) and move to the 2nd dimension
        # psiHat -> b x 2 x 2(L - 1) x 2L -> 2b x 2(L - 1) x 2L
        psiHat = torch.reshape(
            torch.permute(torch.view_as_real(psiHat), (0, 3, 1, 2)),
            (2 * b, L, 2 * L),
        )

        # Forward DLT
        Psi = self.dlt.fdlt(psiHat)

        # Convert back to complex and return
        # Psi: b x M x L (complex)
        return torch.view_as_complex(
            torch.permute(torch.reshape(Psi, (b, 2, L, L)), (0, 2, 3, 1))
        )

    def isht(self, Psi):
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
            torch.permute(torch.view_as_real(Psi), (0, 3, 1, 2)), (2 * b, L, L)
        )

        # Inverse DLT
        psiHat = self.dlt.idlt(Psi)

        # Convert back to complex
        psiHat = torch.view_as_complex(
            torch.permute(torch.reshape(psiHat, (b, 2, L, 2 * L)), (0, 2, 3, 1))
        )

        # need the missing zero row
        psiHat = torch.cat(
            [
                psiHat,
                torch.full(
                    (b, 1, 2 * L), 0.0, device=psiHat.device, dtype=psiHat.dtype
                ),
            ],
            dim=1,
        )

        # Inverse FFT and return
        # psi: b x theta x phi (complex)
        return torch.fft.irfft(psiHat, dim=1, norm="forward")


@torch.jit.script
def theta_phi_to_xyz(theta_phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates to cartesian coordinates.

    Args:
        theta (Tensor): shape (..., ) of polar declination angles
        phi (Tensor): shape (..., ) of azimuthal angles

    Returns:
        Tensor: torch tensor of shape (..., 3) containing the cartesian
        coordinates
    """
    theta, phi = torch.unbind(theta_phi, dim=-1)
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=-1,
    )


# import plotly.graph_objects as go

# # from plotly.subplots import make_subplots
# from ebsdtorch.ebsd.master_pattern import MasterPattern
# from ebsdtorch.io.read_master_pattern import read_master_pattern


# # Load master pattern
# mp_fname = "../EMs/EMplay/old/Si-master-20kV.h5"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# mp = read_master_pattern(mp_fname).to(device)

# # # # just compare the coefficients from the real and complex valued transforms
# # # L = 5
# # # sht = RSHT(L, device).float()
# # # csht = CSHT(L, device).float()

# # # # Generate a sampling grid
# # # dh_grid = grid_DriscollHealy(L, torch.float32).to(device)
# # # xyz_grid = theta_phi_to_xyz(dh_grid).float()

# # # # Interpolate master pattern
# # # mp_0 = (
# # #     mp.interpolate(
# # #         xyz_grid,
# # #         mode="bicubic",
# # #         padding_mode="border",
# # #         align_corners=False,
# # #         normalize_coords=True,
# # #         virtual_binning=4 if L < 128 else 1,
# # #     )
# # #     .squeeze(0)
# # #     .float()
# # # )

# # # # Apply the SHT
# # # mp_0_sht = sht.fsht(mp_0.unsqueeze(0))
# # # mp_0_isht = sht.isht(mp_0_sht).squeeze().abs()
# # # print("-" * 80)
# # # mp_0_csht = csht.fsht(mp_0.unsqueeze(0))
# # # mp_0_icsht = csht.isht(mp_0_csht).squeeze().abs()

# # # # print(f"mp_0_sht: \n{mp_0_sht.abs().detach().cpu().numpy().round(3)}")
# # # # print(f"mp_0_csht: \n{mp_0_csht.abs().detach().cpu().numpy().round(3)}")

# # # # print(f"mp_0_isht: \n{mp_0_isht.abs().detach().cpu().numpy().round(3)}")
# # # # print(f"mp_0_icsht: \n{mp_0_icsht.abs().detach().cpu().numpy().round(3)}")

# # # # # again but only for the magnitude
# # # # print(f"mp_0_sht: \n{mp_0_sht.abs().detach().cpu().numpy().round(3)}")
# # # # print(f"mp_0_csht: \n{mp_0_csht.abs().detach().cpu().numpy().round(3)}")

# # # -------------------------------

# # Precompute the inverse SHT results for different bandlimits
# # bandlimits = [64, 65, 66, 67, 68, 69, 70, 71]
# bandlimits = [126, 127, 128, 129, 130, 131, 132, 133]
# # bandlimits = [256, 257, 258, 259, 260, 261, 262, 263]
# # bandlimits = [512, 513, 514, 515, 516, 517, 518, 519]
# precomputed_results = {}

# hd_bandlimit = 500
# sht_full = RSHT(hd_bandlimit, device).float()

# for L in bandlimits:
#     print(f"Computing bandlimit {L}")
#     # Generate a sampling grid
#     dh_grid_hd = grid_DriscollHealy(hd_bandlimit, torch.float32).to(device)
#     xyz_grid_hd = theta_phi_to_xyz(dh_grid_hd).float()

#     # Interpolate master pattern
#     mp_0 = (
#         mp.interpolate(
#             xyz_grid_hd,
#             mode="bicubic",
#             padding_mode="border",
#             align_corners=False,
#             normalize_coords=True,
#             virtual_binning=1,
#         )
#         .squeeze(0)
#         .float()
#     )

#     sht = RSHT(L, device).float()

#     # print total size of sht parameters in GB
#     # not just parameters, but also buffers
#     buffer_size = 0
#     for buffer in sht.buffers():
#         # check if its a sparse tensor
#         try:
#             buffer_size += buffer._nnz() * buffer.element_size()
#         except:
#             buffer_size += buffer.numel() * buffer.element_size()
#     print(f"Size of SHT buffers (GB): {buffer_size / 1024**3}")

#     # Apply the SHT
#     mp_0_sht = sht_full.fsht(mp_0.unsqueeze(0))

#     # trim the coefficients
#     mp_0_sht = mp_0_sht[:, :L, :L].contiguous()

#     # Apply the inverse SHT
#     mp_0_isht = sht.isht(mp_0_sht).squeeze().abs()

#     print(f"coeff shape: {mp_0_sht.shape}")

#     # print min/max of the reconstructed signal
#     print(f"min: {mp_0_isht.min().item()}, max: {mp_0_isht.max().item()}")

#     # make lower resolution xyz grid for plotting
#     dh_grid = grid_DriscollHealy(L, torch.float32).to(device)
#     xyz_grid = theta_phi_to_xyz(dh_grid).float()

#     # Append extra azimuthal angle to close the sphere
#     xyz_grid_plot = torch.cat((xyz_grid, xyz_grid[0:1, :, :]), dim=-3)
#     mp_0_plot = torch.cat((mp_0, mp_0[0:1, :]), dim=0)
#     mp_0_isht_plot = torch.cat((mp_0_isht, mp_0_isht[0:1, :]), dim=0)

#     # append extra polar angle to close the sphere
#     xyz_grid_plot = torch.cat((xyz_grid_plot, xyz_grid_plot[:, 0:1, :]), dim=-2)
#     mp_0_plot = torch.cat((mp_0_plot, mp_0_plot[:, 0:1]), dim=-1)
#     mp_0_isht_plot = torch.cat((mp_0_isht_plot, mp_0_isht_plot[:, 0:1]), dim=-1)

#     # the extra polar angle points have the wrong z sign
#     xyz_grid_plot[:, -1, 2] = -xyz_grid_plot[:, -1, 2]

#     # normalize the mp_0 and mp_0_isht to [0, 1]
#     mp_0_plot = (mp_0_plot - mp_0_plot.min()) / (mp_0_plot.max() - mp_0_plot.min())
#     mp_0_isht_plot = (mp_0_isht_plot - mp_0_isht_plot.min()) / (
#         mp_0_isht_plot.max() - mp_0_isht_plot.min()
#     )

#     # send to cpu as numpy
#     xyz_grid_plot = xyz_grid_plot.cpu().numpy()
#     mp_0_plot = 1 - mp_0_plot.cpu().numpy()
#     mp_0_isht_plot = 1 - mp_0_isht_plot.cpu().numpy()

#     # Cache the results
#     precomputed_results[L] = {
#         "xyz_grid_plot": xyz_grid_plot,
#         "mp_0_plot": (mp_0_plot * 255).astype("uint8"),
#         "mp_0_isht_plot": (mp_0_isht_plot * 255).astype("uint8"),
#     }

# # Create the initial plot with all traces and set only the first one to visible
# # fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]])
# fig = go.Figure()

# for i, L in enumerate(bandlimits):
#     result = precomputed_results[L]
#     visibility = True if i == 0 else False

#     # surf = go.Surface(
#     #     x=result["xyz_grid_plot"][..., 0],
#     #     y=result["xyz_grid_plot"][..., 1],
#     #     z=result["xyz_grid_plot"][..., 2],
#     #     surfacecolor=result["mp_0_plot"],
#     #     colorscale="Greys",
#     #     name=f"Original {L}",
#     #     visible=visibility,
#     # )
#     # surf_isht = go.Surface(
#     #     x=result["xyz_grid_plot"][..., 0],
#     #     y=result["xyz_grid_plot"][..., 1],
#     #     z=result["xyz_grid_plot"][..., 2],
#     #     surfacecolor=result["mp_0_isht_plot"],
#     #     colorscale="Greys",
#     #     name=f"Inverse SHT {L}",
#     #     visible=visibility,
#     # )

#     # fig.add_trace(surf, row=1, col=1)
#     # fig.add_trace(surf_isht, row=1, col=2)

#     surf_isht = go.Surface(
#         x=result["xyz_grid_plot"][..., 0],
#         y=result["xyz_grid_plot"][..., 1],
#         z=result["xyz_grid_plot"][..., 2],
#         surfacecolor=result["mp_0_isht_plot"],
#         colorscale="Greys",
#         name=f"Inverse SHT {L}",
#         visible=visibility,
#     )
#     fig.add_trace(surf_isht)

# # Create slider steps to toggle visibility
# steps = []
# for i, L in enumerate(bandlimits):
#     step = dict(
#         method="update",
#         args=[
#             {"visible": [False] * (len(bandlimits))},
#             {"title": f"Bandlimit: {L}"},
#         ],
#         label=f"{L}",
#     )
#     step["args"][0]["visible"][i] = True
#     steps.append(step)

# # Add sliders to figure
# sliders = [
#     dict(active=0, currentvalue={"prefix": "Bandlimit: "}, pad={"t": 50}, steps=steps)
# ]

# fig.update_layout(sliders=sliders, scene=dict(aspectratio=dict(x=1, y=1, z=1)))

# # fig.update_layout(
# #     scene=dict(
# #         aspectratio=dict(x=1, y=1, z=1),
# #     )
# # )

# # Show the figure
# fig.show()
