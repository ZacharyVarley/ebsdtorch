"""

This module contains the functions cross correlate two functions on the 2-sphere
over the group SO(3), using spherical harmonics. 

Notes for understanding the code:

- Symmetry-less real valued cross correlation volume: 
- (2L - 1) X (2L - 1) X (L) in the order of (n, k, m)
- PyTorch real FFT calls have trailing half dimension

These papers don't explicitly label the summation over m and n:

1) Gutman, Boris, et al. "Shape registration with spherical cross correlation."
   2nd MICCAI Workshop on Mathematical Foundations of Computational Anatomy.
   2008.

2) Sorgi, Lorenzo, and Kostas Daniilidis. "Template gradient matching in
   spherical images." Image Processing: Algorithms and Systems III. Vol. 5298.
   SPIE, 2004.

3) Hielscher, Ralf, Felix Bartel, and Thomas Benjamin Britton. "Gazing at
   crystal balls: Electron backscatter diffraction pattern analysis and cross
   correlation on the sphere." Ultramicroscopy 207 (2019): 112836.


so one might think m ranges from 0 to L and not -L to L.

But this reference in equation (6) explicitly states the range of m and n:

Carpentier, Thibaut, and Aaron Einbond. "Spherical correlation as a similarity
measure for 3-D radiation patterns of musical instruments." Acta Acustica 7
(2023): 40.

Then if you look at the EMSphInx code:

https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/sht_xcorr.hpp

One sees that m only ranges from 0 to L but the inverse FFT is done with an 
output spatial size of 2L - 1 nonetheless.


"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.fft import ifftn, irfftn, ifftshift
from ebsdtorch.harmonics.sht import CSHT, RSHT, grid_DriscollHealy, theta_phi_to_xyz
from ebsdtorch.harmonics.wigner_d_logspace import (
    wigner_d_eq_half_pi,
    read_mn_wigner_d_half_pi_table,
    read_lmn_wigner_d_half_pi_table,
)
from ebsdtorch.io.read_master_pattern import read_master_pattern


@torch.jit.script
def rs2cc_(
    B: int,
    f: Tensor,
    g: Tensor,
    wigner_d_precompute: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, B, B) for real S2 function
        g: complex tensor of coefficients shape (b, B, B) for real S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume of variable shape


    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """
    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    b = max(b1, b2)

    # iterate over l as the 4D volume shape L^4 will often not fit in memory
    cc = torch.zeros(
        (b, B, 2 * B - 1, 2 * B - 1), dtype=torch.complex64, device=f.device
    )
    gconj = g.conj()

    for l in range(B):
        m_inds_half = torch.arange(0, l + 1, dtype=torch.int32, device=f.device)
        # get the relevant harmonic coefficients for g and f for the current l
        g_n = torch.cat(
            [
                (g[:, m_inds_half[1:], l] * (-1.0) ** m_inds_half[1:][None, :]).flip(1),
                gconj[:, m_inds_half, l],
            ],
            dim=1,
        )[
            :, None, None, :
        ]  # (..., m, k, n*)
        f_m = f[:, m_inds_half, l][:, :, None, None]  # (..., m*, k, n)

        m, k = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        if l != 0:
            m = m[l:]
            k = k[l:]

        # get the precomputed Wigner d-matrices for the current l
        d_lmk = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=m,
            n_coords=k,
            l=l,
        )[
            None, :, :, None
        ]  # (B, m*, k*, n)
        k, n = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        d_lkn = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=k,
            n_coords=n,
            l=l,
        )[
            None, None, :, :
        ]  # (B, m, k*, n*)

        cc[:, : (l + 1), (-l + B - 1) : (l + B), (-l + B - 1) : (l + B)] += (
            f_m * g_n * (d_lmk * d_lkn).to(torch.complex64)
        )

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = ifftshift(cc, dim=(-1, -2))

    # dim=(..., -3) means the half dimension is the -3 dimension (m)
    cc = irfftn(
        cc,
        s=(2 * B - 1, 2 * B - 1, 2 * B - 1),
        dim=(-1, -2, -3),
        norm="forward",
    )

    return cc


@torch.jit.script
def rs2cc_fast_(
    B: int,
    f: Tensor,
    g: Tensor,
    d_lmk: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, B, B) for real S2 function
        g: complex tensor of coefficients shape (b, B, B) for real S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume

    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """

    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    m_inds_half = torch.arange(1, B, dtype=torch.int32, device=f.device)
    # augment g and f with zeros for the negative m values
    g_n_aug = torch.cat(
        [
            g[:, 1:, :].flip(1) * ((-1.0) ** m_inds_half.flip(0))[None, :, None],
            g.conj(),
        ],
        dim=1,
    )

    f_lmkb = torch.einsum("lmk,bml->blmk", d_lmk[:, (B - 1) :, :], f)
    g_lknb = torch.einsum("lkn,bnl->blkn", d_lmk[:, :, (B - 1) :], g.conj())
    cc = torch.einsum("blmk,blkn->bmkn", f_lmkb, g_lknb)

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = ifftshift(cc, dim=(-2))

    # dim=(..., -3) means the half dimension is the -3 dimension (m)
    cc = irfftn(cc, s=(2 * B - 1,) * 3, dim=(-1, -2, -3), norm="forward")

    return cc


@torch.jit.script
def cs2cc_(
    B: int,
    f: Tensor,
    g: Tensor,
    wigner_d_precompute: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, 2B - 1, B) for complex S2 function
        g: complex tensor of coefficients shape (b, 2B - 1, B) for complex S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume of variable shape


    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """

    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    b = max(b1, b2)

    # iterate over l as the 4D volume shape L^4 will often not fit in memory
    cc = torch.zeros(
        (b, 2 * B - 1, 2 * B - 1, 2 * B - 1), dtype=torch.complex64, device=f.device
    )

    gconj = g.conj()

    for l in range(B):
        m_inds_full = (
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device) + B - 1
        )
        # get the relevant harmonic coefficients for g and f for the current l
        g_n = gconj[:, m_inds_full, l][:, None, None, :]  # (..., m, k, n)
        f_m = f[:, m_inds_full, l][:, :, None, None]  # (..., m, k, n)

        m, k = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        # get the precomputed Wigner d-matrices for the current l
        d_lmk = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=m,
            n_coords=k,
            l=l,
        )[
            None, :, :, None
        ]  # (B, m*, k*, n)
        k, n = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        d_lkn = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=k,
            n_coords=n,
            l=l,
        )[
            None, None, :, :
        ]  # (B, m, k*, n*)
        cc[
            :, (-l + B - 1) : (l + B), (-l + B - 1) : (l + B), (-l + B - 1) : (l + B)
        ] += (f_m * g_n * (d_lmk * d_lkn).to(torch.complex64))

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = torch.fft.ifftshift(cc, dim=(-3, -2, -1))

    # Inverse 2D FFT along the last two dimensions
    cc = ifftn(
        cc,
        dim=(-3, -2, -1),
        norm="forward",
    ).abs()

    return cc


# # test it by doing autocorrelation of a master pattern with itself
# L = 256
# L_trunc = 128
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")

# precision = "single"
# dtype = torch.float32

# # precision = "double"
# # dtype = torch.float64

# wigner_table = wigner_d_eq_half_pi(
#     L + 2,
#     dtype=dtype,
#     device=device,
# )

# sht_real = RSHT(L, device, precision=precision)
# sht_comp = CSHT(L, device, precision=precision)

# sht_real_trunc = RSHT(L_trunc, device, precision=precision)
# sht_comp_trunc = CSHT(L_trunc, device, precision=precision)

# # dh_grid = grid_DriscollHealy(L).to(device)
# dh_grid = grid_DriscollHealy(L, dtype=dtype).to(device)
# xyz_grid = theta_phi_to_xyz(dh_grid)

# # mp_fname = "../EMs/EMplay/old/Si-master-20kV.h5"
# # mp = read_master_pattern(mp_fname).to(device).to(dtype)

# from kikuchipy.data import ebsd_master_pattern
# import numpy as np
# from ebsdtorch.io.read_master_pattern import MasterPattern

# mp_kp = ebsd_master_pattern(
#     "Ni", allow_download=True, show_progressbar=True, projection="lambert"
# )
# hsphere = torch.tensor(np.array(mp_kp.data[-1]), device=device, dtype=torch.float32)
# hsphere = (hsphere - hsphere.min()) / (hsphere.max() - hsphere.min())
# mp_tensors = (hsphere, hsphere)
# mp = MasterPattern(mp_tensors, laue_group=11)

# mp.normalize("minmax")

# # Interpolate master pattern
# mp_spherical_img = mp.interpolate(
#     xyz_grid,
#     mode="bicubic",
#     padding_mode="border",
#     align_corners=False,
#     normalize_coords=True,
#     virtual_binning=1,
# ).squeeze(0)

# # mp_spherical_img -= mp_spherical_img.mean()

# mp_spherical_r_coeffs = sht_real.fsht(mp_spherical_img.unsqueeze(0))
# mp_spherical_c_coeffs = sht_comp.fsht(mp_spherical_img.unsqueeze(0))

# print(f"mp_spherical_r_coeffs shape: {mp_spherical_r_coeffs.shape}")
# print(f"mp_spherical_c_coeffs shape: {mp_spherical_c_coeffs.shape}")

# # truncate to L
# mp_spherical_r_coeffs = mp_spherical_r_coeffs[:, :L_trunc, :L_trunc].contiguous()
# mp_spherical_c_coeffs = mp_spherical_c_coeffs[
#     :, (-L_trunc + L) : (L_trunc + L - 1), :L_trunc
# ].contiguous()

# print(f"truncate mp_spherical_r_coeffs shape: {mp_spherical_r_coeffs.shape}")
# print(f"truncate mp_spherical_c_coeffs shape: {mp_spherical_c_coeffs.shape}")

# # recover the image from the coefficients and use that image to estimate the maximum possible dot
# mp_spherical_img_recon_r = sht_real_trunc.isht(mp_spherical_r_coeffs)
# mp_spherical_img_recon_c = sht_comp_trunc.isht(mp_spherical_c_coeffs)

# ideal_max_r = (mp_spherical_img_recon_r).mean().item()
# ideal_max_c = (mp_spherical_img_recon_c).abs().mean().item()

# print(f"ideal_max_r: {ideal_max_r}")
# print(f"ideal_max_c: {ideal_max_c}")

# d_lmk = torch.zeros(
#     (L_trunc, 2 * L_trunc - 1, 2 * L_trunc - 1),
#     dtype=torch.float32,
#     device=device,
# )

# ll, nn, kk = torch.meshgrid(
#     torch.arange(0, L_trunc, dtype=torch.int32, device=device),
#     torch.arange(-L_trunc + 1, L_trunc, dtype=torch.int32, device=device),
#     torch.arange(-L_trunc + 1, L_trunc, dtype=torch.int32, device=device),
#     indexing="ij",
# )

# valid = (nn.abs() <= ll) & (kk.abs() <= ll)
# d_lmk[valid] = read_lmn_wigner_d_half_pi_table(
#     wigner_table,
#     coords=torch.stack([ll[valid], nn[valid], kk[valid]], dim=-1),
# )

# d_lmk = d_lmk.to(torch.complex64)

# cc_r = rs2cc_fast_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs, d_lmk)
# # cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs, wigner_table)
# cc_c = cs2cc_(L_trunc, mp_spherical_c_coeffs, mp_spherical_c_coeffs, wigner_table)

# print(f"cc_r shape: {cc_r.shape}")
# print(f"cc_c shape: {cc_c.shape}")

# print(f"cc_r min/max: {cc_r.min().item()}, {cc_r.max().item()}")
# print(f"cc_c min/max: {cc_c.min().item()}, {cc_c.max().item()}")

# # # min across both
# # minval = min(cc_r.min().item(), cc_c.min().item(
# # maxval = max(cc_r.max().item(), cc_c.max().item())

# # cc_c = (cc_c - minval) / (maxval - minval)
# # cc_r = (cc_r - minval) / (maxval - minval)

# cc_c = (cc_c - cc_c.min()) / (cc_c.max() - cc_c.min())
# cc_r = (cc_r - cc_r.min()) / (cc_r.max() - cc_r.min())

# # make a gif of the slices, only showing every 4th slice
# cc_c = (cc_c * 255).byte().squeeze(0).cpu().numpy()
# cc_r = (cc_r * 255).byte().squeeze(0).cpu().numpy()

# # from PIL import Image
# # import cv2
# # import numpy as np

# # # make compressed video instead of gif
# # fourcc = cv2.VideoWriter_fourcc(*"MJPG")


# # # use a function instead for compactness
# # def write_video(fourcc, fname, cc, dim):
# #     if dim == 0:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[2], cc.shape[1]), isColor=False
# #         )
# #         for i in range(cc.shape[0]):
# #             out.write(cc[i, :, :])
# #         out.release()
# #     elif dim == 1:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[2], cc.shape[0]), isColor=False
# #         )
# #         for i in range(cc.shape[1]):
# #             out.write(cc[:, i, :])
# #         out.release()
# #     elif dim == 2:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[1], cc.shape[0]), isColor=False
# #         )
# #         for i in range(cc.shape[2]):
# #             out.write(cc[:, :, i])
# #         out.release()
# #     else:
# #         raise ValueError(f"dim {dim} is not valid")


# # # write each mp4
# # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# # write_video(fourcc, "cc_r_m.mp4", cc_r, 0)
# # write_video(fourcc, "cc_r_k.mp4", cc_r, 1)
# # write_video(fourcc, "cc_r_n.mp4", cc_r, 2)

# # write_video(fourcc, "cc_c_m.mp4", cc_c, 0)
# # write_video(fourcc, "cc_c_k.mp4", cc_c, 1)
# # write_video(fourcc, "cc_c_n.mp4", cc_c, 2)


# # make a batch of coefficients of real valued signal and measure the speed
# # of the cross correlation via 10 iterations
# import time

# n_iters = 10
# batch_size = 1
# mp_spherical_r_coeffs_batch = mp_spherical_r_coeffs.repeat(batch_size, 1, 1)

# cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table)
# cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table)

# start = time.time()

# for i in range(n_iters):
#     cc_r = rs2cc_(
#         L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table
#     )

# torch.cuda.synchronize()

# duration = time.time() - start

# print(f"SLOW: CC per second: {n_iters * batch_size / duration}")

# start = time.time()

# cc_r = rs2cc_fast_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, d_lmk)

# for i in range(n_iters):
#     cc_r = rs2cc_fast_(
#         L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, d_lmk
#     )

# torch.cuda.synchronize()

# duration = time.time() - start

# print(f"FAST: CC per second: {n_iters * batch_size / duration}")


# # # plot with plotly now
# # import plotly.graph_objects as go
# # import numpy as np

# # X, Y, Z = np.meshgrid(
# #     np.arange(cc_c.shape[0]),
# #     np.arange(cc_c.shape[1]),
# #     np.arange(cc_c.shape[2]),
# #     indexing="ij",
# # )

# # print(f"X shape: {X.shape}")
# # print(f"Y shape: {Y.shape}")
# # print(f"Z shape: {Z.shape}")
# # print(f"cc_c shape: {cc_c.shape}")

# # fig = go.Figure(
# #     data=[
# #         go.Volume(
# #             x=X.flatten().astype(np.int8),
# #             y=Y.flatten().astype(np.int8),
# #             z=Z.flatten().astype(np.int8),
# #             value=cc_c.flatten(),
# #             isomin=0,
# #             isomax=150,
# #             opacity=0.15,  # needs to be small to see through all surfaces
# #             surface_count=5,  # needs to be a large number for good volume rendering
# #         )
# #     ]
# # )

# # # fig.show()

# # # write html
# # fig.write_html("cc_c.html")
