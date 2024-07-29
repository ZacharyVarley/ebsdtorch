"""

Often I see batches being looped over to compute mutual information. One
can use scatter to compute mutual information per batch in parallel. The 
main disadvantage is that scatter requires int64 index tensors even if 
the largest bin index is less than 2^31 or even 2^15.

"""

import torch
from typing import Tuple
from torch import Tensor


@torch.jit.script
def mutual_information(
    a: Tensor,
    b: Tensor,
    n_bins: int = 64,
    normalize: bool = True,
) -> Tensor:
    """
    Use scatter to simultaneously compute batchwise mutual information.

    Args:
        a (Tensor): The first tensor (B, n_pix)
        b (Tensor): The second tensor (B, n_pix) - B will broadcast.
        n_bins (int): The number of bins to use.
        normalize (bool): Minmax normalize values before discretization.

    Returns:
        Tensor: The mutual information (B,)

    """
    B1, n_pix = a.shape
    B2, n_pix2 = b.shape

    # get the min and max values
    a_minvals = a.min(dim=-1, keepdim=True).values
    a_maxvals = a.max(dim=-1, keepdim=True).values
    b_minvals = b.min(dim=-1, keepdim=True).values
    b_maxvals = b.max(dim=-1, keepdim=True).values

    if normalize:
        a_bin = ((a - a_minvals) / (a_maxvals - a_minvals) * n_bins * 0.999999).long()
        b_bin = ((b - b_minvals) / (b_maxvals - b_minvals) * n_bins * 0.999999).long()
    else:
        a_bin = (a * n_bins * 0.999999).long()
        b_bin = (b * n_bins * 0.999999).long()

    joint_bin = a_bin * (n_bins - 1) + b_bin

    # get the histogram
    if B1 == B2:
        B = B1
    else:
        if B1 != 1 and B2 != 1:
            raise ValueError("Batch dimensions must match or broadcast.")
        B = max(B1, B2)

    hist_joint = torch.zeros(B, n_bins * n_bins, device=a.device)
    hist_a = torch.zeros(B1, n_bins, device=a.device)
    hist_b = torch.zeros(B2, n_bins, device=a.device)

    # scatter the joint histogram
    hist_joint = hist_joint.scatter_(
        1,
        joint_bin,
        1,
        reduce="add",
    )
    hist_a = hist_a.scatter_(1, a_bin, 1, reduce="add")
    hist_b = hist_b.scatter_(1, b_bin, 1, reduce="add")

    # normalize the histograms
    hist_joint = hist_joint / hist_joint.sum(dim=-1, keepdim=True)
    hist_a = hist_a / hist_a.sum(dim=-1, keepdim=True)
    hist_b = hist_b / hist_b.sum(dim=-1, keepdim=True)

    # compute the mutual information
    log_p_joint = torch.log(hist_joint)
    log_p_joint[torch.isinf(log_p_joint) | torch.isnan(log_p_joint)] = 0.0

    log_p_a = torch.log(hist_a)
    log_p_a[torch.isinf(log_p_a) | torch.isnan(log_p_a)] = 0.0

    log_p_b = torch.log(hist_b)
    log_p_b[torch.isinf(log_p_b) | torch.isnan(log_p_b)] = 0.0

    ent_joint = -(hist_joint * log_p_joint).sum(dim=-1)
    ent_a = -(hist_a * log_p_a).sum(dim=-1)
    ent_b = -(hist_b * log_p_b).sum(dim=-1)

    return ent_a + ent_b - ent_joint


"""

This file contains a PyTorch implementation of the binned normalized mutual
information metric for comparing two images. This file also contains a PyTorch
implementation of densely (like cross correlation) calculated normalized mutual
information metric for comparing two images as one of them shifts in discrete
pixel steps.

------------------------------------------------------------------------------
-------------------------------- Important -----------------------------------
------------------------------------------------------------------------------
The dense binned nmi approximation approach requires a quadratic number of cross
correlations with respect to the binning. For 16 bins, this is 256 cross
correlations. For reference the new approach that I have implemented using a
series approximation, and requires 6 cross correlations for both 3rd and 4th
order. The 4th order is more accurate, but the 3rd order is faster. Further, it
plays nicely with the automatic differentation engine (no Parzen windowing).
------------------------------------------------------------------------------
-------------------------------- Important -----------------------------------
------------------------------------------------------------------------------

The binned approach comes from the following literature:

--- 1997: Mutual information established for multimodal image registration.

Maes, Frederik, Andre Collignon, Dirk Vandermeulen, Guy Marchal, and Paul
Suetens. "Multimodality image registration by maximization of mutual
information." IEEE transactions on Medical Imaging 16, no. 2 (1997): 187-198.

--- 2005: A little known publication of Jonas August and Takeo Kanade at CMU
explains how mutual information can be densely computed via FFTs to quickly
count bin occupancy in the joint histogram over discrete translations.

August, Jonas, and Takeo Kanade. "The role of non-overlap in image
registration." In Biennial International Conference on Information Processing in
Medical Imaging, pp. 713-724. Berlin, Heidelberg: Springer Berlin Heidelberg,
2005. 

--- 2021: The same 2005 idea is iterated and extended by Öfverstedt et al.

Öfverstedt, Johan, Joakim Lindblad, and Nataša Sladoje. "Fast computation of
mutual information in the frequency domain with applications to global
multimodal image alignment." Pattern Recognition Letters 159 (2022): 196-203.

"""

from typing import Optional
import torch
from torch import Tensor
from torch.fft import rfft2, irfft2, fftshift


@torch.jit.script
def binned_nmi(
    x: Tensor,
    y: Tensor,
    binsx: int,
    binsy: int,
    mode: str = "nats",
    norm_mode: str = "none",
    exclude_lowest_bin: bool = False,
) -> float:
    """
    Calculate the mutual information between two variables x and y using PyTorch,
    with an option to exclude the lowest bin.

    Args:
        x: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        y: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        binsx: number of bins for x
        binsy: number of bins for y
        mode: 'nats' or 'bits'. This determines the units of the mutual information.
        normalization_mode: 'Yao', 'Kvalseth Sum', 'Kvalseth Min', 'Kvalseth Max', 'Strehl-Ghosh', 'Logarithmic', 'none'
        exclude_lowest_bin: If True, the lowest bin will be excluded from the calculation.
    """

    x = x.view(-1)
    y = y.view(-1)

    x_binned = torch.floor(x * binsx)
    y_binned = torch.floor(y * binsy)

    # Create a combined variable for 1D histogram
    xy_combined = x_binned * binsy + y_binned

    # Calculate the 1D histogram using a flattening trick so that it can run on the GPU
    hist = torch.histc(xy_combined, bins=(binsx * binsy), min=0, max=(binsx * binsy))
    joint_hist = hist.view(binsx, binsy)

    # Exclude the lowest bin if the flag is set
    if exclude_lowest_bin:
        joint_hist_trimmed = joint_hist[1:, 1:]
    else:
        joint_hist_trimmed = joint_hist

    # if you do this step before trimming the lowest bin, then you will get
    # a very different result and it will not be clear why :(
    joint_hist_trimmed /= joint_hist_trimmed.sum()

    # Calculate the marginal histograms
    marginal_x = joint_hist_trimmed.sum(dim=1)
    marginal_y = joint_hist_trimmed.sum(dim=0)

    # normalize the marginal histograms
    marginal_x /= marginal_x.sum()
    marginal_y /= marginal_y.sum()

    # Calculate the mutual information
    marginal_x_non_zero = marginal_x[marginal_x > 0]
    marginal_y_non_zero = marginal_y[marginal_y > 0]
    joint_non_zero = joint_hist_trimmed[joint_hist_trimmed > 0]

    # Calculate the entropies
    if mode == "bits":
        ent_x = -torch.sum(marginal_x_non_zero * torch.log2(marginal_x_non_zero))
        ent_y = -torch.sum(marginal_y_non_zero * torch.log2(marginal_y_non_zero))
        ent_joint = -torch.sum(joint_non_zero * torch.log2((joint_non_zero)))
    else:
        ent_x = -torch.sum(marginal_x_non_zero * torch.log(marginal_x_non_zero))
        ent_y = -torch.sum(marginal_y_non_zero * torch.log(marginal_y_non_zero))
        ent_joint = -torch.sum(joint_non_zero * torch.log((joint_non_zero)))

    # formula with -1 as the values have not been multiplied by -1
    mi = ent_x + ent_y - ent_joint

    # if the normalization mode is "none" then return the distances
    if norm_mode == "none":
        nmi = mi
    elif norm_mode == "Kvalseth Sum":  # divide by marginal arithmetic mean
        nmi = 2.0 * mi / (ent_x + ent_y)
    elif norm_mode == "Kvalseth Min":  # divide by max marginal
        nmi = mi / torch.min(ent_x, ent_y)
    elif norm_mode == "Kvalseth Max":  # divide by min marginal
        nmi = mi / torch.max(ent_x, ent_y)
    elif norm_mode == "Yao":  # divide by joint
        nmi = mi / ent_joint
    elif norm_mode == "Strehl-Ghosh":  # divide by marginal geometric mean
        nmi = mi / torch.sqrt(ent_x * ent_y)
    elif norm_mode == "Logarithmic":  # divide by marginal log mean
        # handle case if ent_src and ent_dst are the same
        nmi = mi / torch.where(
            (ent_x - ent_y).abs() < 1e-6,
            ent_y,
            (ent_x - ent_y) / (torch.log(ent_x) - torch.log(ent_y)),
        )
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}")
    return nmi


@torch.jit.script
def batch_binned_nmi(
    x: Tensor,
    y: Tensor,
    binsx: int,
    binsy: int,
    mode: str = "nats",
    norm_mode: str = "none",
    exclude_lowest_bin: bool = False,
) -> Tensor:
    """
    Calculate the mutual information for batches of x and y along the first dimension,
    with broadcasting if one of them has a batch size of 1.

    Args:
        x: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        y: torch Tensor of shape (batch_size, ...), assumes binning 0 to 1
        binsx: number of bins for x
        binsy: number of bins for y
        mode: 'nats' or 'bits'. This determines the units of the mutual information.
        normalization_mode: 'Yao', 'Kvalseth Sum', 'Kvalseth Min', 'Kvalseth Max', 'Strehl-Ghosh', 'Logarithmic', 'none'
        exclude_lowest_bin: If True, the lowest bin will be excluded from the calculation.
    """

    batch_size_x = x.shape[0]
    batch_size_y = y.shape[0]

    # Determine the batch size for iteration
    if batch_size_x != batch_size_y:
        if batch_size_x == 1 or batch_size_y == 1:
            batch_size = max(batch_size_x, batch_size_y)
        else:
            raise ValueError(
                "Batch sizes must be equal, or one of them must be 1 for broadcasting."
            )
    else:
        batch_size = batch_size_x

    mi_values = torch.empty(batch_size, dtype=x.dtype, device=x.device)

    for i in range(batch_size):
        xi = x if batch_size_x == 1 else x[i]
        yi = y if batch_size_y == 1 else y[i]
        mi_values[i] = binned_nmi(
            xi,
            yi,
            binsx,
            binsy,
            mode=mode,
            norm_mode=norm_mode,
            exclude_lowest_bin=exclude_lowest_bin,
        )

    return mi_values


@torch.jit.script
def cmif(
    images_dst: Tensor,
    images_src: Tensor,
    dst_ind: Optional[Tensor] = None,
    src_ind: Optional[Tensor] = None,
    bins_dst: int = 4,
    bins_src: int = 4,
    chunk_size: int = 8,
    constant_marginal: bool = False,
    norm_mode: str = "none",
) -> Tensor:
    """
    Calculate the cross mutual information function between two images using the 2D FFT.

    Args:
        images_src: a stack of source images of shape (..., H, W)
        images_dst: a stack of destination images of shape (..., H, W)
        bins_src: the number of bins to use for the source image
        bins_dst: the number of bins to use for the destination image
        overlap_min_fraction: the minimum fraction of overlap between the images
        background_src: the background level for the source image. If -1 then
            none of the pixel values are treated as background.
        background_dst: the background level for the destination image. If -1 then
            none of the pixel values are treated as background.
        logmode: the log mode to use for the entropy calculation
            Options are "log", "log2", "log10"
        normalization_mode: the normalization mode to use for the mutual information.
            Options are "none", "Kvalseth Sum", "Kvalseth Min", "Kvalseth Max", "Yao",
            "Strehl-Ghosh", "Logarithmic" (see Notes)

    Returns:
        Tensor: the mutual information map between the images


    Notes:

    For mutual information normalization approaches, see:

    Amelio, Alessia, and Clara Pizzuti. "Correction for closeness: Adjusting
    normalized mutual information measure for clustering comparison."
    Computational Intelligence 33.3 (2017): 579-601.

    """
    h1, w1 = images_src.shape[-2:]
    h2, w2 = images_dst.shape[-2:]

    # check that the images are the same size
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images different size: src {h1}x{w1} and dst {h2}x{w2}")

    dst_data_shape = images_dst.shape[:-2]
    src_data_shape = images_src.shape[:-2]

    # # minmax normalize the images
    # images_dst -= (
    #     images_dst.view(dst_data_shape + (-1,)).min(dim=-1).values[..., None, None]
    # )
    # images_dst /= (
    #     images_dst.view(dst_data_shape + (-1,)).max(dim=-1).values[..., None, None]
    # )

    # images_src -= (
    #     images_src.view(src_data_shape + (-1,)).min(dim=-1).values[..., None, None]
    # )
    # images_src /= (
    #     images_src.view(src_data_shape + (-1,)).max(dim=-1).values[..., None, None]
    # )

    images_dst = images_dst.view(-1, 1, h1, w1)
    images_src = images_src.view(-1, 1, h1, w1)

    dst_levels = (bins_dst * images_dst).byte().repeat(1, bins_dst, 1, 1)
    dst_levels = (
        dst_levels
        == torch.arange(bins_dst, dtype=torch.uint8, device=images_dst.device)[
            None, :, None, None
        ]
    )

    src_levels = (bins_src * images_src).byte().repeat(1, bins_src, 1, 1)
    src_levels = (
        src_levels
        == torch.arange(bins_src, dtype=torch.uint8, device=images_src.device)[
            None, :, None, None
        ]
    )

    # we need to zero mean and unit norm the images (only within the mask)
    if dst_ind is None:
        dst_ind = torch.ones_like(images_dst)
    else:
        dst_ind = dst_ind.view(-1, 1, h1, w1)

    if src_ind is None:
        src_ind = torch.ones_like(images_src)
    else:
        src_ind = src_ind.view(-1, 1, h1, w1)

    # add the indicator images to the beginning of the levels
    dst_levels = torch.cat((dst_ind, dst_levels), dim=1)
    src_levels = torch.cat((src_ind, src_levels), dim=1)

    # calculate the 2D FFT of the indicator images
    dst_levels_fft = rfft2(dst_levels.float())
    src_levels_fft = rfft2(src_levels.float()).conj()

    # if binning is more than 16, have to do this in chunks
    if bins_src > 8 or bins_dst > 8:
        if constant_marginal:
            # the marginal is calculated from the entire original image and it is a scalar
            src_discrete = (bins_src * images_src).byte()
            dst_discrete = (bins_dst * images_dst).byte()

            # calculate the marginal histograms
            marg_hists_dst = torch.histc(
                dst_discrete[src_ind].float(), bins=bins_dst, min=0, max=bins_dst - 1
            )
            marg_hists_src = torch.histc(
                src_discrete[dst_ind].float(), bins=bins_src, min=0, max=bins_src - 1
            )

            # normalize the marginal histograms
            marg_hists_src /= marg_hists_src.sum()
            marg_hists_dst /= marg_hists_dst.sum()

            # calculate the marginal log probabilities
            marg_src_p_log_p = torch.where(
                marg_hists_src > 0.0,
                marg_hists_src * torch.log(marg_hists_src),
                torch.zeros_like(marg_hists_src),
            )
            marg_dst_p_log_p = torch.where(
                marg_hists_dst > 0.0,
                marg_hists_dst * torch.log(marg_hists_dst),
                torch.zeros_like(marg_hists_dst),
            )

            # find the entropies of the marginals
            ent_src = -torch.sum(marg_src_p_log_p)
            ent_dst = -torch.sum(marg_dst_p_log_p)

        else:
            # first compute the marginal histograms
            marg_hists_dst = fftshift(
                irfft2(src_levels_fft[:, 0:1, :, :] * dst_levels_fft[:, 1:, :, :]),
                dim=(-2, -1),
            )
            marg_hists_src = fftshift(
                irfft2(src_levels_fft[:, 1:, :, :] * dst_levels_fft[:, 0:1, :, :]),
                dim=(-2, -1),
            )

            # normalize the marginal histograms
            marg_hists_src /= marg_hists_src.sum(dim=1, keepdim=True)
            marg_hists_dst /= marg_hists_dst.sum(dim=1, keepdim=True)

            # calculate the marginal log probabilities
            marg_src_p_log_p = torch.where(
                marg_hists_src > 0.0,
                marg_hists_src * torch.log(marg_hists_src),
                torch.zeros_like(marg_hists_src),
            )
            marg_dst_p_log_p = torch.where(
                marg_hists_dst > 0.0,
                marg_hists_dst * torch.log(marg_hists_dst),
                torch.zeros_like(marg_hists_dst),
            )

            # find the entropies of the marginals
            ent_src = -torch.sum(marg_src_p_log_p, dim=1)
            ent_dst = -torch.sum(marg_dst_p_log_p, dim=1)

        # total number of pixel pair values is the olap value so we can normalize the histogram bars in the joint as they come
        ent_joint = torch.zeros(
            (images_dst.shape[0], images_src.shape[0], h1, w1), device=images_dst.device
        )
        tally_joint = torch.zeros(
            (images_dst.shape[0], images_src.shape[0], h1, w1), device=images_dst.device
        )

        for i in range(1, bins_src, chunk_size):
            for j in range(1, bins_dst, chunk_size):
                # print which chunk begin and end for each
                src_chunk = src_levels_fft[:, i : i + chunk_size, :, :]
                dst_chunk = dst_levels_fft[:, j : j + chunk_size, :, :]
                cross_corr_f = torch.einsum("bihw,cjhw->bcijhw", dst_chunk, src_chunk)
                cross_corr_real = fftshift(irfft2(cross_corr_f), dim=(-2, -1))
                # where the bin occupancy was less than 1, set it to zero
                cross_corr_real[cross_corr_real < 1.0] = 0.0
                # update the tally
                tally_joint += cross_corr_real.sum(dim=(2, 3))

        for i in range(1, bins_src, chunk_size):
            for j in range(1, bins_dst, chunk_size):
                src_chunk = src_levels_fft[:, i : i + chunk_size, :, :]
                dst_chunk = dst_levels_fft[:, j : j + chunk_size, :, :]
                cross_corr_f = torch.einsum("bihw,cjhw->bcijhw", dst_chunk, src_chunk)
                cross_corr_real = fftshift(irfft2(cross_corr_f), dim=(-2, -1))
                # where the bin occupancy was less than 1, set it to zero
                cross_corr_real[cross_corr_real < 1.0] = 0.0
                # increment the joint entropy
                joint_p = cross_corr_real / tally_joint
                joint_p_log_p = torch.where(
                    joint_p > 0.0,
                    joint_p * torch.log(joint_p),
                    torch.zeros_like(joint_p),
                )
                ent_joint += -torch.sum(joint_p_log_p, dim=(2, 3))

    else:
        if constant_marginal:
            # the marginal is calculated from the entire original image and it is a scalar
            src_discrete = (bins_src * images_src * 0.99999).long()
            dst_discrete = (bins_dst * images_dst * 0.99999).long()

            # have to use scatter instead
            marg_hists_dst = torch.zeros(
                images_dst.shape[0], bins_dst, device=images_dst.device
            )
            marg_hists_src = torch.zeros(
                images_src.shape[0], bins_src, device=images_src.device
            )

            # reshape to be (B, n_bins, -1)
            src_discrete = src_discrete.view(-1, h1 * w1)
            dst_discrete = dst_discrete.view(-1, h1 * w1)

            # scatter the joint histogram
            marg_hists_dst = marg_hists_dst.scatter_(
                -1,
                dst_discrete,
                1,
                reduce="add",
            )
            marg_hists_src = marg_hists_src.scatter_(
                -1,
                src_discrete,
                1,
                reduce="add",
            )

            # normalize the marginal histograms
            marg_hists_src /= marg_hists_src.sum(dim=1, keepdim=True)
            marg_hists_dst /= marg_hists_dst.sum(dim=1, keepdim=True)

            # calculate the marginal log probabilities
            marg_src_p_log_p = torch.where(
                marg_hists_src > 0.0,
                marg_hists_src * torch.log(marg_hists_src),
                torch.zeros_like(marg_hists_src),
            )
            marg_dst_p_log_p = torch.where(
                marg_hists_dst > 0.0,
                marg_hists_dst * torch.log(marg_hists_dst),
                torch.zeros_like(marg_hists_dst),
            )

            # find the entropies of the marginals
            ent_src = -torch.sum(marg_src_p_log_p, dim=1)
            ent_dst = -torch.sum(marg_dst_p_log_p, dim=1)

            # element-wise multiply every possible pair of slices
            # the result is a tensor of shape (B, 1 + num_bins_src, 1 + num_bins_dst, H, W)
            # where the "1 +" are coming from the indicator bins for the image foreground
            cross_corr_f = torch.einsum(
                "bihw,cjhw->bcijhw", dst_levels_fft, src_levels_fft
            )

            # Inverse 2D real valued fourier transform and shift and abs
            cross_corr_real = fftshift(irfft2(cross_corr_f), dim=(-2, -1))

            # where the bin occupancy was less than 1, set it to zero
            cross_corr_real[cross_corr_real < 1.0] = 0.0

            # extract out the joint histograms
            joint_hists = cross_corr_real[:, :, 1:, 1:, :, :]

            # individually normalize the histograms
            joint_hists /= joint_hists.sum(dim=(2, 3), keepdim=True)

            # calculate the entropy of the joint histograms
            # must be careful to ignore the zero bins
            joint_p_log_p = torch.where(
                joint_hists > 0.0,
                joint_hists * torch.log(joint_hists),
                torch.zeros_like(joint_hists),
            )

            # find the entropies of the marginals and the joint
            ent_joint = -torch.sum(joint_p_log_p, dim=(2, 3))
        else:
            # element-wise multiply every possible pair of slices
            # the result is a tensor of shape (B, 1 + num_bins_src, 1 + num_bins_dst, H, W)
            # where the "1 +" are coming from the indicator bins for the image foreground
            cross_corr_f = torch.einsum(
                "bihw,cjhw->bcijhw", dst_levels_fft, src_levels_fft
            )

            # Inverse 2D real valued fourier transform and shift and abs
            cross_corr_real = fftshift(irfft2(cross_corr_f), dim=(-2, -1)).real

            # where the bin occupancy was less than 1, set it to zero
            cross_corr_real[cross_corr_real < 1.0] = 0.0

            # extract out the joint histograms
            joint_hists = cross_corr_real[:, :, 1:, 1:, :, :]

            # extract out the marginal histograms
            marg_hists_src = cross_corr_real[:, :, 1:, 0, :, :]
            marg_hists_dst = cross_corr_real[:, :, 0, 1:, :, :]

            # individually normalize the histograms
            joint_hists /= joint_hists.sum(dim=(2, 3), keepdim=True)
            marg_hists_src /= marg_hists_src.sum(dim=2, keepdim=True)
            marg_hists_dst /= marg_hists_dst.sum(dim=2, keepdim=True)

            # calculate the entropy of the joint histograms
            # must be careful to ignore the zero bins
            joint_p_log_p = torch.where(
                joint_hists > 0.0,
                joint_hists * torch.log(joint_hists),
                torch.zeros_like(joint_hists),
            )
            marg_src_p_log_p = torch.where(
                marg_hists_src > 0.0,
                marg_hists_src * torch.log(marg_hists_src),
                torch.zeros_like(marg_hists_src),
            )
            marg_dst_p_log_p = torch.where(
                marg_hists_dst > 0.0,
                marg_hists_dst * torch.log(marg_hists_dst),
                torch.zeros_like(marg_hists_dst),
            )

            # find the entropies of the marginals and the joint
            ent_src = -torch.sum(marg_src_p_log_p, dim=2)
            ent_dst = -torch.sum(marg_dst_p_log_p, dim=2)
            ent_joint = -torch.sum(joint_p_log_p, dim=(2, 3))

    # mutual information is the sum of the marginal entropies minus the joint entropy
    if constant_marginal:
        mi = ent_dst[:, None, None, None] + ent_src[None, :, None, None] - ent_joint
    else:
        mi = ent_dst + ent_src - ent_joint

    # replace any NaNs or Infs or Negatives with zeros
    mi[torch.isnan(mi) | torch.isinf(mi) | (mi < 0.0)] = 0.0

    # if the normalization mode is "none" then return the distances
    if norm_mode == "none":
        nmi = mi
    elif norm_mode == "Kvalseth Sum":  # divide by marginal arithmetic mean
        nmi = 2.0 * mi / (ent_src + ent_dst)
    elif norm_mode == "Kvalseth Min":  # divide by max marginal
        nmi = mi / torch.min(ent_src, ent_dst)
    elif norm_mode == "Kvalseth Max":  # divide by min marginal
        nmi = mi / torch.max(ent_src, ent_dst)
    elif norm_mode == "Yao":  # divide by joint
        nmi = mi / ent_joint
    elif norm_mode == "Strehl-Ghosh":  # divide by marginal geometric mean
        nmi = mi / torch.sqrt(ent_src * ent_dst)
    elif norm_mode == "Logarithmic":  # divide by marginal log mean
        # handle case if ent_src and ent_dst are the same
        nmi = mi / torch.where(
            (ent_src - ent_dst).abs() < 1e-6,
            ent_dst,
            (ent_src - ent_dst) / (torch.log(ent_src) - torch.log(ent_dst)),
        )
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}")
    return nmi


# # test it
# # visualize the master pattern using PIL
# from PIL import Image
# from ebsdtorch.io.read_master_pattern import read_master_pattern

# mp_fname = "../EMs/EMplay/Ti-alpha-master-20kV.h5"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp = read_master_pattern(mp_fname).to(device)
# mp.normalize("minmax")
# mp.normalize("zeromean")
# # mp.apply_clahe()

# # visualize the first pattern
# import matplotlib.pyplot as plt

# # visualize the master pattern using PIL
# img = mp.master_pattern
# img -= img.min()
# img /= img.max()
# img = (img * 255).byte().cpu()
# img = Image.fromarray(img.numpy())
# img.save("mp_ti.png")

# # do cmif on the master pattern and itself
# cmif_values = cmif(
#     mp.master_pattern[None, None],
#     mp.master_pattern[None, None],
#     bins_dst=4,
#     bins_src=4,
#     norm_mode="none",
#     constant_marginal=False,
# )
# cmif_values = (cmif_values.squeeze() * 255).byte().cpu().numpy()
# img = Image.fromarray(cmif_values)
# img.save("cmif_ti.png")
