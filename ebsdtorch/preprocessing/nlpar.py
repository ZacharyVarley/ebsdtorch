"""
This module implements non-local pattern averaging (NLPAR) to denoise EBSD
patterns by averaging over a local kernel.

The NLPAR algorithm is based on the following paper:

Brewick, Patrick T., Stuart I. Wright, and David J. Rowenhorst. "NLPAR:
Non-local smoothing for enhanced EBSD pattern indexing." Ultramicroscopy 200
(2019): 50-61.

The steps to NLPAR:

INPUT: SCAN_H x SCAN_W x PAT_H x PAT_W tensor of EBSD patterns

1) For all patterns look at the 8 sourrounding patterns in a 3x3 window
Noise estimate: sigma = sqrt(min(distances(central_pat, 8_neighbors)) / 2*n_pixels_per_pat)
2) Compute the normalized distances to each pattern in the neighborhood:
d_bar = (sum(pairwise pixel distances) - 2 * n_pixels_per_pat * sigma^2) / (sqrt(2*n_pixels_per_pat) * 2 * sigma^2)
dbar = sum(pairwise pixel distances) / (sqrt(2*n_pixels_per_pat) * 2 * sigma^2) - (n_pixels_per_pat^0.5 / (2^0.5))
3) Compute the weights for each pattern in the neighborhood:
softmax within the neighborhood of the zero-clamped normalized distances
4) Compute the denoised pattern as the weighted average of the neighborhood patterns

"""

from typing import Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad


@torch.jit.script
def nlpar(
    data: Tensor,
    k_rad: int = 5,
    coeff: float = 0.375,
    coeff_fit_tol: float = 1e-6,
    center_logit_bias: float = 0.0,
) -> Tensor:
    """
    Non-local pattern averaging (NLPAR) denoising algorithm.

    Args:
        data (torch.Tensor): The EBSD pattern data tensor.
        k_rad (int): The radius of the kernel.
        coeff (float): higher -> original pattern has more weight
        coeff_fit_tol (float): tolerance (Newton's method) to hit coeff
        center_logit_bias (float): higher -> constant bias towards original pattern

    Returns:
        Tensor: The denoised EBSD pattern data tensor.

    """
    assert len(data.shape) == 4, "The EBSD pattern data tensor must have 4 dimensions."
    # get the dimensions of the data tensor
    h_scan, w_scan = data.shape[:-2]
    h_pat, w_pat = data.shape[-2:]

    # make a mask for the data array to remove contributions from the padded values
    out_bounds_mask = torch.zeros(
        (h_scan, w_scan), device=data.device, dtype=torch.bool
    )
    out_bounds_mask = torch.nn.functional.pad(
        out_bounds_mask,
        (k_rad,) * 4,
        mode="constant",
        value=True,
    )
    # pad the data tensor with inf so we can remove those values later
    data_padded = torch.nn.functional.pad(
        data,
        (0,) * 4 + (k_rad,) * 4,
        mode="constant",
        value=0.0,
    )
    # get convenient variables
    k_size = 2 * k_rad + 1
    n_pixels_per_pat = h_pat * w_pat
    n_pats_per_kernel = k_size * k_size

    # initialize the squared differences tensor
    dists = torch.empty(
        (h_scan, w_scan, n_pats_per_kernel),
        device=data_padded.device,
        dtype=torch.float32,
    )

    # make the overall mask for the data array
    out_bounds_mask_all = torch.empty(
        (h_scan, w_scan, n_pats_per_kernel),
        device=data_padded.device,
        dtype=torch.bool,
    )

    # get kernel coordinates
    offset_coords = torch.stack(
        torch.meshgrid(
            torch.arange(-k_rad, k_rad + 1),
            torch.arange(-k_rad, k_rad + 1),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    # find indices of 8 nearest neighbors
    # closest point is self so we skip it
    offset_coords = offset_coords[
        torch.argsort(torch.sum(torch.abs(offset_coords), dim=-1))
    ]

    starts_i = k_rad + offset_coords[:, 0]
    ends_i = k_rad + offset_coords[:, 0] + h_scan
    starts_j = k_rad + offset_coords[:, 1]
    ends_j = k_rad + offset_coords[:, 1] + w_scan

    # iterate and compute MSE
    for i in range(n_pats_per_kernel):
        dists[:, :, i] = torch.sum(
            (
                data_padded[starts_i[i] : ends_i[i], starts_j[i] : ends_j[i], :, :]
                - data_padded[k_rad : (k_rad + h_scan), k_rad : (k_rad + w_scan), :, :]
            )
            ** 2,
            dim=(-1, -2),
        )
        out_bounds_mask_all[:, :, i] = out_bounds_mask[
            starts_i[i] : ends_i[i], starts_j[i] : ends_j[i]
        ]

    # estimate the noise level for the entire scan
    sigma_hat = torch.sqrt(
        torch.min(dists[:, :, 1:9], dim=-1).values / (2 * n_pixels_per_pat)
    )
    # pad sigma_hat so we can use offsets
    sigma_hat_padded = torch.nn.functional.pad(
        sigma_hat,
        (k_rad,) * 4,
        mode="constant",
        value=1.0,
    )
    # iterate and normalize the squared differences
    for i in range(n_pats_per_kernel):
        sigma_other = sigma_hat_padded[starts_i[i] : ends_i[i], starts_j[i] : ends_j[i]]
        dists[:, :, i] = (
            dists[:, :, i] - n_pixels_per_pat * (sigma_hat**2 + sigma_other**2)
        ) / (2**0.5 * n_pixels_per_pat**0.5 * (sigma_hat**2 + sigma_other**2))

    # compute the weights as the clamped negative distances
    weights = -dists.clamp_min_(0.0)

    # need average first bin in softmax normed pdf to equal coeff specified
    # by definition only applicable to the 3x3 NN kernel even when doing a larger kernel
    # f  = exp(w[0]/x^2) / sum(exp(w[i]/x^2)) - coeff
    # f' = -2 * exp(w[0]/x^2) * (sum(w[0] * exp(w[i]/x^2)) - sum(w[i] * exp(w[i]/x^2))) / x^3 * sum(exp(w[i]/x^2))^2
    # Newton's method:
    # x = x - f / f'
    guess_coeff = 1.0
    nn_weights = weights[:, :, :9]
    for _ in range(10):
        # notice that guess_coeff_sq is only calculated at the beginning
        guess_coeff_sq = guess_coeff**2
        obj = (
            torch.mean(torch.softmax(nn_weights / guess_coeff_sq, dim=-1)[:, :, 0])
            - coeff
        )
        grad = (
            -2
            * torch.exp(nn_weights[:, :, 0] / guess_coeff_sq)
            * (
                torch.sum(
                    nn_weights[:, :, [0]] * torch.exp(nn_weights / guess_coeff_sq),
                    dim=-1,
                )
                - torch.sum(nn_weights * torch.exp(nn_weights / guess_coeff_sq), dim=-1)
            )
            / (
                guess_coeff**3
                * torch.sum(torch.exp(nn_weights / guess_coeff_sq), dim=-1) ** 2
            )
        )
        obj_mean = torch.mean(obj)
        grad_mean = torch.mean(grad)
        guess_coeff -= obj_mean / grad_mean
        if torch.abs(obj_mean) < coeff_fit_tol:
            break

    # apply the computed coefficient to the weights
    weights /= guess_coeff**2

    # bias the original pattern
    if center_logit_bias > 0:
        weights[:, :, 0] += center_logit_bias

    # use the mask to change the weights of the pad values to -inf
    # compute the softmax it will take -inf to 0 weight contribution
    weights = weights.masked_fill_(out_bounds_mask_all, -torch.inf)
    weights = torch.nn.functional.softmax(weights, dim=-1)

    # unfolding could be too expensive so we have to use a loop again
    denoised = torch.zeros_like(data)
    for i in range(n_pats_per_kernel):
        denoised += data_padded[
            starts_i[i] : ends_i[i], starts_j[i] : ends_j[i], :, :
        ] * weights[:, :, i].unsqueeze(-1).unsqueeze(-1)
    return denoised
