import torch
from torch import Tensor
from typing import Optional, Tuple
from ebsdtorch.crystallography.unit_cell import Cell


@torch.jit.script
def square_to_hemisphere(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to hemisphere via equal-area bijection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    """
    # map to first quadrant and swap x and y if needed
    x_abs, y_abs = (
        torch.abs(pts[..., 0]) * (torch.pi / 2) ** 0.5,
        torch.abs(pts[..., 1]) * (torch.pi / 2) ** 0.5,
    )
    swap = x_abs >= y_abs
    x_new = torch.where(swap, x_abs, y_abs)
    y_new = torch.where(swap, y_abs, x_abs)

    # only one case now
    x_hs = (
        (2 * x_new / torch.pi)
        * torch.sqrt(torch.pi - x_new**2)
        * torch.cos(torch.pi * y_new / (4 * x_new))
    )
    y_hs = (
        (2 * x_new / torch.pi)
        * torch.sqrt(torch.pi - x_new**2)
        * torch.sin(torch.pi * y_new / (4 * x_new))
    )
    z_out = 1 - (2 * x_new**2 / torch.pi)

    # swap back and copy sign
    x_out = torch.where(swap, x_hs, y_hs)
    y_out = torch.where(swap, y_hs, x_hs)
    x_out = x_out.copysign_(pts[..., 0])
    y_out = y_out.copysign_(pts[..., 1])

    out = torch.stack((x_out, y_out, z_out), dim=-1)

    # make sure (0, 0) is mapped to (0, 0, 1) instead of nans
    mask = (pts[..., 0] == 0) & (pts[..., 1] == 0)
    out[mask] = torch.tensor([0.0, 0.0, 1.0], device=pts.device, dtype=pts.dtype)

    return out


@torch.jit.script
def hexagon_to_hemisphere(pts: Tensor) -> Tensor:
    """
    Map hexagon to the Northern hemisphere via inverse square lambert
    projection, assuming the input is a regular hexagon with a tip at (1, 0).

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    Roşca, D., and M. De Graef. "Area-preserving projections from hexagonal and
    triangular domains to the sphere and applications to electron back-scatter
    diffraction pattern simulations." Modelling and Simulation in Materials
    Science and Engineering 21.5 (2013): 055021.

    Different to original F90 code in EMsoft:

    1) Fold/rotate everything into sextant 1 to remove triple branch
    2) One rescaling at the end for the hexagon area
    3) Undo the swap at the end

    """

    # initial swap to make corner at (1, 0) instead of (0, 1)
    x_abs, y_abs = torch.abs(pts[..., 1]), torch.abs(pts[..., 0])

    # if we are in I_1 sextant, rotate to I_0
    rotate = y_abs > (x_abs / 3**0.5)

    # apply 60 degree clockwise rotation
    x_new = torch.where(rotate, x_abs * 0.5 + y_abs * 0.5 * (3**0.5), x_abs)
    y_new = torch.where(rotate, -x_abs * 0.5 * (3**0.5) + y_abs * 0.5, y_abs)

    # I_0 sextant equal-area bijection having swapped x & y
    factor = (3.0**0.25) * ((2.0 / torch.pi) ** 0.5) * x_new
    trig_arg = (y_new * torch.pi) / (2.0 * (3.0**0.5) * x_new)
    x_hs = factor * torch.cos(trig_arg)
    y_hs = factor * torch.sin(trig_arg)

    # undo the rotation
    x_out = torch.where(rotate, x_hs * 0.5 - y_hs * 0.5 * (3**0.5), x_hs)
    y_out = torch.where(rotate, x_hs * 0.5 * (3**0.5) + y_hs * 0.5, y_hs)

    # copy sign (with swap anticipated) and rescale vertices to touch unit circle
    x_out = x_out.copysign_(pts[..., 1]) / ((3 * 3**0.5 / (2 * torch.pi)) ** 0.5)
    y_out = y_out.copysign_(pts[..., 0]) / ((3 * 3**0.5 / (2 * torch.pi)) ** 0.5)

    # undo original swap and compute the z coordinate on the hemisphere
    out = torch.stack((y_out, x_out, 1.0 - x_out**2 - y_out**2), dim=-1)

    # make sure (0, 0) is mapped to (0, 0, 1) instead of nans
    mask = (pts[..., 0] == 0) & (pts[..., 1] == 0)
    out[mask] = torch.tensor([0.0, 0.0, 1.0], device=pts.device, dtype=pts.dtype)

    return out


def kvectors_grid(
    cell: Cell,
    grid_half_width: int,
    stnum: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """

    Generate a grid of k-vectors using Rosca-Lambert hexagon or square equal
    area bijection with the unit hemisphere in 3D space. The k-vectors are
    normalized to the unit sphere and then multiplied by the DSM to get the
    k-vectors in Cartesian reciprocal space.

    Args:
        :cell: Unit cell object
        :grid_half_width: Half width of the grid: width = 2*grid_half_width + 1
        :device: Device to put the kvectors on
        :stnum: Sampling type number (1-19) for overriding the default

    Returns:
        :k: k-vectors in reciprocal space
        :k_ij: grid indices in the square or hexagon

    """
    device = cell.device
    xtal_system = cell.xtal_system
    pg = cell.pg_num
    sg = cell.sg_num
    sg_2nd = cell.sg_setting != 1

    # use hexagon-circle equal area bijection for sampling?
    use_hex = xtal_system in [4, 5]

    # mapping from point group to sampling scheme (-1 for special cases)
    # 36 entries and stnum cases 11 and 13 are currently infeasible (in EMsoft)
    point_group_to_sample_case = (
        [1, 2, 3, 4, 5, 5, 5, 6, 5, 5, 6, 6, 7, -1, 9, -1]
        + [-1, -1, -1, -1, 15, 12, 17, 16, 18, -1, 19, 3, 6, 6, 8, 9, -1, -1, -1, -1]
        + []
    )

    if stnum is None:
        stnum = point_group_to_sample_case[pg - 1]
        if stnum == -1:
            if 143 <= sg <= 167:  # trigonal
                if 143 <= sg <= 146:  # 3
                    stnum = 10
                elif sg == 147 <= 155:  # bar3 and 32
                    stnum = 12
                elif sg in [156, 158, 160, 161]:  # 3m
                    stnum = 14
                elif sg == 157 | sg == 159:  # 3m
                    stnum = 15
                elif sg in [162, 163]:  # bar3m
                    stnum = 17
                # elif sg in [164, 165, 166, 167]:  # bar3m
                else:  # only remaining option
                    stnum = 16
            else:
                if pg == 14:
                    if 115 <= sg <= 120:
                        stnum = 6
                    else:
                        stnum = 8
                elif pg == 26:
                    if sg in [187, 188]:
                        stnum = 16
                    else:
                        stnum = 17

    # grid has all points on just one hemisphere
    ki, kj = torch.meshgrid(
        [
            torch.arange(-grid_half_width, grid_half_width + 1, device=device),
            torch.arange(-grid_half_width, grid_half_width + 1, device=device),
        ],
        indexing="ij",
    )
    k_ij = torch.stack([ki, kj], dim=-1).reshape(-1, 2)
    # append hemisphere flag to k_ij
    k_ij = torch.cat((k_ij, torch.full_like(k_ij[:, 0:1], 1, device=device)), dim=1)
    xy = k_ij.float() / float(grid_half_width)

    if stnum == 1:  # triclinic 1
        kstar = square_to_hemisphere(xy)
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 2:  # triclinic -1
        # just Northern hemisphere
        kstar = square_to_hemisphere(xy)
    elif stnum == 3:  # monoclinic 2
        # keep x >= 0 with SH
        mask = xy[:, 0] >= 0
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 4:  # monoclinic m
        # keep y >= 0 with SH
        mask = k_ij[:, 1] >= 0
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 5:  # monoclinic 2/m, orthorhombic 222, mm2, tetragonal 4, -4
        # keep x >= 0 and y >= 0 with SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0)
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 6:  # orthorhombic mmm, tetragonal 4/m, 422, -4m2, cubic m-3, 432
        # keep x >= 0 and y >= 0 and x >= y lack SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0)
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
    elif stnum == 7:  # tetragonal 4mm
        # keep x >= 0 and y >= 0 and x >= y with SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0) & (k_ij[:, 0] >= k_ij[:, 1])
        kstar = square_to_hemisphere(xy[mask])
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = k_ij[mask]
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 8:  #  tetragonal -42m, cubic -43m
        # keep x >= 0 and abs(y) <= abs(x) lacks SH
        mask = (k_ij[:, 0] >= 0) & (torch.abs(k_ij[:, 1]) <= k_ij[:, 0])
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
    elif stnum == 9:  # tetragonal 4/mmm, cubic m-3m
        # keep x >= 0 and y >= 0 and x >= y lacks SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0) & (k_ij[:, 0] >= k_ij[:, 1])
        kstar = square_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
    elif stnum == 10:  # hexagonal 3
        # keep x >= 0 and y >= 0 in hexagon with SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0)
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 11:  # rhombohedral 3
        raise NotImplementedError(
            "Rhombohedral 3 not implemented: use hexagonal setting."
        )
    elif stnum == 12:  # hexagonal -3, 321, -6; [not implemented: rhombohedral 32]
        if (sg >= 143) & (sg <= 167) & sg_2nd:
            raise NotImplementedError(
                "Rhombohedral 32 not implemented: use hexagonal setting."
            )
        # keep x >= 0 and y >= 0 and x >= y lacks SH
        mask = (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0) & (k_ij[:, 0] >= k_ij[:, 1])
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
    # examining "PGSamplingType" & "getHexvsRho" in EMsoft: case 13 is infeasible
    elif stnum == 13:  # [not implemented: rhombohedral -3], hexagonal 312
        if (sg >= 143) & (sg <= 167) & sg_2nd:
            raise NotImplementedError(
                "Rhombohedral -3 not implemented: use hexagonal setting."
            )
        else:
            raise NotImplementedError("This case is infeasible.")
    elif stnum == 14:  # trigonal 3m
        if (sg >= 143) & (sg <= 167) & sg_2nd:
            raise NotImplementedError(
                "Rhombohedral -3 not implemented: use hexagonal setting."
            )
        # keep kj >= 0 and ki >= (j-1)//2 and ki <= 2*j with SH
        mask = (
            (k_ij[:, 1] >= 0)
            # & (k_ij[:, 0] >= ((k_ij[:, 1] - 1) // 2)) # bug suspected in EMsoft code
            & (k_ij[:, 0] >= (k_ij[:, 1] + 1) // 2)
            & (k_ij[:, 0] <= 2 * k_ij[:, 1])
        )
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 15:  # hexagonal 31m, 6
        # keep kj >= 0 and ki >= kj with SH
        mask = (k_ij[:, 1] >= 0) & (k_ij[:, 0] >= k_ij[:, 1])
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 16:  #  hexagonal -3m1, 622, -6m2 [not implemented: rhombohedral -3m]
        if (sg >= 143) & (sg <= 167) & sg_2nd:
            raise NotImplementedError(
                "Rhombohedral -3m not implemented: use hexagonal setting."
            )
        # keep kj >= 0 and ki >= 0 and xx >= 0 ang >= pi/6 lacks SH
        xx = k_ij[:, 0] - 0.5 * k_ij[:, 1]
        yy = k_ij[:, 1] * (3**0.5) / 2.0
        ang = torch.atan2(yy, xx)
        mask = (
            (k_ij[:, 0] >= 0)
            & (k_ij[:, 1] >= 0)
            & (ang >= (torch.pi / 5.9999))
            & (xx >= 0)
        )
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
    elif stnum == 17:  # hexagonal -31m, 6/m, -62m
        # keep kj >= 0 and ki >= 0 and xx >= 0 and ang <= pi/3 lacks SH
        xx = k_ij[:, 0] - 0.5 * k_ij[:, 1]
        yy = k_ij[:, 1] * (3**0.5) / 2.0
        ang = torch.atan2(yy, xx)
        mask = (
            (k_ij[:, 0] >= 0) & (k_ij[:, 1] >= 0) & (ang <= (torch.pi / 3)) & (xx >= 0)
        )
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
    elif stnum == 18:  # hexagonal 6mm
        # keep kj >= 0 and ki >= 0 and xx >= 0 and ang <= pi/6 with SH
        xx = k_ij[:, 0] - 0.5 * k_ij[:, 1]
        yy = k_ij[:, 1] * (3**0.5) / 2.0
        ang = torch.atan2(yy, xx)
        mask = (
            (k_ij[:, 0] >= 0)
            & (k_ij[:, 1] >= 0)
            & (ang <= (torch.pi / 6.0001))
            & (xx >= 0)
        )
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
        # add the other hemisphere
        kstar = torch.cat((kstar, kstar.clone()), dim=0)
        kstar[len(kstar) // 2 :, 2] *= -1
        # duplicate the kij for the other hemisphere
        k_ij = torch.cat((k_ij, k_ij.clone()), dim=0)
        k_ij[len(k_ij) // 2 :, 2] *= -1
    elif stnum == 19:  # hexagonal 6/mmm
        # keep kj >= 0 and ki >= 0 and xx >= 0 and ang <= pi/6 lacks SH
        xx = k_ij[:, 0] - 0.5 * k_ij[:, 1]
        yy = k_ij[:, 1] * (3**0.5) / 2.0
        ang = torch.atan2(yy, xx)
        mask = (
            (k_ij[:, 0] >= 0)
            & (k_ij[:, 1] >= 0)
            & (ang <= (torch.pi / 6.0001))
            & (xx >= 0)
        )
        kstar = hexagon_to_hemisphere(xy[mask])
        k_ij = k_ij[mask]
        # filter again if z < 0 (outside of the hexagon gets mapped down)
        mask_in_hex = kstar[:, 2] >= 0
        kstar = kstar[mask_in_hex]
        k_ij = k_ij[mask_in_hex]
    else:
        raise ValueError(f"Sampling type {stnum} not implemented.")

    # cartesian normalization and division by electron wavelength
    # kstar = kstar / torch.norm(kstar, dim=1, keepdim=True)

    # # I will just do this when doing diffraction computations
    # # as scalar mult commutes with matrix mult (dsm below)
    # kstar = kstar / e_lambda_nm

    # get k in reciprocal space by post-multiplying with DSM
    # k = torch.matmul(kstar, cell.dsm.to(kstar))

    # # k_n (normal component) is always a scalar 1/e_lambda_nm for EBSD
    # kn = 1.0 / e_lambda_nm

    return kstar, k_ij


# if __name__ == "__main__":

#     # visualize the mapping for all 32 point groups
#     import matplotlib.pyplot as plt
#     import numpy as np

#     grid_half_width = 500
#     device = torch.device("cuda")

#     # # loop over all point groups
#     # for stnum in range(1, 20):
#     #     if stnum == 11 or stnum == 13:
#     #         continue
#     #     kvecs, k_ij = kvectors_grid(
#     #         cell=Cell(
#     #             sg_num=225,
#     #             atom_data=[
#     #                 (28, 0.0, 0.0, 0.0, 1.0, 0.00328),
#     #             ],
#     #             abc=(0.3524, 0.3524, 0.3524),
#     #             abc_units="nm",
#     #             angles=(90.0, 90.0, 90.0),
#     #             angles_units="deg",
#     #         ),
#     #         grid_half_width=grid_half_width,
#     #         stnum=stnum,
#     #     )
#     #     kvecs = kvecs.cpu().numpy()
#     #     k_ij = k_ij.cpu().numpy()
#     #     img_NH = np.zeros((2 * grid_half_width + 1, 2 * grid_half_width + 1))
#     #     img_SH = np.zeros((2 * grid_half_width + 1, 2 * grid_half_width + 1))
#     #     indices_NH_mask = k_ij[:, 2] == 1
#     #     indices_SH_mask = k_ij[:, 2] == -1
#     #     img_NH[
#     #         k_ij[indices_NH_mask, 0] + grid_half_width,
#     #         k_ij[indices_NH_mask, 1] + grid_half_width,
#     #     ] = 1.0
#     #     img_SH[
#     #         k_ij[indices_SH_mask, 0] + grid_half_width,
#     #         k_ij[indices_SH_mask, 1] + grid_half_width,
#     #     ] = 1.0

#     #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     #     min_val = min(img_NH.min(), img_SH.min())
#     #     max_val = max(img_NH.max(), img_SH.max())
#     #     ax[0].imshow(img_NH, cmap="gray", vmin=min_val, vmax=max_val)
#     #     ax[0].set_title("Northern Hemisphere")
#     #     ax[0].axis("off")
#     #     ax[1].imshow(img_SH, cmap="gray", vmin=min_val, vmax=max_val)
#     #     ax[1].set_title("Southern Hemisphere")
#     #     ax[1].axis("off")
#     #     plt.suptitle(f"STNUM: {stnum}")
#     #     plt.show()

#     #     # also do a plot with the k-vectors stereographically projected
#     #     # for each hemisphere
#     #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     #     ax[0].scatter(kvecs[indices_NH_mask, 0], kvecs[indices_NH_mask, 1], s=1, c="r")
#     #     ax[0].set_title("Northern Hemisphere")
#     #     ax[0].axis("equal")
#     #     ax[0].axis("off")
#     #     ax[1].scatter(kvecs[indices_SH_mask, 0], kvecs[indices_SH_mask, 1], s=1, c="r")
#     #     ax[1].set_title("Southern Hemisphere")
#     #     ax[1].axis("equal")
#     #     ax[1].axis("off")
#     #     plt.suptitle(f"STNUM: {stnum}")
#     #     plt.show()


#     # Nickel
#     cell = Cell(
#         sg_num=225,
#         atom_data=[
#             (28, 0.0, 0.0, 0.0, 1.0, 0.00328),
#         ],
#         abc=(0.3524, 0.3524, 0.3524),
#         abc_units="nm",
#         angles=(90.0, 90.0, 90.0),
#         angles_units="deg",
#     )

#     # make a plot for all stnum cases at once (19 X 4 plots)
#     fig, ax = plt.subplots(4, 19, figsize=(20, 4))

#     for stnum in range(1, 20):
#         # add title to the first row
#         ax[0, stnum - 1].set_title(f"CASE: {stnum}", fontsize=10)
#         if stnum == 11 or stnum == 13:
#             # just shut off the axes for the empty plots
#             ax[0, stnum - 1].axis("off")
#             ax[1, stnum - 1].axis("off")
#             ax[2, stnum - 1].axis("off")
#             ax[3, stnum - 1].axis("off")
#             continue
#         kvecs, k_ij = kvectors_grid(
#             cell,
#             grid_half_width,
#             stnum=stnum,
#         )
#         print(
#             f"Number of k-vectors for stnum {stnum}: {len(kvecs)}, fraction of sphere: {len(kvecs) / (2  * (2 * grid_half_width + 1) ** 2)}"
#         )
#         kvecs = kvecs.cpu().numpy()
#         k_ij = k_ij.cpu().numpy()
#         img_NH = np.zeros((2 * grid_half_width + 1, 2 * grid_half_width + 1))
#         img_SH = np.zeros((2 * grid_half_width + 1, 2 * grid_half_width + 1))
#         indices_NH_mask = k_ij[:, 2] == 1
#         indices_SH_mask = k_ij[:, 2] == -1
#         img_NH[
#             k_ij[indices_NH_mask, 0] + grid_half_width,
#             k_ij[indices_NH_mask, 1] + grid_half_width,
#         ] = 1.0
#         img_SH[
#             k_ij[indices_SH_mask, 0] + grid_half_width,
#             k_ij[indices_SH_mask, 1] + grid_half_width,
#         ] = 1.0

#         min_val = min(img_NH.min(), img_SH.min())
#         max_val = max(img_NH.max(), img_SH.max())
#         ax[0, stnum - 1].imshow(img_NH, cmap="gray", vmin=min_val, vmax=max_val)
#         ax[0, stnum - 1].axis("off")
#         ax[1, stnum - 1].imshow(img_SH, cmap="gray", vmin=min_val, vmax=max_val)
#         ax[1, stnum - 1].axis("off")

#         # also do a plot with the k-vectors stereographically projected
#         # for each hemisphere
#         ax[2, stnum - 1].scatter(
#             kvecs[indices_NH_mask, 0], kvecs[indices_NH_mask, 1], s=1, c="r"
#         )
#         ax[2, stnum - 1].axis("equal")
#         ax[2, stnum - 1].axis("off")
#         ax[3, stnum - 1].scatter(
#             kvecs[indices_SH_mask, 0], kvecs[indices_SH_mask, 1], s=1, c="r"
#         )
#         ax[3, stnum - 1].axis("equal")
#         ax[3, stnum - 1].axis("off")

#         if stnum == 1:
#             # manually place y-labels for the first column
#             # turn axes back on to be able to set labels
#             for p in range(4):
#                 ax[p, stnum - 1].axis("on")
#                 ax[p, stnum - 1].set_xticks([])
#                 ax[p, stnum - 1].set_yticks([])
#                 ax[p, stnum - 1].grid(False)
#                 ax[p, stnum - 1].set_ylabel(["NH", "SH", "NH", "SH"][p], fontsize=10)
#                 if p == 2 or p == 3:
#                     # remove bounding box around the scatter plots
#                     ax[p, stnum - 1].spines["top"].set_visible(False)
#                     ax[p, stnum - 1].spines["right"].set_visible(False)
#                     ax[p, stnum - 1].spines["bottom"].set_visible(False)
#                     ax[p, stnum - 1].spines["left"].set_visible(False)

#     plt.tight_layout()
#     plt.savefig("kvectors_mapping.png")
#     plt.show()
