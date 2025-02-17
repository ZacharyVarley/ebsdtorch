from typing import Tuple
import torch
from torch import Tensor


@torch.jit.script
def square_to_hemisphere(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square
    lambert projection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    """

    # get shape of input
    shape_in = pts.shape[:-1]
    n_pts = int(torch.prod(torch.tensor(shape_in)))
    pts = pts.view(-1, 2)
    pi = torch.pi
    a = pts[:, 0] * (pi / 2) ** 0.5
    b = pts[:, 1] * (pi / 2) ** 0.5
    m = torch.abs(b) <= torch.abs(a)
    output = torch.empty((n_pts, 3), dtype=pts.dtype, device=pts.device)
    output[m, 0] = (
        (2 * a[m] / pi)
        * torch.sqrt(pi - a[m] ** 2)
        * torch.cos((pi * b[m]) / (4 * a[m]))
    )
    output[m, 1] = (
        (2 * a[m] / pi)
        * torch.sqrt(pi - a[m] ** 2)
        * torch.sin((pi * b[m]) / (4 * a[m]))
    )
    output[m, 2] = 1 - (2 * a[m] ** 2 / pi)
    output[~m, 0] = (
        (2 * b[~m] / pi)
        * torch.sqrt(pi - b[~m] ** 2)
        * torch.sin((pi * a[~m]) / (4 * b[~m]))
    )
    output[~m, 1] = (
        (2 * b[~m] / pi)
        * torch.sqrt(pi - b[~m] ** 2)
        * torch.cos((pi * a[~m]) / (4 * b[~m]))
    )
    output[~m, 2] = 1 - (2 * b[~m] ** 2 / pi)
    return output.reshape(shape_in + (3,))


@torch.jit.script
def square_to_hemisphere_v2(x: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square
    lambert projection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    This version is more efficient than the previous one, as it just plops
    everything into the first quadrant, then swaps the x and y coordinates
    if needed so that we always have x >= y. Then swaps back at the end and
    copy the sign of the original x and y coordinates.

    """
    pi = torch.pi
    # map to first quadrant and swap x and y if needed
    x_abs, y_abs = (
        torch.abs(x[..., 0]) * (pi / 2) ** 0.5,
        torch.abs(x[..., 1]) * (pi / 2) ** 0.5,
    )
    cond = x_abs >= y_abs
    x_new = torch.where(cond, x_abs, y_abs)
    y_new = torch.where(cond, y_abs, x_abs)

    # only one case now
    x_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.cos(pi * y_new / (4 * x_new))
    )
    y_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.sin(pi * y_new / (4 * x_new))
    )
    z_out = 1 - (2 * x_new**2 / pi)

    # swap back and copy sign
    x_out = torch.where(cond, x_hs, y_hs)
    y_out = torch.where(cond, y_hs, x_hs)
    x_out = x_out.copysign_(x[..., 0])
    y_out = y_out.copysign_(x[..., 1])

    return torch.stack((x_out, y_out, z_out), dim=-1)


# # test speed of both functions on cuda

# import matplotlib.pyplot as plt
# import time

# # uniform meshgrid in (-1, 1) x (-1, 1)
# n = 100
# n_runs = 10
# x = torch.linspace(-1, 1, n)
# xx, yy = torch.meshgrid(x, x)
# device = torch.device("cuda")
# pts = torch.stack((xx.flatten(), yy.flatten()), dim=-1).to(device)

# # time the first function
# total = 0
# _ = square_to_hemisphere(pts)  # warm up
# _ = square_to_hemisphere(pts)  # warm up
# for _ in range(n_runs):
#     start = time.time()
#     out = square_to_hemisphere(pts)
#     torch.cuda.synchronize()
#     total += time.time() - start
# print(f"Time for square_to_hemisphere: {total / n_runs:.8f} s")

# # time the second function
# total = 0
# _ = square_to_hemisphere_v2(pts)  # warm up
# _ = square_to_hemisphere_v2(pts)  # warm up
# for _ in range(n_runs):
#     start = time.time()
#     out_v2 = square_to_hemisphere_v2(pts)
#     torch.cuda.synchronize()
#     total += time.time() - start
# print(f"Time for square_to_hemisphere_v2: {total / n_runs:.8f} s")

# # check if the two functions give the same result
# print(
#     f"Max difference between the two functions: {torch.max(torch.abs(out - out_v2)):.15f}"
# )

# # plot the 2D points on the square, then on the hemisphere for each method so 3 plots total
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# # use complex plane coloring for the 2D square
# colors = torch.atan2(pts[:, 1], pts[:, 0])
# axs[0].scatter(pts[:, 0], pts[:, 1], c=colors, cmap="hsv")
# axs[0].set_title("2D square")
# # use z coordinate for the hemisphere
# axs[1].scatter(out[:, 0], out[:, 1], c=colors, cmap="hsv")
# axs[1].set_title("Hemisphere (square_to_hemisphere)")
# axs[1].set_xlim(-1.5, 1.5)
# axs[1].set_ylim(-1.5, 1.5)
# # use z coordinate for the hemisphere
# axs[2].scatter(out_v2[:, 0], out_v2[:, 1], c=colors, cmap="hsv")
# axs[2].set_title("Hemisphere (square_to_hemisphere_v2)")
# axs[2].set_xlim(-1.5, 1.5)
# axs[2].set_ylim(-1.5, 1.5)
# plt.show()
# # plt.savefig("square_to_hemisphere.png")


@torch.jit.script
def hexagon_to_hemisphere(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square
    lambert projection, assuming the input is a regular hexagon with the Eastern
    tip at (0, 1).

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points


    Roşca, D., and M. De Graef. "Area-preserving projections from hexagonal and
    triangular domains to the sphere and applications to electron back-scatter
    diffraction pattern simulations." Modelling and Simulation in Materials
    Science and Engineering 21.5 (2013): 055021.


    Different to original F90 code in EMsoft:

    1) Just rotate everything into the first sextant, then rotate back at the end.
    2) Apply a single correction for the area of the hexagon at the end

    """
    pi = torch.pi

    # initial swap to make corner at (1, 0) instead of (0, 1)
    y_abs, x_abs = torch.abs(pts).unbind(-1)

    # if we are in I_1 sextant, rotate to I_0
    rotate = y_abs > (x_abs / 3**0.5)

    # apply 60 degree clockwise rotation
    x_new = torch.where(rotate, x_abs * 0.5 + y_abs * 0.5 * (3**0.5), x_abs)
    y_new = torch.where(rotate, -x_abs * 0.5 * (3**0.5) + y_abs * 0.5, y_abs)

    # I_0 sextant equal-area bijection having swapped x & y
    factor = (3.0**0.25) * ((2.0 / pi) ** 0.5) * x_new
    trig_arg = (y_new * pi) / (2.0 * (3.0**0.5) * x_new)
    x_hs = factor * torch.cos(trig_arg)
    y_hs = factor * torch.sin(trig_arg)

    # undo the rotation
    x_out = torch.where(rotate, x_hs * 0.5 - y_hs * 0.5 * (3**0.5), x_hs)
    y_out = torch.where(rotate, x_hs * 0.5 * (3**0.5) + y_hs * 0.5, y_hs)

    # copy sign (with swap anticipated) and rescale to unit sphere
    x_out = x_out.copysign_(pts[..., 1]) / ((3 * 3**0.5 / (2 * pi)) ** 0.5)
    y_out = y_out.copysign_(pts[..., 0]) / ((3 * 3**0.5 / (2 * pi)) ** 0.5)

    # undo original swap and compute the z coordinate on the hemisphere
    return torch.stack((y_out, x_out, 1.0 - x_out**2 - y_out**2), dim=-1)


# import matplotlib.pyplot as plt
# import time

# # visualize just the hexagon to hemisphere mapping
# n = 2001
# # make regular hexagonal grid
# x = torch.linspace(-2, 2, n)
# xx, yy = torch.meshgrid(x, x)
# device = torch.device("cuda")

# pts = torch.stack((xx.flatten(), yy.flatten()), dim=-1).to(device).double()

# # time the hexagon to hemisphere function
# total = 0
# _ = hexagon_to_hemisphere(pts)  # warm up
# _ = hexagon_to_hemisphere(pts)  # warm up
# for _ in range(1):
#     start = time.time()
#     out_hex = hexagon_to_hemisphere(pts)
#     torch.cuda.synchronize()
#     total += time.time() - start

# print(f"Time for hexagon_to_hemisphere: {total / 10:.4f} s")

# # plot the 2D points on the hexagon, then on the hemisphere
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# # use complex plane coloring for the 2D hexagon
# colors = torch.atan2(pts[:, 1], pts[:, 0])

# colors = colors.cpu().numpy()
# pts = pts.cpu().numpy()
# out_hex = out_hex.cpu().numpy()


# # just plot the edge of the hexagon and circle
# mask = (out_hex[..., 2] < 0.01) & (out_hex[..., 2] > 0.000)

# pts = pts[mask]
# out_hex = out_hex[mask]
# colors = colors[mask]


# axs[0].scatter(pts[:, 0], pts[:, 1], c=colors, cmap="hsv")
# axs[0].set_title("2D hexagon")
# axs[0].set_xlim(-2, 2)
# axs[0].set_ylim(-2, 2)
# axs[0].set_aspect("equal")
# # use z coordinate for the hemisphere
# axs[1].scatter(out_hex[:, 0], out_hex[:, 1], c=colors, cmap="hsv")
# axs[1].set_title("Hemisphere (hexagon_to_hemisphere)")
# axs[1].set_xlim(-1.5, 1.5)
# axs[1].set_ylim(-1.5, 1.5)
# axs[1].set_aspect("equal")
# plt.show()


# # implementing kvectors and need to visualize which points stay with a loop:
# """
# istart = 0
# iend = npx
# jstart = 0
# jend = npx
# do j=jstart,jend
#     do i=istart+(j-1)/2,2*j
#     xy = (/ dble(i),  dble(j) /) * delta
#     if (InsideHexGrid(xy)) then
#         call AddkVector(ktail,cell,numk,xy,i,j,hexgrid, addSH = yes)
#     end if
#     end do
# end do

# Recursive function InsideHexGrid(xy) result(res)
# !DEC$ ATTRIBUTES DLLEXPORT :: InsideHexGrid

# use Lambert

# IMPLICIT NONE

# real(kind=dbl),INTENT(IN)       :: xy(2)
# logical                         :: res
# real(kind=dbl)                  :: ax, ay

# ! assume it is a good point
# res = .TRUE.

# ! first of all, take the absolute values and see if the transformed point lies inside the
# ! rectangular box with edge lengths (1,sqrt(3)/2)
# ax = abs(xy(1)-0.5D0*xy(2))
# ay = abs(xy(2)*LPs%srt)

# if ((ax.gt.1.D0).or.(ay.gt.LPs%srt)) res = .FALSE.
# ! then check for the inclined edge
# if (res) then
#   if (ax+ay*LPs%isrt .gt. 1.D0) res = .FALSE.
# end if

# end function InsideHexGrid

# """


# # writing just the loop traversal to see which points get called in the first sweep
# # and which points get called in the second sweep
# def inside_hex_grid(xy: Tuple[float, float]) -> bool:
#     ax = abs(xy[0] - 0.5 * xy[1])
#     ay = abs(xy[1] * (3**0.5 / 2))

#     if (ax > 1) or (ay > (3**0.5 / 2)):
#         return False
#     if (ax + ay * (1 / (3**0.5))) > 1:
#         return False
#     return True


# npx = 11
# img = torch.zeros((2 * npx + 1, 2 * npx + 1, 3), dtype=torch.float32)
# import numpy as np

# delta = 1 / npx
# for j in range(0, npx + 1):
#     for i in range((j - 1) // 2, 2 * j + 1):
#         xy = (i * delta, j * delta)
#         if inside_hex_grid(xy):
#             img[i + npx, j + npx] = torch.tensor([1, 1, 1])

# plt.imshow(img)
# plt.show()
