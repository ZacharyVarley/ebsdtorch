import torch


@torch.jit.script
def square_lambert(pts: torch.Tensor) -> torch.Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square lambert projection.
    :param pts: torch tensor of shape (n, 3) containing the points
    :return: torch tensor of shape (n, 2) containing the projected points
    """

    # constants
    TWO_DIV_SQRT8 = 0.7071067811865475   # 2 / sqrt(8)
    TWOSQRT2_DIV_PI = 0.9003163161571062 # 2 * sqrt(2) / pi

    # Define output tensor
    out = torch.empty((pts.shape[0], 2), dtype=pts.dtype, device=pts.device)

    # Define conditions and calculations
    cond = torch.abs(pts[:, 1]) <= torch.abs(pts[:, 0])
    factor = torch.sqrt(2.0 * (1.0 - torch.abs(pts[:, 2])))

    # instead of precalcuating each branch, just use the condition to select the correct branch
    out[cond, 0] = torch.sign(pts[cond, 0]) * factor[cond] * TWO_DIV_SQRT8
    out[cond, 1] = (
        torch.sign(pts[cond, 0])
        * factor[cond]
        * torch.atan2(
            pts[cond, 1] * torch.sign(pts[cond, 0]),
            pts[cond, 0] * torch.sign(pts[cond, 0]),
        )
        * TWOSQRT2_DIV_PI
    )
    out[~cond, 0] = (
        torch.sign(pts[~cond, 1])
        * factor[~cond]
        * torch.atan2(
            pts[~cond, 0] * torch.sign(pts[~cond, 1]),
            pts[~cond, 1] * torch.sign(pts[~cond, 1]),
        )
        * TWOSQRT2_DIV_PI
    )
    out[~cond, 1] = torch.sign(pts[~cond, 1]) * factor[~cond] * TWO_DIV_SQRT8

    # where close to (0, 0, 1), map to (0, 0)
    at_pole = torch.abs(pts[:, 2]) > 0.99999999
    out[at_pole] = 0.0
    
    return out


# @torch.jit.script
# def square_lambert_two(pts: torch.Tensor) -> torch.Tensor:
#     """
#     Map points on the sphere to the unit square using the square lambert projection.
#     :param pts: torch tensor of shape (n, 3) containing the points
#     :return: torch tensor of shape (n, 2) containing the projected points
#     """
#     ROOTPI_DIV2 = 0.886226925452758  # sqrt(pi) / 2
#     ROOT_PIDIV2 = 1.253314137315500  # sqrt(pi / 2)

#     # Define output tensor
#     out = torch.empty((pts.shape[0], 2), dtype=pts.dtype, device=pts.device)

#     # Define conditions and calculations
#     condition = torch.abs(pts[:, 1]) <= torch.abs(pts[:, 0])
#     factor = torch.sqrt(2.0 * (1.0 - torch.abs(pts[:, 2])))

#     # instead of precalcuating each branch, just use the condition to select the correct branch
#     out[condition, 0] = torch.sign(pts[condition, 0]) * factor[condition] * ROOTPI_DIV2
#     out[condition, 1] = (
#         torch.sign(pts[condition, 0])
#         * factor[condition]
#         * torch.atan(pts[condition, 1] / pts[condition, 0] / ROOTPI_DIV2)
#     )

#     out[~condition, 0] = (
#         torch.sign(pts[~condition, 1])
#         * factor[~condition]
#         * torch.atan(pts[~condition, 0] / pts[~condition, 1] / ROOTPI_DIV2)
#     )
#     out[~condition, 1] = (
#         torch.sign(pts[~condition, 1]) * factor[~condition] * ROOTPI_DIV2
#     )

#     return out / ROOT_PIDIV2


# # generate 10 points on the sphere and then use each implementation of the square lambert projection
# # to map them to the unit square and compare the results

# # generate the points
# pts = torch.randn((10000, 3), dtype=torch.float32)
# pts = pts / torch.norm(pts, dim=1, keepdim=True)

# # move all the points to the northern hemisphere
# pts[:, 2] = torch.abs(pts[:, 2])

# # color the points according to their xyz coordinates
# colors = torch.stack((
#     pts[:, 0] + pts[:, 0].min() / pts[:, 0].max() - pts[:, 0].min(),
#     pts[:, 1] + pts[:, 1].min() / pts[:, 1].max() - pts[:, 1].min(),
#     pts[:, 2] + pts[:, 2].min() / pts[:, 2].max() - pts[:, 2].min()), dim=1)

# # map the points to the unit square
# pts1 = square_lambert(pts)
# pts2 = square_lambert_two(pts)

# # plot the results using plotly

# import plotly.graph_objects as go

# # make two plots for each implementation
# # plot the points on the sphere with their colors and the points on the square with their colors
# fig = go.Figure(
#     data=[
#         go.Scatter3d(
#             x=pts[:, 0],
#             y=pts[:, 1],
#             z=pts[:, 2],
#             mode="markers",
#             marker=dict(size=2, color=colors),
#         ),
#         go.Scatter3d(
#             x=pts1[:, 0],
#             y=pts1[:, 1],
#             z=torch.zeros_like(pts1[:, 0]),
#             mode="markers",
#             marker=dict(size=2, color=colors),
#         ),
#     ]
# )
# fig.show()

