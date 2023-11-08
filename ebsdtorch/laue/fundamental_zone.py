import torch

from ebsdtorch.laue.orientations import quaternion_multiply, quaternion_real_of_prod, quaternion_apply

# sqrt(2) / 2 and sqrt(3) / 2
R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
R3 = 0.8660254037844386467637231707529361834714026269051903140279034897

# 7 subsets of O Laue groups (O, T, D4, D2, C4, C2, C1)
LAUE_O = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2, R2, 0, 0],
        [R2, -R2, 0, 0],
        [R2, 0, R2, 0],
        [R2, 0, -R2, 0],
        [0, R2, 0, R2],
        [0, -R2, 0, R2],
        [0, 0, R2, R2],
        [0, 0, -R2, R2],
    ],
    dtype=torch.float64,
)
LAUE_T = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ],
    dtype=torch.float64,
)
LAUE_D4 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
    ],
    dtype=torch.float64,
)
LAUE_D2 = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float64
)
LAUE_C4 = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 0, 1], [R2, 0, 0, R2], [R2, 0, 0, -R2]], dtype=torch.float64
)
LAUE_C2 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float64)
LAUE_C1 = torch.tensor([[1, 0, 0, 0]], dtype=torch.float64)


# subsets of D6 Laue groups (D6, D3, C6, C3) - C1 was already defined above
LAUE_D6 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 0, 0, 1],
        [R3, 0, 0, 0.5],
        [R3, 0, 0, -0.5],
        [0, 1, 0, 0],
        [0, -0.5, R3, 0],
        [0, 0.5, R3, 0],
        [0, R3, 0.5, 0],
        [0, -R3, 0.5, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float64,
)
LAUE_D3 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 1, 0, 0],
        [0, -0.5, R3, 0],
        [0, 0.5, R3, 0],
    ],
    dtype=torch.float64,
)
LAUE_C6 = torch.tensor(
    [
        [1, 0, 0, 0],
        [0.5, 0, 0, R3],
        [0.5, 0, 0, -R3],
        [0, 0, 0, 1],
        [R3, 0, 0, 0.5],
        [R3, 0, 0, -0.5],
    ],
    dtype=torch.float64,
)
LAUE_C3 = torch.tensor(
    [[1, 0, 0, 0], [0.5, 0, 0, R3], [0.5, 0, 0, -R3]], dtype=torch.float64
)

LAUE_GROUPS = [
    LAUE_C1,
    LAUE_C2,
    LAUE_C3,
    LAUE_C4,
    LAUE_C6,
    LAUE_D2,
    LAUE_D3,
    LAUE_D4,
    LAUE_D6,
    LAUE_T,
    LAUE_O,
]

LAUE_MULTS = [
    2,
    4,
    6,
    8,
    12,
    8,
    12,
    16,
    24,
    24,
    48,
]

@torch.jit.script
def _ori_to_so3_fz(
    orientations: torch.Tensor, laue_group: torch.Tensor
) -> torch.Tensor:
    """
    :param misorientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # get the important shapes
    data_shape = orientations.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = quaternion_real_of_prod(
        orientations.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = quaternion_multiply(
        orientations.reshape(N, 4), laue_group[row_maximum_indices]
    )

    return output.reshape(data_shape)


@torch.jit.script
def _oris_are_in_so3_fz(orientations: torch.Tensor, laue_group: torch.Tensor) -> torch.Tensor:
    """
    :param misorientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # get the important shapes
    data_shape = orientations.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = quaternion_real_of_prod(
        orientations.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # first element is always the identity so if the index is 0, then it is in the fundamental zone
    return (row_maximum_indices == 0).reshape(data_shape[:-1])


def oris_are_in_so3_fz(orientations: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param orientations: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _oris_are_in_so3_fz(
        orientations,
        LAUE_GROUPS[laue_group - 1].to(orientations.dtype).to(orientations.device),
    )


def oris_to_so3_fz(quaternions: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param quaternions: quaternions to move to fundamental zone of shape (..., 4)
    :param laue_group: laue group of quaternions to move to fundamental zone
    :return: orientations in fundamental zone of shape (..., 4)
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _ori_to_so3_fz(
        quaternions,
        LAUE_GROUPS[laue_group - 1].to(quaternions.dtype).to(quaternions.device),
    )


@torch.jit.script
def _points_are_in_s2_fz(points: torch.Tensor, laue_id: int) -> torch.Tensor:
    """
    :param points: points for filtering to fundamental zone of shape (N, 3)
    :param laue_id: laue group of points to move to fundamental zone
    :return: points in fundamental zone of shape (M, 3) where M <= N
    """
    # define some constants
    PI_2 = torch.pi / 2
    PI_3 = torch.pi / 3
    PI_4 = torch.pi / 4
    PI_6 = torch.pi / 6
    PI_n23 = -2.0 * torch.pi / 3

    # set epsilon
    EPS = 1e-12

    # use rules to find the equivalent points in the fundamental zone
    x, y, z = torch.unbind(points, dim=-1)

    eta = torch.atan2(y, x)
    chi = torch.acos(z)

    if laue_id == 1 or laue_id == 2:  # triclinic, monoclinic
        cond = eta.ge(0.0) & eta.le(torch.pi + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 3 or laue_id == 4:  # orthorhombic, tetragonal-low
        cond = eta.ge(0.0) & eta.le(PI_2 + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 5:  # tetragonal-high
        cond = eta.ge(0.0) & eta.le(PI_4 + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 6:  # trigonal-low
        cond = eta.ge(PI_n23) & (eta.le(0.0 + EPS)) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 7:  # trigonal-high
        cond = eta.ge(-PI_2) & eta.le(-PI_6 + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 8:  # hexagonal-low
        cond = eta.ge(0.0) & eta.le(PI_3 + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 9:  # hexagonal-high
        cond = eta.ge(0.0) & eta.le(PI_6 + EPS) & chi.ge(0.0) & chi.le(PI_2 + EPS)
    elif laue_id == 10:  # cubic-low
        # where eta is over 45 degrees, subtract from 90 degrees
        cond = (
            torch.where(
                eta.ge(PI_4),
                chi
                <= torch.acos(torch.sqrt(1.0 / (2.0 + torch.tan(PI_2 - eta) ** 2)))
                + EPS,
                chi <= torch.acos(torch.sqrt(1.0 / (2.0 + torch.tan(eta) ** 2))) + EPS,
            )
            & eta.ge(0.0)
            & eta.le(PI_2 + EPS)
            & chi.ge(0.0)
        )

    elif laue_id == 11:  # cubic-high
        # where eta is over 45 degrees, subtract from 90 degrees
        cond = (
            torch.where(
                eta.ge(PI_4),
                chi
                <= torch.acos(torch.sqrt(1.0 / (2.0 + torch.tan(PI_2 - eta) ** 2)))
                + EPS,
                chi <= torch.acos(torch.sqrt(1.0 / (2.0 + torch.tan(eta) ** 2))) + EPS,
            )
            & eta.ge(0.0)
            & eta.le(PI_4 + EPS)
            & chi.ge(0.0)
        )
    else:
        raise ValueError(f"Laue id {laue_id} not in [1, 11]")

    return cond


@torch.jit.script
def _points_to_s2_fz(
    points: torch.Tensor, laue_group: torch.Tensor, laue_id: int
) -> torch.Tensor:
    """
    :param points: points to move to fundamental zone of shape (..., 3)
    :param laue_group: laue group of points to move to fundamental zone
    :param laue_id: laue group of points to move to fundamental zone
    :return: points in fundamental zone of shape (..., 3)
    """

    # get the important shapes
    data_shape = points.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    equivalent_points = quaternion_apply(
        laue_group.reshape(-1, 4), points.view(N, 1, 3)
    )

    # concatenate all of the points with their inverted coordinates
    equivalent_points = torch.cat([equivalent_points, -equivalent_points], dim=1)

    cond = _points_are_in_s2_fz(equivalent_points, laue_id)

    return equivalent_points[cond].reshape(data_shape)


@torch.jit.script
def _eqiv_points_s2(
    points: torch.Tensor, laue_group: torch.Tensor
) -> torch.Tensor:
    """
    :param points: points to move to fundamental zone of shape (..., 3)
    :param laue_group: laue group of points to move to fundamental zone
    :param laue_id: laue group of points to move to fundamental zone
    :return: points in fundamental zone of shape (..., 3)
    """

    # get the important shapes
    data_shape = points.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    equivalent_points = quaternion_apply(
        laue_group.reshape(-1, 4), points.view(N, 1, 3)
    )

    # concatenate all of the points with their inverted coordinates
    equivalent_points = torch.cat([equivalent_points, -equivalent_points], dim=1)

    return equivalent_points.reshape(data_shape[:-1], len(laue_group), 3)


def eqiv_points_s2(points: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param points: points on the sphere to find equivalent points of shape (..., 3)
    :param laue_group: laue group of points to find equivalent points
    :return: equivalent points of shape (..., card, 3), card is Laue group cardinality
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _eqiv_points_s2(
        points,
        LAUE_GROUPS[laue_group - 1].to(points.dtype).to(points.device),
    )


def points_to_s2_fz(points: torch.Tensor, laue_group: int) -> torch.Tensor:
    """
    :param points: points to move to fundamental zone of shape (..., 3)
    :param laue_group: laue group of points to move to fundamental zone
    :return: points in fundamental zone of shape (..., 3)
    """
    # assert LAUE_GROUP is an int between 1 and 11 inclusive
    if not isinstance(laue_group, int) or laue_group < 1 or laue_group > 11:
        raise ValueError(f"Laue group {laue_group} not laue number in [1, 11]")
    # find all equivalent quaternions
    return _points_to_s2_fz(
        points,
        LAUE_GROUPS[laue_group - 1].to(points.dtype).to(points.device),
        laue_group,
    )


# # test if moving points on a sphere in 3D to the fundamental zone
# # makes a small fundamental zone on the sphere

# import numpy as np

# # generate random points on a sphere
# N = 100000
# points = torch.randn(N, 3)
# points = points / torch.linalg.norm(points, dim=1, keepdim=True)

# from sampling import s2_fibonacci_lattice

# points = s2_fibonacci_lattice(N).to(torch.float32).to(points.device)
# points = points / torch.linalg.norm(points, dim=1, keepdim=True)

# # move points to fundamental zone
# # points_fz = points_to_s2_fz(points, 11)

# are_in_fz = _points_are_in_s2_fz(points, 9)
# print(len(are_in_fz))
# print(f"{are_in_fz.sum()} points are in the fundamental zone")
# points_fz = points[are_in_fz]
# points_not_fz = points[~are_in_fz]

# # plot the points using plotly
# import plotly.graph_objects as go

# fig = go.Figure()
# fig.add_trace(
#     go.Scatter3d(
#         x=points_fz[:, -3].numpy(),
#         y=points_fz[:, -2].numpy(),
#         z=points_fz[:, -1].numpy(),
#         mode="markers",
#         marker=dict(size=2, color="red"),
#     )
# )
# fig.add_trace(
#     go.Scatter3d(
#         x=points_not_fz[:, -3].numpy(),
#         y=points_not_fz[:, -2].numpy(),
#         z=points_not_fz[:, -1].numpy(),
#         mode="markers",
#         marker=dict(size=2, color="blue"),
#     )
# )
# fig.show()
