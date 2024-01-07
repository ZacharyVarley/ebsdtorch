"""

This file implements functions for moving points on the 2-sphere and unit
quaternions to the arbitrarily chosen unique set of points / orientations under
the symmetry of one of the given Laue groups.

Quaternion operators for the Laue groups were taken from the following paper:

Larsen, Peter Mahler, and Søren Schmidt. "Improved orientation sampling for
indexing diffraction patterns of polycrystalline materials." Journal of Applied
Crystallography 50, no. 6 (2017): 1571-1582.

"""
import torch
from torch import Tensor

from ebsdtorch.s2_and_so3.orientations import (
    quaternion_multiply,
    quaternion_real_of_prod,
    quaternion_apply,
    quaternion_invert,
    misorientation_angle,
    norm_standard_quaternion,
    xyz_to_theta_phi,
)

from ebsdtorch.s2_and_so3.sampling import so3_cubochoric_grid, s2_fibonacci_lattice


@torch.jit.script
def get_laue_mult(laue_group: int) -> int:
    """
    This function returns the multiplicity of the Laue group specified by the
    laue_group parameter. The ordering of the Laue groups is as follows:

    1. C1
    2. C2
    3. C3
    4. C4
    5. C6
    6. D2
    7. D3
    8. D4
    9. D6
    10. T
    11. O

    Args:
        laue_group: integer between 1 and 11 inclusive

    Returns:
        integer containing the multiplicity of the Laue group

    """

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

    return LAUE_MULTS[laue_group - 1]


@torch.jit.script
def laue_elements(laue_id: int) -> Tensor:
    """
    This function returns the elements of the Laue group specified by the
    laue_id parameter. The elements are returned as a tensor of shape
    (cardinality, 4) where the first element is always the identity.

    Args:
        laue_id: integer between inclusive [1, 11]

    1. C1
    2. C2
    3. C3
    4. C4
    5. C6
    6. D2
    7. D3
    8. D4
    9. D6
    10. T
    11. O

    Returns:
        torch tensor of shape (cardinality, 4) containing the elements of the

    """

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
    R3 = 0.8660254037844386467637231707529361834714026269051903140279034897
    # 7 subsets of O Laue groups (O, T, D4, D2, C4, C2, C1)
    LAUE_O = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [R2, R2, 0.0, 0.0],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, -R2, 0.0],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
        ],
        dtype=torch.float64,
    )
    LAUE_T = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
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
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
        ],
        dtype=torch.float64,
    )
    LAUE_D2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    LAUE_C4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
        ],
        dtype=torch.float64,
    )
    LAUE_C2 = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
    )
    LAUE_C1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    # subsets of D6 Laue groups (D6, D3, C6, C3) - C1 was already defined above
    LAUE_D6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
            [0.0, R3, 0.5, 0.0],
            [0.0, -R3, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    LAUE_D3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
        ],
        dtype=torch.float64,
    )
    LAUE_C6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )
    LAUE_C3 = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, R3], [0.5, 0.0, 0.0, -R3]],
        dtype=torch.float64,
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

    return LAUE_GROUPS[laue_id - 1]


@torch.jit.script
def so3_to_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    This function moves the given quaternions to the fundamental zone of the
    given Laue group. The space of a single orientations is different than that
    of two relative orientations. This function moves the quaternions to the
    fundamental zone of the space of single orientations.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        orientations in fundamental zone of shape (..., 4)

    Notes:

    Imagine orienting a cube. The description of the orientation of the cube is
    different than the description of the relative orientation of the cube with
    respect to another cube, as there are two entities with its symmetry. This
    has been called the "exchange symmetry" of misorientation space.

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.device).to(quats.dtype)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = quaternion_real_of_prod(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = quaternion_multiply(quats.reshape(N, 4), laue_group[row_maximum_indices])

    return output.reshape(data_shape)


@torch.jit.script
def so3_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given quaternions are in the fundamental zone of the given
    Laue group. This computes the orientation fundamental zone, not the
    misorientation fundamental zone.


    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)


    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.device).to(quats.dtype)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equiv_quats_real_part = quaternion_real_of_prod(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equiv_quats_real_part, dim=-1)

    # first element is always the identity for the enumerations of the Laue operators
    # so if its index is 0, then a given orientation was already in the fundamental zone
    return (row_maximum_indices == 0).reshape(data_shape[:-1])


@torch.jit.script
def disori_angle(quats1: Tensor, quats2: Tensor, laue_id: int) -> Tensor:
    """

    Return the disorientation angle in radians between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)

    Returns:
        disorientation angle in radians of shape (...)

    """
    # multiply without symmetry
    misori_quats = quaternion_multiply(quats1, quaternion_invert(quats2))

    # move the misorientation quaternions to the fundamental zone (disorientation)
    disori_quats_fz = so3_to_fz_laue(misori_quats, laue_id)

    # find the disorientation angle
    return misorientation_angle(norm_standard_quaternion(disori_quats_fz))


@torch.jit.script
def s2_in_fz_laue(points: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given 3D points are in the fundamental zone of the given
    Laue group. This computes the sphere fundamental zone, not the misorientation
    nor orientation fundamental zone. The 2-spherical fundamental zone is also
    called the fundamental sector.

    """
    # define some constants
    PI_2 = torch.pi / 2.0
    PI_3 = torch.pi / 3.0
    PI_4 = torch.pi / 4.0
    PI_6 = torch.pi / 6.0
    PI_n23 = -2.0 * torch.pi / 3.0

    # set epsilon
    EPS = 1e-12

    # use rules to find the equivalent points in the fundamental zone
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

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
def s2_to_fz_laue(points: Tensor, laue_group: Tensor, laue_id: int) -> Tensor:
    """
    Move the given 3D points to the fundamental zone of the given Laue group. This
    computes the sphere fundamental zone, not the misorientation nor orientation
    fundamental zone. The 2-spherical fundamental zone is also called the
    fundamental sector.

    Args:
        points: points to move to fundamental zone of shape (..., 3)
        laue_group: laue group of points to move to fundamental zone
        laue_id: laue group of points to move to fundamental zone

    Returns:
        points in fundamental zone of shape (..., 3)

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

    # find the points that are in the s2 fundamental zone
    cond = s2_in_fz_laue(equivalent_points, laue_id)

    return equivalent_points[cond].reshape(data_shape)


@torch.jit.script
def s2_equiv_laue(points: Tensor, laue_id: int) -> Tensor:
    """
    Return the equivalent points in the 2-spherical fundamental zone of
    the given Laue group.

    Args:
        points: points to move to fundamental zone of shape (..., 3)
        laue_id: laue group of points to move to fundamental zone

    Returns:
        points in fundamental zone of shape (..., |laue_group|, 3)

    """

    # get the important shapes
    data_shape = points.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    laue_group = laue_elements(laue_id).to(points.device).to(points.dtype)

    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    equivalent_points = quaternion_apply(
        laue_group.reshape(-1, 4), points.view(N, 1, 3)
    )

    # concatenate all of the points with their inverted coordinates
    equivalent_points = torch.cat([equivalent_points, -equivalent_points], dim=1)

    return equivalent_points.reshape(data_shape[:-1] + (len(laue_group), 3))


@torch.jit.script
def so3_sample_fz_laue(
    laue_id: int,
    target_n_samples: int,
    device: torch.device,
) -> Tensor:
    """

    A function to sample the fundamental zone of SO(3) for a given Laue group.
    This function uses the cubochoric grid sampling method, although other methods
    could be used. A slight oversampling is used to ensure that the number of
    samples closest to the target number of samples is used, as rejection sampling
    is used here.

    Args:
        laue_id: integer between 1 and 11 inclusive
        target_n_samples: number of samples to use on the fundamental sector of SO(3)
        device: torch device to use

    Returns:
        torch tensor of shape (n_samples, 4) containing the sampled orientations

    """
    # get the multiplicity of the laue group
    laue_mult = get_laue_mult(laue_id)

    # multiply by half the Laue multiplicity (inversion is not included in the operators)
    required_oversampling = (
        torch.tensor([int(target_n_samples * 1.018)]) * 0.5 * laue_mult
    )

    # take the cube root to get the edge length
    edge_length = int(torch.ceil(torch.pow(required_oversampling, 1 / 3)))
    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[so3_in_fz_laue(so3_samples, laue_id)]

    # randomly permute the samples
    so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


@torch.jit.script
def s2_sample_fz_laue(
    laue_group: int,
    target_n_samples: int,
    device: torch.device,
) -> Tensor:
    """

    A function to sample the fundamental zone of S2 for a given Laue group.
    This function uses the fibonacci lattice sampling method, although other methods
    could be used. A slight oversampling is used to ensure that the number of
    samples closest to the target number of samples is used, as rejection sampling
    is used here.

    Args:
        laue_group: integer between 1 and 11 inclusive
        target_n_samples: number of samples to use on the fundamental sector of S2
        device: torch device to use

    Returns:
        torch tensor of shape (n_samples, 3) containing the sampled orientations

    """

    laue_mult = get_laue_mult(laue_group)

    # get the sampling locations on the fundamental sector of S2
    s2_samples = s2_fibonacci_lattice(target_n_samples * laue_mult, device=device)

    # filter out all but the S2 fundamental sector of the laue group
    s2_samples_fz = s2_samples[s2_in_fz_laue(s2_samples, laue_group)]

    return s2_samples_fz


@torch.jit.script
def so3_color_fz_laue(
    quaternions: Tensor, reference_direction: Tensor, laue_group: Tensor, laue_id: int
) -> Tensor:
    """

    Return the coloring of each orientation.

    """

    reference_direction_moved = quaternion_apply(quaternions, reference_direction)

    reference_direction_moved_fz = s2_to_fz_laue(
        reference_direction_moved, laue_group, laue_id
    )

    theta_phi = xyz_to_theta_phi(reference_direction_moved_fz)  # INCORRECT!!!
    theta_phi[:, 0] = theta_phi[:, 0] * 2.0 + 0.5

    angle = torch.fmod(theta_phi[:, 0] / (2.0 * torch.pi), 1.0)

    hsl = torch.stack((angle, torch.ones_like(angle), theta_phi[:, 1]), dim=-1)

    l2 = hsl[..., 2] * 2.0
    s2 = hsl[..., 1] * torch.where(l2 <= 1.0, l2, 2.0 - l2)
    s2[torch.isnan(s2)] = 0.0
    val = (l2 + s2) / 2.0

    hsv = torch.stack((hsl[..., 0], s2, val), dim=-1)

    h6_f = torch.floor(hsv[..., 0] * 6.0)
    h6_byte = h6_f.byte()
    f = hsv[..., 0] * 6.0 - h6_f
    p = hsv[..., 2] * (1.0 - hsv[..., 1])
    q = hsv[..., 2] * (1.0 - hsv[..., 1] * f)
    t = hsv[..., 2] * (1.0 - hsv[..., 1] * (1.0 - f))

    output = torch.zeros_like(hsv)

    m0 = h6_byte == 0
    m1 = h6_byte == 1
    m2 = h6_byte == 2
    m3 = h6_byte == 3
    m4 = h6_byte == 4
    m5 = h6_byte == 5

    output[m0] = torch.stack((hsv[m0, 2], t[m0], p[m0]), dim=-1)
    output[m1] = torch.stack((q[m1], hsv[m1, 2], p[m1]), dim=-1)
    output[m2] = torch.stack((p[m2], hsv[m2, 2], t[m2]), dim=-1)
    output[m3] = torch.stack((p[m3], q[m3], hsv[m3, 2]), dim=-1)
    output[m4] = torch.stack((t[m4], p[m4], hsv[m4, 2]), dim=-1)
    output[m5] = torch.stack((hsv[m5, 2], p[m5], q[m5]), dim=-1)

    return (output * 255.0).byte()


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
