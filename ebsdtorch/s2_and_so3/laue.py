"""

This file implements operations for points on the 2-sphere and orientations
represented by unit quaternions, all under given Laue group(s).

Notes:

Imagine a cube floating in 3D space. The orientation fundamental zone is a
unique selection of all possible orientings under the symmetry of the cube. This
space is smaller than the space of all possible orientations of something
without symmetry. Now imagine two cubes floating in 3D space. The misorientation
fundamental zone is a unique selection of all relative orientations of the two
cubes that are unique under the symmetry of the two cubes (the "exchange
symmetry"). The symmetry of the two objects need not be the same. For example,
the first object could be a cube and the second object could be a tetrahedron.
The misorientation fundamental zone is smaller than all of orientation space.

For a detailed, see the following paper:

Krakow, Robert, Robbie J. Bennett, Duncan N. Johnstone, Zoja Vukmanovic,
Wilberth Solano-Alvarez, Steven J. Lainé, Joshua F. Einsle, Paul A. Midgley,
Catherine MF Rae, and Ralf Hielscher. "On three-dimensional misorientation
spaces." Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences 473, no. 2206 (2017): 20170274.

Quaternion operators for the Laue groups were taken from the following paper:

Larsen, Peter Mahler, and Søren Schmidt. "Improved orientation sampling for
indexing diffraction patterns of polycrystalline materials." Journal of Applied
Crystallography 50, no. 6 (2017): 1571-1582.

"""

import torch
from torch import Tensor

from ebsdtorch.s2_and_so3.orientations import (
    qu_prod,
    qu_prod_pos_real,
    qu_apply,
    qu_conj,
    qu_angle,
    qu_norm_std,
)
from ebsdtorch.s2_and_so3.sphere import xyz_to_theta_phi
from ebsdtorch.s2_and_so3.sampling import so3_cubochoric_grid, s2_fibonacci_lattice
from ebsdtorch.s2_and_so3.square_projection import inv_rosca_lambert


@torch.jit.script
def get_laue_mult(laue_group: int) -> int:
    """
    Multiplicity of a given Laue group (including inversion):

    Laue =11] m3‾m, 4‾3m, 432       Cubic      high
    Laue =10] m3‾, 23               Cubic      low
    Laue = 9] 6/mmm, 6‾m2, 6mm, 622 Hexagonal  high
    Laue = 8] 6/m, 6‾, 6            Hexagonal  low
    Laue = 7] 3‾m, 3m, 32           Trigonal   high
    Laue = 6] 3‾, 3                 Trigonal   low
    Laue = 5] 4/mmm, 4‾2m, 4mm, 422 Tetragonal high
    Laue = 4] 4/m, 4‾, 4            Tetragonal low
    Laue = 3] mmm, mm2, 222         Orthorhombic
    Laue = 2] 2/m, m, 2             Monoclinic
    Laue = 1] 1‾, 1                 Triclinic

    Args:
        laue_group: integer between 1 and 11 inclusive

    Returns:
        integer containing the multiplicity of the Laue group

    """

    LAUE_MULTS = [
        2,  #   1 - Triclinic
        4,  #   2 - Monoclinic
        8,  #   3 - Orthorhombic
        8,  #   4 - Tetragonal low
        16,  #  5 - Tetragonal high
        6,  #   6 - Trigonal low
        12,  #  7 - Trigonal high
        12,  #  8 - Hexagonal low
        24,  #  9 - Hexagonal high
        24,  # 10 - Cubic low
        48,  # 11 - Cubic high
    ]

    return LAUE_MULTS[laue_group - 1]


@torch.jit.script
def laue_elements(laue_id: int) -> Tensor:
    """
    Generators for Laue group specified by the laue_id parameter. The first
    element is always the identity.

    Laue =11] m3‾m, 4‾3m, 432       Cubic      high
    Laue =10] m3‾, 23               Cubic      low
    Laue = 9] 6/mmm, 6‾m2, 6mm, 622 Hexagonal  high
    Laue = 8] 6/m, 6‾, 6            Hexagonal  low
    Laue = 7] 3‾m, 3m, 32           Trigonal   high
    Laue = 6] 3‾, 3                 Trigonal   low
    Laue = 5] 4/mmm, 4‾2m, 4mm, 422 Tetragonal high
    Laue = 4] 4/m, 4‾, 4            Tetragonal low
    Laue = 3] mmm, mm2, 222         Orthorhombic
    Laue = 2] 2/m, m, 2             Monoclinic
    Laue = 1] 1‾, 1                 Triclinic

    Args:
        laue_id: integer between inclusive [1, 11]

    Returns:
        torch tensor of shape (cardinality, 4) containing the elements of the


    Notes:

    https://en.wikipedia.org/wiki/Space_group

    """

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 1.0 / (2.0**0.5)
    R3 = (3.0**0.5) / 2.0

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

    LAUE_C3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
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

    LAUE_C4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
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

    LAUE_C2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_GROUPS = [
        LAUE_C1,  #  1 - Triclinic
        LAUE_C2,  #  2 - Monoclinic
        LAUE_D2,  #  3 - Orthorhombic
        LAUE_C4,  #  4 - Tetragonal low
        LAUE_D4,  #  5 - Tetragonal high
        LAUE_C3,  #  6 - Trigonal low
        LAUE_D3,  #  7 - Trigonal high
        LAUE_C6,  #  8 - Hexagonal low
        LAUE_D6,  #  9 - Hexagonal high
        LAUE_T,  #  10 - Cubic low
        LAUE_O,  #  11 - Cubic high
    ]

    return LAUE_GROUPS[laue_id - 1]


@torch.jit.script
def ori_to_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    This function moves the given quaternions to the fundamental zone of the
    given Laue group. This computes the orientation fundamental zone, not the
    misorientation fundamental zone.

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
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    )

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = qu_prod(quats.reshape(N, 4), laue_group[row_maximum_indices])

    return output.reshape(data_shape)


@torch.jit.script
def ori_equiv_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Find the equivalent orientations under the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        Slices of equivalent quaternions of shape (..., |laue_group|, 4)

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions = qu_prod(quats.reshape(N, 1, 4), laue_group.reshape(-1, 4))

    return equivalent_quaternions.reshape(data_shape[:-1] + (len(laue_group), 4))


@torch.jit.script
def ori_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given quaternions are in the orientation fundamental zone
    of the given Laue group, not the misorientation fundamental zone.

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
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equiv_quats_real_part = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equiv_quats_real_part, dim=-1)

    # first element is always the identity for the enumerations of the Laue operators
    # so if its index is 0, then a given orientation was already in the fundamental zone
    return (row_maximum_indices == 0).reshape(data_shape[:-1])


@torch.jit.script
def ori_angle_laue(quats1: Tensor, quats2: Tensor, laue_id: int) -> Tensor:
    """

    Return the misalignment angle in radians between the given quaternions. This
    is not the disorientation angle, which is the angle between the two quaternions
    with both pre and post multiplication by the respective Laue groups.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)

    Returns:
        orientation angle in radians of shape (...)

    """

    # multiply without symmetry
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # move the orientation quaternions to the fundamental zone
    ori_quats_fz = ori_to_fz_laue(misori_quats, laue_id)

    # find the disorientation angle
    return qu_angle(qu_norm_std(ori_quats_fz))


@torch.jit.script
def disorientation(quats1: Tensor, quats2: Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation quaternion between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats1.shape

    # check that the shapes are the same
    if data_shape != quats2.shape:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, but got {data_shape} and {quats2.shape}"
        )

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1).to(quats1.dtype).to(quats1.device)

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2).to(quats2.dtype).to(quats2.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quaternions = qu_prod(
        laue_group_2.reshape(1, -1, 1, 4),
        qu_prod(misori_quats.view(N, 1, 1, 4), laue_group_1.reshape(1, 1, -1, 4)),
    )

    # flatten along the laue group dimensions
    equivalent_quaternions = equivalent_quaternions.reshape(N, -1, 4)

    # find the quaternion with the largest real part value (smallest angle)
    row_maximum_indices = torch.argmax(
        equivalent_quaternions[..., 0].abs(),
        dim=-1,
    )

    # TODO - Multiple equivalent quaternions can have the same angle. This function
    # should choose the one with an axis that is in the fundamental sector of the sphere
    # under the symmetry given by the intersection of the two Laue groups.

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = equivalent_quaternions[torch.arange(N), row_maximum_indices]

    return output.reshape(data_shape)


@torch.jit.script
def disori_angle_laue(quats1: Tensor, quats2: Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation angle in radians between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats1.shape

    # check that the shapes are the same
    if data_shape != quats2.shape:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, but got {data_shape} and {quats2.shape}"
        )

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1).to(quats1.dtype).to(quats1.device)

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2).to(quats2.dtype).to(quats2.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quaternions = qu_prod(
        laue_group_2.reshape(1, -1, 1, 4),
        qu_prod(misori_quats.view(N, 1, 1, 4), laue_group_1.reshape(1, 1, -1, 4)),
    )

    # flatten along the laue group dimensions
    equivalent_quaternions = equivalent_quaternions.reshape(N, -1, 4)

    # find the quaternion with the largest real part value (smallest angle)
    row_maximum_indices = torch.argmax(
        equivalent_quaternions[..., 0].abs(),
        dim=-1,
    )

    # TODO - Multiple equivalent quaternions can have the same angle. This function
    # should choose the one with an axis that is in the fundamental sector of the sphere
    # under the symmetry given by the intersection of the two Laue groups.

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = equivalent_quaternions[torch.arange(N), row_maximum_indices]

    return output.reshape(data_shape)


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
def s2_to_fz_laue(points: Tensor, laue_id: int) -> Tensor:
    """
    Move 3D Cartesian points on 2-sphere to fundamental zone for the Laue group.

    Args:
        points: points to move to fundamental zone of shape (..., 3) laue_group:
        laue group of points to move to fundamental zone laue_id: laue group of
        points to move to fundamental zone

    Returns:
        points in fundamental zone of shape (..., 3)

    """

    # get the important shapes
    data_shape = points.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # get the laue group elements
    laue_group = laue_elements(laue_id).to(points.dtype).to(points.device)

    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    equivalent_points = qu_apply(laue_group.reshape(-1, 4), points.view(N, 1, 3))

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
    laue_group = laue_elements(laue_id).to(points.dtype).to(points.device)

    # reshape so that points is (N, 1, 3) and laue_group is (1, card, 4) then use broadcasting
    equivalent_points = qu_apply(laue_group.reshape(-1, 4), points.view(N, 1, 3))

    # concatenate all of the points with their inverted coordinates
    equivalent_points = torch.cat([equivalent_points, -equivalent_points], dim=1)

    return equivalent_points.reshape(data_shape[:-1] + (len(laue_group), 3))


@torch.jit.script
def sample_ori_fz_laue(
    laue_id: int,
    target_n_samples: int,
    device: torch.device,
) -> Tensor:
    """

    A function to sample the fundamental zone of SO(3) for a given Laue group.
    This function uses the cubochoric grid sampling method, although other methods
    could be used. Rejection sampling is used so the number of samples will almost
    certainly be different than the target number of samples.

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
    required_oversampling = target_n_samples * 0.5 * laue_mult

    # take the cube root to get the edge length
    edge_length = int(required_oversampling ** (1.0 / 3.0))
    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[ori_in_fz_laue(so3_samples, laue_id)]

    # randomly permute the samples
    so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


@torch.jit.script
def sample_ori_fz_laue_angle(
    laue_id: int,
    target_mean_disorientation: float,
    device: torch.device,
    permute: bool = True,
) -> Tensor:
    """

    A function to sample the fundamental zone of SO(3) for a given Laue group.
    This function uses the cubochoric grid sampling method, although other methods
    could be used. A target number of samples is used, as rejection sampling
    is used here, so the number of samples will almost certainly be different.

    Args:
        laue_id: integer between 1 and 11 inclusive
        target_mean_disorientation: target mean disorientation in radians
        device: torch device to use
        permute: whether or not to randomly permute the samples

    Returns:
        torch tensor of shape (n_samples, 4) containing the sampled orientations

    """
    # get the multiplicity of the laue group
    laue_mult = get_laue_mult(laue_id)

    # use empirical fit to get the number of samples
    n_so3_without_symmetry = (
        2 * (131.97049) / (target_mean_disorientation - 0.03732) + 1
    ) ** 3
    edge_length = int((n_so3_without_symmetry / (0.5 * laue_mult)) ** (1.0 / 3.0) + 1.0)
    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[ori_in_fz_laue(so3_samples, laue_id)]

    # randomly permute the samples
    if permute:
        so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


@torch.jit.script
def sample_s2_fz_laue_fibonacci(
    laue_id: int,
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
        laue_id: integer between 1 and 11 inclusive
        target_n_samples: number of samples to use on the fundamental sector of S2
        device: torch device to use

    Returns:
        torch tensor of shape (n_samples, 3) containing the sampled orientations

    """

    laue_mult = get_laue_mult(laue_id)

    # get the sampling locations on the fundamental sector of S2
    s2_samples = s2_fibonacci_lattice(target_n_samples * laue_mult, device=device)

    # filter out all but the S2 fundamental sector of the laue group
    s2_samples_fz = s2_samples[s2_in_fz_laue(s2_samples, laue_id)]

    return s2_samples_fz


@torch.jit.script
def sample_s2_fz_laue_rosca(
    laue_id: int,
    target_n_samples: int,
    device: torch.device,
) -> Tensor:
    """

    A function to sample the fundamental zone of S2 for a given Laue group. This
    function uses the Rosca-Lambert equal area bijection between the square and
    the Northern hemisphere of the unit sphere to transform a uniform grid of 2D
    points to the sphere, where rejection sampling discards points not in the
    fundamental zone. The number of samples will almost certainly be different
    than the target number of samples.

    Args:
        laue_id: integer between 1 and 11 inclusive
        target_n_samples: number of samples to use on the fundamental sector of S2
        device: torch device to return the tensor on

    Returns:
        torch tensor of shape (n_samples, 3) containing the sampled orientations

    """

    laue_mult = get_laue_mult(laue_id)

    # estimate the edge length of the square
    edge_length = int((target_n_samples * laue_mult) ** 0.5)

    # make a meshgrid and flatten it on the square [-1, 1] x [-1, 1]
    x = torch.linspace(-1.0, 1.0, edge_length, device=device)
    y = torch.linspace(-1.0, 1.0, edge_length, device=device)
    xx, yy = torch.meshgrid(x, y)
    square_points = torch.stack((xx.flatten(), yy.flatten()), dim=-1)

    # use the Rosca-Lambert equal area bijection to map the square to the sphere
    s2_samples = inv_rosca_lambert(square_points)

    # filter out all but the S2 fundamental sector of the laue group
    s2_samples_fz = s2_samples[s2_in_fz_laue(s2_samples, laue_id)]

    return s2_samples_fz


@torch.jit.script
def so3_color_fz_laue(
    quaternions: Tensor,
    reference_direction: Tensor,
    laue_id: int,
) -> Tensor:
    """

    Return the coloring of each orientation.

    """

    reference_direction_moved = qu_apply(quaternions, reference_direction)

    reference_direction_moved_fz = s2_to_fz_laue(reference_direction_moved, laue_id)

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
