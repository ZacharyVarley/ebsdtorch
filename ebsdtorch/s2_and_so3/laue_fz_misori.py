"""

Functions for misorientation fundamental zones under the 11 Laue point groups.

The misorientation fundamental zone is the unique subset of orientation space
that contains all possible misorientations of two different entities each with a
given symmetry.

Krakow, Robert, Robbie J. Bennett, Duncan N. Johnstone, Zoja Vukmanovic,
Wilberth Solano-Alvarez, Steven J. Lainé, Joshua F. Einsle, Paul A. Midgley,
Catherine MF Rae, and Ralf Hielscher. "On three-dimensional misorientation
spaces." Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences 473, no. 2206 (2017): 20170274.

"""

import torch
from torch import Tensor

from ebsdtorch.s2_and_so3.quaternions import (
    qu_prod,
    qu_triple_prod_pos_real,
    qu_apply,
    qu_conj,
)
from ebsdtorch.s2_and_so3.sampling import so3_cubochoric_grid
from ebsdtorch.s2_and_so3.laue_generators import get_laue_mult, laue_elements


@torch.jit.script
def misori_to_fz_laue(qu: Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation quaternion between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4) in the form (w, x, y, z)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(qu.shape[:-1]))

    # retrieve the laue group elements
    laue_group_1 = laue_elements(laue_id_1).to(qu.dtype).to(qu.device)
    laue_group_2 = laue_elements(laue_id_2).to(qu.dtype).to(qu.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quaternions = qu_prod(
        laue_group_2.reshape(1, -1, 1, 4),
        qu_prod(qu.view(N, 1, 1, 4), laue_group_1.reshape(1, 1, -1, 4)),
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

    return output.reshape(qu.shape)


@torch.jit.script
def disorientation_naive(
    quats1: Tensor, quats2: Tensor, laue_id_1: int, laue_id_2: int
):
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
    equivalent_quat_pos_real = qu_triple_prod_pos_real(
        laue_group_2.reshape(1, -1, 1, 4),
        misori_quats.view(N, 1, 1, 4),
        laue_group_1.reshape(1, 1, -1, 4),
    )

    # flatten along the laue group dimensions
    equivalent_quat_pos_real = equivalent_quat_pos_real.reshape(N, -1)

    # find the largest real part magnitude and return the angle
    cosine_half_angle = torch.max(equivalent_quat_pos_real, dim=-1).values

    return 2.0 * torch.acos(cosine_half_angle)
