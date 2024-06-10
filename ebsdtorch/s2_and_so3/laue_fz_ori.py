"""

Functions for orientation fundamental zones under the 11 Laue point groups.

The orientation fundamental zone is a unique subset of the orientation space
that is used to represent all possible orientations of a crystal possessing a
given symmetry. Covered briefly in the beginning of the following paper:

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
    qu_prod_pos_real,
    qu_triple_prod_pos_real,
    qu_apply,
    qu_conj,
    qu_angle,
    qu_norm_std,
)
from ebsdtorch.s2_and_so3.sampling import so3_cubochoric_grid
from ebsdtorch.s2_and_so3.laue_generators import get_laue_mult, laue_elements
from ebsdtorch.s2_and_so3.sphere import xyz_to_theta_phi
from ebsdtorch.s2_and_so3.laue_fz_s2 import s2_to_fz_laue


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

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

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
def ori_in_fz_laue_brute(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given unit quaternions with positive real part are in the
    orientation fundamental zone of the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)

    Raises:
        ValueError: if the laue_id is not supported

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
def ori_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given unit quaternions with positive real part are in the
    orientation fundamental zone of the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)

    Raises:
        ValueError: if the laue_id is not supported

    """
    # all of the bound equality checks have to be inclusive to
    # match the behavior of the brute force method
    if laue_id == 11:
        # O: cubic high
        # max(abs(x,y,z)) < R2M1*abs(w) and sum(abs(x,y,z)) < abs(w)
        xyz_abs = torch.abs(quats[..., 1:])
        return (
            torch.max(xyz_abs, dim=-1).values <= (quats[..., 0] * (2**0.5 - 1))
        ) & (torch.sum(xyz_abs, dim=-1) <= quats[..., 0])
    elif laue_id == 10:
        # T: cubic low
        # sum(abs(x,y,z)) < abs(w)
        return torch.sum(torch.abs(quats[..., 1:]), dim=-1) <= quats[..., 0]
    elif laue_id == 9:
        # D6: hexagonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN75 * n then rot = m else rot = R3O2 * m + 0.5 * n
        # if abs(z) < TAN15 * abs(w) and rot < abs(w) then in FZ
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2 + 3**0.5) * n, m, (3**0.5 / 2) * m + 0.5 * n)
        return (torch.abs(quats[..., 3]) <= (2 - 3**0.5) * quats[..., 0]) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 8:
        # C6: hexagonal low
        # if abs(z) < TAN15 * abs(w)
        return torch.abs(quats[..., 3]) <= ((2 - 3**0.5) * quats[..., 0])
    elif laue_id == 7:
        # D3: trigonal high
        # if abs(x) > abs(y) * R3:
        #   rot = abs(x)
        # else:
        # rot = R3O2 * abs(y) + 0.5 * abs(x)
        # FZ: if abs(z) < TAN30 * abs(w) and rot < abs(w)
        rot = torch.where(
            torch.abs(quats[..., 1]) >= torch.abs(quats[..., 2]) * (3**0.5),
            torch.abs(quats[..., 1]),
            (3**0.5 / 2) * torch.abs(quats[..., 2]) + 0.5 * torch.abs(quats[..., 1]),
        )
        return (torch.abs(quats[..., 3]) <= ((1.0 / 3**0.5) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 6:
        # C3: trigonal low
        # FZ: abs(z) < TAN30 * abs(w)
        return torch.abs(quats[..., 3]) <= (1.0 / 3**0.5) * quats[..., 0]
    elif laue_id == 5:
        # D4: tetragonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN67_5 * n then rot = m else rot = R2O2 * m + R2O2 * n
        # FZ: abs(z) < TAN22_5 * abs(w) and rot < abs(w)
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2**0.5 + 1) * n, m, (1 / 2**0.5) * m + (1 / 2**0.5) * n)
        return (torch.abs(quats[..., 3]) <= ((2**0.5 - 1) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 4:
        # C4: tetragonal low
        # FZ: abs(z) < TAN22_5 * abs(w)
        return torch.abs(quats[..., 3]) <= (2**0.5 - 1) * quats[..., 0]
    elif laue_id == 3:
        # D2: orthorhombic
        # FZ: max(abs(x,y,z)) < abs(w)
        return torch.max(torch.abs(quats[..., 1:]), dim=-1).values <= quats[..., 0]
    elif laue_id == 2:
        # C2: monoclinic
        # FZ: abs(z) < abs(w)
        return torch.abs(quats[..., 3]) <= quats[..., 0]
    elif laue_id == 1:
        # C1: triclinic
        return torch.full(quats.shape[:-1], True, dtype=torch.bool, device=quats.device)
    else:
        raise ValueError(f"Laue group {laue_id} is not supported")


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
    angular_resolution_deg: float,
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
        2 * (131.97049) / (angular_resolution_deg - 0.03732) + 1
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
def ori_color_laue(
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
