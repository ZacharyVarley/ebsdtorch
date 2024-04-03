"""

This module provides functions for working with the fundamental zone of the
2-sphere under the symmetry of a given Laue group. 

This is adopted from the code following line 1800 in the file
mod_IPFsupport.f90:

https://github.com/EMsoft-org/EMsoftOO/blob/develop/Source/EMsoftOOLib/mod_IPFsupport.f90

"""

import torch
from torch import Tensor

from ebsdtorch.s2_and_so3.laue_generators import laue_elements, get_laue_mult
from ebsdtorch.s2_and_so3.quaternions import qu_apply
from ebsdtorch.s2_and_so3.sphere import inv_rosca_lambert
from ebsdtorch.s2_and_so3.sampling import s2_fibonacci


@torch.jit.script
def s2_in_fz_laue(points: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given 3D points are in the fundamental zone of the given
    Laue group. This computes the sphere fundamental zone, not the misorientation
    nor orientation fundamental zone. The 2-spherical fundamental zone is also
    called the fundamental sector.

    Args:
        points: points to check if in fundamental zone of shape (..., 3)

    Returns:
        boolean tensor of shape (..., ) indicating if the points are in the
        fundamental zone

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
    s2_samples = s2_fibonacci(target_n_samples * laue_mult, device=device)

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
