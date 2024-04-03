"""

Uniform sampling of the 2-sphere and SO(3) using various methods.

"""

import torch
from torch import Tensor
from ebsdtorch.s2_and_so3.orientations import cu2qu
from ebsdtorch.s2_and_so3.quaternions import qu_std
from ebsdtorch.s2_and_so3.sphere import theta_phi_to_xyz


@torch.jit.script
def s2_fibonacci(
    n: int,
    device: torch.device,
    mode: str = "avg",
) -> Tensor:
    """
    Sample n points on the unit sphere using the Fibonacci spiral method.

    Args:
        n (int): the number of points to sample
        device (torch.device): the device to use
        mode (str): the mode to use for the Fibonacci lattice. "avg" will
            optimize for average spacing, while "max" will optimize for
            maximum spacing. Default is "avg".

    References:
    https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/


    """
    # initialize the golden ratio
    phi = (1 + 5**0.5) / 2
    # initialize the epsilon parameter
    if mode == "avg":
        epsilon = 0.36
    elif mode == "max":
        if n >= 600000:
            epsilon = 214.0
        elif n >= 400000:
            epsilon = 75.0
        elif n >= 11000:
            epsilon = 27.0
        elif n >= 890:
            epsilon = 10.0
        elif n >= 177:
            epsilon = 3.33
        elif n >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
    else:
        raise ValueError('mode must be either "avg" or "max"')
    # generate the points (they must be doubles for large numbers of points)
    indices = torch.arange(n, dtype=torch.float64, device=device)
    theta = 2 * torch.pi * indices / phi
    phi = torch.acos(1 - 2 * (indices + epsilon) / (n - 1 + 2 * epsilon))
    points = theta_phi_to_xyz(theta, phi)
    return points


@torch.jit.script
def so3_fibonacci(
    n: int,
    device: torch.device,
) -> Tensor:
    """
    Super Fibonacci sampling of orientations.

    See the following paper for more information:

    Alexa, Marc. "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO
    (3)." In Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition, pp. 8291-8300. 2022.

    Args:
        n (int): the number of orientations to sample
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D super Fibonacci sampling quaternions (n, 4)

    """

    PHI = 2.0**0.5
    # positive real solution to PSI^4 = PSI + 4
    PSI = 1.533751168755204288118041

    indices = torch.arange(n, device=device, dtype=torch.float64)
    s = indices + 0.5
    t = s / n
    d = 2 * torch.pi * s
    r = torch.sqrt(t)
    R = torch.sqrt(1 - t)
    alpha = d / PHI
    beta = d / PSI
    qu = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        dim=1,
    )

    return qu


@torch.jit.script
def so3_cu_rand(n: int, device: torch.device) -> Tensor:
    """
    3D random sampling in cubochoric coordinates lifted to SO(3) as quaternions.

    Args:
        n (int): the number of orientations to sample device
        device (torch.device): the device to use

    Returns:
        torch.Tensor: Quaternions of shape (n, 4) in form (w, x, y, z)

    """
    box_sampling = torch.rand(n, 3, device=device) * torch.pi ** (
        2.0 / 3.0
    ) - 0.5 * torch.pi ** (2.0 / 3.0)
    qu = cu2qu(box_sampling)
    qu = qu_std(qu / torch.norm(qu, dim=-1, keepdim=True))
    return qu


@torch.jit.script
def so3_cubochoric_grid(edge_length: int, device: torch.device):
    """
    Generate a 3D grid sampling in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        edge_length (int): the number of points along each axis of the cube
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D grid sampling in cubochoric coordinates (n, 3)

    """
    cu = torch.linspace(
        -0.5 * torch.pi ** (2 / 3),
        0.5 * torch.pi ** (2 / 3),
        edge_length,
        device=device,
    )
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing="ij"), dim=-1).reshape(-1, 3)
    qu = cu2qu(cu)
    qu = qu_std(qu)
    return qu


def so3_uniform_quat(n: int, device: torch.device) -> Tensor:
    """
    Generate uniformly distributed elements of SO(3) as quaternions. This
    routine includes both quaternion hemispheres and will return quaternions
    with negative real part.

    Args:
        n (int): the number of orientations to sample
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D random sampling in cubochoric coordinates (n, 4)


    Notes:

    This function is based on the following work of Ken Shoemake:

    Shoemake, Ken. "Uniform random rotations." Graphics Gems III (IBM Version).
    Morgan Kaufmann, 1992. 124-132.

    """

    # h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))

    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    w = torch.rand(n, device=device)

    h = torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v),
            torch.sqrt(u) * torch.sin(2 * torch.pi * w),
            torch.sqrt(u) * torch.cos(2 * torch.pi * w),
        ],
        dim=1,
    )

    return h
