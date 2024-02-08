import torch
from ebsdtorch.s2_and_so3.orientations import cu2qu, standardize_quaternion
from ebsdtorch.s2_and_so3.sphere import theta_phi_to_xyz


@torch.jit.script
def s2_fibonacci_lattice(
    n: int,
    device: torch.device,
    mode: str = "avg",
) -> torch.Tensor:
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
def halton_sequence(
    size: int,
    base: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate the Halton sequence of given size and base using torch.

    Args:
        size (int): the number of points to generate
        base (int): the base to use for the Halton sequence
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the Halton sequence of size `size` and base `base`

    """

    digits = (
        int(
            torch.log10(torch.tensor(float(size * base), device=device))
            / torch.log10(torch.tensor(float(base), device=device))
        )
        + 1
    )
    indices = torch.arange(1, size + 1, device=device).reshape(-1, 1)
    digits_array = base ** torch.arange(digits, device=device, dtype=torch.float32)
    divisors = base ** (torch.arange(digits, device=device, dtype=torch.float32) + 1)
    coefficients = (indices // digits_array) % base
    radical_inverse_values = coefficients.mv(1.0 / divisors)
    return radical_inverse_values


@torch.jit.script
def halton_sequence_id(
    id_start: int,
    id_stop: int,
    base: int,
    device: torch.device,
) -> torch.Tensor:
    """

    Generate the Halton sequence of given size and base using torch.

    Args:
        id_start (int): the starting index
        id_stop (int): the stopping index
        base (int): the base to use for the Halton sequence
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the Halton sequence of size `size` and base `base`

    """

    if id_start > 1e15 or id_stop > 1e15:
        raise ValueError("id_start and id_stop must be less than 999999999999999")
    size = id_stop - id_start
    digits = (
        int(
            torch.log10(torch.tensor(float(size * base), device=device))
            / torch.log10(torch.tensor(float(base), device=device))
        )
        + 1
    )
    indices = torch.arange(id_start, id_stop + 1, device=device).reshape(-1, 1)
    digits_array = base ** torch.arange(digits, device=device, dtype=torch.float64)
    divisors = base ** (torch.arange(digits, device=device, dtype=torch.float64) + 1)
    coefficients = (indices // digits_array) % base
    radical_inverse_values = coefficients.mv(1.0 / divisors)
    return radical_inverse_values


@torch.jit.script
def halton_id_3d(
    id_start: int,
    id_stop: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a 3D Halton sequence in the unit cube.

    Args:
        id_start (int): the starting index
        id_stop (int): the stopping index
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D Halton sequence points (n, 3)

    """
    return torch.stack(
        [
            halton_sequence_id(id_start, id_stop, 2, device),
            halton_sequence_id(id_start, id_stop, 3, device),
            halton_sequence_id(id_start, id_stop, 5, device),
        ],
        dim=1,
    )


@torch.jit.script
def so3_halton_cubochoric(
    id_start: int,
    id_stop: int,
    device: torch.device,
):
    """

    Generate a 3D Halton sequence in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        id_start (int): the starting index
        id_stop (int): the stopping index
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D Halton sequence points in cubochoric coordinates (n, 3)


    """
    cu = halton_id_3d(id_start, id_stop, device=device) * torch.pi ** (
        2 / 3
    ) - 0.5 * torch.pi ** (2 / 3)
    qu = cu2qu(cu)
    qu = standardize_quaternion(qu / torch.norm(qu, dim=-1, keepdim=True))
    return qu


@torch.jit.script
def so3_cubochoric_rand(n: int, device: torch.device) -> torch.Tensor:
    """
    Generate a 3D random sampling in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        n (int): the number of orientations to sample
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D random sampling in cubochoric coordinates (n, 3)

    """
    box_sampling = torch.rand(n, 3, device=device) * torch.pi ** (
        2.0 / 3.0
    ) - 0.5 * torch.pi ** (2.0 / 3.0)
    qu = cu2qu(box_sampling)
    qu = standardize_quaternion(qu / torch.norm(qu, dim=-1, keepdim=True))
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
    qu = standardize_quaternion(qu)
    return qu
