import torch

from ebsdtorch.laue.orientations import cu2ho, ho2qu, standardize_quaternion


@torch.jit.script
def s2_fibonacci_lattice(n: int, mode: str = "avg") -> torch.Tensor:
    """
    Sample n points on the unit sphere using the Fibonacci spiral method.
    :param n: number of points to sample
    :return: torch tensor of shape (n, 3) containing the points

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
    # generate the points
    indices = torch.arange(n, dtype=torch.float64)
    theta = 2 * torch.pi * indices / phi
    phi = torch.acos(1 - 2 * (indices + epsilon) / (n - 1 + 2 * epsilon))
    points = torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=1,
    )
    return points.float()


@torch.jit.script
def halton_sequence(size: int, base: int, device:torch.device) -> torch.Tensor:
    """Generate the Halton sequence of given size and base using torch."""
    digits = int(torch.log10(torch.tensor(float(size * base), device=device)) / 
                 torch.log10(torch.tensor(float(base), device=device))) + 1
    indices = torch.arange(1, size + 1, device=device).reshape(-1, 1)
    digits_array = base ** torch.arange(digits, device=device, dtype=torch.float32)
    divisors = base ** (torch.arange(digits, device=device, dtype=torch.float32) + 1)
    coefficients = (indices // digits_array) % base
    radical_inverse_values = coefficients.mv(1.0 / divisors)
    return radical_inverse_values


@torch.jit.script
def halton_sequence_id(id_start: int, id_stop: int, base: int, device:torch.device) -> torch.Tensor:
    """Generate the Halton sequence of given size and base using torch."""
    if id_start > 1e15 or id_stop > 1e15:
        raise ValueError("id_start and id_stop must be less than 999999999999999")
    size = id_stop - id_start
    digits = int(torch.log10(torch.tensor(float(size * base), device=device)) / 
                    torch.log10(torch.tensor(float(base), device=device))) + 1
    indices = torch.arange(id_start, id_stop + 1, device=device).reshape(-1, 1)
    digits_array = base ** torch.arange(digits, device=device, dtype=torch.float64)
    divisors = base ** (torch.arange(digits, device=device, dtype=torch.float64) + 1)
    coefficients = (indices // digits_array) % base
    radical_inverse_values = coefficients.mv(1.0 / divisors)
    return radical_inverse_values


@torch.jit.script
def halton_id_3d(id_start: int, id_stop: int, device: torch.device) -> torch.Tensor:
    """Generate a 3D Halton sequence given the start and stop indices """
    return torch.stack([halton_sequence_id(id_start, id_stop, 2, device),
                        halton_sequence_id(id_start, id_stop, 3, device),
                        halton_sequence_id(id_start, id_stop, 5, device)], dim=1)


@torch.jit.script
def so3_halton_cubochoric(id_start: int, id_stop: int, device: torch.device):
    """
    Generate a 3D Halton sequence in the cubochoric mapping of SO(3). Orientations 
    are returned as unit quaternions with positive scalar part (w, x, y, z)
    """
    cu = halton_id_3d(id_start, id_stop, device=device) * torch.pi ** (
        2 / 3
    ) - 0.5 * torch.pi ** (2 / 3)
    ho = cu2ho(cu)
    qu = ho2qu(ho)
    qu = standardize_quaternion(qu / torch.norm(qu, dim=-1, keepdim=True))
    return qu


@torch.jit.script
def so3_cubochoric_grid(
    edge_length: int, 
    device: torch.device):
    """
    Generate a 3D grid in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)
    """
    cu = torch.linspace(-0.5 * torch.pi ** (2 / 3), 0.5 * torch.pi ** (2 / 3), edge_length, device=device)
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing='ij'), dim=-1).reshape(-1, 3)
    ho = cu2ho(cu)
    qu = ho2qu(ho)
    qu = standardize_quaternion(qu / torch.norm(qu, dim=-1, keepdim=True))
    return qu