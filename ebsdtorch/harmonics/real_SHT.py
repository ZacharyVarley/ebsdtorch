"""

Real Valued Spherical Harmonic Transformations



Downstream SO(3) FFT uses ZYZ angles, so use YZ polar/azimuthal not XZ. This
file implements the spherical harmonic transform for real-valued functions on
the sphere. The rotation of spherical harmonics is achieved by the Wigner
D-matrix. There are many ways to compute the Wigner D-matrix via recursion, one
begins with the D_{m'm}^{j} with m' = 0 and m = 0, and then recurses to higher
order azimuthal numbers, but a faster recursion algorithm is used in the
spherical package, and decrements the azimuthal number down from the chosen
maximum. TS2Kit's implementation uses naive Python loops to find Wigner d and
Wigner D. I am joining efficient recursion with TS2Kit's spherical harmonic
transform implementation:

https://github.com/twmitchel/TS2Kit/blob/main/TS2Kit.pdf

And spherical package:

https://github.com/moble/spherical


Side note thoughts on EMsoft's EMSphInx:
https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/square_sht.hpp
EMSphInx computes quadrature weights for latitude rings that make pixels
relatively equal area. I am not doing that because the quadrature weight
calculation is O(B^4) as far as I understand. Might add this later.

Side note thoughts on Euler angles and SO(3) FFT:

The author of spherical package, Michael Boyle, absolutely detests Euler angles,
but as far as I understand, the SO(3) FFT uses ZYZ angles because it must. Cross
correlation over SO(3) is simply an inverse 3D FFT with real valued (in ZYZ
case) Wignder D-matrix coefficients prefactors. This is because the volume is
periodic about each axis, which is coincident with choosing to use Euler angles
from the beginning... I think. No other orientation representations have 1D
rings along a given dimension.

"""

import torch
from torch import Tensor


@torch.jit.script
def grid_DriscollHealy(bandlimit: int) -> Tensor:
    """
    Generate Driscoll-Healy grid for spherical harmonics.

    Args: bandlimit: Bandlimit of the grid.

    Returns: (Tensor): shape (2*bandlimit, bandlimit) with coords [theta, phi].

    """
    k = torch.arange(0, 2 * bandlimit).double()
    theta = 2 * torch.pi * k / (2 * bandlimit)
    phi = torch.pi * (2 * k + 1) / (4 * bandlimit)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack([theta, phi], dim=-1)
