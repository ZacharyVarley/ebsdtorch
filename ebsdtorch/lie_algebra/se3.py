"""

This file contains an implementation of the Lie algebra or tangent space of
rigid motion in 3D Euclidean space from the following technical report:

Blanco-Claraco, J.L., 2021. A tutorial on $\mathbf {SE}(3) $ transformation
parameterizations and on-manifold optimization. arXiv preprint arXiv:2103.15980.

I need to explicitly implement the Jacobian of SE(3) and SO(3) maps as well, as
it is known to be numerically unstable when just relying on the PyTorch autograd
operating on the naive implementation:

Teed, Zachary, and Jia Deng. "Tangent space backpropagation for 3d
transformation groups." In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 10338-10347. 2021.

The Lie algebra of the special Euclidean group in 3D, $\mathbf {se}(3)$, is
barely different than the individual translation and SO(3) rotation Lie
algebras. They mix according to the Jacobian of SO3... see section 10.6.9 of: 

Chirikjian, G. S. (2012). Algebraic and Geometric Coding Theory. Stochastic
Models, Information Theory, and Lie Groups, Volume 2: Analytic Methods and
Modern Applications, 313-336.

I am mostly cleaning up the PyTorch3D (from Meta / Facebook) implementation:

https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/se3.py

That said, there are some newer approaches: dual quaternions. Dual quaternions
can represent SE(3) and there has been recent efforts to craft a numerically
stable exponential and logarithm map for them:

https://dyalab.mines.edu/papers/dantam2018practical.pdf

My hunch is that in the same way that the folks found that keeping the map in
quaternion form was more numerically stable for SO(3):

https://github.com/nurlanov-zh/so3_log_map

... it might be the case that doing everything with dual quaternions is more
stable for SE(3). I will have to look into this later.

#########################################################################
###############################  WARNING  ###############################
#########################################################################

Small angles (approx. < 0.001 radians) ... if you need them either use float64
quatnernions instead of float32, or float32 / float64 rotation matrices. If you
know that you will be far away from the identity rotation, or you don't care
about small angles below 0.5 degree, you can use quaternions and float32.

If your angles are smaller than about 0.001 rad, you lose most of the effective
significant digits when storing the real part of the quaternion in float32. This
is one reason why the Lie algebra is often implemented with rotation matrices,
even though they are not as efficient to work with when using them to rotate
points in 3D. Another reason could be that the subsequent transformations of
points (in the case of PyTorch3D) are not necessarily rotations overall.

For basics on the Lie algebra of SO(3) as an example:

https://arxiv.org/pdf/1606.05285

Bloesch, M., Sommer, H., Laidlow, T., Burri, M., Nuetzi, G., Fankhauser, P.,
Bellicoso, D., Gehring, C., Leutenegger, S., Hutter, M. and Siegwart, R., 2016.
A primer on the differential calculus of 3d orientations. arXiv preprint
arXiv:1606.05285.

"""

from typing import Tuple
import torch
from torch import Tensor
from ebsdtorch.s2_and_so3.orientations import ax2qu, qu2ax


@torch.jit.script
def skew(omega: Tensor) -> Tensor:
    """
    Compute the skew-symmetric matrix of a vector.

    Args:
        omega: torch tensor of shape (..., 3) containing the scaled axis of rotation

    Returns:
        torch tensor of shape (..., 3, 3) containing the skew-symmetric matrix

    """

    data_shape = omega.shape[:-1]
    data_n = int(torch.prod(torch.tensor(data_shape)))
    out = torch.zeros((data_n, 3, 3), dtype=omega.dtype, device=omega.device)
    out[..., 0, 1] = -omega[..., 2].view(-1)
    out[..., 0, 2] = omega[..., 1].view(-1)
    out[..., 1, 2] = -omega[..., 0].view(-1)
    out[..., 1, 0] = omega[..., 2].view(-1)
    out[..., 2, 0] = -omega[..., 1].view(-1)
    out[..., 2, 1] = omega[..., 0].view(-1)
    return out.view(data_shape + (3, 3))


@torch.jit.script
def w_exp(omega: Tensor) -> Tensor:
    """
    Compute the matrix exponential of a skew-symmetric matrix. This is
    Rodrigues' formula. Angles below 0.01 radians use Taylor expansion.

    Args:
        omega: torch tensor of shape (..., 3) containing the omega

    Returns:
        torch tensor of shape (..., 3, 3) containing the skew matrix exponential

    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)
    skew_mat = skew(omega)
    skew_sq = torch.matmul(skew_mat, skew_mat)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.001)[..., 0, 0]

    # This prefactor is only used for the calculation of exp(skew)
    # sin(theta) / theta
    # expression: 1 - theta^2 / 6 + theta^4 / 120 - theta^6 / 5040 ...
    prefactor1 = 1 - theta[~stable] ** 2 / 6 + theta[~stable] ** 4 / 120

    # This prefactor is shared between calculations of exp(skew) and v
    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    prefactor2 = 1 / 2 - theta[~stable] ** 2 / 24 + theta[~stable] ** 4 / 720

    skew_exp = torch.empty_like(skew_mat)
    skew_exp[stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + (torch.sin(theta[stable]) / theta[stable]) * skew_mat[stable]
        + (1 - torch.cos(theta[stable])) / theta[stable] ** 2 * skew_sq[stable]
    )
    skew_exp[~stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + prefactor1 * skew_mat[~stable]
        + prefactor2 * skew_sq[~stable]
    )
    return skew_exp


@torch.jit.script
def w_exp_vmat(omega: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute both the matrix exponential of a skew-symmetric matrix
    and the V matrix. Angles below 0.01 radians use Taylor expansion.

    Args:
        omega: torch tensor of shape (..., 3) containing the omega

    Returns:
        torch tensor shape (..., 3, 3) of skew matrix exponential
        torch tensor shape (..., 3, 3) of v matrices

    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)
    skew_mat = skew(omega)
    skew_sq = torch.matmul(skew_mat, skew_mat)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.001)[..., 0, 0]

    # This prefactor is only used for the calculation of exp(skew)
    # sin(theta) / theta
    # expression: 1 - theta^2 / 6 + theta^4 / 120 - theta^6 / 5040 ...
    prefactor1 = 1 - theta[~stable] ** 2 / 6

    # This prefactor is shared between calculations of exp(skew) and v
    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    prefactor2 = 1 / 2 - theta[~stable] ** 2 / 24

    # This prefactor is only used for the calculation of v
    # (theta - sin(theta)) / theta^3
    # expression: 1/6 - theta^2 / 120 + theta^4 / 5040 - theta^6 / 362880 ...
    prefactor3 = 1 / 6 - theta[~stable] ** 2 / 120

    skew_exp = torch.empty_like(skew_mat)
    skew_exp[stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + (torch.sin(theta[stable]) / theta[stable]) * skew_mat[stable]
        + (1 - torch.cos(theta[stable])) / theta[stable] ** 2 * skew_sq[stable]
    )
    skew_exp[~stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + prefactor1 * skew_mat[~stable]
        + prefactor2 * skew_sq[~stable]
    )
    # skew_exp = torch.matrix_exp(skew_mat)

    v = torch.empty_like(skew_mat)
    v[stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + (1 - torch.cos(theta[stable])) / theta[stable] ** 2 * skew_mat[stable]
        + ((theta[stable] - torch.sin(theta[stable])) / theta[stable] ** 3)
        * skew_sq[stable]
    )
    v[~stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + prefactor2 * skew_mat[~stable]
        + prefactor3 * skew_sq[~stable]
    )

    return skew_exp, v


@torch.jit.script
def w_to_v(omega: Tensor) -> Tensor:
    """
    Compute the V matrix for omega.

    Args:
        omega: torch tensor of shape (..., 3) containing the omega vectors

    Returns:
        torch tensor of shape (..., 3, 3) containing the v matrix

    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)
    skew_omega = skew(omega)
    skew_omega2 = torch.matmul(skew_omega, skew_omega)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.05)[..., 0, 0]

    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    factor1 = 1 / 2 - theta[~stable] ** 2 / 24 + theta[~stable] ** 4 / 720

    # (theta - sin(theta)) / theta^3
    # expression: 1/6 - theta^2 / 120 + theta^4 / 5040 - theta^6 / 362880 ...
    factor2 = 1 / 6 - theta[~stable] ** 2 / 120 + theta[~stable] ** 4 / 5040

    v = torch.empty_like(skew_omega)

    v[stable] = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + ((1 - torch.cos(theta[stable])) / theta[stable] ** 2) * skew_omega[stable]
        + ((theta[stable] - torch.sin(theta[stable])) / theta[stable] ** 3)
        * skew_omega2[stable]
    )
    v[~stable] = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + factor1 * skew_omega[~stable]
        + factor2 * skew_omega2[~stable]
    )
    return v


@torch.jit.script
def w_to_v_inv(omega: Tensor) -> Tensor:
    """
    Compute the inverse of the V matrix for an omega.

    Args:
        omega: torch tensor of shape (..., 3) containing omega vectors

    Returns:
        torch tensor of shape (..., 3, 3) containing the inverse of the v matrix

    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)
    skew_omega = skew(omega)
    skew_omega2 = torch.matmul(skew_omega, skew_omega)

    v_inv = torch.empty_like(skew_omega)
    stable = (theta > 0.05)[..., 0, 0]
    v_inv[stable] = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        - 0.5 * skew_omega[stable]
        + (
            1
            - (
                theta[stable]
                * torch.cos(theta[stable] / 2.0)
                / (2 * torch.sin(theta[stable] / 2.0))
            )
        )
        / theta[stable] ** 2
        * skew_omega2[stable]
    )
    # (1 - theta * cos(theta / 2) / (2 * sin(theta / 2))) / theta^2
    # expression: 1/12 + 1/720 * theta^2 + 1/30240 * theta^4
    factor_approx = 1 / 12 + theta[~stable] ** 2 / 720 + theta[~stable] ** 4 / 30240

    v_inv[~stable] = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        - 0.5 * skew_omega[~stable]
        + factor_approx * skew_omega2[~stable]
    )

    return v_inv


@torch.jit.script
def se3_exp_map_quat(vecs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the SE3 matrix from omega and tvec.

    Args:
        vec: torch tensor of shape (..., 6) containing omega and tvec

    Returns:
        torch tensor of shape (..., 4), (..., 3) containing quaternion and translation

    """
    data_shape = vecs.shape[:-1]

    omegas = vecs[..., :3]
    tvecs = vecs[..., 3:]

    # the angle is the norm of the 3D vector
    norm = torch.linalg.norm(omegas, dim=-1, keepdim=True)
    stable = norm > 0.05
    # use logsumexp trick
    theta = torch.where(
        stable,
        norm,
        torch.exp(
            torch.logsumexp(
                torch.log(torch.abs(omegas) + 1e-10) * 2.0, dim=-1, keepdim=True
            )
            / 2.0
        ),
    )

    axis_angle = torch.cat([omegas / theta, theta], dim=-1)
    quat = ax2qu(axis_angle)

    v = w_to_v(omegas)
    tvecs = torch.matmul(v, tvecs[..., None]).view(data_shape + (3,))

    return quat, tvecs


@torch.jit.script
def se3_log_map_quat(quats: Tensor, tvecs: Tensor) -> Tensor:
    """
    Compute the omega and tvec from an SE3 matrix.

    Args:
        se3: torch tensor of shape (..., ) containing the SE3 matrix

    Returns:
        torch tensor of shape (..., 6) containing the omega and tvec

    """
    data_shape = quats.shape[:-1]
    axis_angle = qu2ax(quats)

    omegas = axis_angle[..., :3] * axis_angle[..., 3:4]

    v_inv = w_to_v_inv(omegas)
    tvecs = torch.matmul(v_inv, tvecs[..., None]).view(data_shape + (3,))

    return torch.cat([omegas, tvecs], dim=-1)


@torch.jit.script
def se3_exp_map_om(vecs: Tensor) -> Tensor:
    """
    Compute the SE3 matrix from omega and tvec.

    Args:
        vec: torch tensor of shape (..., 6) containing the omega and tvec

    Returns:
        torch tensor of shape (..., 4, 4) containing the SE3 matrix

    """
    data_shape = vecs.shape[:-1]
    omegas = vecs[..., :3]
    tvecs = vecs[..., 3:]
    rexp, v = w_exp_vmat(omegas)
    se3 = torch.zeros(data_shape + (4, 4), dtype=vecs.dtype, device=vecs.device)
    se3[..., :3, :3] = rexp
    se3[..., :3, 3] = torch.matmul(v, tvecs[..., None]).view(data_shape + (3,))
    se3[..., 3, 3] = 1.0
    return se3


@torch.jit.script
def se3_exp_map_split(vecs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the SE3 matrix from omega and tvec.

    Args:
        vec: torch tensor of shape (..., 6) containing the omega and tvec

    Returns:
        torch tensor of shape (..., 3, 3) containing the rotation matrix
        torch tensor of shape (..., 3) containing the translation vector

    """
    data_shape = vecs.shape[:-1]
    omegas = vecs[..., :3]
    tau = vecs[..., 3:]
    rexp, v = w_exp_vmat(omegas)
    tvec = torch.matmul(v, tau[..., None]).view(data_shape + (3,))
    return rexp, tvec


@torch.jit.script
def se3_log_map_om(se3: Tensor) -> Tensor:
    """
    Compute omega and tvec from an SE3 matrix.

    Args:
        se3: torch tensor of shape (..., 4, 4) containing the SE3 matrix

    Returns:
        torch tensor of shape (..., 6) containing the omega and tvec

    """
    data_shape = se3.shape[:-2]

    # omega is (theta / 2 sin(theta)) * (R32 - R23, R13 - R31, R21 - R12)
    omegas = torch.zeros(data_shape + (3,), dtype=se3.dtype, device=se3.device)

    # find the trace of the rotation matrix portion
    rtrace = torch.diagonal(se3[..., :3, :3], dim1=-2, dim2=-1).sum(-1)
    # rtrace = torch.einsum("...ii", se3[..., :3, :3])

    # find the angles
    acos_arg = 0.5 * (rtrace - 1.0)
    acos_arg = torch.clamp(acos_arg, -1.0, 1.0)
    theta = torch.acos(acos_arg)

    # where the angle is small, treat (theta/sin(theta)) as 1
    stable = theta > 0.01
    omegas[..., 0] = se3[..., 2, 1] - se3[..., 1, 2]
    omegas[..., 1] = se3[..., 0, 2] - se3[..., 2, 0]
    omegas[..., 2] = se3[..., 1, 0] - se3[..., 0, 1]
    factor = torch.where(stable, 0.5 * theta / torch.sin(theta), 0.5)
    omegas = factor[:, None] * omegas

    v_inv = w_to_v_inv(omegas)
    tvecs = torch.matmul(v_inv, se3[..., :3, 3][..., None]).view(data_shape + (3,))

    return torch.cat([omegas, tvecs], dim=-1)


@torch.jit.script
def se3_log_map_R_tvec(R: Tensor, tvec: Tensor) -> Tensor:
    """
    Compute omega and tvec from an SE3 matrix.

    Args:
        R: torch tensor of shape (..., 3, 3) containing the rotation matrix
        tvec: torch tensor of shape (..., 3) containing the translation vector

    Returns:
        torch tensor of shape (..., 6) containing the omega and tvec

    """
    data_shape = R.shape[:-2]

    # omega is (theta / 2 sin(theta)) * (R32 - R23, R13 - R31, R21 - R12)
    omegas = torch.zeros(data_shape + (3,), dtype=R.dtype, device=R.device)

    # find the trace of the rotation matrix portion
    rtrace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    # rtrace = torch.einsum("...ii", se3[..., :3, :3])

    # find the angles
    acos_arg = 0.5 * (rtrace - 1.0)
    acos_arg = torch.clamp(acos_arg, -1.0, 1.0)
    theta = torch.acos(acos_arg)

    # where the angle is small, treat (theta/sin(theta)) as 1
    stable = theta > 0.0001
    omegas[..., 0] = R[..., 2, 1] - R[..., 1, 2]
    omegas[..., 1] = R[..., 0, 2] - R[..., 2, 0]
    omegas[..., 2] = R[..., 1, 0] - R[..., 0, 1]
    factor = torch.where(stable, 0.5 * theta / torch.sin(theta), 0.5)
    omegas = factor[:, None] * omegas

    v_inv = w_to_v_inv(omegas)
    tau = torch.matmul(v_inv, tvec[..., None]).view(data_shape + (3,))

    return torch.cat([omegas, tau], dim=-1)
