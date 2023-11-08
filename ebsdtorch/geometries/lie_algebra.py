import torch
from torch import Tensor

'''
This class implements a Lie algebra parametrization of SE(3)

This is the old way of doing it with homogeneous coordinates:
https://www.ethaneade.com/lie.pdf for a good intro reference and optimization
https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf

That said, there are some newer approaches: dual quaternions.
Dual quaternions can represent SE(3) and there has been recent efforts
to craft a numerically stable exponential and logarithm map for them:
https://dyalab.mines.edu/papers/dantam2018practical.pdf


'''


@torch.jit.script
def exp_map(v: Tensor) -> Tensor:
    r"""Exponential map for the Lie algebra of SE(3).

    Parameters
    ----------
    v : torch.Tensor
        A vector in the Lie algebra se(3)

    Returns
    -------
    torch.Tensor
        The corresponding element of SE(3)

    """



class SE3_Lie(torch.nn.Module):
    def __init__(self,
                 matrix: Tensor = None,
                 ):
        super().__init__()
        

    