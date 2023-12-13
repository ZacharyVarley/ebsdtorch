from typing import Tuple
import torch
from torch import Tensor

@torch.jit.script
def x_norm(x: Tensor, 
           i_x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    x is a 64 bit floating point tensor of shape (..., ) and i_x is a 32 bit integer tensor of shape (..., )

    Returns normalized X-number as the mantissa and the exponent.
    
    Done according to Toshio Fukushima's definition F90 code:

    Table 7 Fortran subroutine to normalize a weakly normalized X-number
    subroutine xnorm(x,ix)
    integer ix,IND
    real*8 x,w,BIG,BIGI,BIGS,BIGSI
    parameter (IND=960,BIG=2.d0**IND,BIGI=2.d0**(-IND))
    parameter (BIGS=2.d0**(IND/2),BIGSI=2.d0**(-IND/2))
    w=abs(x)
    if(w.ge.BIGS) then
        x=x*BIGI; ix=ix+1
    elseif(w.lt.BIGSI) then
        x=x*BIG; ix=ix-1
    endif
    return; end

    The constants related with the normalization bounds, B^(1/2) and B^(-1/2),
    are termed as BIGS and BIGSI, respectively 

    """

    # Constants
    IND = torch.tensor(960, dtype=torch.int32)
    BIG = torch.pow(2.0, IND.double())
    BIGI = torch.pow(2.0, -IND.double())
    BIGS = torch.pow(2.0, IND.double()/2.0)
    BIGSI = torch.pow(2.0, -IND.double()/2.0)

    w = torch.abs(x)
    x = torch.where(w >= BIGS, x * BIGI, x)
    x = torch.where(w < BIGSI, x * BIG, x)
    i_x = torch.where(w >= BIGS, i_x + 1, i_x)
    i_x = torch.where(w < BIGSI, i_x - 1, i_x)

    return x, i_x


@torch.jit.script
def x2f(x: torch.Tensor,
        ix: torch.Tensor,
) -> torch.Tensor:
    """

    Return a 64 bit floating point number from a 64 bit floating point tensor and a 32 bit integer tensor.

    Done according to Toshio Fukushima's definition F90 code:

    Table 6 Fortran function to translate an X-number into an F-number
    real*8 function x2f(x,ix)
    integer ix,IND
    real*8 x,BIG,BIGI
    parameter (IND=960,BIG=2.d0**IND,BIGI=2.d0**(-IND))
    if(ix.eq.0) then
        x2f=x
    elseif(ix.lt.0) then
        x2f=x*BIGI
    else
        x2f=x*BIG
    endif
    return; end

    The radix B and its reciprocal B^(-1) are named BIG and BIGI in the
    program. An integer constant IND is the index of power of 2 to define
    the radix
    
    """

    # Constants
    IND = torch.tensor(960, dtype=torch.int32)
    BIG = 2.0**IND
    BIGI = 2.0**(-IND)

    return torch.where(ix == 0, x, torch.where(ix < 0, x * BIGI, x * BIG))


@torch.jit.script
def xlsum2(f: Tensor,
           g: Tensor,
           x: Tensor,
           ix: Tensor,
           y: Tensor,
           iy: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Return the linear combination of two X-numbers, X, and Y, with prefactors f and g.

    Done according to Toshio Fukushima's definition F90 code:

    Table 8 Fortran subroutine to compute the two-term linear sum of X-
    numbers with F-number coefficients
    subroutine xlsum2(f,g,x,ix,y,iy,z,iz)
    integer ix,iy,iz,IND,id
    real*8 f,g,x,y,z,BIGI
    parameter (IND=960,BIGI=2.d0**(-IND))
    id=ix-iy
    if(id.eq.0) then
        z=f*x+g*y; iz=ix
    elseif(id.eq.1) then
        z=f*x+g*(y*BIGI); iz=ix
    elseif(id.eq.-1) then
        z=g*y+f*(x*BIGI); iz=iy
    elseif(id.gt.1) then
        z=f*x; iz=ix
    else
        z=g*y; iz=iy
    endif
        call xnorm(z,iz)
    return; end

    """

    # Constants
    IND = torch.tensor(960, dtype=torch.int32)
    BIGI = 2.0**(-IND)

    id = ix - iy
    z = torch.where(id == 0, f * x + g * y, torch.where(id == 1, f * x + g * y * BIGI, torch.where(id == -1, g * y + f * x * BIGI, torch.where(id > 1, f * x, g * y))))
    iz = torch.where(id == 0, ix, torch.where(id == 1, ix, torch.where(id == -1, iy, torch.where(id > 1, ix, iy))))
    z, iz = x_norm(z, iz)

    return z, iz

# # test out the functions
# x = torch.tensor([2.0,], dtype=torch.float64)
# print(f"x: {x}")
# ix = torch.tensor([0,], dtype=torch.int32)
# x, ix = x_norm(x, ix)
# print(f"x-number mantissas: {x}")
# print(f"x-number exponents: {ix}")
# x_reconstructed = x2f(x, ix)
# print(f"x reconstructed: {x_reconstructed}")

# f = torch.tensor(1.0, dtype=torch.float64)
# g = torch.tensor(1.0, dtype=torch.float64)

# for index in range(1, 2000):
#     x, ix = xlsum2(f, g, x, ix, x, ix)
#     if index % 100 == 99:
#         x_reconstructed = x2f(x, ix)
#         print(f"power: {index+1} x: {x} ix: {ix} x reconstructed: {x_reconstructed}")