from typing import Tuple
import torch
from torch import Tensor
import numpy as np


"""
    See the following publications for an explanation of the recursive algorithm:

    "Numerical computation of spherical harmonics of arbitrary degree 
    and order by extending exponent of floating point numbers"

    https://doi.org/10.1007/s00190-011-0519-2

    "Numerical computation of Wigner's d-function of arbitrary high 
    degree and orders by extending exponent of floating point numbers"

    http://dx.doi.org/10.13140/RG.2.2.31922.20160 
    https://www.researchgate.net/publication/309652602

    Both by Toshio Fukushima.

    ---------------------------------------------
    

    Using equations (10-16) from:
    "Numerical computation of Wigner's d-function of arbitrary high 
    degree and orders by extending exponent of floating point numbers"

    d^j_km = a_jkm * d^(j-1) - b_jkm * d^(j-2)

    Note the following symmetry relations of the Wigner d-function:

    d^j_k_m(-BETA) = d^j_m_k(BETA)                       # Negating BETA is equivalent to swapping k and m
    d^j_-k_-m = (-1)^(k-m) d^j_km                        # Negating k and m yields -1 to (k-m) power prefactor
    d^j_k_-m(BETA) = (-1)^(j + k + 2m) d^j_km(π - BETA)  # Negating m yields -1 to (j + k + 2m) power prefactor and angle supplement
    d^j_-k_m(BETA) = (-1)^(j + 2k + 3m) d^j_km(π - BETA) # Negating k yields -1 to (j + 2k + 3m) power prefactor and angle supplement
    d^j_m_k = (-1)^(k-m) d^j_km                          # Swapping k and m yields -1 to (k-m) power prefactor

    This means we can just calculate the non-negative values of k and m and then fill in the rest using the symmetry relations.

    where

    a_jkm = (4*j - 2) * u_jkm * w_jkm
    b_jkm = v_jkm * w_jkm

    where 
    u_jkm = [ 2j *(2j - 2) - (2k) (2m) ] - 2j (2j - 2) * 2 * sin^2(BETA / 2)    IF 0 < BETA < PI/2
    u_jkm = -1.0 * (2k) (2m)                                                    IF BETA = PI/2
    u_jkm = 2j (2j - 2) cos(BETA) - (2k) (2m)                                   IF PI/2 < BETA < PI

    v_jkm = 2j * SQRT [ (2j +2k - 2) (2j - 2k - 2) (2j +2m - 2) (2j - 2m - 2) ]

    w_jkm = 1 / [ (2j - 2) * SQRT [ (2j + 2k) (2j - 2k) (2j + 2m) (2j - 2m) ] ]

    Returns:
        d_lkm: Wigner d-function

    
    For recurrence relation, we require seed values d_kkm and d_(k + 1)km

    There are taken from equations (17-27) of the same paper...

    d_kkm = c_(k + m) * s_(k - m) * e_(km) and d_(k + 1)km = a_km * d_kkm

    where

    c_n = cos(BETA / 2)^n
    c_0 = 1 AND c_1 = cos(BETA / 2) AND c_n = c_(n - 1) * c_1

    s_n = sin(BETA / 2)^n
    s_0 = 1 AND s_1 = sin(BETA / 2) AND s_n = s_(n - 1) * s_1

    e_(km) = SQRT[ (2k)! / ((k+m)!(k-m)!) ] where m <= k !!! need recursive alternative
    e_mm = 1 AND e_Lm = 2 * SQRT[ (2L * (2L - 1)) / ((2L + 2m)(2L - 2m)) ] * e_(L - 1)m

    a_km = SQRT[2 * (2k + 1) / ((2k + 2m + 2) (2k - 2m + 2)) ] * u_km

    where

    u_km = (2k - 2m - 2) - (2k - 2) * tc    IF 0 < BETA < PI/2
    u_km = -(2m)                            IF BETA = PI/2
    u_km = (2k - 2) * t - (2m)              IF PI/2 < BETA < PI

    when BETA is PI/2, we have a simplification of prefactor c_(k + m) * s_(k - m)

    f_k = c_(k+m) * s_(k-m) = 2**(-k)                   IF k values are integers
    f_k = c_(k+m) * s_(k-m) = SQRT(2) 2**(-k - 1/2)     IF k values are half-integers

    f_0 = 1 AND f_L = 0.5 * f_(L - 1)                   IF L is integer
    f_0 = 0.5 * SQRT(2) AND f_L = 0.5 * f_(L - 1)       IF L is half-integer


    **************************************************************
    **************************************************************

    Only using integer values for k and m, so I will rewrite 
    according to Will Lenthe's code...

    Some of the formulae do not simply convert 2j to j etc.

    **************************************************************
    **************************************************************

    a_jkm = (2j - 1) * u_jkm * w_jkm
    b_jkm = v_jkm * w_jkm

    where

    u_jkm =  j * (j - 1) - k * m  - j * (j - 1) * t_c    IF 0 < BETA < PI/2
    u_jkm =  -k * m                                      IF BETA = PI/2
    u_jkm =  j * (j - 1) * t - k * m                     IF PI/2 < BETA < PI

    v_jkm = j * SQRT [ (j + k - 1) (j - k - 1) (j + m - 1) (j - m - 1) ]

    w_jkm = 1 / [ (j - 1) * SQRT [ (j + k) (j - k) (j + m) (j - m) ] ]

    And the seed values are provided by:

    d_kkm = c_(k + m) * s_(k - m) * e_(km) and d_(k + 1)km = a_km * d_kkm

    where

    c_n = cos(BETA / 2)^n
    c_0 = 1 AND c_1 = cos(BETA / 2) AND c_n = c_(n - 1) * c_1

    s_n = sin(BETA / 2)^n
    s_0 = 1 AND s_1 = sin(BETA / 2) AND s_n = s_(n - 1) * s_1

    e_mm = 1 AND e_Lm = 2 * SQRT[ (L * (2L - 1)) / ((L + m)(L - m)) ] * e_(L - 1)m for L = (m+1), (m+2), ..., k

    a_km = SQRT[ (2k + 1) / ((k + m + 1) (k - m + 1)) ] * u_km


    where

    u_km = (k - m + 1) - (k + 1) * tc    IF 0 < BETA < PI/2
    u_km = -m                            IF BETA = PI/2
    u_km = (k + 1) * t - m               IF PI/2 < BETA < PI

    
"""

from extended_precision import x2f, xlsum2, x_norm

@torch.jit.script
def u_jkm(j: Tensor,
          k: Tensor,
          m: Tensor,
          beta: Tensor,
          tc: Tensor,
          t: Tensor,
) -> Tensor:
    """
    Calculate u_jkm according to the following formula:

    u_jkm =  j * (j - 1) - k * m  - j * (j - 1) * t_c    IF 0 < BETA < PI/2
    u_jkm =  -k * m                                      IF BETA = PI/2
    u_jkm =  j * (j - 1) * t - k * m                     IF PI/2 < BETA < PI

    """
    u_jkm = torch.where((0.0 < beta) & (beta < np.pi / 2.0), j * (j - 1.0) - k * m - j * (j - 1.0) * tc, \
        torch.where(beta == np.pi / 2.0, -k * m, j * (j - 1.0) * t - k * m))
    return u_jkm


@torch.jit.script
def v_jkm(j: Tensor,
          k: Tensor,
          m: Tensor,
) -> Tensor:
    """
    Calculate v_jkm according to the following formula:

    v_jkm = j * SQRT [ (j + k - 1) (j - k - 1) (j + m - 1) (j - m - 1) ]

    """
    return j * torch.sqrt((j + k - 1.0) * (j - k - 1.0) * (j + m - 1.0) * (j - m - 1.0))


@torch.jit.script
def w_jkm(j: Tensor,
          k: Tensor,
          m: Tensor,
) -> Tensor:
    """
    Calculate w_jkm according to the following formula:

    w_jkm = 1 / [ (j - 1) * SQRT [ (j + k) (j - k) (j + m) (j - m) ] ]

    """
    return 1.0 / ((j - 1.0) * torch.sqrt((j + k) * (j - k) * (j + m) * (j - m)))




"""

Implement Toshio's F90 code:

Table 4: Double precision X-number Fortran subroutine to return Wigner’s d-functions, d( j)
km(β).

This is an adaptation of the X-number formulation to wdvc listed in Table 3. This time, the input
double precision seed value dkkm is replaced with its X-number mantissa and exponent, xdkkm
and idkkm. Also, some of the working integer variables are declared as 64 bit integers in order to
avoid the overflow in their multiplications.

subroutine xwdvc(jmax2,k2,m2,tc,xdkkm,idkkm,dkm)
real*8 tc,xdkkm,dkm(0:*),xd0,x2f,a,xd1,w,b,xd2
integer jmax2,k2,m2,idkkm,id0,id1,id2
integer*8 m2k2,j2,j2j22,j2pk2,j2mk2,j2pm2,j2mm2
xd0=xdkkm; id0=idkkm; dkm(k2/2)=x2f(xd0,id0); if(jmax2.le.k2) return
m2k2=int8(m2)*int8(k2); j2=k2+2; j2mm2=j2-m2
a=sqrt(dble(j2-1)/dble((j2+m2)*(j2mm2)))*(dble(j2mm2)-dble(j2)*tc)
xd1=xd0*a; id1=id0; call xnorm(xd1,id1); dkm(j2/2)=x2f(xd1,id1)
do j2=k2+4,jmax2,2
    j22=j2-2; j2j22=j2*j22
    j2pk2=j2+k2; j2mk2=j2-k2; j2pm2=j2+m2; j2mm2=j2-m2
    w=1.d0/(dble(j22)*sqrt(dble(j2pk2*j2mk2)*dble(j2pm2*j2mm2)))
    b=w*dble(j2)*sqrt(dble((j2pk2-2)*(j2mk2-2))*dble((j2pm2-2)*(j2mm2-2)))
    a=w*dble(j2+j22)*(dble(j2j22-m2k2)-dble(j2j22)*tc)
    call xlsum2(a,xd1,-b,xd0,xd2,id1,id0,id2)
    dkm(j2/2)=x2f(xd2,id2); xd0=xd1; id0=id1; xd1=xd2; id1=id2
enddo
return; end

"""

@torch.jit.script
def xwdvc(jmax2: Tensor,
          k2: Tensor,
          m2: Tensor,
          tc: Tensor,
          xd0: Tensor,
          id0: Tensor,
) -> Tensor:
    dkm = torch.zeros(int(jmax2 - k2) // 2 + 1, dtype=torch.float64)
    dkm[0] = x2f(xd0, id0)[0]
    if jmax2 <= k2:
        return dkm
    
    m2k2 = m2 * k2
    j2 = k2 + 2
    j2mm2 = j2 - m2
    a = ((j2 - 1).double() / ((j2 + m2) * j2mm2).double())**0.5 * (j2mm2.double() - j2.double() * tc)
    xd1 = xd0 * a
    id1 = id0
    xd1, id1 = x_norm(xd1, id1)
    dkm[1] = x2f(xd1, id1)[0]
    index = 2
    for j2 in torch.arange(int(k2 + 4), int(jmax2) + 2, dtype=torch.int64, step=2):
        j22 = j2 - 2
        j2j22 = j2 * j22
        j2pk2 = j2 + k2
        j2mk2 = j2 - k2
        j2pm2 = j2 + m2
        j2mm2 = j2 - m2
        # w=1.d0/(dble(j22)*sqrt(dble(j2pk2*j2mk2)*dble(j2pm2*j2mm2)))
        w = 1.0 / (j22.double() * ((j2pk2 * j2mk2).double() * (j2pm2 * j2mm2).double())**0.5)
        # b=w*dble(j2)*sqrt(dble((j2pk2-2)*(j2mk2-2))*dble((j2pm2-2)*(j2mm2-2)))
        b = w * j2.double() * (((j2pk2 - 2) * (j2mk2 - 2)).double() * ((j2pm2 - 2) * (j2mm2 - 2)).double())**0.5
        # a=w*dble(j2+j22)*(dble(j2j22-m2k2)-dble(j2j22)*tc)
        a = w * (j2 + j22).double() * ((j2j22 - m2k2).double() - j2j22.double() * tc)
        # d_n is a * d_n-1 - b * d_n-2
        # d_n-2 is xd0, id0 and d_n-1 is xd1, id1 so it is easy to swap a and b here
        xd2, id2 = xlsum2(-b, a, xd0, id0, xd1, id1)
        dkm[index] = x2f(xd2, id2)[0]
        xd0 = xd1
        id0 = id1
        xd1 = xd2
        id1 = id2
        index += 1

    return dkm


"""

Now we implement the logic for seed generation and print out the results 
using his example code ported from F90 as inspiration. We will make β and J
a input parameters...


Table 5: Test driver of xwdvc. It print outs the values of Wigner's d-functions for all the de-
gree/orders satisfying the condition, 0 ≤ m ≤ k ≤ j ≤ J, when the case J = 9/2 and β = π/4.
program prtxwdc
real*8 PI,beta,betah,ch,sh,tc,cn,sn,fm,fk,xekm,f,xdkkm,fj
integer JX,JX2,jmax2,jg,j0,icn,isn,n,m2,k2,kpm,kmm,iekm,idkkm
parameter (JX=16384,JX2=JX*2)
real*8 dkm(0:JX),xc(0:JX2),xs(0:JX); integer ic(0:JX2),is(0:JX)
PI=atan(1.d0)*4.d0; jmax2=9; beta=PI*0.25d0; jg=jmax2/2; j0=jmax2-jg*2
betah=beta*0.5d0; ch=cos(betah); sh=sin(betah); tc=2.d0*sh*sh
cn=1.d0; icn=0; sn=1.d0; isn=0; xc(0)=cn; ic(0)=icn; xs(0)=sn; is(0)=isn
do n=1,jmax2
    cn=ch*cn; call xnorm(cn,icn); xc(n)=cn; ic(n)=icn
enddo
do n=1,jg
    sn=sh*sn; call xnorm(sn,isn); xs(n)=sn; is(n)=isn
enddo
do m2=j0,jmax2,2
    fm=dble(m2)*0.5d0
    do k2=m2,jmax2,2
        fk=dble(k2)*0.5d0; kpm=(k2+m2)/2; kmm=(k2-m2)/2
        if(k2.eq.m2) then
            xekm=1.d0; iekm=0
        else
            f=dble(k2*(k2-1))/dble(kpm*kmm)
            xekm=xekm*sqrt(f); call xnorm(xekm,iekm)
        endif
        xdkkm=xc(kpm)*xs(kmm); idkkm=ic(kpm)+is(kmm)
        call xnorm(xdkkm,idkkm)
        xdkkm=xdkkm*xekm; idkkm=idkkm+iekm
        call xnorm(xdkkm,idkkm)
        call xwdvc(jmax2,k2,m2,tc,xdkkm,idkkm,dkm)
        do j2=k2,jmax2,2
            fj=dble(j2)*0.5d0
            write(*,”(0p3f10.1,1pe25.15)”) fj,fk,fm,dkm(j2/2)
        enddo
    enddo
enddo
end program prtxwdc

"""

@torch.jit.script
def xwdvc_driver(jmax2: Tensor,
                 beta: Tensor,
) -> None:
    """
    This is the driver for the xwdvc function. It will print out the values of Wigner's d-functions 
    for all the degree/orders satisfying the condition, 0 ≤ m ≤ k ≤ j ≤ J, given J and β.

    """
    # keep everything as tensors and use int() to get the values for range if needed
    JX = 16384
    JX2 = JX * 2
    jg = jmax2 // 2
    j0 = jmax2 - jg * 2
    betah = beta * 0.5
    ch = torch.cos(betah)
    sh = torch.sin(betah)
    tc = 2.0 * sh * sh
    xc = torch.ones(JX2, dtype=torch.float64)
    i_c = torch.zeros(JX2, dtype=torch.int64)
    xs = torch.ones(JX, dtype=torch.float64)
    i_s = torch.zeros(JX, dtype=torch.int64)

    cn = torch.ones(1, dtype=torch.float64)
    icn = torch.zeros(1, dtype=torch.int64)

    for n in range(1, int(jmax2)):
        cn = ch * cn
        cn, icn = x_norm(cn, icn)
        xc[n] = cn[0]
        i_c[n] = icn[0]

    sn = torch.ones(1, dtype=torch.float64)
    isn = torch.zeros(1, dtype=torch.int64)

    for n in range(1, int(jg)):
        sn = sh * sn
        sn, isn = x_norm(sn, isn)
        xs[n] = sn[0]
        i_s[n] = isn[0]

    xekm = torch.ones(1, dtype=torch.float64)
    iekm = torch.zeros(1, dtype=torch.int64)

    for m2 in torch.arange(int(j0), int(jmax2) + 2, step=2, dtype=torch.int64):
        fm = m2.double() * 0.5
        for k2 in torch.arange(int(m2), int(jmax2) + 2, step=2, dtype=torch.int64):
            fk = k2.double() * 0.5
            kpm = (k2 + m2) // 2
            kmm = (k2 - m2) // 2
            if k2 == m2:
                xekm = torch.ones(1, dtype=torch.float64)
                iekm = torch.zeros(1, dtype=torch.int64)
            else:
                f = (k2.double() * (k2.double() - 1)) / (kpm.double() * kmm.double())
                xekm = xekm * f**0.5
                xekm, iekm = x_norm(xekm, iekm)
            xdkkm = xc[kpm] * xs[kmm]
            idkkm = i_c[kpm] + i_s[kmm]
            xdkkm, idkkm = x_norm(xdkkm, idkkm)
            xdkkm = xdkkm * xekm
            idkkm = idkkm + iekm
            xdkkm, idkkm = x_norm(xdkkm, idkkm)
            dkm = xwdvc(jmax2, k2, m2, tc, xdkkm, idkkm)
            index = 0
            for j2 in torch.arange(int(k2), int(jmax2) + 2, step=2, dtype=torch.int64):
                fj = j2.double() * 0.5
                # print(f"fj: {fj.item()}, fk: {fk.item()}, fm: {fm.item()}, dkm: {dkm[j2 // 2].item()} kpm: {kpm.item()}, kmm: {kmm.item()}, xc[kpm]: {xc[kpm].item()}, xs[kmm]: {xs[kmm].item()}, xdkkm: {xdkkm.item()}, idkkm: {idkkm.item()}, xekm: {xekm.item()}, iekm: {iekm.item()}")
                print(f"fj: {fj.item()}, fk: {fk.item()}, fm: {fm.item()}, dkm: {dkm[index].item()} item number {index + 1} out  of {dkm.shape[0]}")
                index += 1

# test out the functions
xwdvc_driver(
    jmax2=torch.tensor([10,], dtype=torch.int64),
    beta=torch.tensor([torch.pi * 7117.0 / 16384.0,], dtype=torch.float64),
)