import torch
import numpy as np
from scipy.special import gammaln

sqrt2 = np.sqrt(2.0)


## Recursive computation of d^l_mn (torch.pi/2)
def triHalfRecur(l, m, n):
    denom = (-1 + l) * np.sqrt((l - m) * (l + m) * (l - n) * (l + n))
    c1 = (1 - 2 * l) * m * n / denom
    c2 = (
        -1.0
        * l
        * np.sqrt(((l - 1) * (l - 1) - m * m) * ((l - 1) * (l - 1) - n * n))
        / denom
    )
    return c1, c2


def generateLittleHalf(B):
    # m, n -> m + (B-1), n + (B-1)
    d = torch.empty(B, 2 * B - 1, 2 * B - 1).double().fill_(0)
    # Fill first two levels (l = 0, l = 1)
    d[0, B - 1, B - 1] = 1

    d[1, -1 + (B - 1), -1 + (B - 1)] = 0.5
    d[1, -1 + (B - 1), B - 1] = 1.0 / sqrt2
    d[1, -1 + (B - 1), B] = 0.5

    d[1, (B - 1), -1 + (B - 1)] = -1.0 / sqrt2
    d[1, (B - 1), B] = 1.0 / sqrt2

    d[1, B, -1 + (B - 1)] = 0.5
    d[1, B, (B - 1)] = -1.0 / sqrt2
    d[1, B, B] = 0.5

    ## Fill rest of values through Kostelec-Rockmore recursion
    for l in range(2, B):
        for m in range(0, l):
            for n in range(0, l):
                if (m == 0) and (n == 0):
                    d[l, B - 1, B - 1] = -1.0 * ((l - 1) / l) * d[l - 2, B - 1, B - 1]
                else:
                    c1, c2 = triHalfRecur(l, m, n)
                    d[l, m + (B - 1), n + (B - 1)] = (
                        c1 * d[l - 1, m + (B - 1), n + (B - 1)]
                        + c2 * d[l - 2, m + (B - 1), n + (B - 1)]
                    )
        for m in range(0, l + 1):
            lnV = 0.5 * (
                gammaln(2 * l + 1) - gammaln(l + m + 1) - gammaln(l - m + 1)
            ) - l * np.log(2.0)
            d[l, m + (B - 1), l + (B - 1)] = np.exp(lnV)
            d[l, l + (B - 1), m + (B - 1)] = np.power(-1.0, l - m) * np.exp(lnV)
        for m in range(0, l + 1):
            for n in range(0, l + 1):
                val = d[l, m + (B - 1), n + (B - 1)]
                if (m != 0) or (n != 0):
                    d[l, -m + (B - 1), -n + (B - 1)] = np.power(-1.0, m - n) * val
                    d[l, -m + (B - 1), n + (B - 1)] = np.power(-1.0, l - n) * val
                    d[l, m + (B - 1), -n + (B - 1)] = np.power(-1.0, l + m) * val

    print("Computed littleHalf_{}".format(B), flush=True)
    return d


def generateLittleHalf_dict(B):
    d_dict = {}
    # m, n -> m + (B-1), n + (B-1)
    d = torch.empty(B, 2 * B - 1, 2 * B - 1).double().fill_(0)
    # Fill first two levels (l = 0, l = 1)
    d[0, B - 1, B - 1] = 1

    d[1, -1 + (B - 1), -1 + (B - 1)] = 0.5
    d[1, -1 + (B - 1), B - 1] = 1.0 / sqrt2
    d[1, -1 + (B - 1), B] = 0.5

    d[1, (B - 1), -1 + (B - 1)] = -1.0 / sqrt2
    d[1, (B - 1), B] = 1.0 / sqrt2

    d[1, B, -1 + (B - 1)] = 0.5
    d[1, B, (B - 1)] = -1.0 / sqrt2
    d[1, B, B] = 0.5

    ## Fill rest of values through Kostelec-Rockmore recursion
    for l in range(2, B):
        for m in range(0, l):
            for n in range(0, l):
                if (m == 0) and (n == 0):
                    d[l, B - 1, B - 1] = -1.0 * ((l - 1) / l) * d[l - 2, B - 1, B - 1]
                    d_dict[(l, m, n)] = -1.0 * ((l - 1) / l) * d[l - 2, B - 1, B - 1]
                else:
                    c1, c2 = triHalfRecur(l, m, n)
                    d[l, m + (B - 1), n + (B - 1)] = (
                        c1 * d[l - 1, m + (B - 1), n + (B - 1)]
                        + c2 * d[l - 2, m + (B - 1), n + (B - 1)]
                    )
                    d_dict[(l, m, n)] = (
                        c1 * d[l - 1, m + (B - 1), n + (B - 1)]
                        + c2 * d[l - 2, m + (B - 1), n + (B - 1)]
                    )

        for m in range(0, l + 1):
            lnV = 0.5 * (
                gammaln(2 * l + 1) - gammaln(l + m + 1) - gammaln(l - m + 1)
            ) - l * np.log(2.0)
            d[l, m + (B - 1), l + (B - 1)] = np.exp(lnV)
            d[l, l + (B - 1), m + (B - 1)] = np.power(-1.0, l - m) * np.exp(lnV)
            d_dict[(l, m, l)] = np.exp(lnV)
            d_dict[(l, l, m)] = np.power(-1.0, l - m) * np.exp(lnV)

        for m in range(0, l + 1):
            for n in range(0, l + 1):
                val = d[l, m + (B - 1), n + (B - 1)]
                if (m != 0) or (n != 0):
                    d[l, -m + (B - 1), -n + (B - 1)] = np.power(-1.0, m - n) * val
                    d[l, -m + (B - 1), n + (B - 1)] = np.power(-1.0, l - n) * val
                    d[l, m + (B - 1), -n + (B - 1)] = np.power(-1.0, l + m) * val
                    d_dict[(l, -m, -n)] = np.power(-1.0, m - n) * val
                    d_dict[(l, -m, n)] = np.power(-1.0, l - n) * val
                    d_dict[(l, m, -n)] = np.power(-1.0, l + m) * val

    print("Computed littleHalf_{}".format(B), flush=True)
    return d_dict


def dltWeightsDH(B):
    W = torch.empty(2 * B).double().fill_(0)
    for k in range(0, 2 * B):
        C = (2.0 / B) * np.sin(torch.pi * (2 * k + 1) / (4.0 * B))
        wk = 0.0
        for p in range(0, B):
            wk += (1.0 / (2 * p + 1)) * np.sin(
                (2 * k + 1) * (2 * p + 1) * torch.pi / (4.0 * B)
            )
        W[k] = C * wk
    print("Computed dltWeights_{}".format(B), flush=True)
    return W


## Inverse (orthogonal) DCT Matrix of dimension N x N
def idctMatrix(N):
    DI = torch.empty(N, N).double().fill_(0)
    for k in range(0, N):
        for n in range(0, N):
            DI[k, n] = np.cos(torch.pi * n * (k + 0.5) / N)
    DI[:, 0] = DI[:, 0] * np.sqrt(1.0 / N)
    DI[:, 1:] = DI[:, 1:] * np.sqrt(2.0 / N)
    print("Computed idctMatrix_{}".format(N), flush=True)
    return DI


## Inverse (orthogonal) DST Matrix of dimension N x N
def idstMatrix(N):
    DI = torch.empty(N, N).double().fill_(0)
    for k in range(0, N):
        for n in range(0, N):
            if n == (N - 1):
                DI[k, n] = np.power(-1.0, k)
            else:
                DI[k, n] = np.sin(torch.pi * (n + 1) * (k + 0.5) / N)
    DI[:, N - 1] = DI[:, N - 1] * np.sqrt(1.0 / N)
    DI[:, : (N - 1)] = DI[:, : (N - 1)] * np.sqrt(2.0 / N)
    print("Computed idstMatrix_{}".format(N), flush=True)
    return DI


# Normalization coeffs for m-th frequency (C_m)
def normCm(B):
    Cm = torch.empty(2 * B - 1).double().fill_(0)
    for m in range(-(B - 1), B):
        Cm[m + (B - 1)] = np.power(-1.0, m) * np.sqrt(2.0 * torch.pi)
    print("Computed normCm_{}".format(B), flush=True)
    return Cm


def wigner_d(B):
    return generateLittleHalf_dict(B)


# Computes sparse matrix of Wigner-d function cosine + sine series coefficients
def wigner_D(B):
    d = generateLittleHalf(B).cpu().numpy()
    print(d)
    H = 0
    W = 0
    indH = []
    indW = []
    val = []
    mln_dict = {}
    N = 2 * B
    for m in range(-(B - 1), B):
        for l in range(np.absolute(m), B):
            for n in range(0, l + 1):
                iH = l + H
                iW = n + W

                # Cosine series
                if (m % 2) == 0:
                    if n == 0:
                        c = np.sqrt((2 * l + 1) / 2.0) * np.sqrt(N)
                    else:
                        c = np.sqrt((2 * l + 1) / 2.0) * np.sqrt(2.0 * N)

                    if (m % 4) == 2:
                        c *= -1.0

                    coeff = (
                        c * d[l, n + (B - 1), -m + (B - 1)] * d[l, n + (B - 1), B - 1]
                    )

                # Sine series
                else:
                    if n == l:
                        coeff = 0.0

                    else:
                        c = np.sqrt((2 * l + 1) / 2.0) * np.sqrt(2.0 * N)

                        if (m % 4) == 1:
                            c *= -1.0

                        coeff = (
                            c
                            * d[l, (n + 1) + (B - 1), -m + (B - 1)]
                            * d[l, (n + 1) + (B - 1), B - 1]
                        )

                if np.absolute(coeff) > 1.0e-15:
                    indH.append(iH)
                    indW.append(iW)
                    val.append(coeff)

                if coeff != 0:
                    # print("l={}, m={}, n={}, coeff={}".format(l, m, n, coeff))
                    mln_dict[(l, m, n)] = coeff

        H += B
        W += N

    return mln_dict


# test against spherical package
from spherical import Wigner

order_max = 5

w = Wigner(ell_max=order_max)
d_spherical = w.d(np.exp(1.0j * torch.pi / 2))

print(d_spherical)

# define the quaternion for torch.pi/2 rotation around y axis
quat = np.array([0, 0.7071068, 0, 0.7071068], dtype=np.float64)
# get the Wigner-D matrix
D_spherical = w.D(R=quat)

# call the above trimmed version of wignerCoeffs
d_ts2kit = wigner_d(order_max)
D_ts2kit = wigner_D(order_max)


for key in D_ts2kit.keys():
    l, m, n = key
    if m > 0 and n > 0:
        print("-" * 80)
        print("l={}, m={}, n={}".format(l, m, n))
        print("d_spherical = {}".format(d_spherical[w.dindex(l, m, n)]))
        print("D_spherical = {}".format(D_spherical[w.Dindex(l, m, n)]))
        print("d_ts2kit = {}".format(d_ts2kit[(l, m, n)]))
        print("D_ts2kit = {}".format(D_ts2kit[(l, m, n)]))
