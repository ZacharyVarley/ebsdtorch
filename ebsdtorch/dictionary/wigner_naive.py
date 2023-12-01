import numpy as np
from scipy.special import jv, legendre, sph_harm, jacobi
try:
    from scipy.misc import factorial, comb
except:
    from scipy.special import factorial, comb
from numpy import floor, sqrt, sin, cos, exp, power
from math import pi
from scipy.special import jacobi


def wigner_d_naive_v3(l: int,
                      m: int,
                      n: int,
                      approx_lim=10000000):
    """
    Wigner "small d" matrix. (Euler z-y-z convention)
    example:
        l = 2
        m = 1
        n = 0
        beta = linspace(0,pi,100)
        wd210 = wignerd(l,m,n)(beta)

    some conditions have to be met:
         l >= 0
        -l <= m <= l
        -l <= n <= l

    The approx_lim determines at what point
    bessel functions are used. Default is when:
        l > m+10
          and
        l > n+10

    for integer l and n=0, we can use the spherical harmonics. If in
    addition m=0, we can use the ordinary legendre polynomials.
    """

    if (l < 0) or (abs(m) > l) or (abs(n) > l):
        raise ValueError("wignerd(l = {0}, m = {1}, n = {2}) value error.".format(l, m, n) \
            + " Valid range for parameters: l>=0, -l<=m,n<=l.")

    jmn_terms = {
        l + n : (m - n, m - n),
        l - n : (n - m, 0.),
        l + m : (n - m, 0.),
        l - m : (m - n, m - n),
        }
    
    # jmn_terms = np.array(list(jmn_terms.items()), dtype=np.float64)

    k = min(jmn_terms)
    a, lmb = jmn_terms[k]

    k = np.int64(k)
    a = np.int64(a)
    a_float = np.float128(a)
    lmb = np.float128(lmb)

    b = np.int64(2 * l - 2 * k - a)
    b_float = np.float64(2. * l - 2. * k - a)

    if (a < 0) or (b < 0):
        raise ValueError("wignerd(l = {0}, m = {1}, n = {2}) value error.".format(l, m, n) \
            + " Encountered negative values in (a,b) = ({0},{1})".format(a,b))

    coeff1 = power(-1., lmb) 
    coeff2 = sqrt(comb(2. * l - k, k + a))
    coeff3 = (1. / sqrt(comb(k + b, b)))

    print("coeff1: {0}".format(coeff1))
    print("coeff2: {0}".format(coeff2))
    print("coeff3: {0}".format(coeff3))

    coeff = coeff1 * coeff2 * coeff3

    #print 'jacobi (exact)'
    return lambda beta: coeff \
        * power(sin(0.5*beta),a_float) \
        * power(cos(0.5*beta),b_float) \
        * jacobi(k,a,b_float)(cos(beta))


test_parameters = [
    (365, 102, 20, -4.23570250037880395095020243575390e-02, 8161.0 / 16384.0),
    (294, 247, 188, 1.11943794723176255836019618855372e-01, 7417.0 / 16384.0),
    (3777, 1014, 690, 1.68450832524798173944840155878705e-03, 12233.0 / 16384.0),
]

for test_parameter in test_parameters:
    j = test_parameter[0]
    k = test_parameter[1]
    m = test_parameter[2]
    correct_value = test_parameter[3]
    beta = np.pi * test_parameter[4]
    d_value = wigner_d_naive_v3(j, k, m)(beta)
    print("Correct   : " + "{:.20f}".format(correct_value))
    print("Computed  : " + "{:.20f}".format(d_value))
    print("Difference: " + "{:.20f}".format(abs(d_value - correct_value)))
    print(" ------------------- ")


# print test with print out value of d(1, 0, 0), beta = (pi * 7117.0 / 16384.0)
print(f"d(1, 0, 0) beta = (pi * 7117.0 / 16384.0) = {wigner_d_naive_v3(1, 0, 0)(np.pi * 7117.0 / 16384.0)}")

# print test with print out value of d(1, 1, 0), beta = (pi * 7117.0 / 16384.0)
print(f"d(1, 1, 0) beta = (pi * 7117.0 / 16384.0) = {wigner_d_naive_v3(1, 1, 0)(np.pi * 7117.0 / 16384.0)}")

# print test with print out value of d(1, 1, 0), beta = (pi * 7117.0 / 16384.0)
print(f"d(4, 1, 1) beta = (pi * 7117.0 / 16384.0) = {wigner_d_naive_v3(4, 1, 1)(np.pi * 7117.0 / 16384.0)}")
