import math
import random

"""

Lenstra elliptic-curve factorization method 

Originally from:

https://stackoverflow.com/questions/4643647/fast-prime-factorization-module

and

https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization


Useful for reshaping arrays into approximately square shapes for GPU processing.
Also useful for finding special bandwidth numbers for FFTs on SO(3).

"""


def FactorECM(N0):
    def factor_trial_division(x):
        factors = []
        while (x & 1) == 0:
            factors.append(2)
            x >>= 1
        for d in range(3, int(math.sqrt(x)) + 1, 2):
            while x % d == 0:
                factors.append(d)
                x //= d
        if x > 1:
            factors.append(x)
        return sorted(factors)

    def is_probably_prime_fermat(n, trials=32):
        for _ in range(trials):
            if pow(random.randint(2, n - 2), n - 1, n) != 1:
                return False
        return True

    def gen_primes_sieve_of_eratosthenes(end):
        composite = [False] * end
        for p in range(2, int(math.sqrt(end)) + 1):
            if composite[p]:
                continue
            for i in range(p * p, end, p):
                composite[i] = True
        return [p for p in range(2, end) if not composite[p]]

    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def egcd(a, b):
        ro, r, so, s = a, b, 1, 0
        while r != 0:
            ro, (q, r) = r, divmod(ro, r)
            so, s = s, so - q * s
        return ro, so, (ro - so * a) // b

    def modular_inverse(a, n):
        g, s, _ = egcd(a, n)
        if g != 1:
            raise ValueError(a)
        return s % n

    def elliptic_curve_add(N, A, B, X0, Y0, X1, Y1):
        if X0 == X1 and Y0 == Y1:
            l = ((3 * X0**2 + A) * modular_inverse(2 * Y0, N)) % N
        else:
            l = ((Y1 - Y0) * modular_inverse(X1 - X0, N)) % N
        x = (l**2 - X0 - X1) % N
        y = (l * (X0 - x) - Y0) % N
        return x, y

    def elliptic_curve_mul(N, A, B, X, Y, k):
        k -= 1
        BX, BY = X, Y
        while k != 0:
            if k & 1:
                X, Y = elliptic_curve_add(N, A, B, X, Y, BX, BY)
            BX, BY = elliptic_curve_add(N, A, B, BX, BY, BX, BY)
            k >>= 1
        return X, Y

    def factor_ecm(N, bound=512, icurve=0):
        def next_factor_ecm(x):
            return factor_ecm(x, bound=bound + 512, icurve=icurve + 1)

        def prime_power(p, bound2=int(math.sqrt(bound) + 1)):
            mp = p
            while mp * p < bound2:
                mp *= p
            return mp

        if N < (1 << 16):
            return factor_trial_division(N)

        if is_probably_prime_fermat(N):
            return [N]

        while True:
            X, Y, A = [random.randrange(N) for _ in range(3)]
            B = (Y**2 - X**3 - A * X) % N
            if 4 * A**3 - 27 * B**2 != 0:
                break

        for p in gen_primes_sieve_of_eratosthenes(bound):
            k = prime_power(p)
            try:
                X, Y = elliptic_curve_mul(N, A, B, X, Y, k)
            except ValueError as ex:
                g = gcd(ex.args[0], N)
                if g != N:
                    return sorted(next_factor_ecm(g) + next_factor_ecm(N // g))
                else:
                    return next_factor_ecm(N)
        return next_factor_ecm(N)

    return factor_ecm(N0)


def nearly_square_factors(n):
    """
    For a given highly composite number n, find two factors that are roughly
    close to each other. This is useful for reshaping some arrays into square
    shapes for GPU processing.
    """
    factor_a = 1
    factor_b = 1
    factors = FactorECM(n)
    for factor in factors[::-1]:
        if factor_a > factor_b:
            factor_b *= factor
        else:
            factor_a *= factor
    return factor_a, factor_b
