import torch
import numpy as np


@torch.jit.script
def u_jkm_0(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            tc: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner U coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        tc: torch.Tensor
            1 - cos(BETA) which is the same as 2 * sin(BETA / 2) ** 2

    """
    return -tc * ((j - 1) * j) - (k * m - (j - 1) * j)

@torch.jit.script
def u_jkm_1(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            ) -> int:
    """
    Calculate the Wigner U coefficient for j, k, m when BETA is at PI/2

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    """
    return - (k * m)

@torch.jit.script
def u_jkm_2(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            t: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner U coefficient for j, k, m when BETA is in range (PI/2, PI]

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        tc: torch.Tensor
            cos(BETA)

    """
    return t  * ((j - 1) * j) -  k * m


@torch.jit.script
def v_jkm(j: torch.Tensor,
          k: torch.Tensor,
          m: torch.Tensor,
            ) -> torch.Tensor:
    """
    Calculate the Wigner V coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        
    Returns:
        The recursion coefficient u_jkm
     
    """
    return ( (j+k-1) * (j-k-1) * (j+m-1) * (j-m-1) )**0.5 * j


@torch.jit.script
def w_jkm(
        j: torch.Tensor,
        k: torch.Tensor,
        m: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the Wigner W coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    Returns:
        The recursion coefficient u_jkm

        return Real(1) / ( std::sqrt( Real( (j+k  ) * (j-k  ) * (j+m  ) * (j-m  ) ) ) * (j-1) );

    """

    return 1.0 / ( ( (j+k) * (j-k) * (j+m) * (j-m) )**0.5 * (j-1) )


@torch.jit.script
def a_jkm_0(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            tc: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        tc: torch.Tensor
            1 - cos(BETA) which is the same as 2 * sin(BETA / 2) ** 2

    Returns:
        The recursion coefficient u_jkm for BETA in range [0, PI/2)

        Given by: w_jkm<Real>(j, k, m) * ( u_jkm_0<Real>(j, k, m, tc) * (2*j-1) )

    """
    
    return w_jkm(j, k, m) * ( u_jkm_0(j, k, m, tc) * (2*j-1) )


@torch.jit.script
def a_jkm_1(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    Returns:
        The recursion coefficient u_jkm for BETA at PI/2

        Given by:  w_jkm<Real>(j, k, m) * ( u_jkm_1      (j, k, m    ) * (2*j-1) )

    """
    return w_jkm(j, k, m) * ( u_jkm_1(j, k, m) * (2*j-1) )


@torch.jit.script
def a_jkm_2(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            t: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        t: torch.Tensor
            cos(BETA)

    Returns:
        The recursion coefficient u_jkm for BETA in range (PI/2, PI]

        Given by: w_jkm<Real>(j, k, m) * ( u_jkm_2<Real>(j, k, m, t ) * (2*j-1) )

    """
    return w_jkm(j, k, m) * ( u_jkm_2(j, k, m, t) * (2*j-1) )


@torch.jit.script
def b_jkm(j: torch.Tensor,
            k: torch.Tensor,
            m: torch.Tensor,
            ) -> torch.Tensor:
    """
    Calculate the Wigner B coefficient for j, k, m when BETA

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
    
    Returns:
        The recursion coefficient u_jkm

        Given by:  w_jkm<Real>(j, k, m) * v_jkm<Real>(j, k, m)

    """
    return w_jkm(j, k, m) * v_jkm(j, k, m)



@torch.jit.script
def u_km_0(k: torch.Tensor,
           m: torch.Tensor,
           tc: torch.Tensor
            ) -> torch.Tensor:
    """
    template <typename Real> inline Real u_km_0 (const int64_t k, const int64_t m, const Real tc) {return     (-tc * (k + 1) - (m - 1 - k));}
		
    
    Calculate the Wigner U coefficient for k, m when BETA is in range [0, PI/2)

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        tc: torch.Tensor
            1 - cos(BETA) which is the same as 2 * sin(BETA / 2) ** 2

    Returns:
        The recursion coefficient u_km

        Given by: -tc * (k + 1) - (m - 1 - k)

    """
    return -tc * (k + 1) - (m - 1 - k)


@torch.jit.script
def u_km_1(k: torch.Tensor,
           m: torch.Tensor,
            ) -> int:
    """
	template <typename Real> inline Real u_km_1 (const int64_t k, const int64_t m) {return Real( -  m );}
        
    
    Calculate the Wigner U coefficient for k, m when BETA is at PI/2

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    Returns:
        The recursion coefficient u_km

        Given by: -m

    """
    return -m


@torch.jit.script
def u_km_2(k: torch.Tensor,
           m: torch.Tensor,
           t: torch.Tensor
            ) -> torch.Tensor:
    """
	template <typename Real> inline Real u_km_2 (             const int64_t k, const int64_t m, const Real t ) {return     ( t  * (k + 1) -  m         );}
        
    
    Calculate the Wigner U coefficient for k, m when BETA is in range (PI/2, PI]

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        t: torch.Tensor
            cos(BETA)

    Returns:
        The recursion coefficient u_km

        Given by: t  * (k + 1) -  m

    """
    return t  * (k + 1) -  m


"""
Implementations of these lines:

//@brief     : compute recursion seed coefficient a_{k,m} (equation 22)
		//@param k   : first order in d^j_{k,m}
		//@param m   : second order in d^j_{k,m}
		//@param t/tc: cos(beta) / 1 - cos(beta)
		//@return    : recursion seed
		//@note      : _0, _1, _2 for 0 <= beta < pi / 2, beta == pi / 2, and pi / 2 < beta <= pi / 2 respectively
		template <typename Real> inline Real a_km_0 (             const int64_t k, const int64_t m, const Real tc) {return std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_0<Real>(k, m, tc);}
		template <typename Real> inline Real a_km_1 (             const int64_t k, const int64_t m               ) {return std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_1<Real>(k, m    );}
		template <typename Real> inline Real a_km_2 (             const int64_t k, const int64_t m, const Real t ) {return std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_2<Real>(k, m, t );}

"""

@torch.jit.script
def a_km_0(k: torch.Tensor,
           m: torch.Tensor,
           tc: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for k, m when BETA is in range [0, PI/2)

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        tc: torch.Tensor
            1 - cos(BETA) which is the same as 2 * sin(BETA / 2) ** 2

    Returns:
        The recursion coefficient a_km

        Given by: std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_0<Real>(k, m, tc)

    """
    return ( ( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) )**0.5 * u_km_0(k, m, tc)


@torch.jit.script
def a_km_1(k: torch.Tensor,
           m: torch.Tensor,
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for k, m when BETA is at PI/2

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    Returns:
        The recursion coefficient a_km

        Given by: std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_1<Real>(k, m    )

    """
    return ( ( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) )**0.5 * u_km_1(k, m)


@torch.jit.script
def a_km_2(k: torch.Tensor,
           m: torch.Tensor,
           t: torch.Tensor
            ) -> torch.Tensor:
    """
    Calculate the Wigner A coefficient for k, m when BETA is in range (PI/2, PI]

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        t: torch.Tensor
            cos(BETA)

    Returns:
        The recursion coefficient a_km

        Given by: std::sqrt( Real( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) ) * u_km_2<Real>(k, m, t )

    """
    return ( ( 2*k+1 ) / ( (k+m+1) * (k-m+1) ) )**0.5 * u_km_2(k, m, t)


"""
Implementations of these lines:

//@brief: compute recursion seed coefficient e_{k,m} = \sqrt{\frac{(2k)!}{(k+m)!(k-m)!}} recursively (m <= k) (equation 21)
		//@param k: k in d^k_{k,m}
		//@param m: m in d^k_{k,m}
		//@return: e_km where d^k_{k,m} = 2^-k * e_km
		template <typename Real> inline Real e_km(const int64_t k, const int64_t m) {
			Real e_lm = 1;//e_mm;
			for(int64_t l = m+1; l <= k; l++) e_lm *= std::sqrt( Real( l*(2*l-1) ) / ( 2 * (l+m) * (l-m) ) ) * 2;
			return e_lm;//e_km
		}
"""

@torch.jit.script
def e_km(k: torch.Tensor,
         m: torch.Tensor,
         ) -> torch.Tensor:
    """
    Calculate the Wigner E coefficient for k, m

    Args:
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient

    Returns:
        The recursion coefficient e_km

        Given by: e_km where d^k_{k,m} = 2^-k * e_km

    """
    e_lm = torch.tensor([1.0], dtype=torch.float64)
    for l in range(int(m+1), int(k+1)):
        e_lm *= ( l*(2*l-1) / ( 2 * (l+m) * (l-m) ) )**0.5 * 2
    return e_lm



"""
Implementations of these lines:

Real d(const int64_t j, const int64_t k, const int64_t m, const Real t, const bool nB) {
			//require 0 <= m <= k <= j and beta >= 0 (handle other cases with symmetry)
			if(nB) {                   //d^j_{ k, m}(-beta) =                d^j_{m,k}(     beta)
				return d(j, m, k, t, false);                //equation 5
			} else if(k < 0 && m < 0) {//d^j_{-k,-m}( beta) = (-1)^(   k- m) d^j_{k,m}(     beta)
				const int sign = (k-m) % 2 == 0 ? 1 : -1;
				return d<Real>(j, -k, -m,  t, false) * sign;//equation 6
			} else if(         m < 0) {//d^j_{ k,-m}( beta) = (-1)^(j+ k+2m) d^j_{k,m}(pi - beta)
				const int sign = (j+k) % 2 == 0 ? 1 : -1;
				return d<Real>(j,  k, -m, -t, false) * sign;//equation 7
			} else if(k < 0         ) {//d^j_{-k, m}( beta) = (-1)^(j+2k+3m) d^j_{k,m}(pi - beta)
				const int sign = (j+m) % 2 == 0 ? 1 : -1;
				return d<Real>(j, -k,  m, -t, false) * sign;//equation 8
			} else if(k     <  m    ) {//d^j_{ m, k}( beta) = (-1)^(   k- m) d^j_{k,m}(     beta)
				const int sign = (k-m) % 2 == 0 ? 1 : -1;
				return d<Real>(j,  m,  k,  t, false) * sign;//equation 9
			}

			if(j < k) return NAN;

			//determine if beta is < (0), > (2), or = (1) to pi/2
			const size_t type = t > 0 ? 0 : (t < 0 ? 2 : 1);
			const Real tc = Real(1) - t;

			//compute powers of cos/sin of beta / 2
			const Real c2 = std::sqrt( (Real(1) + t) / 2 );//cos(acos(t)) == cos(beta / 2), always positive since at this point 0 <= beta <= pi
			const Real s2 = std::sqrt( (Real(1) - t) / 2 );//sin(acos(t)) == sin(beta / 2), always positive since at this point 0 <= beta <= pi
			const Real cn = std::pow(c2, Real(k+m));//equation 20 for n = k+m
			const Real sn = std::pow(s2, Real(k-m));//equation 20 for n = k-m

			//compute first term for three term recursion 
			const Real d_kkm  = cn * sn * e_km<Real>(k, m);//equation 18, d^k_{k, m}(beta)
			if(j == k  ) return d_kkm;//if j == k we're done

			//compute second term for three term recursion 
			Real a_km;
			switch(type) {
				case 0: a_km = a_km_0<Real>(k, m, tc); break;//beta <  pi/2
				case 1: a_km = a_km_1<Real>(k, m    ); break;//beta == pi/2
				case 2: a_km = a_km_2<Real>(k, m, t ); break;//beta >  pi/2
			}
			const Real d_k1km = d_kkm * a_km;//equation 19, d^{k+1}_{k, m}(beta)
			if(j == k+1) return d_k1km;//if j == k + 1 we're done

"""

@torch.jit.script
def d(j: torch.Tensor,
      k: torch.Tensor,
      m: torch.Tensor,
      t: torch.Tensor,
      nB: bool
      ) -> torch.Tensor:
    """
    Calculate the Wigner D coefficient for j, k, m

    Args:
        j: torch.Tensor
            The degree of the Wigner coefficient
        k: torch.Tensor
            The primary order of the Wigner coefficient
        m: torch.Tensor
            The secondary order of the Wigner coefficient
        t: torch.Tensor
            cos(BETA)
        nB: bool
            True if BETA is negative, False otherwise

    Returns:
        coefficient d_jkm

    """
    # if nB:
    #     return d(j, m, k, t, False)
    # elif k < 0 and m < 0:
    #     sign = 1 if (k-m) % 2 == 0 else -1
    #     return d(j, -k, -m,  t, False) * sign
    # elif m < 0:
    #     sign = 1 if (j+k) % 2 == 0 else -1
    #     return d(j,  k, -m, -t, False) * sign
    # elif k < 0:
    #     sign = 1 if (j+m) % 2 == 0 else -1
    #     return d(j, -k,  m, -t, False) * sign
    # elif k < m:
    #     sign = 1 if (k-m) % 2 == 0 else -1
    #     return d(j,  m,  k,  t, False) * sign
    
    if j < k:
        return torch.tensor([np.nan], dtype=torch.float16)
    
    #determine if beta is < (0), > (2), or = (1) to pi/2
    beta_type = 0 if t > 0 else (2 if t < 0 else 1)
    tc = 1 - t

    # compute powers of cos/sin of beta / 2
    c2 = ((1 + t) / 2)**0.5
    s2 = ((1 - t) / 2)**0.5

    # use recursion to compute c2**(k+m) and s2**(k-m)

    # cn = c2**(k+m)
    cn = torch.pow(c2, k+m)
    # sn = s2**(k-m)
    sn = torch.pow(s2, k-m)


    d_kkm = cn * sn * e_km(k, m)
    if j == k:
        return d_kkm

    if beta_type == 0:
        a_km = a_km_0(k, m, tc)
    elif beta_type == 1:
        a_km = a_km_1(k, m)
    else:
        a_km = a_km_2(k, m, t)

    d_k1km = d_kkm * a_km

    if j == k+1:
        return d_k1km
    
    """
    Implementations of these lines:

    //recursively compute by degree to j
			Real d_ikm;
			Real d_i2km = d_kkm ;
			Real d_i1km = d_k1km;
			switch(type) {
				case 0://beta <  pi/2
					for(int64_t i = k + 2; i <= j; i++) {
						d_ikm = a_jkm_0<Real>(i, k, m, tc) * d_i1km - b_jkm<Real>(i, k, m) * d_i2km;//equation 10, d^i_{k, m}(beta)
						d_i2km = d_i1km;
						d_i1km = d_ikm ;
					}
					break;
				case 1://beta == pi/2
					for(int64_t i = k + 2; i <= j; i++) {
						d_ikm = a_jkm_1<Real>(i, k, m    ) * d_i1km - b_jkm<Real>(i, k, m) * d_i2km;//equation 10, d^i_{k, m}(beta)
						d_i2km = d_i1km;
						d_i1km = d_ikm ;
					}
					break;
				case 2://beta >  pi/2
					for(int64_t i = k + 2; i <= j; i++) {
						d_ikm = a_jkm_2<Real>(i, k, m, t ) * d_i1km - b_jkm<Real>(i, k, m) * d_i2km;//equation 10, d^i_{k, m}(beta)
						d_i2km = d_i1km;
						d_i1km = d_ikm ;
					}
					break;
			}
			return d_ikm;

    """

    d_ikm = torch.tensor([0.0], dtype=torch.float64)
    d_i2km = d_kkm
    d_i1km = d_k1km

    if beta_type == 0:
        for i in range(int(k+2), int(j+1)):
            i = torch.tensor([i], dtype=torch.int64)
            d_ikm = a_jkm_0(i, k, m, tc) * d_i1km - b_jkm(i, k, m) * d_i2km
            d_i2km = d_i1km
            d_i1km = d_ikm

    elif beta_type == 1:
        for i in range(int(k+2), int(j+1)):
            i = torch.tensor([i], dtype=torch.int64)
            d_ikm = a_jkm_1(i, k, m) * d_i1km - b_jkm(i, k, m) * d_i2km
            d_i2km = d_i1km
            d_i1km = d_ikm

    else:
        for i in range(int(k+2), int(j+1)):
            i = torch.tensor([i], dtype=torch.int64)
            d_ikm = a_jkm_2(i, k, m, t) * d_i1km - b_jkm(i, k, m) * d_i2km
            d_i2km = d_i1km
            d_i1km = d_ikm

    return d_ikm


# # test out computing the Wigner d coefficient
# j = torch.tensor([365], dtype=torch.int64)
# k = torch.tensor([102], dtype=torch.int64)
# m = torch.tensor([20], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 8161.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct   : " + "{:.20f}".format(-4.23570250037880395095020243575390e-02))
# print("Computed  : " + "{:.20f}".format(d_value))
# print("Difference: " + "{:.20f}".format(abs(d_value + 4.23570250037880395095020243575390e-02)))
# print(" ------------------- ")

# # test out computing the Wigner d coefficient
# j = torch.tensor([294], dtype=torch.int64)
# k = torch.tensor([247], dtype=torch.int64)
# m = torch.tensor([188], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 7417.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct   : " + "{:.20f}".format(-1.11943794723176255836019618855372e-01))
# print("Computed  : " + "{:.20f}".format(d_value))
# print("Difference: " + "{:.20f}".format(abs(d_value + 1.11943794723176255836019618855372e-01)))
# print(" ------------------- ")

# # test out computing the Wigner d coefficient 
# j = torch.tensor([3777], dtype=torch.int64)
# k = torch.tensor([1014], dtype=torch.int64)
# m = torch.tensor([690], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 12233.0 / 16384.0], dtype=torch.float64)
# beta = torch.tensor([beta], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct   : " + "{:.20f}".format(1.68450832524798173944840155878705e-03))
# print("Computed  : " + "{:.20f}".format(d_value))
# print("Difference: " + "{:.20f}".format(abs(d_value - 1.68450832524798173944840155878705e-03)))
# print(" ------------------- ")


# # print out value of d(1, 0, 0, t, False) for t = cos(7117.0 / 16384.0)

# # test out computing the Wigner d coefficient
# j = torch.tensor([0], dtype=torch.int64)
# k = torch.tensor([0], dtype=torch.int64)
# m = torch.tensor([0], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 7117.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct for 0 0 0 : " + "{:.20f}".format(d_value))

# # test out computing the Wigner d coefficient
# j = torch.tensor([1], dtype=torch.int64)
# k = torch.tensor([0], dtype=torch.int64)
# m = torch.tensor([0], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 7117.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct for 1 0 0 : " + "{:.20f}".format(d_value))

# # test out computing the Wigner d coefficient
# j = torch.tensor([2], dtype=torch.int64)
# k = torch.tensor([0], dtype=torch.int64)
# m = torch.tensor([0], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 7117.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct for 2 0 0 : " + "{:.20f}".format(d_value))

# # test out computing the Wigner d coefficient
# j = torch.tensor([5], dtype=torch.int64)
# k = torch.tensor([5], dtype=torch.int64)
# m = torch.tensor([0], dtype=torch.int64)
# beta = torch.tensor([torch.pi * 7117.0 / 16384.0], dtype=torch.float64)
# t = torch.cos(beta)
# d_value = d(j, k, m, t, False).item()
# print("Correct  for 5 5 0: " + "{:.20f}".format(d_value))

# for k in range(1):
#     for m in range(1):
#         for j in range(3):
#             if abs(m) <= j and abs(k) <= j and abs(m) <= abs(k):
#                 # doing equivalent of print(f"d({j}, {k}, {m}) = {wigner_d_naive_v3(j, k, m)(np.pi / 4.0)}")
#                 j = torch.tensor([j], dtype=torch.int64)
#                 k = torch.tensor([k], dtype=torch.int64)
#                 m = torch.tensor([m], dtype=torch.int64)
#                 beta = torch.tensor([torch.pi * 7417.0 / 16384.0,], dtype=torch.float64)
#                 t = torch.cos(beta)
#                 d_value = d(j, k, m, t, False)
#                 print(f'd({j.item()}, {k.item()}, {m.item()}) = {d_value.item()}')


test_parameters = [
    # (400, 400, 400, 1.0, 0.0),
    (365, 102, 20, -4.23570250037880395095020243575390e-02, 8161.0 / 16384.0),
    (294, 247, 188, -1.11943794723176255836019618855372e-01, 7417.0 / 16384.0),
]

for test_parameter in test_parameters:
    j = torch.tensor([test_parameter[0],], dtype=torch.int64)
    k = torch.tensor([test_parameter[1],], dtype=torch.int64)
    m = torch.tensor([test_parameter[2],], dtype=torch.int64)
    correct_value = test_parameter[3]
    beta = torch.tensor([torch.pi * test_parameter[4],], dtype=torch.float64)
    t = torch.cos(beta)
    d_value = d(j, k, m, t, False).item()
    print("Correct   : " + "{:.20f}".format(correct_value))
    print("Computed  : " + "{:.20f}".format(d_value))
    print("Difference: " + "{:.20f}".format(abs(d_value - correct_value)))
    print(" ------------------- ")
    