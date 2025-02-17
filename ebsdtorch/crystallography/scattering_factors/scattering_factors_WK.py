"""
:Author: Zachary T. Varley
:Date: December 2024
:License: MIT

______________________________________________________________________________
Rational Coefficients for Exponential Integrals E1(x) and Ei(x) come from Boost:
https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/expint.hpp

and fall under the Boost Software License:

Copyright John Maddock 2007. 

Copyright Matt Borland 2024. 

Use, modification and distribution are subject to the Boost Software License,
Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
______________________________________________________________________________

This module provides a PyTorch implementation of the Born-WK approximation of
the imaginary portion of the electron scattering factor:

Weickenmeier, A., and H. Kohl. “Computation of Absorptive Form Factors for
High-Energy Electron Diffraction.” Acta Crystallographica Section A Foundations
of Crystallography, vol. 47, no. 5, Sept. 1991, pp. 590–97. DOI.org (Crossref),
https://doi.org/10.1107/S0108767391004804.

If in the future torch.special adds a compiled expi, that should be used. As far
as I can tell, the WK parameterization is not worth it for the real part of the
scattering factor due to fitting artifacts from deliberately using exponential
fittings for an analytical form of the imaginary part for TDS. The imaginary
part of the scattering factor is the same for WK and manually (very slow)
integrating the LVD fits. I plan to use LVD for real part and WK for imaginary
part in a hybrid scattering factor function. I want to compare this to the
optical-theorem adherent values from the Moliere integral and see if TDS can
easily be incorporated to Moliere's Bessel function integral.

We clamp the imaginary part of the scattering factor to zero, quoting section 2
of the paper:

Thomas, M., et al. “Parameterized Absorptive Electron Scattering Factors.” Acta
Crystallographica Section A Foundations and Advances, vol. 80, no. 2, Mar. 2024,
pp. 146–50. DOI.org (Crossref), https://doi.org/10.1107/S2053273323010963.

"In equation (1), multiplying by the temperature factor, exp(-B_iso s^2),
ensures that the function smoothly asymptotes to zero. Without it, the
absorptive form factor instead behaves as -exp(s) for large s paired with any
B_iso > 0.05 A^2 (Peng et al., 1996b). This results in large negative βf' at
large s, which implies amplification, rather than absorption, of the electron
beam and we consider this to be unphysical. We therefore set βf' to zero where
equation (2) returns a negative value."

"""

import torch
from torch import Tensor
from typing import List, Union

wk_A_param = torch.tensor(
    [
        [0.00427, 0.00957, 0.00802, 0.00209],
        [0.01217, 0.02616, -0.00884, 0.01841],
        [0.00251, 0.03576, 0.00988, 0.02370],
        [0.01596, 0.02959, 0.04024, 0.01001],
        [0.03652, 0.01140, 0.05677, 0.01506],
        [0.04102, 0.04911, 0.05296, 0.00061],
        [0.04123, 0.05740, 0.06529, 0.00373],
        [0.03547, 0.03133, 0.10865, 0.01615],
        [0.03957, 0.07225, 0.09581, 0.00792],
        [0.02597, 0.02197, 0.13762, 0.05394],
        [0.03283, 0.08858, 0.11688, 0.02516],
        [0.03833, 0.17124, 0.03649, 0.04134],
        [0.04388, 0.17743, 0.05047, 0.03957],
        [0.03812, 0.17833, 0.06280, 0.05605],
        [0.04166, 0.17817, 0.09479, 0.04463],
        [0.04003, 0.18346, 0.12218, 0.03753],
        [0.04245, 0.17645, 0.15814, 0.03011],
        [0.05011, 0.16667, 0.17074, 0.04358],
        [0.04058, 0.17582, 0.20943, 0.02922],
        [0.04001, 0.17416, 0.20986, 0.05497],
        [0.09685, 0.14777, 0.20981, 0.04852],
        [0.06667, 0.17356, 0.22710, 0.05957],
        [0.05118, 0.16791, 0.26700, 0.06476],
        [0.03204, 0.18460, 0.30764, 0.05052],
        [0.03866, 0.17782, 0.31329, 0.06898],
        [0.05455, 0.16660, 0.33208, 0.06947],
        [0.05942, 0.17472, 0.34423, 0.06828],
        [0.06049, 0.16600, 0.37302, 0.07109],
        [0.08034, 0.15838, 0.40116, 0.05467],
        [0.02948, 0.19200, 0.42222, 0.07480],
        [0.16157, 0.32976, 0.18964, 0.06148],
        [0.16184, 0.35705, 0.17618, 0.07133],
        [0.06190, 0.18452, 0.41600, 0.12793],
        [0.15913, 0.41583, 0.13385, 0.10549],
        [0.16514, 0.41202, 0.12900, 0.13209],
        [0.15798, 0.41181, 0.14254, 0.14987],
        [0.16535, 0.44674, 0.24245, 0.03161],
        [0.16039, 0.44470, 0.24661, 0.05840],
        [0.16619, 0.44376, 0.25613, 0.06797],
        [0.16794, 0.44505, 0.27188, 0.07313],
        [0.16552, 0.45008, 0.30474, 0.06161],
        [0.17327, 0.44679, 0.32441, 0.06143],
        [0.16424, 0.45046, 0.33749, 0.07766],
        [0.18750, 0.44919, 0.36323, 0.05388],
        [0.16081, 0.45211, 0.40343, 0.06140],
        [0.16599, 0.43951, 0.41478, 0.08142],
        [0.16547, 0.44658, 0.45401, 0.05959],
        [0.17154, 0.43689, 0.46392, 0.07725],
        [0.15752, 0.44821, 0.48186, 0.08596],
        [0.15732, 0.44563, 0.48507, 0.10948],
        [0.16971, 0.42742, 0.48779, 0.13653],
        [0.14927, 0.43729, 0.49444, 0.16440],
        [0.18053, 0.44724, 0.48163, 0.15995],
        [0.13141, 0.43855, 0.50035, 0.22299],
        [0.31397, 0.55648, 0.39828, 0.04852],
        [0.32756, 0.53927, 0.39830, 0.07607],
        [0.30887, 0.53804, 0.42265, 0.09559],
        [0.28398, 0.53568, 0.46662, 0.10282],
        [0.35160, 0.56889, 0.42010, 0.07246],
        [0.33810, 0.58035, 0.44442, 0.07413],
        [0.35449, 0.59626, 0.43868, 0.07152],
        [0.35559, 0.60598, 0.45165, 0.07168],
        [0.38379, 0.64088, 0.41710, 0.06708],
        [0.40352, 0.64303, 0.40488, 0.08137],
        [0.36838, 0.64761, 0.47222, 0.06854],
        [0.38514, 0.68422, 0.44359, 0.06775],
        [0.37280, 0.67528, 0.47337, 0.08320],
        [0.39335, 0.70093, 0.46774, 0.06658],
        [0.40587, 0.71223, 0.46598, 0.06847],
        [0.39728, 0.73368, 0.47795, 0.06759],
        [0.40697, 0.73576, 0.47481, 0.08291],
        [0.40122, 0.78861, 0.44658, 0.08799],
        [0.41127, 0.76965, 0.46563, 0.10180],
        [0.39978, 0.77171, 0.48541, 0.11540],
        [0.39130, 0.80752, 0.48702, 0.11041],
        [0.40436, 0.80701, 0.48445, 0.12438],
        [0.38816, 0.80163, 0.51922, 0.13514],
        [0.39551, 0.80409, 0.53365, 0.13485],
        [0.40850, 0.83052, 0.53325, 0.11978],
        [0.40092, 0.85415, 0.53346, 0.12747],
        [0.41872, 0.88168, 0.54551, 0.09404],
        [0.43358, 0.88007, 0.52966, 0.12059],
        [0.40858, 0.87837, 0.56392, 0.13698],
        [0.41637, 0.85094, 0.57749, 0.16700],
        [0.38951, 0.83297, 0.60557, 0.20770],
        [0.41677, 0.88094, 0.55170, 0.21029],
        [0.50089, 1.00860, 0.51420, 0.05996],
        [0.47470, 0.99363, 0.54721, 0.09206],
        [0.47810, 0.98385, 0.54905, 0.12055],
        [0.47903, 0.97455, 0.55883, 0.14309],
        [0.48351, 0.98292, 0.58877, 0.12425],
        [0.48664, 0.98057, 0.61483, 0.12136],
        [0.46078, 0.97139, 0.66506, 0.13012],
        [0.49148, 0.98583, 0.67674, 0.09725],
        [0.50865, 0.98574, 0.68109, 0.09977],
        [0.46259, 0.97882, 0.73056, 0.12723],
        [0.46221, 0.95749, 0.76259, 0.14086],
        [0.48500, 0.95602, 0.77234, 0.13374],
    ]
)

wk_B_param = torch.tensor(
    [
        [4.17218, 16.05892, 26.78365, 69.45643],
        [1.83008, 7.20225, 16.13585, 18.75551],
        [0.02620, 2.00907, 10.80597, 130.49226],
        [0.38968, 1.99268, 46.86913, 108.84167],
        [0.50627, 3.68297, 27.90586, 74.98296],
        [0.41335, 10.98289, 34.80286, 177.19113],
        [0.29792, 7.84094, 22.58809, 72.59254],
        [0.17964, 2.60856, 11.79972, 38.02912],
        [0.16403, 3.96612, 12.43903, 40.05053],
        [0.09101, 0.41253, 5.02463, 17.52954],
        [0.06008, 2.07182, 7.64444, 146.00952],
        [0.07424, 2.87177, 18.06729, 97.00854],
        [0.09086, 2.53252, 30.43883, 98.26737],
        [0.05396, 1.86461, 22.54263, 72.43144],
        [0.05564, 1.62500, 24.45354, 64.38264],
        [0.05214, 1.40793, 23.35691, 53.59676],
        [0.04643, 1.15677, 19.34091, 52.88785],
        [0.07991, 1.01436, 15.67109, 39.60819],
        [0.03352, 0.82984, 14.13679, 200.97722],
        [0.02289, 0.71288, 11.18914, 135.02390],
        [0.12527, 1.34248, 12.43524, 131.71112],
        [0.05198, 0.86467, 10.59984, 103.56776],
        [0.03786, 0.57160, 8.30305, 91.78068],
        [0.00240, 0.44931, 7.92251, 86.64058],
        [0.01836, 0.41203, 6.73736, 76.30466],
        [0.03947, 0.43294, 6.26864, 71.29470],
        [0.03962, 0.43253, 6.05175, 68.72437],
        [0.03558, 0.39976, 5.36660, 62.46894],
        [0.05475, 0.45736, 5.38252, 60.43276],
        [0.00137, 0.26535, 4.48040, 54.26088],
        [0.10455, 2.18391, 9.04125, 75.16958],
        [0.09890, 2.06856, 9.89926, 68.13783],
        [0.01642, 0.32542, 3.51888, 44.50604],
        [0.07669, 1.89297, 11.31554, 46.32082],
        [0.08199, 1.76568, 9.87254, 38.10640],
        [0.06939, 1.53446, 8.98025, 33.04365],
        [0.07044, 1.59236, 17.53592, 215.26198],
        [0.06199, 1.41265, 14.33812, 152.80257],
        [0.06364, 1.34205, 13.66551, 125.72522],
        [0.06565, 1.25292, 13.09355, 109.50252],
        [0.05921, 1.15624, 13.24924, 98.69958],
        [0.06162, 1.11236, 12.76149, 90.92026],
        [0.05081, 0.99771, 11.28925, 84.28943],
        [0.05120, 1.08672, 12.23172, 85.27316],
        [0.04662, 0.85252, 10.51121, 74.53949],
        [0.04933, 0.79381, 9.30944, 41.17414],
        [0.04481, 0.75608, 9.34354, 67.91975],
        [0.04867, 0.71518, 8.40595, 64.24400],
        [0.03672, 0.64379, 7.83687, 73.37281],
        [0.03308, 0.60931, 7.04977, 64.83582],
        [0.04023, 0.58192, 6.29247, 55.57061],
        [0.02842, 0.50687, 5.60835, 48.28004],
        [0.03830, 0.58340, 6.47550, 47.08820],
        [0.02097, 0.41007, 4.52105, 37.18178],
        [0.07813, 1.45053, 15.05933, 199.48830],
        [0.08444, 1.40227, 13.12939, 160.56676],
        [0.07206, 1.19585, 11.55866, 127.31371],
        [0.05717, 0.98756, 9.95556, 117.31874],
        [0.08249, 1.43427, 12.37363, 150.55968],
        [0.07081, 1.31033, 11.44403, 144.17706],
        [0.07442, 1.38680, 11.54391, 143.72185],
        [0.07155, 1.34703, 11.00432, 140.09138],
        [0.07794, 1.55042, 11.89283, 142.79585],
        [0.08508, 1.60712, 11.45367, 116.64063],
        [0.06520, 1.32571, 10.16884, 134.69034],
        [0.06850, 1.43566, 10.57719, 131.88972],
        [0.06264, 1.26756, 9.46411, 107.50194],
        [0.06750, 1.35829, 9.76480, 127.40374],
        [0.06958, 1.38750, 9.41888, 122.10940],
        [0.06574, 1.31578, 9.13448, 120.98209],
        [0.06517, 1.29452, 8.67569, 100.34878],
        [0.06213, 1.30860, 9.18871, 91.20213],
        [0.06292, 1.23499, 8.42904, 77.59815],
        [0.05693, 1.15762, 7.83077, 67.14066],
        [0.05145, 1.11240, 8.33441, 65.71782],
        [0.05573, 1.11159, 8.00221, 57.35021],
        [0.04855, 0.99356, 7.38693, 51.75829],
        [0.04981, 0.97669, 7.38024, 44.52068],
        [0.05151, 1.00803, 8.03707, 45.01758],
        [0.04693, 0.98398, 7.83562, 46.51474],
        [0.05161, 1.02127, 9.18455, 64.88177],
        [0.05154, 1.03252, 8.49678, 58.79463],
        [0.04200, 0.90939, 7.71158, 57.79178],
        [0.04661, 0.87289, 6.84038, 51.36000],
        [0.04168, 0.73697, 5.86112, 43.78613],
        [0.04488, 0.83871, 6.44020, 43.51940],
        [0.05786, 1.20028, 13.85073, 172.15909],
        [0.05239, 1.03225, 11.49796, 143.12303],
        [0.05167, 0.98867, 10.52682, 112.18267],
        [0.04931, 0.95698, 9.61135, 95.44649],
        [0.04748, 0.93369, 9.89867, 102.06961],
        [0.04660, 0.89912, 9.69785, 100.23434],
        [0.04323, 0.78798, 8.71624, 92.30811],
        [0.04641, 0.85867, 9.51157, 111.02754],
        [0.04918, 0.87026, 9.41105, 104.98576],
        [0.03904, 0.72797, 8.00506, 86.41747],
        [0.03969, 0.68167, 7.29607, 75.72682],
        [0.04291, 0.69956, 7.38554, 77.18528],
    ]
)


@torch.jit.script
def evaluate_polynomial(coeffs: List[float], x: Tensor) -> Tensor:
    """Evaluate polynomial using Horner's method"""
    result = torch.full_like(x, coeffs[-1])
    for i in range(len(coeffs) - 2, -1, -1):
        result = (result * x) + coeffs[i]
    return result


@torch.jit.script
def exp1(x: Tensor) -> Tensor:
    """
    Compute the exponential integral E1(x) for real x.
    This is a PyTorch port of the rational approximation implementation.

    """
    # don't fill tensor to save allocation overhead
    result = torch.empty_like(x)

    # Handle special cases
    result = torch.where(x == 0, torch.tensor(float("inf")), result)
    result = torch.where(x < 0, torch.tensor(float("nan")), result)

    mask_range1 = (x > 0) & (x <= 1)
    if mask_range1.any():
        Y = 0.66373538970947265625
        P = [
            0.0865197248079397976498,
            0.0320913665303559189999,
            -0.245088216639761496153,
            -0.0368031736257943745142,
            -0.00399167106081113256961,
            -0.000111507792921197858394,
        ]
        Q = [
            1.0,
            0.37091387659397013215,
            0.056770677104207528384,
            0.00427347600017103698101,
            0.000131049900798434683324,
            -0.528611029520217142048e-6,
        ]

        x_masked1 = x[mask_range1]
        range1_result = evaluate_polynomial(P, x_masked1) / evaluate_polynomial(
            Q, x_masked1
        )
        range1_result = range1_result + x_masked1 - torch.log(x_masked1) - Y
        result[mask_range1] = range1_result

    mask_range2 = x > 1
    if mask_range2.any():
        P = [
            -0.121013190657725568138e-18,
            -0.999999999999998811143,
            -43.3058660811817946037,
            -724.581482791462469795,
            -6046.8250112711035463,
            -27182.6254466733970467,
            -66598.2652345418633509,
            -86273.1567711649528784,
            -54844.4587226402067411,
            -14751.4895786128450662,
            -1185.45720315201027667,
        ]
        Q = [
            1.0,
            45.3058660811801465927,
            809.193214954550328455,
            7417.37624454689546708,
            38129.5594484818471461,
            113057.05869159631492,
            192104.047790227984431,
            180329.498380501819718,
            86722.3403467334749201,
            18455.4124737722049515,
            1229.20784182403048905,
            -0.776491285282330997549,
        ]

        x_masked2 = x[mask_range2]
        recip = 1 / x_masked2
        range2_result = 1 + evaluate_polynomial(P, recip) / evaluate_polynomial(
            Q, recip
        )
        range2_result = range2_result * torch.exp(-x_masked2) * recip
        result[mask_range2] = range2_result

    return result


@torch.jit.script
def expi(x: Tensor) -> Tensor:
    """
    Compute the exponential integral Ei(x) for real x.
    PyTorch implementation with TorchScript support.
    """
    result = torch.zeros_like(x)

    # Handle special cases
    result = torch.where(x == 0, torch.tensor(float("-inf")), result)
    neg_mask = x < 0
    if neg_mask.any():
        result[neg_mask] = -exp1(-x[neg_mask])

    # Constants for different ranges
    mask_range1 = (x > 0) & (x <= 6)
    if mask_range1.any():
        Y = 0.66373538970947265625
        P = [
            2.98677224343598593013,
            0.356343618769377415068,
            0.780836076283730801839,
            0.114670926327032002811,
            0.0499434773576515260534,
            0.00726224593341228159561,
            0.00115478237227804306827,
            0.000116419523609765200999,
            0.798296365679269702435e-5,
            0.2777056254402008721e-6,
        ]
        Q = [
            1.0,
            -1.17090412365413911947,
            0.62215109846016746276,
            -0.195114782069495403315,
            0.0391523431392967238166,
            -0.00504800158663705747345,
            0.000389034007436065401822,
            -0.138972589601781706598e-4,
        ]

        r = 0.372507410781366634461991866580119133535689497771654051555657435242200120636201854384926049951548942392
        x_masked1 = x[mask_range1]
        t = (x_masked1 / 3) - 1
        range1_result = evaluate_polynomial(P, t) / evaluate_polynomial(Q, t)
        t = x_masked1 - r
        range1_result = range1_result * t

        # Use torch.where for the conditional logic
        small_t_mask = torch.abs(t) < 0.1
        range1_result = torch.where(
            small_t_mask,
            range1_result + torch.log1p(t / r),
            range1_result + torch.log(x_masked1 / r),
        )
        result[mask_range1] = range1_result

    mask_range2 = (x > 6) & (x <= 10)
    if mask_range2.any():
        Y = 1.158985137939453125
        P = [
            0.00139324086199402804173,
            -0.0349921221823888744966,
            -0.0264095520754134848538,
            -0.00761224003005476438412,
            -0.00247496209592143627977,
            -0.000374885917942100256775,
            -0.554086272024881826253e-4,
            -0.396487648924804510056e-5,
        ]
        Q = [
            1.0,
            0.744625566823272107711,
            0.329061095011767059236,
            0.100128624977313872323,
            0.0223851099128506347278,
            0.00365334190742316650106,
            0.000402453408512476836472,
            0.263649630720255691787e-4,
        ]

        x_masked2 = x[mask_range2]
        t = x_masked2 / 2 - 4
        range2_result = Y + evaluate_polynomial(P, t) / evaluate_polynomial(Q, t)
        range2_result = range2_result * torch.exp(x_masked2) / x_masked2
        range2_result = range2_result + x_masked2
        result[mask_range2] = range2_result

    mask_range3 = (x > 10) & (x <= 20)
    if mask_range3.any():
        Y = 1.086973190307617188
        P = [
            -0.00893891094356945667451,
            -0.0484607730127134045806,
            -0.0652810444222236895772,
            -0.0478447572647309671455,
            -0.0226059218923777094596,
            -0.00720603636917482065907,
            -0.00155941947035972031334,
            -0.000209750022660200888349,
            -0.138652200349182596186e-4,
        ]
        Q = [
            1.0,
            1.97017214039061194971,
            1.86232465043073157508,
            1.09601437090337519977,
            0.438873285773088870812,
            0.122537731979686102756,
            0.0233458478275769288159,
            0.00278170769163303669021,
            0.000159150281166108755531,
        ]

        x_masked3 = x[mask_range3]
        t = x_masked3 / 5 - 3
        range3_result = Y + evaluate_polynomial(P, t) / evaluate_polynomial(Q, t)
        range3_result = range3_result * torch.exp(x_masked3) / x_masked3
        range3_result = range3_result + x_masked3
        result[mask_range3] = range3_result

    # z > 20
    mask_range4 = x > 20
    if mask_range4.any():
        Y = 1.013065338134765625
        P = [
            -0.0130653381347656243849,
            0.19029710559486576682,
            94.7365094537197236011,
            -2516.35323679844256203,
            18932.0850014925993025,
            -38703.1431362056714134,
        ]
        Q = [
            1.0,
            61.9733592849439884145,
            -2354.56211323420194283,
            22329.1459489893079041,
            -70126.245140396567133,
            54738.2833147775537106,
            8297.16296356518409347,
        ]

        x_masked4 = x[mask_range4]
        t = 1.0 / x_masked4
        range4_result = Y + evaluate_polynomial(P, t) / evaluate_polynomial(Q, t)
        range4_result *= torch.exp(x_masked4) / x_masked4
        range4_result += x_masked4
        result[mask_range4] = range4_result

    return result


@torch.jit.script
def RIH2_torch(X: Tensor) -> Tensor:
    # Define constants
    rih2_table = torch.tensor(
        [
            1.000000,
            1.005051,
            1.010206,
            1.015472,
            1.020852,
            1.026355,
            1.031985,
            1.037751,
            1.043662,
            1.049726,
            1.055956,
            1.062364,
            1.068965,
            1.075780,
            1.082830,
            1.090140,
            1.097737,
            1.105647,
            1.113894,
            1.122497,
            1.131470,
        ],
        dtype=X.dtype,
        device=X.device,
    )
    rih2_table = rih2_table.to(X.device).to(X.dtype)
    idx = (200.0 / X).floor().long()
    idx = torch.clamp(idx, 0, len(rih2_table) - 2)
    sig = rih2_table[idx] + 200.0 * (rih2_table[idx + 1] - rih2_table[idx]) * (
        1.0 / X - 0.5e-3 * idx
    )
    return sig


@torch.jit.script
def RIH1_torch_where(X1: Tensor, X2: Tensor, X3: Tensor) -> Tensor:
    """
    Compute RIH1 for tensors of any shape, with broadcasting.
    Use where() to handle different cases.
    """
    # Case 1: X2 <= 20.0 and X3 <= 20.0
    mask1 = (X2 <= 20.0) & (X3 <= 20.0)
    mask2 = (X2 > 20.0) & (X3 <= 20.0)
    mask3 = (X2 <= 20.0) & (X3 > 20.0)

    return torch.where(
        mask1,
        torch.exp(-X1) * (expi(X2) - expi(X3)),
        torch.where(
            mask2,
            torch.exp(X2 - X1) * RIH2_torch(X2) / X2 - torch.exp(-X1) * expi(X3),
            torch.where(
                mask3,
                torch.exp(-X1) * expi(X2) - torch.exp(X3 - X1) * RIH2_torch(X3) / X3,
                torch.exp(X2 - X1) * RIH2_torch(X2) / X2
                - torch.exp(X3 - X1) * RIH2_torch(X3) / X3,
            ),
        ),
    )


@torch.jit.script
def e_wavelength_eV_to_A(E_eV: Tensor) -> Tensor:
    """
    Calculate electron wavelength in Angstroms using PyTorch, performed in log
    space to prevent any numerical issues with constants.

    """
    # log(λ) = log(h) - 0.5 * log(2me*E_eV) - 0.5 * log(1 + eE/2mc^2) + log(1e10)
    # log(λ) = log(h) - 0.5 * log(2me) + log(1e10) - 0.5 * log(E_eV) - 0.5 * log(1 + e*E_eV/2mc^2)
    # log(λ) = 2.5066892156586356 - 0.5 * log(E_eV) - 0.5 * log(1 + exp(log(E_eV) - 13.83726968949393))
    # log(λ) = 2.5066892156586356 - 0.5 * log(E_eV) - 0.5 * log(1 + E_eV * 9.7847589066012e-07)
    log_wavelength = (
        2.5066892156586356
        - 0.5 * torch.log(E_eV)
        - 0.5 * torch.log1p(E_eV * 9.7847589066012e-07)
    )
    wavelength = torch.exp(log_wavelength)
    return wavelength


# @torch.jit.script # takes too long to compile
def scatter_factor_WK(
    g: Tensor,  # [N_g]
    Z: Union[int, Tensor],  # [N_Z]
    thermal_sigma: Union[float, Tensor],  # [N_Z]
    voltage_kV: Union[float, Tensor],  # [N_V]
    wk_A_param: Tensor = wk_A_param,  # [98, 4]
    wk_B_param: Tensor = wk_B_param,  # [98, 4]
    include_core: bool = True,
    include_phonon: bool = True,
    return_type: str = "EMsoft",  # "EMsoft" or "ITC"
) -> Tensor:
    """
    Compute the Weickenmeier-Kohl atomic scattering factors for electron
    scattering.

    Args:
        :g:
            Tensor of scattering magnitudes (1/Å) shape [N_g]
        :Z:
            Atomic number(s) (1-98) shape [N_Z] or scalar
        :thermal_sigma:
            RMS displacement for each atom shape [N_Z] or scalar
        :accelerating_voltage:
            Voltage values in eV shape [N_V] or scalar
        :wk_A_param:
            A parameters shape [98, 4]
        :wk_B_param:
            B parameters shape [98, 4]
        :include_core:
            Include core loss contribution
        :include_phonon:
            Include phonon/TDS contribution
        :return_type:
            "Convention" for the scatter factors (EMsoft or International Tables
            of Crystallography). ITC is available to match against tabulated
            values here:
            (https://onlinelibrary.wiley.com/iucr/itc/Cb/ch4o3v0001/ch4o3.pdf)

            Default is "EMsoft". ITC is always real valued.

    Returns:
        Tensor of shape [N_V, N_Z, N_g]: complex with "EMsoft" convention or
        real with "ITC" convention. ITC only used for verifying implementation.

    """
    # Convert scalar inputs to tensors and ensure 1D
    if isinstance(voltage_kV, (int, float)):
        voltage_kV = torch.tensor([voltage_kV], device=g.device, dtype=g.dtype)
    if isinstance(Z, (int, float)):
        Z = torch.tensor([Z], device=g.device, dtype=torch.int64)
    if isinstance(thermal_sigma, (int, float)):
        thermal_sigma = torch.tensor([thermal_sigma], device=g.device, dtype=g.dtype)

    # if any tensors were more than 1D, raise an error
    if len(g.shape) > 1 or len(Z.shape) > 1 or len(thermal_sigma.shape) > 1:
        raise ValueError(
            f"Input tensors must be scalars or 1D tensors, "
            + "got shapes g: {g.shape}, Z: {Z.shape}, thermal_sigma: {thermal_sigma.shape}"
        )

    # if using ITC convention, set phonon and core to False and let the user know
    if return_type == "ITC":
        include_core = False
        include_phonon = False
        # print(
        #     "return_type='ITC' for real-valued WK factors lacks core / phonon contributions.\n"
        #     + "This is used for comparison against tabulated values of f (Å)\n"
        #     + "for absorptive electron scattering factors set return_type='EMsoft'",
        # )

    # Ensure inputs are 1D
    g = g.view(-1)  # [N_g]
    Z = Z.view(-1)  # [N_Z]
    thermal_sigma = thermal_sigma.view(-1)  # [N_Z]
    voltage_kV = voltage_kV.view(-1)  # [N_V]

    # Get dimensions
    N_V = len(voltage_kV)
    N_Z = len(Z)
    N_g = len(g)

    # Prepare g-vector calculations
    if return_type == "ITC":
        G = g.view(1, 1, -1) * 2.0 * torch.pi  # [1, 1, N_g]
        S = g.view(1, 1, -1) / 2.0  # [1, 1, N_g]
    else:
        # I follow MDG in EMsoft, not py4DSTEM folks here
        # the two pi factor was already removed from G
        G = g.view(1, 1, -1)  # [1, 1, N_g]
        S = g.view(1, 1, -1) / (4.0 * torch.pi)  # [1, 1, N_g] !

    # Ensure wk params on right device/dtype
    wk_A_param = wk_A_param.to(G.device).to(G.dtype)
    wk_B_param = wk_B_param.to(G.device).to(G.dtype)

    # Get parameters for atomic numbers
    A = wk_A_param[Z - 1]  # [N_Z, 4]
    B = wk_B_param[Z - 1]  # [N_Z, 4]

    # Calculate thermal factors
    dwf = torch.exp(-0.5 * thermal_sigma.view(-1, 1) ** 2 * G.view(1, -1) ** 2).view(
        1, N_Z, N_g
    )  # [1, N_Z, N_g]

    # vectorize old naive loop
    S = S.view(1, N_g, 1)  # [1, N_g, 1]
    argu = B.view(N_Z, 1, 4) * S**2  # [N_Z, N_g, 4]
    term1 = A.view(N_Z, 1, 4) * B.view(N_Z, 1, 4) * (1.0 - 0.5 * argu)
    term2 = A.view(N_Z, 1, 4) * (1.0 - torch.exp(-argu)) / S**2
    term3 = A.view(N_Z, 1, 4) / S**2

    # don't multiply by DWF here yet
    # don't include extra 4.0 * torch.pi factor like WK
    f_real = (
        torch.where(
            argu < 0.1,
            term1,
            torch.where(
                argu <= 20.0,
                term2,
                term3,
            ),
        )
        .sum(dim=-1)
        .view(1, N_Z, N_g)
    )  # [1, N_Z, N_g]

    # Calculate wavelength for each voltage
    # k0 = 2.0 * torch.pi / e_wavelength_eV_to_A(1000.0 * voltage_kV)  # [N_V]

    # k0 = .5068 * SQRT(1022.*ACCVLT + ACCVLT*ACCVLT)
    k0 = 0.5068 * torch.sqrt(1022.0 * voltage_kV + voltage_kV * voltage_kV)  # [N_V]

    # Reshape for broadcasting
    k0 = k0.view(-1, 1, 1)  # [N_V, 1, 1]
    voltage_kV = voltage_kV.view(-1, 1, 1)  # [N_V, 1, 1]

    # Core contribution
    if include_core:
        Z_float = Z.float().view(1, -1, 1)  # [1, N_Z, 1]
        A0 = 0.5289
        DE = 6.0e-3 * Z_float
        theta_e = (
            DE
            / (2.0 * voltage_kV)
            * (2.0 * voltage_kV + 1022.0)
            / (voltage_kV + 1022.0)
        )  # [N_V, N_Z, 1]

        R = 0.885 * A0 / torch.pow(Z_float, 1.0 / 3.0)  # [1, N_Z, 1]
        TA = 1.0 / (k0 * R)  # [N_V, N_Z, 1]
        TB = G / (2.0 * k0)  # [N_V, 1, N_g]

        OMEGA = 2.0 * TB / TA  # [N_V, N_Z, N_g]
        KAPPA = theta_e / TA  # [N_V, N_Z, 1]

        K2 = KAPPA * KAPPA
        O2 = OMEGA * OMEGA

        X1 = (
            OMEGA
            / ((1.0 + O2) * torch.sqrt(O2 + 4.0 * K2))
            * torch.log((OMEGA + torch.sqrt(O2 + 4.0 * K2)) / (2.0 * KAPPA))
        )

        X2 = (
            1.0
            / torch.sqrt((1.0 + O2) * (1.0 + O2) + 4.0 * K2 * O2)
            * torch.log(
                (
                    1.0
                    + 2.0 * K2
                    + O2
                    + torch.sqrt((1.0 + O2) * (1.0 + O2) + 4.0 * K2 * O2)
                )
                / (2.0 * KAPPA * torch.sqrt(1.0 + K2))
            )
        )

        X3 = torch.where(
            OMEGA > 1e-2,
            1.0
            / (OMEGA * torch.sqrt(O2 + 4.0 * (1.0 + K2)))
            * torch.log(
                (OMEGA + torch.sqrt(O2 + 4.0 * (1.0 + K2)))
                / (2.0 * torch.sqrt(1.0 + K2))
            ),
            1.0 / (4.0 * (1.0 + K2)),
        )

        HI = 2.0 * Z_float / (TA * TA) * (-X1 + X2 - X3)
        Fcore = 4.0 / (A0 * A0) * 2.0 * torch.pi / (k0 * k0) * HI
    else:
        Fcore = torch.zeros(N_V, N_Z, N_g, dtype=g.dtype, device=g.device)

    if include_phonon:
        # Get indices for the 10 pairs we actually need
        # i_idx, j_idx = torch.triu_indices(ro=4, 4, offset=0, device=g.device)
        i_idx = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], device=g.device, dtype=torch.int64
        )
        j_idx = torch.tensor(
            [0, 1, 2, 3, 1, 2, 3, 2, 3, 3], device=g.device, dtype=torch.int64
        )
        # Get only the parameter pairs we need
        A_i = A[:, i_idx].view(-1, 1, 10) * (4.0 * torch.pi) ** 2  # [N_Z, 1, 10]
        A_j = A[:, j_idx].view(-1, 1, 10) * (4.0 * torch.pi) ** 2  # [N_Z, 1, 10]
        B_i = B[:, i_idx].view(-1, 1, 10) / (4.0 * torch.pi) ** 2  # [N_Z, 1, 10]
        B_j = B[:, j_idx].view(-1, 1, 10) / (4.0 * torch.pi) ** 2  # [N_Z, 1, 10]

        # Calculate thermal factor
        M = 0.5 * thermal_sigma.view(-1, 1, 1) ** 2  # [N_Z, 1, 1]
        G2 = G.view(1, -1, 1) ** 2  # [1, N_g, 1]
        exp_DW = torch.exp(-M * G2)  # [N_Z, N_g, 1]

        # Apply A coefficients and thermal factor
        result = torch.zeros(N_Z, N_g, 10, dtype=g.dtype, device=g.device)

        # Combine according to eq 15
        # Create multiplier for i≠j terms (2.0 where i≠j, 1.0 where i=j)
        multiplier = torch.where(i_idx == j_idx, 1.0, 2.0).view(1, 1, -1)

        # Handle g=0 case
        g_zero = g == 0

        if g_zero.any():
            # I¹ᵢⱼ(0) from eq 18
            I1_0 = torch.pi * (
                B_i * torch.log((B_i + B_j) / B_i) + B_j * torch.log((B_i + B_j) / B_j)
            )

            # I²ᵢⱼ(M,0) from eq 19
            I2_0 = torch.pi * (
                (B_i + 2 * M) * torch.log((B_i + B_j + 2 * M) / (B_i + 2 * M))
                + B_j * torch.log((B_i + B_j + 2 * M) / (B_j + 2 * M))
                + 2 * M * torch.log(2 * M / (B_j + 2 * M))
            )

            result[:, g_zero, :] = (
                multiplier * A_i * A_j * (exp_DW[:, g_zero, :] * I1_0 - I2_0)
            )

        # Handle g≠0 case
        g_nonzero = ~g_zero
        if g_nonzero.any():
            # I¹ᵢⱼ(g) from eq 20
            G2_NZ = G2[:, g_nonzero]  # [1, N_g (nonzero), 1]
            pi_by_g2 = torch.pi / G2_NZ
            BiBj_term = -B_i * B_j * G2_NZ / (B_i + B_j)
            Bi2_term = B_i * B_i * G2_NZ / (B_i + B_j)
            Bj2_term = B_j * B_j * G2_NZ / (B_i + B_j)

            I1_g = pi_by_g2 * (
                2 * 0.57721566490153286060651209008240243  # 2C (Euler constant)
                + torch.log(B_i * G2_NZ)
                + torch.log(B_j * G2_NZ)
                - 2.0 * expi(BiBj_term)
                # # This is not numerically stable, so we use RIH1_torch_where
                # + torch.exp(-B_i * G2_NZ) * (expi(Bi2_term) - expi(B_i * G2_NZ))
                # + torch.exp(-B_j * G2_NZ) * (expi(Bj2_term) - expi(B_j * G2_NZ))
                + RIH1_torch_where(B_i * G2_NZ, Bi2_term, B_i * G2_NZ)
                + RIH1_torch_where(B_j * G2_NZ, Bj2_term, B_j * G2_NZ)
            )

            # I²ᵢⱼ(M,g) from eq 21
            BiM = B_i + M
            BjM = B_j + M
            BijM = B_i + B_j + 2 * M

            I2_g = pi_by_g2 * (
                2.0
                * (
                    expi(-M * BiM * G2_NZ / (B_i + 2 * M))
                    + expi(-M * BjM * G2_NZ / (B_j + 2 * M))
                    - expi(-BiM * BjM * G2_NZ / BijM)
                    - expi(-0.5 * M * G2_NZ)
                )
                # # This is not numerically stable, so we use RIH1_torch_where
                # + torch.exp(-M * G2_NZ)
                # * (
                #     2.0 * expi(0.5 * M * G2_NZ)
                #     - expi(M * M * G2_NZ / (B_i + 2 * M))
                #     - expi(M * M * G2_NZ / (B_j + 2 * M))
                # )
                # + torch.exp(-BiM * G2_NZ)
                # * (
                #     expi(BiM * BiM * G2_NZ / BijM)
                #     - expi(BiM * BiM * G2_NZ / (B_i + 2 * M))
                # )
                # + torch.exp(-BjM * G2_NZ)
                # * (
                #     expi(BjM * BjM * G2_NZ / BijM)
                #     - expi(BjM * BjM * G2_NZ / (B_j + 2 * M))
                # )
                + RIH1_torch_where(
                    M * G2_NZ, 0.5 * M * G2_NZ, M * M * G2_NZ / (B_i + 2 * M)
                )
                + RIH1_torch_where(
                    M * G2_NZ, 0.5 * M * G2_NZ, M * M * G2_NZ / (B_j + 2 * M)
                )
                + RIH1_torch_where(
                    BiM * G2_NZ,
                    BiM * BiM * G2_NZ / BijM,
                    BiM * BiM * G2_NZ / (B_i + 2 * M),
                )
                + RIH1_torch_where(
                    BjM * G2_NZ,
                    BjM * BjM * G2_NZ / BijM,
                    BjM * BjM * G2_NZ / (B_j + 2 * M),
                )
            )

            result[:, g_nonzero, :] = (
                multiplier * A_i * A_j * (exp_DW[:, g_nonzero, :] * I1_g - I2_g)
            )

        # Sum over pairs and prepare for broadcasting
        Fphon = result.sum(dim=-1).view(1, N_Z, N_g)
    else:
        Fphon = torch.zeros(1, N_Z, N_g, dtype=g.dtype, device=g.device)

    # Relativistic correction
    gamma = (voltage_kV + 511.0) / 511.0  # [N_V, 1, 1]

    # EMsoft: with relativistic correction gamma, and DWF multiplied in
    if return_type == "EMsoft":
        # Expand Freal to match voltage dimension and multiply by DWF and gamma
        f_real_corrected = f_real * dwf * gamma  # [N_V, N_Z, N_g]

        # Combine all contributions (Fphon already has DWF included)
        f_imag = (
            (Fcore * dwf + Fphon) * gamma**2 / (4.0 * torch.pi * k0)
        )  # [N_V, N_Z, N_g]

        f_scatt = torch.view_as_complex(torch.stack((f_real_corrected, f_imag), dim=-1))

        # unlike py4DSTEM's current code no factor of 0.04787801 or (0.4787801)
        # also we do not yet remove physicist factor:
        # 2.0 * sngl(cRestmass*cCharge/cPlanck**2)*1.0E-18 = 0.664840340614319
        # both will be done when computing Ucg

    # ITC: real-valued without gamma, without DWF
    else:
        f_scatt = f_real

    return f_scatt
