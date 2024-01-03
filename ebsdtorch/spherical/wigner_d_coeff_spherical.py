import numpy as np
from spherical import Wigner


test_parameters = [
    (365, 102, 20, -4.23570250037880395095020243575390e-02, 8161.0 / 16384.0),
    (294, 247, 188, -1.11943794723176255836019618855372e-01, 7417.0 / 16384.0),
    # (1540, 1127, 1120, 4.0189543855173193313064415985304e-02, 2032.0 / 16384.0),
    # (6496, 141, 94, 1.91605798359216822133779150869763e-03, 10134.0 / 16384.0),
]

for test_parameter in test_parameters:
    j = test_parameter[0]
    k = test_parameter[1]
    m = test_parameter[2]
    correct_value = test_parameter[3]
    beta = np.pi * test_parameter[4]
    w = Wigner(ell_max=j)
    d_value = w.d(expiβ=(np.complex128(np.exp(1j * beta))))[w.dindex(j, k, m)] * (
        -1
    ) ** (k + m)
    print("Correct   : " + "{:.20f}".format(correct_value))
    print("Computed  : " + "{:.20f}".format(d_value))
    print("Difference: " + "{:.20f}".format(abs(d_value - correct_value)))
    print(" ------------------- ")
