import torch
import pytest
from ebsdtorch.harmonics.wigner_d_logspace import (
    wigner_d_eq_half_pi,
    wigner_d_lt_half_pi,
    read_lmn_wigner_d_half_pi_table,
)

test_parameters = [
    # test case taken from http://dx.doi.org/10.13140/RG.2.2.31922.20160
    (
        365,
        102,
        20,
        torch.tensor(-4.23570250037880395095020243575390e-02, dtype=torch.float64),
        torch.tensor(torch.pi * 8161.0 / 16384.0, dtype=torch.float64),
    ),
]


@pytest.mark.parametrize(
    "j, k, m, correct_value, beta",
    test_parameters,
)
def test_wigner(j, k, m, correct_value, beta):
    table = wigner_d_lt_half_pi(
        beta, order_max=370, dtype=torch.float64, device=torch.device("cpu")
    )
    d_value = read_lmn_wigner_d_half_pi_table(table, coords=torch.tensor([j, k, m]))
    assert torch.isclose(d_value, correct_value, atol=1e-14)
