import torch
import pytest
from ebsdtorch.wigner.wigner_d import build_jkm_volume

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
def test_build_jkm_volume(j, k, m, correct_value, beta):
    volume = build_jkm_volume(j, k, beta, torch.device("cpu"))
    d_value = volume[j, k, m]
    assert torch.isclose(d_value, correct_value, atol=1e-14)
