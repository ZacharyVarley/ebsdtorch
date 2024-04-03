import torch
import pytest
from ebsdtorch.harmonics.real_SOFT import CCSOFT


@pytest.fixture
def sht_modules():
    device = torch.device("cpu")
    B = 8
    F = CCSOFT(B).to(device)
    return F


def test_real_SHT(sht_modules):
    F = sht_modules
    B = 8
    device = torch.device("cpu")

    # make a complex signal with batch size 10
    Psi = (
        torch.view_as_complex(
            2 * (torch.rand(10, B, 2 * B - 1, 2, device=device).double() - 0.5)
        )
        .to(device)
        .to(torch.complex64)
    )

    # set SH coefficients to zero for l < |k|
    jj, kk = torch.meshgrid(
        torch.arange(0, B, device=device),
        torch.arange(-(B - 1), B, device=device),
        indexing="ij",
    )
    invalid_mask = jj * jj < kk * kk
    jj_invalid = jj[invalid_mask]
    kk_invalid = kk[invalid_mask]
    Psi[:, jj_invalid, kk_invalid + (B - 1)] = 0.0

    # autocorrelate Psi with itself
    Psi2 = F(Psi, Psi)

    # check that the output has the correct shape
    assert Psi2.shape == (10, 2 * B - 1, 2 * B - 1, 2 * B - 1)
