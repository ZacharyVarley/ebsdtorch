import torch
import pytest
from ebsdtorch.harmonics.real_SHT import FTSHT, ITSHT


@pytest.fixture
def sht_modules():
    device = torch.device("cpu")
    B = 8
    F = FTSHT(B).to(device)
    I = ITSHT(B).to(device)

    return F, I


def test_real_SHT(sht_modules):
    F, I = sht_modules
    B = 8
    device = torch.device("cpu")

    # make a complex signal with batch size 10
    Psi = torch.view_as_complex(
        2 * (torch.rand(10, 2 * B - 1, B, 2, device=device).double() - 0.5)
    ).to(device)

    # set SH coefficients to zero for l < |k|
    kk, jj = torch.meshgrid(
        torch.arange(-(B - 1), B, device=device),
        torch.arange(0, B, device=device),
        indexing="ij",
    )
    invalid_mask = jj * jj < kk * kk
    jj_invalid = jj[invalid_mask]
    kk_invalid = kk[invalid_mask]
    Psi[:, kk_invalid + (B - 1), jj_invalid] = 0.0

    # run forward and inverse transform
    Psi2 = F(I(Psi))

    error_fp64 = torch.mean(torch.abs(Psi - Psi2))
    assert error_fp64 < 1e-14

    # run the same test with float precision
    Psi_f = Psi.cfloat()
    F_f = F.to(device).float()
    I_f = I.to(device).float()

    Psi2_f = F_f(I_f(Psi_f))

    error_fp32 = torch.mean(torch.abs(Psi_f - Psi2_f))
    assert error_fp32 < 1e-7
