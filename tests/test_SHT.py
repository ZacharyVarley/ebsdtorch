import torch
import pytest
from ebsdtorch.harmonics.sht import RSHT, CSHT

import torch
import pytest
from ebsdtorch.harmonics.sht import RSHT, CSHT


@pytest.fixture(params=[3, 4, 5])
def bandlimit(request):
    return request.param


@pytest.fixture(params=["single", "double"])
def precision(request):
    return request.param


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def rsht_module(bandlimit, device, precision):
    return RSHT(bandlimit, device=device, precision=precision)


@pytest.fixture
def csht_module(bandlimit, device, precision):
    return CSHT(bandlimit, device=device, precision=precision)


def test_real_SHT(rsht_module, bandlimit, device, precision):
    sht = rsht_module
    dtype = torch.float32 if precision == "single" else torch.float64

    psi = 2 * (
        torch.rand(1, 2 * bandlimit, 2 * bandlimit, device=device, dtype=dtype) - 0.5
    )

    # Make an exactly representable band limited signal
    psi_band = sht.isht(sht.fsht(psi))

    # Run forward and inverse transform
    psi_recon = sht.isht(sht.fsht(psi_band))

    # Compute error
    error = torch.mean(torch.abs(psi_band - psi_recon))

    tolerance = 1e-6 if precision == "single" else 1e-14
    assert (
        error < tolerance
    ), f"Error {error} exceeds tolerance {tolerance} for bandlimit {bandlimit}, precision {precision}"


def test_complex_SHT(csht_module, bandlimit, device, precision):
    sht = csht_module
    dtype = torch.float32 if precision == "single" else torch.float64

    Psi = torch.view_as_complex(
        2
        * (
            torch.rand(1, 2 * bandlimit - 1, bandlimit, 2, device=device, dtype=dtype)
            - 0.5
        )
    ).to(device)

    mm, ll = torch.meshgrid(
        torch.arange(0, 2 * bandlimit - 1, device=device, dtype=torch.int64),
        torch.arange(0, bandlimit, device=device, dtype=torch.int64),
        indexing="ij",
    )

    valid = torch.abs(mm - bandlimit + 1) <= ll
    Psi[:, ~valid] = 0

    # Run forward and inverse transform
    Psi2 = sht.fsht(sht.isht(Psi))

    # View as real to compute error
    error = torch.mean(torch.abs(torch.view_as_real(Psi) - torch.view_as_real(Psi2)))

    tolerance = 1e-6 if precision == "single" else 1e-14
    assert (
        error < tolerance
    ), f"Error {error} exceeds tolerance {tolerance} for bandlimit {bandlimit}, precision {precision}"
