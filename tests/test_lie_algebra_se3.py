import torch
import pytest

from ebsdtorch.geometry.lie_algebra_se3 import (
    se3_exp_map_om,
    se3_log_map_om,
    se3_exp_map_quat,
    se3_log_map_quat,
)


@pytest.fixture
def se3_test_data():
    torch.manual_seed(0)
    dtype = torch.float32
    omegas = torch.randn((10, 3), dtype=dtype)
    omegas = omegas / torch.linalg.norm(omegas, dim=-1, keepdim=True)
    theta = torch.abs(torch.rand((10, 1), dtype=dtype)) * torch.pi / 2.0
    omegas = omegas * theta
    tvecs = torch.randn((10, 3))
    vec = torch.cat([omegas, tvecs], dim=-1)
    return vec


def test_se3_exp_map_quat(se3_test_data):
    vec = se3_test_data
    quats, tvecs = se3_exp_map_quat(vec)
    vec_reconstructed = se3_log_map_quat(quats, tvecs)
    assert torch.allclose(vec, vec_reconstructed, atol=1e-5)


def test_se3_exp_map_om(se3_test_data):
    vec = se3_test_data
    se3_mat = se3_exp_map_om(vec)
    vec_reconstructed = se3_log_map_om(se3_mat)
    assert torch.allclose(vec, vec_reconstructed, atol=1e-5)
