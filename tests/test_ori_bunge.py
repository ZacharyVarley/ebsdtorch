import pytest
import torch
from ebsdtorch.s2_and_so3.orientations import (
    qu2bu,
    bu2qu,
)


@pytest.fixture
def euler_angles_random():
    N = 10
    bu = torch.rand(N, 3, dtype=torch.float64) * 2 * torch.pi
    bu[:, 1] *= 0.5
    return bu


@pytest.fixture
def euler_angles_edge():
    # DEFINED WITH DEGREES AND RETURNED IN RADIANS
    return torch.deg2rad(
        torch.tensor(
            [
                [0, 0, 0],
                [90, 0, 0],
                [180, 0, 0],
                [270, 0, 0],
                [360, 0, 0],
                [0, 90, 0],
                [90, 90, 0],
                [180, 90, 0],
                [270, 90, 0],
                [360, 90, 0],
                [0, 180, 0],
                [90, 180, 0],
                [180, 180, 0],
                [270, 180, 0],
                [360, 180, 0],
                [0, 0, 90],
                [90, 0, 90],
                [180, 0, 90],
                [270, 0, 90],
                [360, 0, 90],
                [0, 90, 90],
                [90, 90, 90],
                [180, 90, 90],
                [270, 90, 90],
                [360, 90, 90],
                [0, 180, 90],
                [90, 180, 90],
                [180, 180, 90],
                [270, 180, 90],
                [360, 180, 90],
                [0, 0, 180],
                [90, 0, 180],
                [180, 0, 180],
                [270, 0, 180],
                [360, 0, 180],
                [0, 90, 180],
                [90, 90, 180],
                [180, 90, 180],
                [270, 90, 180],
                [360, 90, 180],
                [0, 180, 180],
                [90, 180, 180],
                [180, 180, 180],
                [270, 180, 180],
                [360, 180, 180],
                [0, 0, 270],
                [90, 0, 270],
                [180, 0, 270],
                [270, 0, 270],
                [360, 0, 270],
                [0, 90, 270],
                [90, 90, 270],
                [180, 90, 270],
                [270, 90, 270],
                [360, 90, 270],
                [0, 180, 270],
                [90, 180, 270],
                [180, 180, 270],
                [270, 180, 270],
                [360, 180, 270],
                [0, 0, 360],
                [90, 0, 360],
                [180, 0, 360],
                [270, 0, 360],
                [360, 0, 360],
                [0, 90, 360],
                [90, 90, 360],
                [180, 90, 360],
                [270, 90, 360],
                [360, 90, 360],
                [0, 180, 360],
                [90, 180, 360],
                [180, 180, 360],
                [270, 180, 360],
                [360, 180, 360],
            ],
            dtype=torch.float64,
        )
    )


def test_bu2qu_random(euler_angles_random):
    qu = bu2qu(euler_angles_random)
    bu = qu2bu(qu)
    assert torch.allclose(bu, euler_angles_random, atol=1e-5)


def test_qu2bu_edge(euler_angles_edge):
    eu = euler_angles_edge
    qu = bu2qu(eu)
    bu = qu2bu(qu)

    # assert that shapes are same
    assert bu.shape == eu.shape, f"{bu.shape} and {eu.shape}"

    # find largest difference (modulo 2pi for 1st and 3rd angles)
    diff_1 = torch.fmod(torch.abs(bu[..., 0] - eu[..., 0]), 2 * torch.pi)
    diff_2 = torch.abs(bu[..., 1] - eu[..., 1])
    diff_3 = torch.fmod(torch.abs(bu[..., 2] - eu[..., 2]), 2 * torch.pi)
    diff = diff_1 + diff_2 + diff_3

    # check for the case that (ZXZ) = (Z00) or (00Z) as both are the same
    mask_special = (
        ((bu[:, 0] != 0) & (bu[:, 1] == 0) & (bu[:, 2] == 0))
        | ((bu[:, 0] == 0) & (bu[:, 1] == 0) & (bu[:, 2] != 0))
    ) | (
        ((eu[:, 0] != 0) & (eu[:, 1] == 0) & (eu[:, 2] == 0))
        | ((eu[:, 0] == 0) & (eu[:, 1] == 0) & (eu[:, 2] != 0))
    )

    # use the sum to quickly extract the only non-zero angle per row
    # these differences are also meaningless
    diff[mask_special] = torch.sum(bu[mask_special, :], dim=1) - torch.sum(
        eu[mask_special, :], dim=1
    )

    assert torch.all(
        diff < 1e-5
    ), f"{diff.argmax()}: {diff.max()}, {bu[diff.argmax()]}, {eu[diff.argmax()]}"


@pytest.fixture
def test_rand_quaternions():

    # set random seed for reproducibility
    torch.manual_seed(0)

    # define random quaternions
    qu_rand = torch.randn(10, 4)

    # add a small offset away from zero in the same direction per axis
    qu_rand = qu_rand + 1e-3 * torch.sign(qu_rand)

    # normalize the quaternions and set positive real part
    qu_rand = qu_rand / torch.norm(qu_rand, dim=-1, keepdim=True)
    qu_rand = qu_rand * torch.where(qu_rand[:, 0] < 0, -1, 1).unsqueeze(-1)

    return qu_rand


def test_qu2bu(test_rand_quaternions):
    qu = test_rand_quaternions
    bu = qu2bu(qu)
    qu_recon = bu2qu(bu)
    assert torch.allclose(qu, qu_recon, atol=1e-5)
