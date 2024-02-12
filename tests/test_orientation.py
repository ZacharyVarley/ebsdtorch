import pytest
import torch
import numpy as np
from ebsdtorch.s2_and_so3.orientations import (
    # quaternion operations
    quaternion_raw_multiply,
    quaternion_multiply,
    quaternion_invert,
    quaternion_real_of_prod,
    normalize_quaternion,
    standardize_quaternion,
    # quaternion to other
    qu2ax,
    qu2cu,
    qu2eu,
    qu2ho,
    qu2om,
    qu2ro,
    # orientation matrix to other
    om2ax,
    om2cu,
    om2eu,
    om2ho,
    om2qu,
    om2ro,
    # axis angle to other
    ax2qu,
    ax2cu,
    ax2eu,
    ax2ho,
    ax2om,
    ax2ro,
    # Rodrigues vector to other
    ro2qu,
    ro2cu,
    ro2eu,
    ro2ho,
    ro2om,
    ro2ax,
    # Homochoric to other
    ho2qu,
    ho2cu,
    ho2eu,
    ho2om,
    ho2ax,
    ho2ro,
    # Cubochoric to other
    cu2qu,
    cu2ax,
    cu2eu,
    cu2om,
    cu2ro,
    cu2ho,
    # Euler angles to other
    eu2qu,
    eu2ax,
    eu2cu,
    eu2om,
    eu2ro,
    eu2ho,
)


# dynamic test generation
def pytest_generate_tests(metafunc):
    if "convert_ax_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_ax_func_pair", test_ax_functions)
    elif "convert_quat_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_quat_func_pair", test_quat_functions)
    elif "convert_om_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_om_func_pair", test_om_functions)
    elif "convert_ho_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_ho_func_pair", test_ho_functions)
    elif "convert_ro_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_ro_func_pair", test_ro_functions)
    elif "convert_cu_func_pair" in metafunc.fixturenames:
        metafunc.parametrize("convert_cu_func_pair", test_cu_functions)


@pytest.fixture
def test_edge_quaternions():
    # define edge cases (no rotation and +/- 180 degree rotations)
    qu_edge = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # normalize the quaternions
    qu_edge = qu_edge / torch.norm(qu_edge, dim=-1, keepdim=True)

    names = [
        "no rotation",
        "180 around x",
        "180 around y",
        "180 around z",
        "180 around -x",
        "180 around -y",
        "180 around -z",
        "180 around xy",
        "180 around yz",
        "180 around xz",
        "180 around xyz",
    ]

    return qu_edge, names


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

    names = [
        "random 1",
        "random 2",
        "random 3",
        "random 4",
        "random 5",
        "random 6",
        "random 7",
        "random 8",
        "random 9",
        "random 10",
    ]

    return qu_rand, names


test_quat_functions = [
    (qu2ax, ax2qu),
    (qu2cu, cu2qu),
    (qu2eu, eu2qu),
    (qu2ho, ho2qu),
    (qu2om, om2qu),
    (qu2ro, ro2qu),
]


def test_rand_quaternion_operations(test_rand_quaternions, convert_quat_func_pair):
    quats, names = test_rand_quaternions

    conv_forward, conv_backward = convert_quat_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(quats, "ZXZ")
    else:
        other_rep = conv_forward(quats)

    # convert back to quaternion
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(quats - recon).sum(dim=-1)).item()

    # send to numpy
    quats = quats.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        quats, recon, atol=1e-6
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {quats[max_diff]} and out {recon[max_diff]}"


def test_edge_quaternion_operations(test_edge_quaternions, convert_quat_func_pair):
    quats, names = test_edge_quaternions

    conv_forward, conv_backward = convert_quat_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(quats, "ZXZ")
    else:
        other_rep = conv_forward(quats)

    # convert back to quaternion
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(quats - recon).sum(dim=-1)).item()

    # send to numpy
    quats = quats.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        quats, recon, atol=1e-6, rtol=1e-6
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {quats[max_diff]} and out {recon[max_diff]}"


@pytest.fixture
def test_rand_axis_angle():
    # set random seed for reproducibility
    torch.manual_seed(0)

    ax_rand = torch.empty(10, 4)

    # define random axis angle
    ax_rand[:, :3] = torch.randn(10, 3)

    # add a small offset away from zero in the same direction per axis
    ax_rand[:, :3] = ax_rand[:, :3] + 1e-3 * torch.sign(ax_rand[:, :3])

    # normalize the axis
    ax_rand[:, :3] = ax_rand[:, :3] / torch.norm(ax_rand[:, :3], dim=-1, keepdim=True)

    # random angle between 0 and pi
    ax_rand[:, 3] = torch.rand(10) * np.pi * 0.98 + 0.01

    names = [f"random {i}" for i in range(1, 11)]

    return ax_rand, names


test_ax_functions = [
    (ax2qu, qu2ax),
    (ax2cu, cu2ax),
    (ax2eu, eu2ax),
    (ax2ho, ho2ax),
    (ax2om, om2ax),
    (ax2ro, ro2ax),
]


def test_rand_ax_operations(test_rand_axis_angle, convert_ax_func_pair):
    ax_angles, names = test_rand_axis_angle

    conv_forward, conv_backward = convert_ax_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(ax_angles, "ZXZ")
    else:
        other_rep = conv_forward(ax_angles)

    # convert back to axis angle
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(ax_angles - recon).sum(dim=-1)).item()

    # send to numpy
    ax_angles = ax_angles.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        ax_angles, recon, atol=1e-5
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {ax_angles[max_diff]} and out {recon[max_diff]}"


@pytest.fixture
def test_rand_orientation_matrix():
    # set random seed for reproducibility
    torch.manual_seed(0)

    # generate random quaternions and convert to orientation matrix
    qu_rand = torch.randn(10, 4)
    qu_rand = qu_rand / torch.norm(qu_rand, dim=-1, keepdim=True)
    qu_rand = qu_rand * torch.where(qu_rand[:, 0] < 0, -1, 1).unsqueeze(-1)
    om_rand = qu2om(qu_rand)

    names = [f"random {i}" for i in range(1, 11)]

    return om_rand, names


test_om_functions = [
    (om2cu, cu2om),
    (om2eu, eu2om),
    (om2ho, ho2om),
    (om2ro, ro2om),
]


def test_rand_om_operations(test_rand_orientation_matrix, convert_om_func_pair):
    om, names = test_rand_orientation_matrix

    conv_forward, conv_backward = convert_om_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(om, "ZXZ")
    else:
        other_rep = conv_forward(om)

    # convert back to orientation matrix
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(om - recon).sum(dim=(-1, -2))).item()

    # send to numpy
    om = om.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        om, recon, atol=1e-5
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {om[max_diff]} and out {recon[max_diff]}"


@pytest.fixture
def test_rand_ho_vectors():
    # set random seed for reproducibility
    torch.manual_seed(0)

    # start from random quaternions
    qu_rand = torch.randn(10, 4)
    qu_rand = qu_rand / torch.norm(qu_rand, dim=-1, keepdim=True)
    qu_rand = qu_rand * torch.where(qu_rand[:, 0] < 0, -1, 1).unsqueeze(-1)

    # convert to homochoric
    ho_rand = qu2ho(qu_rand)

    names = [f"random {i}" for i in range(1, 11)]

    return ho_rand, names


test_ho_functions = [
    (ho2cu, cu2ho),
    (ho2eu, eu2ho),
    (ho2ro, ro2ho),
]


def test_rand_ho_operations(test_rand_ho_vectors, convert_ho_func_pair):
    ho, names = test_rand_ho_vectors

    conv_forward, conv_backward = convert_ho_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(ho, "ZXZ")
    else:
        other_rep = conv_forward(ho)

    # convert back to homochoric
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(ho - recon).sum(dim=-1)).item()

    # send to numpy
    ho = ho.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        ho, recon, atol=1e-5
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {ho[max_diff]} and out {recon[max_diff]}"


@pytest.fixture
def test_rand_ro_vectors():
    # set random seed for reproducibility
    torch.manual_seed(0)

    # start from random quaternions
    qu_rand = torch.randn(10, 4)
    qu_rand = qu_rand / torch.norm(qu_rand, dim=-1, keepdim=True)
    qu_rand = qu_rand * torch.where(qu_rand[:, 0] < 0, -1, 1).unsqueeze(-1)

    # convert to Rodrigues vector
    ro_rand = qu2ro(qu_rand)

    names = [f"random {i}" for i in range(1, 11)]

    return ro_rand, names


test_ro_functions = [
    (ro2cu, cu2ro),
    (ro2eu, eu2ro),
]


def test_rand_ro_operations(test_rand_ro_vectors, convert_ro_func_pair):
    ro, names = test_rand_ro_vectors

    conv_forward, conv_backward = convert_ro_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(ro, "ZXZ")
    else:
        other_rep = conv_forward(ro)

    # convert back to Rodrigues vector
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(ro - recon).sum(dim=-1)).item()

    # send to numpy
    ro = ro.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        ro, recon, atol=1e-5
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {ro[max_diff]} and out {recon[max_diff]}"


@pytest.fixture
def test_rand_cubochoric_vectors():
    # set random seed for reproducibility
    torch.manual_seed(0)

    # start from random quaternions
    qu_rand = torch.randn(10, 4)
    qu_rand = qu_rand / torch.norm(qu_rand, dim=-1, keepdim=True)
    qu_rand = qu_rand * torch.where(qu_rand[:, 0] < 0, -1, 1).unsqueeze(-1)

    # convert to cubochoric
    cu_rand = qu2cu(qu_rand)

    names = [f"random {i}" for i in range(1, 11)]

    return cu_rand, names


test_cu_functions = [
    (cu2eu, eu2cu),
]


def test_rand_cu_operations(test_rand_cubochoric_vectors, convert_cu_func_pair):
    cu, names = test_rand_cubochoric_vectors

    conv_forward, conv_backward = convert_cu_func_pair

    # convert to other representation
    if "eu" in conv_forward.name:
        other_rep = conv_forward(cu, "ZXZ")
    else:
        other_rep = conv_forward(cu)

    # convert back to cubochoric
    if "eu" in conv_backward.name:
        recon = conv_backward(other_rep, "ZXZ")
    else:
        recon = conv_backward(other_rep)

    # find row of the maximum difference
    max_diff = torch.argmax(torch.abs(cu - recon).sum(dim=-1)).item()

    # send to numpy
    cu = cu.cpu().numpy()
    other_rep = other_rep.cpu().numpy()
    recon = recon.cpu().numpy()

    # check if the difference is within tolerance
    assert np.allclose(
        cu, recon, atol=1e-5
    ), f"{conv_forward.name} failed on {names[max_diff]} with in {cu[max_diff]} and out {recon[max_diff]}"
