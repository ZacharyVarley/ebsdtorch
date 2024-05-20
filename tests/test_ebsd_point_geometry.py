import torch
from ebsdtorch.ebsd.geometry import PointEBSDGeometry
import pytest


@pytest.fixture
def geometry_se3():
    detector_shape = (100, 100)
    sample_x_tilt_degrees = 15.0
    sample_y_tilt_degrees = 70.0
    detector_tilt_degrees = 30.0
    pattern_center_guess = (0.5, 0.5, 0.5)

    geometry = PointEBSDGeometry(
        detector_shape,
        "se3",
        sample_x_tilt_degrees,
        sample_y_tilt_degrees,
        detector_tilt_degrees,
        pattern_center_guess,
    ).to(torch.float32)

    return geometry


@pytest.fixture
def geometry_pc():
    detector_shape = (100, 100)
    sample_x_tilt_degrees = 15.0
    sample_y_tilt_degrees = 70.0
    detector_tilt_degrees = 30.0
    pattern_center_guess = (0.5, 0.5, 0.5)

    geometry = PointEBSDGeometry(
        detector_shape,
        "bruker_pc",
        sample_x_tilt_degrees,
        sample_y_tilt_degrees,
        detector_tilt_degrees,
        pattern_center_guess,
    ).to(torch.float32)

    return geometry


def test_point_ebsd_geometry_se3(geometry_se3):
    # test the forward pass
    rotation_matrix, translation_vector = geometry_se3()
    assert rotation_matrix.shape == (3, 3)
    assert translation_vector.shape == (3,)

    # test the transform method
    pixel_coordinates = torch.tensor(
        [[1.0, 6.0], [2.0, 4.0], [3.0, 3.0]], dtype=torch.float32
    )
    sample_coordinates = geometry_se3.transform(pixel_coordinates)

    # assert the shape
    assert sample_coordinates.shape == (
        3,
        3,
    ), "Shape of sample coordinates is not correct."

    # test the inverse_project method
    pixel_coordinates_bp = geometry_se3.backproject(sample_coordinates)

    # assert the shape
    assert pixel_coordinates_bp.shape == (
        3,
        2,
    ), "Shape of backprojected pixel coordinates is not correct."

    # assert that the backprojection is the inverse of the forward projection
    assert torch.allclose(pixel_coordinates, pixel_coordinates_bp, atol=1e-6)


def test_point_ebsd_geometry_pc(geometry_pc):
    # test the forward pass
    rotation_matrix, translation_vector = geometry_pc()
    assert rotation_matrix.shape == (3, 3)
    assert translation_vector.shape == (3,)

    # test the transform method
    pixel_coordinates = torch.tensor(
        [[1.0, 6.0], [2.0, 4.0], [3.0, 3.0]], dtype=torch.float32
    )
    sample_coordinates = geometry_pc.transform(pixel_coordinates)

    # assert the shape
    assert sample_coordinates.shape == (
        3,
        3,
    ), "Shape of sample coordinates is not correct."

    # test the inverse_project method
    pixel_coordinates_bp = geometry_pc.backproject(sample_coordinates)

    # assert the shape
    assert pixel_coordinates_bp.shape == (
        3,
        2,
    ), "Shape of backprojected pixel coordinates is not correct."

    # assert that the backprojection is the inverse of the forward projection
    assert torch.allclose(pixel_coordinates, pixel_coordinates_bp, atol=1e-6)
