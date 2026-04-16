import cv2

from src.mocopi.visualization import rotate_normalized_point


def test_rotate_normalized_point_90_clockwise():
    xr, yr = rotate_normalized_point((0.25, 0.75), cv2.ROTATE_90_CLOCKWISE)
    assert xr == 0.25
    assert yr == 0.25


def test_rotate_normalized_point_90_counterclockwise():
    xr, yr = rotate_normalized_point((0.25, 0.75), cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert xr == 0.75
    assert yr == 0.75


def test_rotate_normalized_point_180():
    xr, yr = rotate_normalized_point((0.25, 0.75), cv2.ROTATE_180)
    assert xr == 0.75
    assert yr == 0.25


def test_rotate_normalized_point_none():
    xr, yr = rotate_normalized_point((0.25, 0.75), None)
    assert xr == 0.25
    assert yr == 0.75
