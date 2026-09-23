# Тесты замера ошибки продлённой оси по стенам.

from __future__ import annotations

import numpy as np
import pytest

from tools.measure_extension_residual import _axis_error, _correction


def test_axis_error_has_the_same_sign_from_both_walls() -> None:
    true_axis, axis = 0.0, -0.1
    left_wall, right_wall = 2.0, -2.0
    far_left = +1.0 * (left_wall - axis)
    far_right = -1.0 * (right_wall - axis)
    near_left = +1.0 * (left_wall - true_axis)
    near_right = -1.0 * (right_wall - true_axis)
    assert _axis_error(far_left, near_left, "left") == pytest.approx(-0.1)
    assert _axis_error(far_right, near_right, "right") == pytest.approx(-0.1)


def test_correction_recovers_shift_and_slope_on_common_slices() -> None:
    x = np.arange(5.5, 24.0, 1.0)
    rails = np.column_stack([x, 0.01 * x])
    bed = np.column_stack([x + 0.05, 0.01 * x - 0.02 - 0.004 * x])
    slope, shift, lo, hi = _correction(rails, bed, x_min=5.0, slice_m=1.0, min_slices=4)
    assert slope == pytest.approx(-0.004, abs=5e-4)
    assert shift == pytest.approx(-0.02, abs=5e-3)
    assert (lo, hi) == (5.5, 23.5)

