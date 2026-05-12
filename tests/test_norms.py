from __future__ import annotations

import math

import pytest

from nao_e_so_reta.norms import (
    lp_distance_xy,
    lp_norm,
    manhattan_polyline_xy,
    superellipse_boundary_xy,
)


def test_lp_norm_special_cases() -> None:
    assert lp_norm(3, 4, 1) == pytest.approx(7)
    assert lp_norm(3, 4, 2) == pytest.approx(5)
    assert lp_norm(3, 4, math.inf) == pytest.approx(4)


def test_lp_distance_xy() -> None:
    assert lp_distance_xy((0, 0), (3, 4), 2) == pytest.approx(5)


def test_invalid_p() -> None:
    with pytest.raises(ValueError):
        lp_norm(1, 2, 0.5)


def test_superellipse_boundary_closes_curve() -> None:
    pts = superellipse_boundary_xy((10, 20), 5, 2, n_points=40)
    assert len(pts) == 41
    assert pts[0] == pytest.approx(pts[-1])


def test_manhattan_polyline_orders() -> None:
    a = (0, 0)
    b = (10, 5)
    assert manhattan_polyline_xy(a, b, "x-then-y") == [(0, 0), (10, 0), (10, 5)]
    assert manhattan_polyline_xy(a, b, "y-then-x") == [(0, 0), (0, 5), (10, 5)]
