from __future__ import annotations

import pytest

from nao_e_so_reta.analysis import (
    calibration_for_p,
    compare_norms_for_pair,
    optimal_scale_alpha,
    safe_relative_error_pct,
    tortuosity,
)


def test_safe_relative_error_pct() -> None:
    assert safe_relative_error_pct(80, 100) == pytest.approx(-20)
    assert safe_relative_error_pct(100, 0) is None
    assert safe_relative_error_pct(100, None) is None


def test_tortuosity() -> None:
    assert tortuosity(150, 100) == pytest.approx(1.5)
    assert tortuosity(None, 100) is None


def test_compare_norms_for_pair() -> None:
    rows = compare_norms_for_pair((0, 0), (3, 4), graph_distance_m=10, p=2)
    names = [r.name for r in rows]
    assert "L2 / Euclidiana" in names
    l2 = next(r for r in rows if r.name == "L2 / Euclidiana")
    assert l2.distance_m == pytest.approx(5)
    assert l2.relative_error_pct == pytest.approx(-50)


def test_optimal_scale_alpha() -> None:
    assert optimal_scale_alpha([1, 2, 3], [2, 4, 6]) == pytest.approx(2)


def test_calibration_for_p() -> None:
    pairs = [
        ((0, 0), (3, 4), 10.0),
        ((0, 0), (6, 8), 20.0),
    ]
    rec = calibration_for_p(pairs, p=2)
    assert rec.scale_alpha == pytest.approx(2.0)
    assert rec.mape_pct == pytest.approx(0.0)
