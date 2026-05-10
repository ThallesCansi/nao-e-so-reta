from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from nao_e_so_reta.config import XY

DEFAULT_P_VALUES: tuple[float, ...] = (
    1.0,
    1.25,
    1.5,
    1.54,
    1.75,
    2.0,
    3.0,
    5.0,
    10.0,
    math.inf,
)


def validate_p(p: float) -> float:
    """Valida o parâmetro p de uma norma Lp.

    Matematicamente, ||.||_p é norma para p >= 1. Para 0 < p < 1,
    tem-se uma quase-norma que viola a desigualdade triangular.
    """
    if math.isinf(p):
        return p
    p = float(p)
    if p < 1:
        raise ValueError("Para ser uma norma Lp, p deve satisfazer p >= 1.")
    return p


def lp_norm(dx: float, dy: float, p: float) -> float:
    """Norma Lp de um vetor bidimensional (dx, dy)."""
    p = validate_p(p)
    ax = abs(float(dx))
    ay = abs(float(dy))

    if math.isinf(p):
        return max(ax, ay)

    return (ax**p + ay**p) ** (1.0 / p)


def lp_distance_xy(a: XY, b: XY, p: float) -> float:
    """Distância Lp entre dois pontos no plano projetado."""
    return lp_norm(a[0] - b[0], a[1] - b[1], p)


def named_lp_distances(a: XY, b: XY, p: float) -> dict[str, float]:
    """Retorna distâncias canônicas e a distância Lp personalizada."""
    return {
        "L1 / Manhattan": lp_distance_xy(a, b, 1.0),
        "L2 / Euclidiana": lp_distance_xy(a, b, 2.0),
        f"Lp / Minkowski (p={p:.2f})": lp_distance_xy(a, b, p),
        "L∞ / Chebyshev": lp_distance_xy(a, b, math.inf),
    }


def metric_name_from_p(p: float) -> str:
    """Nome estável para uma métrica Lp em tabelas e gráficos."""
    p = validate_p(p)
    if math.isinf(p):
        return "Linf"
    if abs(p - round(p)) < 1e-12:
        return f"L{int(round(p))}"
    return "L" + str(p).replace(".", "_")


def metric_column_from_p(p: float) -> str:
    """Nome da coluna de distância, em metros, para um valor de p."""
    return f"d_{metric_name_from_p(p)}_m"


def superellipse_boundary_xy(
    center: XY,
    radius: float,
    p: float,
    n_points: int = 180,
) -> list[XY]:
    """Fronteira da bola Lp: |x|^p + |y|^p = radius^p."""
    p = validate_p(p)
    if math.isinf(p):
        return square_boundary_xy(center, radius, n_points)

    cx, cy = center
    radius = float(radius)
    if radius <= 0:
        return [(cx, cy)]

    n_points = max(16, int(n_points))
    out: list[XY] = []
    for t in np.linspace(0.0, 2.0 * math.pi, n_points + 1):
        ct = math.cos(float(t))
        st = math.sin(float(t))
        x = radius * math.copysign(abs(ct) ** (2.0 / p), ct)
        y = radius * math.copysign(abs(st) ** (2.0 / p), st)
        out.append((cx + x, cy + y))
    return out


def square_boundary_xy(center: XY, radius: float, n_points: int = 180) -> list[XY]:
    """Aproxima a bola L∞ por uma fronteira quadrada."""
    cx, cy = center
    r = max(0.0, float(radius))
    if r == 0:
        return [(cx, cy)]

    corners = [
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r),
        (cx - r, cy - r),
    ]

    n_per_edge = max(2, int(n_points / 4))
    out: list[XY] = []
    for a, b in zip(corners[:-1], corners[1:]):
        for s in np.linspace(0.0, 1.0, n_per_edge, endpoint=False):
            out.append((a[0] + float(s) * (b[0] - a[0]), a[1] + float(s) * (b[1] - a[1])))
    out.append(corners[-1])
    return out


def manhattan_polyline_xy(a: XY, b: XY, order: str = "x-then-y") -> list[XY]:
    """Polilinha Manhattan no plano projetado.

    Há infinitas geodésicas L1 entre dois pontos em uma malha ortogonal.
    Esta função fornece duas representações canônicas.
    """
    ax, ay = a
    bx, by = b

    if order == "y-then-x":
        return [(ax, ay), (ax, by), (bx, by)]
    if order == "x-then-y":
        return [(ax, ay), (bx, ay), (bx, by)]

    raise ValueError("order deve ser 'x-then-y' ou 'y-then-x'.")


def visual_minkowski_curve_xy(a: XY, b: XY, p: float, n_points: int = 90) -> list[XY]:
    """Curva visual auxiliar entre A e B.

    Esta curva não representa uma geodésica única de Lp. Em espaços normados
    estritamente convexos, p > 1, segmentos de reta são geodésicas. A curva aqui
    é apenas uma camada didática para variar a percepção entre forma em L e forma
    suavizada conforme p muda.
    """
    p = validate_p(p)
    ax, ay = a
    bx, by = b

    t_values = np.linspace(0.0, 1.0, max(2, int(n_points)))
    k = 6.0 if math.isinf(p) else max(1.0, float(p) * 2.5)

    x_norm = t_values
    y_norm = 1.0 - (1.0 - t_values**k) ** (1.0 / k)

    xs = ax + x_norm * (bx - ax)
    ys = ay + y_norm * (by - ay)

    return list(zip(xs.tolist(), ys.tolist()))


def pairwise_lp(values: Iterable[tuple[float, float]], p: float) -> list[float]:
    """Calcula ||(dx,dy)||_p para uma coleção de pares."""
    return [lp_norm(dx, dy, p) for dx, dy in values]
