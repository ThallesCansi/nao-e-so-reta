from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import XY

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
    """Valida o parâmetro p de uma norma Lp."""
    if math.isinf(p):
        return math.inf
    p = float(p)
    if p < 1:
        raise ValueError("Para ser norma Lp, use p >= 1.")
    return p

def p_key(p: float) -> str:
    """Chave curta e estável para nomes internos de colunas."""
    p = validate_p(p)
    if math.isinf(p):
        return "Linf"
    if abs(p - round(p)) < 1e-12:
        return f"L{int(round(p))}"
    return "L" + (f"{p:.10g}".replace(".", "_"))


def p_value_label(p: float) -> str:
    """Texto simples para o valor de p."""
    p = validate_p(p)
    if math.isinf(p):
        return "∞"
    if abs(p - round(p)) < 1e-12:
        return str(int(round(p)))
    return f"{p:.10g}"


def metric_label(p: float) -> str:
    """Nome legível para tabelas."""
    return f"d_{p_value_label(p)}"


def metric_latex(p: float) -> str:
    """Nome em LaTeX para gráficos Matplotlib."""
    p = validate_p(p)
    if math.isinf(p):
        return r"$d_\infty$"
    return rf"$d_{{{p_value_label(p)}}}$"


def tau_label(p: float) -> str:
    """Nome legível para a tortuosidade relativa a d_p."""
    return f"τ_{p_value_label(p)}"


def tau_latex(p: float) -> str:
    """Nome em LaTeX para a tortuosidade relativa a d_p."""
    p = validate_p(p)
    if math.isinf(p):
        return r"$\tau_\infty = d_G/d_\infty$"
    return rf"$\tau_{{{p_value_label(p)}}} = d_G/d_{{{p_value_label(p)}}}$"


def distance_col(p: float) -> str:
    """Coluna interna para a distância d_p em metros."""
    return f"dist_{p_key(p)}_m"


def tau_col(p: float) -> str:
    """Coluna interna para a tortuosidade tau_p = d_G/d_p."""
    return f"tau_{p_key(p)}"


def error_col(p: float) -> str:
    """Coluna interna para erro assinado d_p - d_G em metros."""
    return f"error_{p_key(p)}_m"


def abs_error_col(p: float) -> str:
    """Coluna interna para erro absoluto |d_p - d_G| em metros."""
    return f"abs_error_{p_key(p)}_m"


def ape_col(p: float) -> str:
    """Coluna interna para erro percentual absoluto relativo a d_G."""
    return f"ape_{p_key(p)}_pct"


def lp_norm(dx: float, dy: float, p: float) -> float:
    """Norma Lp de um vetor bidimensional."""
    p = validate_p(p)
    ax = abs(float(dx))
    ay = abs(float(dy))
    if math.isinf(p):
        return max(ax, ay)
    return float((ax**p + ay**p) ** (1.0 / p))


def lp_distance_xy(a: XY, b: XY, p: float) -> float:
    """Distância Lp entre dois pontos no plano projetado em metros."""
    return lp_norm(a[0] - b[0], a[1] - b[1], p)


def lp_distance_arrays(dx: np.ndarray, dy: np.ndarray, p: float) -> np.ndarray:
    """Distância Lp vetorizada para arrays |dx| e |dy|."""
    p = validate_p(p)
    if math.isinf(p):
        return np.maximum(dx, dy)
    return (dx**p + dy**p) ** (1.0 / p)


def p_grid(start: float = 1.0, stop: float = 5.0, step: float = 0.01) -> np.ndarray:
    """Cria uma grade fechada de valores de p."""
    if start < 1 or stop < start or step <= 0:
        raise ValueError("Use 1 <= start <= stop e step > 0.")
    n = int(round((stop - start) / step))
    return np.round(start + step * np.arange(n + 1), 10)


def ensure_p_values(p_values: Iterable[float]) -> tuple[float, ...]:
    """Normaliza e valida uma coleção de valores de p."""
    return tuple(validate_p(p) for p in p_values)
