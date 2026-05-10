from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from nao_e_so_reta.config import XY
from nao_e_so_reta.norms import lp_distance_xy, named_lp_distances


@dataclass(frozen=True)
class MetricComparison:
    name: str
    distance_m: float
    graph_distance_m: float | None
    absolute_error_m: float | None
    relative_error_pct: float | None
    graph_over_metric: float | None


@dataclass(frozen=True)
class CalibrationRecord:
    p: float
    scale_alpha: float
    mae_m: float
    rmse_m: float
    mape_pct: float
    mean_distortion: float
    n_pairs: int


def safe_relative_error_pct(estimate: float, reference: float | None) -> float | None:
    """Erro percentual assinado em relação a uma referência."""
    if reference is None or reference <= 0:
        return None
    return ((estimate - reference) / reference) * 100.0


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Razão protegida contra valores nulos ou zero."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def compare_norms_for_pair(
    a_xy: XY,
    b_xy: XY,
    *,
    graph_distance_m: float | None,
    p: float,
) -> list[MetricComparison]:
    """Compara distâncias Lp com a distância real no grafo."""
    distances = named_lp_distances(a_xy, b_xy, p=p)

    out: list[MetricComparison] = []
    for name, distance in distances.items():
        rel = safe_relative_error_pct(distance, graph_distance_m)
        abs_err = None if graph_distance_m is None else distance - graph_distance_m
        ratio = safe_ratio(graph_distance_m, distance)
        out.append(
            MetricComparison(
                name=name,
                distance_m=float(distance),
                graph_distance_m=graph_distance_m,
                absolute_error_m=abs_err,
                relative_error_pct=rel,
                graph_over_metric=ratio,
            )
        )
    return out


def tortuosity(graph_distance_m: float | None, euclidean_distance_m: float) -> float | None:
    """Tortuosidade clássica: distância na rede dividida pela distância euclidiana."""
    return safe_ratio(graph_distance_m, euclidean_distance_m)


def optimal_scale_alpha(metric_distances: list[float], graph_distances: list[float]) -> float:
    """Ajusta alpha em graph ≈ alpha * metric por mínimos quadrados sem intercepto."""
    if len(metric_distances) != len(graph_distances):
        raise ValueError("As listas devem ter o mesmo tamanho.")
    if not metric_distances:
        raise ValueError("É necessário ao menos um par de distâncias.")

    numerator = sum(x * y for x, y in zip(metric_distances, graph_distances))
    denominator = sum(x * x for x in metric_distances)

    if denominator <= 0:
        raise ValueError("As distâncias métricas não podem ser todas zero.")

    return numerator / denominator


def calibration_for_p(
    pairs: list[tuple[XY, XY, float]],
    p: float,
) -> CalibrationRecord:
    """Avalia um valor de p em uma coleção de pares.

    Cada par tem a forma:
        (a_xy, b_xy, graph_distance_m)
    """
    metric_distances: list[float] = []
    graph_distances: list[float] = []

    for a_xy, b_xy, graph_distance in pairs:
        d = lp_distance_xy(a_xy, b_xy, p=p)
        if d > 0 and graph_distance > 0:
            metric_distances.append(float(d))
            graph_distances.append(float(graph_distance))

    if not metric_distances:
        raise ValueError("Não há pares válidos para calibração.")

    alpha = optimal_scale_alpha(metric_distances, graph_distances)

    residuals = [alpha * x - y for x, y in zip(metric_distances, graph_distances)]
    abs_errors = [abs(r) for r in residuals]
    sq_errors = [r * r for r in residuals]
    pct_errors = [abs(r) / y * 100.0 for r, y in zip(residuals, graph_distances) if y > 0]
    distortions = [y / x for x, y in zip(metric_distances, graph_distances) if x > 0]

    return CalibrationRecord(
        p=float(p) if not math.isinf(p) else math.inf,
        scale_alpha=float(alpha),
        mae_m=float(mean(abs_errors)),
        rmse_m=float(math.sqrt(mean(sq_errors))),
        mape_pct=float(mean(pct_errors)),
        mean_distortion=float(mean(distortions)),
        n_pairs=len(metric_distances),
    )


def calibration_curve(
    pairs: list[tuple[XY, XY, float]],
    p_values: list[float],
) -> list[CalibrationRecord]:
    """Calcula curva de calibração para uma grade de valores de p."""
    return [calibration_for_p(pairs, p=p) for p in p_values]


def best_calibration_record(
    records: list[CalibrationRecord],
    criterion: str = "mape_pct",
) -> CalibrationRecord:
    """Seleciona o melhor p segundo MAE, RMSE ou MAPE."""
    if not records:
        raise ValueError("records não pode ser vazio.")

    valid_criteria = {"mae_m", "rmse_m", "mape_pct"}
    if criterion not in valid_criteria:
        raise ValueError(f"criterion deve estar em {valid_criteria}.")

    return min(records, key=lambda r: getattr(r, criterion))


def records_to_dicts(records: list[CalibrationRecord]) -> list[dict[str, float]]:
    """Converte registros de calibração para DataFrame/CSV."""
    return [
        {
            "p": r.p,
            "alpha": r.scale_alpha,
            "MAE_m": r.mae_m,
            "RMSE_m": r.rmse_m,
            "MAPE_pct": r.mape_pct,
            "mean_distortion": r.mean_distortion,
            "n_pairs": r.n_pairs,
        }
        for r in records
    ]
