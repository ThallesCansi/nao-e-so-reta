from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from nao_e_so_reta.config import LatLon, XY
from nao_e_so_reta.norms import (
    DEFAULT_P_VALUES,
    lp_distance_xy,
    metric_column_from_p,
    metric_name_from_p,
    named_lp_distances,
)
from nao_e_so_reta.routing import (
    nearest_node_from_query,
    node_xy,
    route_length,
    shortest_route,
)


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


@dataclass(frozen=True)
class SinglePairResult:
    """Resultado completo para um único par origem-destino."""

    origin_node: int
    destination_node: int
    route: list[int]
    graph_distance_m: float
    metrics: dict[str, float]
    tortuosity: float


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


def compute_single_pair_result(
    graph: Any,
    origin_node: int,
    destination_node: int,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    weight: str = "length",
) -> SinglePairResult:
    """Calcula rota, distância no grafo e métricas Lp para um par de nós."""
    route = shortest_route(graph, origin_node, destination_node, weight=weight)
    d_graph = route_length(graph, route, weight=weight)

    metrics = {
        metric_column_from_p(p): lp_distance_xy(node_xy(graph, origin_node), node_xy(graph, destination_node), p)
        for p in p_values
    }

    d2 = metrics.get(metric_column_from_p(2.0))
    if d2 is None:
        d2 = lp_distance_xy(node_xy(graph, origin_node), node_xy(graph, destination_node), 2.0)

    return SinglePairResult(
        origin_node=origin_node,
        destination_node=destination_node,
        route=route,
        graph_distance_m=float(d_graph),
        metrics=metrics,
        tortuosity=float(d_graph / d2) if d2 and d2 > 0 else math.nan,
    )


def compute_single_pair_from_queries(
    graph: Any,
    origin_query: str,
    destination_query: str,
    *,
    origin_fallback_latlon: LatLon | None = None,
    destination_fallback_latlon: LatLon | None = None,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    weight: str = "length",
) -> tuple[SinglePairResult, LatLon, LatLon]:
    """Geocodifica origem/destino, encontra nós próximos e calcula métricas."""
    origin_node, origin_point = nearest_node_from_query(
        graph, origin_query, fallback_latlon=origin_fallback_latlon
    )
    destination_node, destination_point = nearest_node_from_query(
        graph, destination_query, fallback_latlon=destination_fallback_latlon
    )
    result = compute_single_pair_result(
        graph,
        origin_node,
        destination_node,
        p_values=p_values,
        weight=weight,
    )
    return result, origin_point, destination_point


def single_pair_to_dataframe(result: SinglePairResult) -> pd.DataFrame:
    """Converte resultado pontual para DataFrame de uma linha."""
    row = {
        "origin_node": result.origin_node,
        "destination_node": result.destination_node,
        "d_graph_m": result.graph_distance_m,
        "tortuosity_dG_dL2": result.tortuosity,
    }
    row.update(result.metrics)
    return pd.DataFrame([row])


def add_lp_columns(
    df: pd.DataFrame,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    origin_x: str = "origin_x",
    origin_y: str = "origin_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
) -> pd.DataFrame:
    """Adiciona colunas d_Lp_m a um DataFrame com coordenadas projetadas."""
    out = df.copy()
    dx = (out[origin_x] - out[target_x]).abs().to_numpy(dtype=float)
    dy = (out[origin_y] - out[target_y]).abs().to_numpy(dtype=float)

    for p in p_values:
        col = metric_column_from_p(p)
        if math.isinf(p):
            out[col] = np.maximum(dx, dy)
        else:
            out[col] = (dx**p + dy**p) ** (1.0 / p)

    return out


def add_graph_distances_grouped(
    graph: Any,
    pairs_df: pd.DataFrame,
    *,
    origin_col: str = "origin",
    target_col: str = "target",
    weight: str = "length",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Adiciona d_graph_m calculando Dijkstra uma vez por origem."""
    out = pairs_df.copy()
    out["d_graph_m"] = np.nan

    grouped = out.groupby(origin_col, sort=False).groups
    iterator = grouped.items()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(grouped), desc="Dijkstra por origem")
        except ImportError:
            pass

    for origin, idx in iterator:
        lengths = nx.single_source_dijkstra_path_length(graph, source=origin, weight=weight)
        targets = out.loc[idx, target_col]
        out.loc[idx, "d_graph_m"] = [float(lengths.get(t, math.nan)) for t in targets]

    return out


def _node_coordinate_columns(graph: Any, pairs_df: pd.DataFrame) -> pd.DataFrame:
    x = nx.get_node_attributes(graph, "x")
    y = nx.get_node_attributes(graph, "y")
    out = pairs_df.copy()
    out["origin_x"] = out["origin"].map(x).astype(float)
    out["origin_y"] = out["origin"].map(y).astype(float)
    out["target_x"] = out["target"].map(x).astype(float)
    out["target_y"] = out["target"].map(y).astype(float)
    return out


def compute_pair_metrics(
    graph: Any,
    pairs_df: pd.DataFrame,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    weight: str = "length",
    show_progress: bool = False,
    drop_unreachable: bool = True,
) -> pd.DataFrame:
    """Calcula d_G e distâncias Lp para uma tabela com colunas origin/target."""
    required = {"origin", "target"}
    missing = required.difference(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df precisa conter as colunas {sorted(required)}. Faltam: {sorted(missing)}")

    out = _node_coordinate_columns(graph, pairs_df)
    out = add_graph_distances_grouped(graph, out, weight=weight, show_progress=show_progress)
    out = add_lp_columns(out, p_values=p_values)

    d2_col = metric_column_from_p(2.0)
    if d2_col not in out.columns:
        out = add_lp_columns(out, p_values=[2.0])
    out["tortuosity_dG_dL2"] = out["d_graph_m"] / out[d2_col]

    if drop_unreachable:
        out = out.dropna(subset=["d_graph_m"]).reset_index(drop=True)
    return out


def summarize_metric_errors(
    results_df: pd.DataFrame,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    graph_col: str = "d_graph_m",
) -> pd.DataFrame:
    """Resume MAE, MAPE, viés, RMSE e razões para cada métrica Lp."""
    if graph_col not in results_df.columns:
        raise ValueError(f"Coluna {graph_col!r} não encontrada.")

    rows: list[dict[str, float | int | str]] = []
    d_graph = results_df[graph_col].to_numpy(dtype=float)
    valid_graph = np.isfinite(d_graph) & (d_graph > 0)

    for p in p_values:
        col = metric_column_from_p(p)
        if col not in results_df.columns:
            continue

        d_metric = results_df[col].to_numpy(dtype=float)
        valid = valid_graph & np.isfinite(d_metric) & (d_metric > 0)
        if not valid.any():
            continue

        errors = d_metric[valid] - d_graph[valid]
        abs_errors = np.abs(errors)
        ape = abs_errors / d_graph[valid]
        ratio_graph_metric = d_graph[valid] / d_metric[valid]
        ratio_metric_graph = d_metric[valid] / d_graph[valid]

        rows.append(
            {
                "metric": metric_name_from_p(p),
                "p": np.inf if math.isinf(p) else float(p),
                "n": int(valid.sum()),
                "MAE_m": float(np.mean(abs_errors)),
                "median_AE_m": float(np.median(abs_errors)),
                "MAPE_percent": float(np.mean(ape) * 100.0),
                "median_APE_percent": float(np.median(ape) * 100.0),
                "bias_m": float(np.mean(errors)),
                "RMSE_m": float(np.sqrt(np.mean(errors**2))),
                "mean_d_metric_m": float(np.mean(d_metric[valid])),
                "mean_d_graph_m": float(np.mean(d_graph[valid])),
                "mean_dG_over_metric": float(np.mean(ratio_graph_metric)),
                "median_dG_over_metric": float(np.median(ratio_graph_metric)),
                "mean_metric_over_dG": float(np.mean(ratio_metric_graph)),
                "median_metric_over_dG": float(np.median(ratio_metric_graph)),
            }
        )

    return pd.DataFrame(rows).sort_values("MAPE_percent").reset_index(drop=True)


def summarize_tortuosity(
    results_df: pd.DataFrame,
    *,
    tortuosity_col: str = "tortuosity_dG_dL2",
) -> pd.Series:
    """Resumo estatístico da tortuosidade d_G/d_2."""
    if tortuosity_col not in results_df.columns:
        raise ValueError(f"Coluna {tortuosity_col!r} não encontrada.")

    values = results_df[tortuosity_col].replace([np.inf, -np.inf], np.nan).dropna()
    return pd.Series(
        {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
        }
    )


def p_grid(start: float = 1.0, stop: float = 10.0, step: float = 0.01) -> np.ndarray:
    """Cria uma grade fechada de valores de p."""
    if start <= 0 or stop < start or step <= 0:
        raise ValueError("Use 0 < start <= stop e step > 0.")
    n = int(round((stop - start) / step))
    return np.round(start + step * np.arange(n + 1), 10)


def evaluate_p_grid(
    results_df: pd.DataFrame,
    p_values_grid: Iterable[float],
    *,
    graph_col: str = "d_graph_m",
    origin_x: str = "origin_x",
    origin_y: str = "origin_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
) -> pd.DataFrame:
    """Avalia uma grade de p sem recalcular menores caminhos no grafo."""
    required = {graph_col, origin_x, origin_y, target_x, target_y}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Faltam colunas em results_df: {sorted(missing)}")

    d_graph = results_df[graph_col].to_numpy(dtype=float)
    dx = (results_df[origin_x] - results_df[target_x]).abs().to_numpy(dtype=float)
    dy = (results_df[origin_y] - results_df[target_y]).abs().to_numpy(dtype=float)
    valid_graph = np.isfinite(d_graph) & (d_graph > 0)

    rows = []
    for p in p_values_grid:
        p = float(p)
        if p <= 0:
            continue
        d_p = (dx**p + dy**p) ** (1.0 / p)
        valid = valid_graph & np.isfinite(d_p) & (d_p > 0)
        if not valid.any():
            continue

        errors = d_p[valid] - d_graph[valid]
        abs_errors = np.abs(errors)
        ape = abs_errors / d_graph[valid]
        rows.append(
            {
                "p": p,
                "n": int(valid.sum()),
                "MAE_m": float(np.mean(abs_errors)),
                "MAPE_percent": float(np.mean(ape) * 100.0),
                "RMSE_m": float(np.sqrt(np.mean(errors**2))),
                "bias_m": float(np.mean(errors)),
            }
        )

    return pd.DataFrame(rows)


def best_p_by(grid_results: pd.DataFrame, criterion: str = "MAPE_percent") -> pd.Series:
    """Retorna a linha da grade com menor valor no critério escolhido."""
    if criterion not in grid_results.columns:
        raise ValueError(f"Critério {criterion!r} não está em grid_results.")
    if grid_results.empty:
        raise ValueError("grid_results está vazio.")
    return grid_results.loc[grid_results[criterion].idxmin()]


def save_dataframe(df: pd.DataFrame, filepath: str | Path, *, index: bool = False) -> Path:
    """Salva DataFrame em CSV criando diretórios se necessário."""
    from pathlib import Path

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path
