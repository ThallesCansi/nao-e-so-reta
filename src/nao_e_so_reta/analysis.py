from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .config import LatLon
from .norms import (
    DEFAULT_P_VALUES,
    abs_error_col,
    ape_col,
    distance_col,
    ensure_p_values,
    error_col,
    lp_distance_arrays,
    lp_distance_xy,
    metric_label,
    p_value_label,
    tau_col,
)
from .routing import nearest_node_from_query, node_xy, route_length, shortest_route


@dataclass(frozen=True)
class MetricComparison:
    """Comparação entre uma métrica plana e a distância no grafo."""

    name: str
    distance_m: float
    graph_distance_m: float | None
    absolute_error_m: float | None
    relative_error_pct: float | None
    graph_over_metric: float | None


@dataclass(frozen=True)
class CalibrationRecord:
    """Registro de calibração compatível com o painel Streamlit."""

    p: float
    scale_alpha: float
    mae_m: float
    rmse_m: float
    mape_pct: float
    mean_distortion: float
    n_pairs: int


@dataclass(frozen=True)
class SinglePairResult:
    """Resultado do estudo pontual entre dois nós."""

    origin_node: int
    destination_node: int
    route: list[int]
    graph_distance_m: float
    distances_m: dict[float, float]
    tortuosities: dict[float, float]


def safe_relative_error_pct(estimate: float, reference: float | None) -> float | None:
    """Erro percentual assinado em relação a uma referência."""
    if reference is None or reference <= 0:
        return None
    return ((estimate - reference) / reference) * 100.0


def safe_ratio(
    numerator: float | np.ndarray | None,
    denominator: float | np.ndarray | None,
):
    """Razão protegida contra divisão por zero."""
    if numerator is None or denominator is None:
        return None
    if np.isscalar(numerator) and np.isscalar(denominator):
        return None if float(denominator) <= 0 else float(numerator) / float(denominator)
    return np.where(np.asarray(denominator) > 0, np.asarray(numerator) / np.asarray(denominator), np.nan)


def compare_norms_for_pair(
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    *,
    graph_distance_m: float | None,
    p: float,
) -> list[MetricComparison]:
    """Compara distâncias Lp com a distância real no grafo."""
    from .norms import named_lp_distances

    rows: list[MetricComparison] = []
    for name, distance in named_lp_distances(a_xy, b_xy, p=p).items():
        rows.append(
            MetricComparison(
                name=name,
                distance_m=float(distance),
                graph_distance_m=graph_distance_m,
                absolute_error_m=None if graph_distance_m is None else float(distance - graph_distance_m),
                relative_error_pct=safe_relative_error_pct(float(distance), graph_distance_m),
                graph_over_metric=safe_ratio(graph_distance_m, float(distance)),
            )
        )
    return rows


def tortuosity(graph_distance_m: float | None, euclidean_distance_m: float) -> float | None:
    """Tortuosidade clássica: distância na rede dividida pela distância euclidiana."""
    return safe_ratio(graph_distance_m, euclidean_distance_m)


def optimal_scale_alpha(metric_distances: list[float], graph_distances: list[float]) -> float:
    """Ajusta alpha em graph ≈ alpha * metric por mínimos quadrados sem intercepto."""
    if len(metric_distances) != len(graph_distances):
        raise ValueError("As listas devem ter o mesmo tamanho.")
    if not metric_distances:
        raise ValueError("É necessário ao menos um par de distâncias.")

    numerator = sum(x * y for x, y in zip(metric_distances, graph_distances, strict=True))
    denominator = sum(x * x for x in metric_distances)
    if denominator <= 0:
        raise ValueError("As distâncias métricas não podem ser todas zero.")
    return numerator / denominator


def calibration_for_p(pairs: list[tuple[tuple[float, float], tuple[float, float], float]], p: float) -> CalibrationRecord:
    """Avalia um valor de p em uma coleção de pares (a_xy, b_xy, d_G)."""
    metric_distances: list[float] = []
    graph_distances: list[float] = []

    for a_xy, b_xy, graph_distance in pairs:
        d_metric = lp_distance_xy(a_xy, b_xy, p=p)
        if d_metric > 0 and graph_distance > 0:
            metric_distances.append(float(d_metric))
            graph_distances.append(float(graph_distance))

    if not metric_distances:
        raise ValueError("Não há pares válidos para calibração.")

    alpha = optimal_scale_alpha(metric_distances, graph_distances)
    residuals = [
        alpha * x - y for x, y in zip(metric_distances, graph_distances, strict=True)
    ]
    abs_errors = [abs(r) for r in residuals]
    pct_errors = [
        abs(r) / y * 100.0 for r, y in zip(residuals, graph_distances, strict=True) if y > 0
    ]
    distortions = [
        y / x for x, y in zip(metric_distances, graph_distances, strict=True) if x > 0
    ]

    return CalibrationRecord(
        p=float(p) if not math.isinf(p) else math.inf,
        scale_alpha=float(alpha),
        mae_m=float(np.mean(abs_errors)),
        rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
        mape_pct=float(np.mean(pct_errors)),
        mean_distortion=float(np.mean(distortions)),
        n_pairs=len(metric_distances),
    )


def calibration_curve(
    pairs: list[tuple[tuple[float, float], tuple[float, float], float]],
    p_values: Iterable[float],
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
    """Calcula d_G, cada d_p e cada tortuosidade tau_p=d_G/d_p para um par."""
    p_values = ensure_p_values(p_values)
    route = shortest_route(graph, origin_node, destination_node, weight=weight)
    d_graph = route_length(graph, route, weight=weight)
    a = node_xy(graph, origin_node)
    b = node_xy(graph, destination_node)

    distances = {p: lp_distance_xy(a, b, p) for p in p_values}
    tortuosities = {p: (d_graph / d if d > 0 else math.nan) for p, d in distances.items()}

    return SinglePairResult(
        origin_node=origin_node,
        destination_node=destination_node,
        route=route,
        graph_distance_m=float(d_graph),
        distances_m=distances,
        tortuosities=tortuosities,
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
    """Geocodifica origem/destino, encontra os nós mais próximos e calcula o estudo pontual."""
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


def single_pair_table(result: SinglePairResult, *, sort_by_abs_error: bool = True) -> pd.DataFrame:
    """Tabela legível para o caso pontual, com uma linha por métrica."""
    rows: list[dict[str, float | str]] = []
    for p, d_metric in result.distances_m.items():
        error = d_metric - result.graph_distance_m
        abs_error = abs(error)
        rows.append(
            {
                "Métrica": metric_label(p),
                "p": p_value_label(p),
                "d_p (km)": d_metric / 1000.0,
                "d_G (km)": result.graph_distance_m / 1000.0,
                "τ_p = d_G/d_p": result.tortuosities[p],
                "Erro absoluto (m)": abs_error,
                "Erro relativo (%)": abs_error / result.graph_distance_m * 100.0,
            }
        )
    out = pd.DataFrame(rows)
    if sort_by_abs_error:
        out = out.sort_values("Erro absoluto (m)").reset_index(drop=True)
    return out


def single_pair_wide_dataframe(result: SinglePairResult) -> pd.DataFrame:
    """DataFrame de uma linha, útil para salvar resultados brutos em CSV."""
    row: dict[str, float | int] = {
        "origin_node": result.origin_node,
        "destination_node": result.destination_node,
        "d_graph_m": result.graph_distance_m,
    }
    for p, d in result.distances_m.items():
        row[distance_col(p)] = d
        row[tau_col(p)] = result.tortuosities[p]
        row[error_col(p)] = d - result.graph_distance_m
        row[abs_error_col(p)] = abs(d - result.graph_distance_m)
        row[ape_col(p)] = abs(d - result.graph_distance_m) / result.graph_distance_m * 100.0
    return pd.DataFrame([row])


def sample_vertex_pairs(
    graph: Any,
    *,
    n_pairs: int = 10_000,
    n_origins: int | None = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Amostra pares de vértices. Agrupar origens reduz o número de execuções de Dijkstra."""
    if n_pairs <= 0:
        raise ValueError("n_pairs deve ser positivo.")
    nodes = np.array(list(graph.nodes), dtype=object)
    if len(nodes) < 2:
        raise ValueError("O grafo precisa ter pelo menos dois vértices.")

    rng = np.random.default_rng(seed)
    if n_origins is None:
        origins = rng.choice(nodes, size=n_pairs, replace=True)
    else:
        n_origins = int(min(max(1, n_origins), len(nodes), n_pairs))
        chosen = rng.choice(nodes, size=n_origins, replace=False)
        counts = np.full(n_origins, n_pairs // n_origins, dtype=int)
        counts[: n_pairs % n_origins] += 1
        origins = np.repeat(chosen, counts)
        rng.shuffle(origins)

    targets = rng.choice(nodes, size=n_pairs, replace=True)
    conflicts = targets == origins
    while conflicts.any():
        targets[conflicts] = rng.choice(nodes, size=int(conflicts.sum()), replace=True)
        conflicts = targets == origins

    return pd.DataFrame({"origin": origins, "target": targets})


def _add_node_coordinates(graph: Any, pairs_df: pd.DataFrame) -> pd.DataFrame:
    x = nx.get_node_attributes(graph, "x")
    y = nx.get_node_attributes(graph, "y")
    out = pairs_df.copy()
    out["origin_x"] = out["origin"].map(x).astype(float)
    out["origin_y"] = out["origin"].map(y).astype(float)
    out["target_x"] = out["target"].map(x).astype(float)
    out["target_y"] = out["target"].map(y).astype(float)
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
    """Adiciona d_G calculando Dijkstra uma vez por origem distinta."""
    out = pairs_df.copy()
    out["d_graph_m"] = np.nan
    groups = out.groupby(origin_col, sort=False).groups
    iterator = groups.items()
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(groups), desc="Dijkstra por origem")
        except ImportError:
            pass

    for origin, idx in iterator:
        lengths = nx.single_source_dijkstra_path_length(graph, source=origin, weight=weight)
        targets = out.loc[idx, target_col]
        out.loc[idx, "d_graph_m"] = [float(lengths.get(target, math.nan)) for target in targets]
    return out


def add_metric_columns(
    df: pd.DataFrame,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
    graph_col: str = "d_graph_m",
) -> pd.DataFrame:
    """Adiciona d_p, tau_p=d_G/d_p e erros para cada p."""
    out = df.copy()
    p_values = ensure_p_values(p_values)
    dx = (out["origin_x"] - out["target_x"]).abs().to_numpy(dtype=float)
    dy = (out["origin_y"] - out["target_y"]).abs().to_numpy(dtype=float)
    d_graph = out[graph_col].to_numpy(dtype=float)

    for p in p_values:
        d = lp_distance_arrays(dx, dy, p)
        err = d - d_graph
        out[distance_col(p)] = d
        out[tau_col(p)] = np.where(np.isfinite(d_graph) & np.isfinite(d) & (d > 0), d_graph / d, np.nan)
        out[error_col(p)] = err
        out[abs_error_col(p)] = np.abs(err)
        out[ape_col(p)] = np.where(d_graph > 0, np.abs(err) / d_graph * 100.0, np.nan)
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
    """Calcula d_G, d_p, tau_p e erros para uma tabela com colunas origin/target."""
    missing = {"origin", "target"}.difference(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df precisa conter origin e target. Faltam: {sorted(missing)}")

    out = _add_node_coordinates(graph, pairs_df)
    out = add_graph_distances_grouped(graph, out, weight=weight, show_progress=show_progress)
    if drop_unreachable:
        out = out.dropna(subset=["d_graph_m"]).reset_index(drop=True)
    out = add_metric_columns(out, p_values=p_values, graph_col="d_graph_m")
    return out


def summarize_metric_errors(
    results_df: pd.DataFrame,
    *,
    p_values: Iterable[float] = DEFAULT_P_VALUES,
) -> pd.DataFrame:
    """Resume erro e tortuosidade relativa para cada métrica."""
    p_values = ensure_p_values(p_values)
    rows: list[dict[str, float | int | str]] = []
    for p in p_values:
        d_col = distance_col(p)
        t_col = tau_col(p)
        e_col = error_col(p)
        ae_col = abs_error_col(p)
        a_col = ape_col(p)
        if d_col not in results_df.columns:
            continue

        values = results_df[[d_col, t_col, e_col, ae_col, a_col, "d_graph_m"]].replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values["d_graph_m"] > 0]
        if values.empty:
            continue

        rows.append(
            {
                "Métrica": metric_label(p),
                "p": p_value_label(p),
                "n": int(len(values)),
                "MAE (m)": float(values[ae_col].mean()),
                "MAPE (%)": float(values[a_col].mean()),
                "RMSE (m)": float(np.sqrt(np.mean(values[e_col] ** 2))),
                "Viés médio (m)": float(values[e_col].mean()),
                "Tortuosidade média": float(values[t_col].mean()),
                "Tortuosidade mediana": float(values[t_col].median()),
                "Tortuosidade p05": float(values[t_col].quantile(0.05)),
                "Tortuosidade p95": float(values[t_col].quantile(0.95)),
                "Distância média d_p (km)": float(values[d_col].mean() / 1000.0),
                "Distância média d_G (km)": float(values["d_graph_m"].mean() / 1000.0),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)


def evaluate_p_grid(
    results_df: pd.DataFrame,
    p_values_grid: Iterable[float],
) -> pd.DataFrame:
    """Avalia uma grade de p usando as coordenadas já calculadas, sem recalcular d_G."""
    required = {"d_graph_m", "origin_x", "origin_y", "target_x", "target_y"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Faltam colunas: {sorted(missing)}")

    dx = (results_df["origin_x"] - results_df["target_x"]).abs().to_numpy(dtype=float)
    dy = (results_df["origin_y"] - results_df["target_y"]).abs().to_numpy(dtype=float)
    d_graph = results_df["d_graph_m"].to_numpy(dtype=float)
    valid_graph = np.isfinite(d_graph) & (d_graph > 0)

    rows = []
    for p in p_values_grid:
        p = float(p)
        if p < 1:
            continue
        d = lp_distance_arrays(dx, dy, p)
        valid = valid_graph & np.isfinite(d) & (d > 0)
        if not valid.any():
            continue
        err = d[valid] - d_graph[valid]
        abs_err = np.abs(err)
        ape = abs_err / d_graph[valid] * 100.0
        tau = d_graph[valid] / d[valid]
        rows.append(
            {
                "p": p,
                "Métrica": metric_label(p),
                "n": int(valid.sum()),
                "MAE (m)": float(abs_err.mean()),
                "MAPE (%)": float(ape.mean()),
                "RMSE (m)": float(np.sqrt(np.mean(err**2))),
                "Viés médio (m)": float(err.mean()),
                "Tortuosidade média": float(tau.mean()),
                "Tortuosidade mediana": float(np.median(tau)),
            }
        )
    return pd.DataFrame(rows)


def best_p_by(grid_results: pd.DataFrame, criterion: str = "MAPE (%)") -> pd.Series:
    """Retorna a linha com menor valor no critério escolhido."""
    if criterion not in grid_results.columns:
        raise ValueError(f"Critério {criterion!r} não está em grid_results.")
    if grid_results.empty:
        raise ValueError("grid_results está vazio.")
    return grid_results.loc[grid_results[criterion].idxmin()]


def top_metrics(summary_df: pd.DataFrame, *, n: int = 5, criterion: str = "MAPE (%)") -> pd.DataFrame:
    """Seleciona as melhores métricas segundo o critério escolhido."""
    if criterion not in summary_df.columns:
        raise ValueError(f"Critério {criterion!r} não está em summary_df.")
    return summary_df.sort_values(criterion).head(n).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, filepath: str | Path, *, index: bool = False) -> Path:
    """Salva DataFrame em CSV, criando diretórios quando necessário."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path
