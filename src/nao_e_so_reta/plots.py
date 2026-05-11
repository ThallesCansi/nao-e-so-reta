from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd

from .norms import distance_col, metric_latex, metric_label, p_value_label, tau_col, tau_latex


def _save(fig, filepath: str | Path | None, *, dpi: int = 250):
    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_route(graph: Any, route: list[int], *, filepath: str | Path | None = None, show: bool = True, close: bool = False):
    """Plota o menor caminho no grafo usando OSMnx."""
    import osmnx as ox

    fig, ax = ox.plot_graph_route(
        graph,
        route,
        node_size=0,
        route_linewidth=4,
        show=show,
        close=close,
    )
    _save(fig, filepath)
    return fig, ax


def plot_metric_scatter(
    results_df: pd.DataFrame,
    p: float,
    *,
    filepath: str | Path | None = None,
    max_points: int = 5000,
    alpha: float = 0.25,
    s: float = 8.0,
):
    """Scatter de d_p contra d_G para uma métrica."""
    import matplotlib.pyplot as plt

    d_col = distance_col(p)
    if d_col not in results_df.columns:
        raise ValueError(f"Coluna {d_col!r} não encontrada.")

    data = results_df[["d_graph_m", d_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=42)

    x = data["d_graph_m"] / 1000.0
    y = data[d_col] / 1000.0
    lim = float(np.nanmax([x.max(), y.max()]))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.scatter(x, y, alpha=alpha, s=s)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1, label=r"$y=x$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Distância intrínseca no grafo, $d_G$ (km)")
    ax.set_ylabel(rf"Distância plana, {metric_latex(p)} (km)")
    ax.set_title(rf"Comparação entre $d_G$ e {metric_latex(p)}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, filepath)
    return fig, ax


def plot_top_metric_scatters(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    n: int = 5,
    criterion: str = "MAPE (%)",
    filepath: str | Path | None = None,
    max_points: int = 4000,
    alpha: float = 0.28,
    s: float = 8.0,
):
    """Plota, em um único gráfico, as n melhores métricas contra d_G."""
    import matplotlib.pyplot as plt

    top = summary_df.sort_values(criterion).head(n)
    data = results_df.copy()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=130)
    x = data["d_graph_m"] / 1000.0
    max_y = float(x.max())

    for _, row in top.iterrows():
        p_raw = row["p"]
        p = math.inf if str(p_raw) == "∞" else float(p_raw)
        d_col = distance_col(p)
        if d_col not in data.columns:
            continue
        y = data[d_col] / 1000.0
        max_y = max(max_y, float(y.max()))
        ax.scatter(x, y, alpha=alpha, s=s, label=metric_latex(p))

    ax.plot([0, max_y], [0, max_y], linestyle="--", linewidth=1.2, label=r"$y=x$")
    ax.set_xlim(0, max_y)
    ax.set_ylim(0, max_y)
    ax.set_xlabel(r"Distância intrínseca no grafo, $d_G$ (km)")
    ax.set_ylabel(r"Distância plana $d_p$ (km)")
    ax.set_title(rf"{n} melhores métricas por {criterion}: $d_p$ versus $d_G$")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Métrica", ncol=2)
    _save(fig, filepath)
    return fig, ax


def plot_error_by_p(
    grid_results: pd.DataFrame,
    *,
    criterion: str = "MAPE (%)",
    filepath: str | Path | None = None,
):
    """Curva de erro em função de p."""
    import matplotlib.pyplot as plt

    if "p" not in grid_results.columns or criterion not in grid_results.columns:
        raise ValueError("grid_results precisa conter 'p' e o critério escolhido.")
    best = grid_results.loc[grid_results[criterion].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.plot(grid_results["p"], grid_results[criterion], linewidth=1.6)
    ax.axvline(float(best["p"]), linestyle="--", linewidth=1, label=rf"melhor $p={float(best['p']):.2f}$")
    ax.set_xlabel(r"Parâmetro $p$")
    ax.set_ylabel(criterion)
    ax.set_title(rf"Erro em função de $p$ ({criterion})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, filepath)
    return fig, ax


def plot_tortuosity_boxplot(
    results_df: pd.DataFrame,
    p_values: Iterable[float],
    *,
    filepath: str | Path | None = None,
):
    """Boxplot das tortuosidades tau_p=d_G/d_p para várias métricas."""
    import matplotlib.pyplot as plt

    labels = []
    values = []
    for p in p_values:
        col = tau_col(p)
        if col not in results_df.columns:
            continue
        s = results_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        labels.append(metric_label(p))
        values.append(s.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Métrica de referência")
    ax.set_ylabel(r"Tortuosidade relativa, $\tau_p = d_G/d_p$")
    ax.set_title(r"Distribuição das tortuosidades relativas")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, filepath)
    return fig, ax


def plot_tortuosity_hist(
    results_df: pd.DataFrame,
    p: float,
    *,
    bins: int = 40,
    filepath: str | Path | None = None,
):
    """Histograma de tau_p=d_G/d_p para uma métrica."""
    import matplotlib.pyplot as plt

    col = tau_col(p)
    if col not in results_df.columns:
        raise ValueError(f"Coluna {col!r} não encontrada.")
    values = results_df[col].replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.hist(values, bins=bins)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel(tau_latex(p))
    ax.set_ylabel("Frequência")
    ax.set_title(rf"Distribuição de {tau_latex(p)}")
    ax.grid(True, alpha=0.3)
    _save(fig, filepath)
    return fig, ax
