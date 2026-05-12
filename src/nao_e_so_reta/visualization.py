from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import pyproj

from nao_e_so_reta.analysis import best_p_by
from nao_e_so_reta.config import XY, LatLon
from nao_e_so_reta.norms import (
    lp_distance_xy,
    manhattan_polyline_xy,
    metric_column_from_p,
    superellipse_boundary_xy,
    visual_minkowski_curve_xy,
)
from nao_e_so_reta.projections import points_xy_to_latlon


def base_map(center: LatLon, zoom: int) -> folium.Map:
    """Cria mapa Folium base."""
    return folium.Map(location=list(center), zoom_start=zoom, control_scale=True)


def add_point_markers(map_obj: folium.Map, markers: list[LatLon]) -> None:
    """Adiciona origem e destino ao mapa."""
    for i, coords in enumerate(markers):
        color = "green" if i == 0 else "red"
        label = "Origem" if i == 0 else "Destino"
        folium.Marker(location=coords, icon=folium.Icon(color=color), tooltip=label).add_to(map_obj)


def add_nearest_node_markers(
    map_obj: folium.Map,
    origin_node_latlon: LatLon,
    destination_node_latlon: LatLon,
) -> None:
    """Mostra os nós efetivamente usados no cálculo."""
    folium.CircleMarker(
        location=origin_node_latlon,
        radius=4,
        color="darkgreen",
        fill=True,
        tooltip="Nó viário mais próximo da origem",
    ).add_to(map_obj)
    folium.CircleMarker(
        location=destination_node_latlon,
        radius=4,
        color="darkred",
        fill=True,
        tooltip="Nó viário mais próximo do destino",
    ).add_to(map_obj)


def add_route_polyline(map_obj: folium.Map, route_latlon: list[LatLon]) -> None:
    """Adiciona rota real da rede."""
    if not route_latlon:
        return
    folium.PolyLine(
        route_latlon,
        color="blue",
        weight=5,
        opacity=0.75,
        tooltip="Distância real no grafo viário",
    ).add_to(map_obj)


def add_theoretical_layers(
    map_obj: folium.Map,
    *,
    origin_click: LatLon,
    destination_click: LatLon,
    origin_xy: XY,
    destination_xy: XY,
    transformer_projected_to_wgs84: pyproj.Transformer,
    p: float,
    show_euclidean: bool,
    show_manhattan: bool,
    show_minkowski_ball: bool,
    show_minkowski_curve: bool,
    n_superellipse_points: int,
    n_visual_curve_points: int,
) -> None:
    """Adiciona camadas teóricas Lp ao mapa."""
    if show_euclidean:
        folium.PolyLine(
            [origin_click, destination_click],
            color="green",
            weight=2,
            dash_array="5, 10",
            tooltip="Euclidiana visual entre cliques",
        ).add_to(map_obj)

    if show_manhattan:
        for order, opacity in [("x-then-y", 0.65), ("y-then-x", 0.35)]:
            path_xy = manhattan_polyline_xy(origin_xy, destination_xy, order=order)
            path_latlon = points_xy_to_latlon(path_xy, transformer_projected_to_wgs84)
            folium.PolyLine(
                path_latlon,
                color="red",
                weight=3,
                opacity=opacity,
                dash_array="5, 10",
                tooltip=f"Manhattan visual ({order})",
            ).add_to(map_obj)

    if show_minkowski_ball:
        radius = lp_distance_xy(origin_xy, destination_xy, p=p)
        boundary_xy = superellipse_boundary_xy(
            origin_xy,
            radius,
            p=p,
            n_points=n_superellipse_points,
        )
        boundary_latlon = points_xy_to_latlon(boundary_xy, transformer_projected_to_wgs84)
        folium.Polygon(
            locations=boundary_latlon,
            color="orange",
            weight=3,
            fill=False,
            dash_array="5, 5",
            tooltip=f"Fronteira da bola Lp, p={p:.2f}",
        ).add_to(map_obj)

    if show_minkowski_curve:
        curve_xy = visual_minkowski_curve_xy(
            origin_xy,
            destination_xy,
            p=p,
            n_points=n_visual_curve_points,
        )
        curve_latlon = points_xy_to_latlon(curve_xy, transformer_projected_to_wgs84)
        folium.PolyLine(
            locations=curve_latlon,
            color="purple",
            weight=4,
            opacity=0.85,
            tooltip=f"Curva didática Minkowski, p={p:.2f}",
        ).add_to(map_obj)


def _line_legend(label: str, color: str, dashed: bool = False) -> str:
    dash_css = (
        f"background-image: repeating-linear-gradient(to right, {color} 0 8px, transparent 8px 14px);"
        if dashed
        else f"background:{color};"
    )
    return (
        "<div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>"
        f"<span style='width:24px; height:4px; border-radius:999px; display:inline-block; {dash_css}'></span>"
        f"<span style='font-size:13px; color:#222;'>{escape(label)}</span>"
        "</div>"
    )


def _dot_legend(label: str, color: str) -> str:
    return (
        "<div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>"
        f"<span style='width:12px; height:12px; border-radius:50%; background:{color}; "
        "border:1px solid #444; display:inline-block;'></span>"
        f"<span style='font-size:13px; color:#222;'>{escape(label)}</span>"
        "</div>"
    )


def add_legend(
    map_obj: folium.Map,
    *,
    has_origin: bool,
    has_destination: bool,
    has_route: bool,
    show_euclidean: bool,
    show_manhattan: bool,
    show_minkowski_ball: bool,
    show_minkowski_curve: bool,
) -> None:
    """Adiciona legenda HTML ao mapa."""
    items: list[str] = []

    if has_origin:
        items.append(_dot_legend("Origem clicada", "green"))
        items.append(_dot_legend("Nó usado na origem", "darkgreen"))
    if has_destination:
        items.append(_dot_legend("Destino clicado", "red"))
        items.append(_dot_legend("Nó usado no destino", "darkred"))
    if has_route:
        items.append(_line_legend("Rota real no grafo", "blue"))
    if show_euclidean:
        items.append(_line_legend("Euclidiana visual", "green", dashed=True))
    if show_manhattan:
        items.append(_line_legend("Manhattan visual", "red", dashed=True))
    if show_minkowski_ball:
        items.append(_line_legend("Bola Lp", "orange", dashed=True))
    if show_minkowski_curve:
        items.append(_line_legend("Curva didática Lp", "purple"))

    if not items:
        return

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #cfcfcf;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
        padding: 12px 14px;
        min-width: 250px;
    ">
        <div style="font-size: 14px; font-weight: 700; margin-bottom: 8px; color: #111;">
            Legenda
        </div>
        {''.join(items)}
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def plot_route(
    graph: Any,
    route: list[int],
    *,
    filepath: str | Path | None = None,
    show: bool = True,
    close: bool = False,
):
    """Plota uma rota no grafo usando OSMnx e salva opcionalmente em arquivo."""
    import osmnx as ox

    fig, ax = ox.plot_graph_route(
        graph,
        list(route),
        node_size=0,
        route_linewidth=4,
        show=show,
        close=close,
    )
    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, ax


def latex_label_from_metric_col(metric_col: str) -> str:
    """Converte nomes de colunas em labels LaTeX para gráficos."""
    if metric_col == "d_graph_m":
        return r"$d_G$"
    if metric_col == metric_column_from_p(math.inf):
        return r"$d_\infty$"
    if metric_col.startswith("d_L") and metric_col.endswith("_m"):
        p_str = metric_col.removeprefix("d_L").removesuffix("_m").replace("_", ".")
        if p_str == "1":
            return r"$d_1$"
        if p_str == "2":
            return r"$d_2$"
        return rf"$d_{{{p_str}}}$"
    return metric_col


def plot_metric_scatter(
    results_df: pd.DataFrame,
    metric_col: str,
    *,
    filepath: str | Path | None = None,
    alpha: float = 0.25,
    s: float = 8.0,
):
    """Gráfico de dispersão d_métrica versus d_G."""
    import matplotlib.pyplot as plt

    if metric_col not in results_df.columns:
        raise ValueError(f"Coluna {metric_col!r} não encontrada.")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = results_df[metric_col]
    y = results_df["d_graph_m"]

    ax.scatter(x, y, alpha=alpha, s=s)
    max_val = float(np.nanmax([x.max(), y.max()]))
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1, label=r"$y=x$")

    metric_label = latex_label_from_metric_col(metric_col)
    ax.set_xlabel(rf"{metric_label} $(\mathrm{{m}})$", fontsize=12)
    ax.set_ylabel(r"$d_G$ $(\mathrm{m})$", fontsize=12)
    ax.set_title(rf"Comparação entre {metric_label} e $d_G$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_error_by_p(
    grid_results: pd.DataFrame,
    *,
    criterion: str = "MAPE_percent",
    filepath: str | Path | None = None,
):
    """Plota erro em função de p para a busca do melhor p."""
    import matplotlib.pyplot as plt

    if "p" not in grid_results.columns or criterion not in grid_results.columns:
        raise ValueError("grid_results precisa conter colunas 'p' e o critério informado.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid_results["p"], grid_results[criterion], linewidth=1.5)
    ax.set_xlabel("p")
    ax.set_ylabel(criterion)
    ax.set_title(f"Erro em função de p ({criterion})")
    ax.grid(True, alpha=0.3)

    best = best_p_by(grid_results, criterion=criterion)
    ax.axvline(float(best["p"]), linestyle="--", linewidth=1)

    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_tortuosity_hist(
    results_df: pd.DataFrame,
    *,
    tortuosity_col: str = "tortuosity_dG_dL2",
    bins: int = 40,
    filepath: str | Path | None = None,
):
    """Histograma da tortuosidade d_G/d_2."""
    import matplotlib.pyplot as plt

    if tortuosity_col not in results_df.columns:
        raise ValueError(f"Coluna {tortuosity_col!r} não encontrada.")

    values = results_df[tortuosity_col].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=bins)
    ax.set_xlabel("Tortuosidade d_G/d_2")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição da tortuosidade")
    ax.grid(True, alpha=0.3)

    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, ax
