from __future__ import annotations

from html import escape

import folium
import pyproj

from nao_e_so_reta.config import LatLon, XY
from nao_e_so_reta.norms import (
    lp_distance_xy,
    manhattan_polyline_xy,
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
