from __future__ import annotations

from pathlib import Path
from typing import Any

import osmnx as ox
import pyproj

from nao_e_so_reta.projections import transformer_projected_to_wgs84


def configure_osmnx(log_console: bool = False, use_cache: bool = True) -> None:
    """Configura opções globais do OSMnx."""
    ox.settings.log_console = log_console
    ox.settings.use_cache = use_cache


def load_graph_from_path_or_place(
    *,
    place_name: str,
    network_type: str,
    graph_path: str | Path,
    log_console: bool = False,
) -> tuple[Any, Any, pyproj.Transformer, str]:
    """Carrega um grafo GraphML local ou baixa do OpenStreetMap.

    Retorna:
        (G_latlon, G_projected, transformer_projected_to_wgs84, source_description)
    """
    configure_osmnx(log_console=log_console)

    path = Path(graph_path)
    if path.exists() and path.is_file():
        graph = ox.load_graphml(path)
        source = f"GraphML local: {path}"
    else:
        graph = ox.graph_from_place(place_name, network_type=network_type)
        source = f"OpenStreetMap via OSMnx: {place_name} / {network_type}"

    projected = ox.project_graph(graph)
    transformer = transformer_projected_to_wgs84(projected.graph["crs"])

    return graph, projected, transformer, source


def save_graphml_for_place(
    *,
    place_name: str,
    network_type: str,
    output_path: str | Path,
    log_console: bool = True,
) -> Path:
    """Baixa uma rede do OSM e salva em GraphML."""
    configure_osmnx(log_console=log_console)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = ox.graph_from_place(place_name, network_type=network_type)
    ox.save_graphml(graph, output)
    return output
