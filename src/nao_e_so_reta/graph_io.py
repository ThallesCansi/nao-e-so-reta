from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import pyproj

from nao_e_so_reta.projections import transformer_projected_to_wgs84


def configure_osmnx(log_console: bool = False, use_cache: bool = True) -> None:
    """Configura opções globais do OSMnx."""
    ox.settings.log_console = log_console
    ox.settings.use_cache = use_cache


def project_graph(graph: Any, to_crs: object | None = None) -> Any:
    """Projeta grafo usando a API pública disponível na versão instalada do OSMnx."""
    if hasattr(ox, "project_graph"):
        return ox.project_graph(graph, to_crs=to_crs)
    if hasattr(ox, "projection") and hasattr(ox.projection, "project_graph"):
        return ox.projection.project_graph(graph, to_crs=to_crs)
    raise AttributeError("Não encontrei função project_graph na instalação atual do OSMnx.")


def to_undirected(graph: Any) -> Any:
    """Converte grafo OSMnx para não direcionado com fallback entre versões."""
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        return ox.convert.to_undirected(graph)
    if hasattr(ox, "utils_graph") and hasattr(ox.utils_graph, "get_undirected"):
        return ox.utils_graph.get_undirected(graph)
    return graph.to_undirected(as_view=False)


def save_graphml(graph: Any, filepath: str | Path) -> None:
    """Salva grafo GraphML criando o diretório de destino."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ox, "save_graphml"):
        ox.save_graphml(graph, filepath=path)
    elif hasattr(ox, "io") and hasattr(ox.io, "save_graphml"):
        ox.io.save_graphml(graph, filepath=path)
    else:
        raise AttributeError("Não encontrei função save_graphml na instalação atual do OSMnx.")


def load_graphml(filepath: str | Path) -> Any:
    """Carrega grafo GraphML com fallback entre versões do OSMnx."""
    if hasattr(ox, "load_graphml"):
        return ox.load_graphml(filepath=filepath)
    if hasattr(ox, "io") and hasattr(ox.io, "load_graphml"):
        return ox.io.load_graphml(filepath=filepath)
    raise AttributeError("Não encontrei função load_graphml na instalação atual do OSMnx.")


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Retorna a maior componente conexa de um grafo não direcionado."""
    if len(graph) == 0:
        raise ValueError("O grafo está vazio.")
    if nx.is_connected(graph):
        return graph.copy()
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def prepare_graph(
    graph: Any,
    *,
    make_undirected: bool = True,
    keep_largest_component: bool = True,
    to_crs: object | None = None,
) -> Any:
    """Projeta o grafo, opcionalmente o torna não direcionado e mantém a maior componente."""
    projected = project_graph(graph, to_crs=to_crs)

    if make_undirected:
        projected = to_undirected(projected)
        if keep_largest_component:
            projected = largest_connected_component(projected)
    elif keep_largest_component:
        warnings.warn(
            "keep_largest_component=True foi solicitado em grafo direcionado. "
            "Para este projeto, recomenda-se make_undirected=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    return projected


def download_graph(
    *,
    place: str | dict | list[str | dict] | None = None,
    center_point: tuple[float, float] | None = None,
    dist: float | None = None,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
    custom_filter: str | None = None,
) -> Any:
    """Baixa um grafo do OSMnx por nome de lugar ou por ponto central e raio."""
    if place is None and center_point is None:
        raise ValueError("Informe `place` ou `center_point`.")

    common_kwargs = {
        "network_type": network_type,
        "simplify": simplify,
        "retain_all": retain_all,
    }
    if custom_filter is not None:
        common_kwargs["custom_filter"] = custom_filter

    if place is not None:
        return ox.graph_from_place(place, **common_kwargs)

    if dist is None:
        raise ValueError("Ao usar `center_point`, também informe `dist` em metros.")
    return ox.graph_from_point(center_point, dist=dist, **common_kwargs)


def load_or_download_graph(
    filepath: str | Path,
    *,
    place: str | dict | list[str | dict] | None = None,
    center_point: tuple[float, float] | None = None,
    dist: float | None = None,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
    custom_filter: str | None = None,
    force_download: bool = False,
    make_undirected: bool = True,
    keep_largest_component: bool = True,
    to_crs: object | None = None,
) -> Any:
    """Carrega um GraphML bruto ou baixa do OSMnx, salvando-o antes do preparo."""
    path = Path(filepath)

    if path.exists() and not force_download:
        raw_graph = load_graphml(path)
    else:
        raw_graph = download_graph(
            place=place,
            center_point=center_point,
            dist=dist,
            network_type=network_type,
            simplify=simplify,
            retain_all=retain_all,
            custom_filter=custom_filter,
        )
        save_graphml(raw_graph, path)

    return prepare_graph(
        raw_graph,
        make_undirected=make_undirected,
        keep_largest_component=keep_largest_component,
        to_crs=to_crs,
    )


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
        graph = load_graphml(path)
        source = f"GraphML local: {path}"
    else:
        graph = ox.graph_from_place(place_name, network_type=network_type)
        source = f"OpenStreetMap via OSMnx: {place_name} / {network_type}"

    projected = project_graph(graph)
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
    save_graphml(graph, output)
    return output
