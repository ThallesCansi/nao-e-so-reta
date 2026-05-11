from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import networkx as nx

try:
    import osmnx as ox
except ImportError:  # pragma: no cover
    ox = None

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None


def _require_osmnx() -> None:
    if ox is None:
        raise ImportError(
            "OSMnx é necessário para operações com grafos. "
            "Instale com: conda install -c conda-forge osmnx geopandas networkx pyproj"
        )


def _require_pyproj() -> None:
    if Transformer is None:
        raise ImportError("pyproj é necessário para transformar coordenadas.")

from .config import LatLon, XY


def _project_graph(graph: Any, to_crs=None) -> Any:
    _require_osmnx()
    if hasattr(ox, "project_graph"):
        return ox.project_graph(graph, to_crs=to_crs)
    return ox.projection.project_graph(graph, to_crs=to_crs)


def _to_undirected(graph: Any) -> Any:
    _require_osmnx()
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        return ox.convert.to_undirected(graph)
    if hasattr(ox, "utils_graph") and hasattr(ox.utils_graph, "get_undirected"):
        return ox.utils_graph.get_undirected(graph)
    return graph.to_undirected(as_view=False)


def _save_graphml(graph: Any, filepath: str | Path) -> None:
    _require_osmnx()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ox, "save_graphml"):
        ox.save_graphml(graph, filepath=filepath)
    else:
        ox.io.save_graphml(graph, filepath=filepath)


def _load_graphml(filepath: str | Path) -> Any:
    _require_osmnx()
    if hasattr(ox, "load_graphml"):
        return ox.load_graphml(filepath=filepath)
    return ox.io.load_graphml(filepath=filepath)


def largest_connected_component(graph: Any) -> Any:
    """Mantém apenas a maior componente conexa de um grafo não direcionado."""
    if len(graph) == 0:
        raise ValueError("O grafo está vazio.")
    if nx.is_connected(graph):
        return graph.copy()
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def prepare_graph(
    graph_raw: Any,
    *,
    make_undirected: bool = True,
    keep_largest_component: bool = True,
    to_crs=None,
) -> Any:
    """Projeta o grafo para metros, torna não direcionado e filtra a maior componente."""
    graph = _project_graph(graph_raw, to_crs=to_crs)
    if make_undirected:
        graph = _to_undirected(graph)
        if keep_largest_component:
            graph = largest_connected_component(graph)
    elif keep_largest_component:
        warnings.warn(
            "keep_largest_component=True foi ignorado para grafo direcionado. "
            "Neste projeto, recomenda-se make_undirected=True.",
            RuntimeWarning,
        )
    return graph


def download_graph(
    *,
    place: str | None = None,
    center_point: LatLon | None = None,
    dist: float | None = None,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
    custom_filter: str | None = None,
) -> Any:
    """Baixa grafo pelo nome do lugar ou por recorte circular em torno de um ponto."""
    _require_osmnx()
    if place is None and center_point is None:
        raise ValueError("Informe place ou center_point.")

    kwargs = dict(network_type=network_type, simplify=simplify, retain_all=retain_all)
    if custom_filter is not None:
        kwargs["custom_filter"] = custom_filter

    if place is not None:
        return ox.graph_from_place(place, **kwargs)

    if dist is None:
        raise ValueError("Ao usar center_point, informe dist em metros.")
    return ox.graph_from_point(center_point, dist=dist, **kwargs)


def load_or_download_graph(
    filepath: str | Path,
    *,
    place: str | None = None,
    center_point: LatLon | None = None,
    dist: float | None = None,
    network_type: str = "drive",
    force_download: bool = False,
    make_undirected: bool = True,
    keep_largest_component: bool = True,
    to_crs=None,
) -> Any:
    """Carrega grafo bruto salvo ou baixa do OSMnx; depois prepara para análise métrica."""
    filepath = Path(filepath)
    if filepath.exists() and not force_download:
        graph_raw = _load_graphml(filepath)
    else:
        graph_raw = download_graph(
            place=place,
            center_point=center_point,
            dist=dist,
            network_type=network_type,
        )
        _save_graphml(graph_raw, filepath)

    return prepare_graph(
        graph_raw,
        make_undirected=make_undirected,
        keep_largest_component=keep_largest_component,
        to_crs=to_crs,
    )


def geocode_point(query: str, fallback_latlon: LatLon | None = None) -> LatLon:
    """Geocodifica texto; se falhar, usa fallback manual em latitude/longitude."""
    _require_osmnx()
    try:
        lat, lon = ox.geocode(query)
        return float(lat), float(lon)
    except Exception as exc:
        if fallback_latlon is None:
            raise RuntimeError(
                f"Falha ao geocodificar {query!r}. Informe fallback_latlon=(lat, lon)."
            ) from exc
        warnings.warn(
            f"Falha ao geocodificar {query!r}. Usando fallback_latlon={fallback_latlon}.",
            RuntimeWarning,
        )
        return fallback_latlon


def node_xy(graph: Any, node: int) -> XY:
    """Coordenadas projetadas de um nó do grafo."""
    return float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])


def node_latlon_from_projected(graph: Any, node: int) -> LatLon:
    """Converte coordenadas projetadas do nó para latitude/longitude."""
    _require_pyproj()
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    x, y = node_xy(graph, node)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def nearest_node_from_latlon(graph: Any, point: LatLon) -> int:
    """Encontra nó mais próximo de um ponto lat/lon em grafo projetado."""
    _require_osmnx()
    _require_pyproj()
    transformer = Transformer.from_crs("EPSG:4326", graph.graph["crs"], always_xy=True)
    x, y = transformer.transform(point[1], point[0])
    return int(ox.distance.nearest_nodes(graph, X=float(x), Y=float(y)))


def nearest_node_from_query(
    graph: Any,
    query: str,
    *,
    fallback_latlon: LatLon | None = None,
) -> tuple[int, LatLon]:
    """Geocodifica uma consulta e retorna o nó mais próximo no grafo projetado."""
    point = geocode_point(query, fallback_latlon=fallback_latlon)
    return nearest_node_from_latlon(graph, point), point


def shortest_route(graph: Any, origin_node: int, destination_node: int, *, weight: str = "length") -> list[int]:
    """Menor caminho entre dois nós."""
    return list(nx.shortest_path(graph, source=origin_node, target=destination_node, weight=weight))


def route_length(graph: Any, route: list[int], *, weight: str = "length") -> float:
    """Comprimento de uma rota; trata arestas paralelas de MultiGraph."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        data = graph.get_edge_data(u, v)
        if data is None:
            raise ValueError(f"Não há aresta entre {u} e {v}.")
        if isinstance(data, dict) and all(isinstance(k, int) for k in data):
            total += min(float(attrs.get(weight, 1.0)) for attrs in data.values())
        else:
            total += float(data.get(weight, 1.0))
    return float(total)
