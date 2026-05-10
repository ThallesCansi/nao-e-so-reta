from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import warnings

import networkx as nx

try:
    import osmnx as ox
except ImportError:  # permite testar módulos puros sem instalar OSMnx
    ox = None

from nao_e_so_reta.config import LatLon, XY
from nao_e_so_reta.projections import transformer_wgs84_to_projected


@dataclass(frozen=True)
class RouteResult:
    origin_node: int
    destination_node: int
    origin_node_latlon: LatLon
    destination_node_latlon: LatLon
    origin_xy: XY
    destination_xy: XY
    route_nodes: list[int]
    route_latlon: list[LatLon]
    length_m: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.length_m is not None


def nearest_node(graph: Any, point: LatLon) -> int:
    """Retorna o nó mais próximo de uma coordenada lat/lon.

    Tenta usar a rotina otimizada do OSMnx. Caso dependências opcionais não
    estejam disponíveis no ambiente, usa busca linear como fallback.
    """
    lat, lon = point

    if ox is not None:
        try:
            return int(ox.distance.nearest_nodes(graph, X=lon, Y=lat))
        except ImportError:
            pass

    nearest = min(
        graph.nodes,
        key=lambda node: (float(graph.nodes[node]["x"]) - lon) ** 2
        + (float(graph.nodes[node]["y"]) - lat) ** 2,
    )
    return int(nearest)


def geocode_point(query: str, fallback_latlon: LatLon | None = None) -> LatLon:
    """Geocodifica uma consulta textual, com fallback opcional em latitude/longitude."""
    if ox is None:
        if fallback_latlon is None:
            raise RuntimeError("OSMnx não está disponível para geocodificação.")
        return fallback_latlon

    try:
        lat, lon = ox.geocode(query)
        return (float(lat), float(lon))
    except Exception as exc:
        if fallback_latlon is None:
            raise RuntimeError(
                f"Falha ao geocodificar '{query}'. Informe fallback_latlon=(lat, lon)."
            ) from exc
        warnings.warn(
            f"Falha ao geocodificar '{query}'. Usando fallback_latlon={fallback_latlon}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback_latlon


def nearest_node_from_latlon(projected_graph: Any, point: LatLon) -> int:
    """Encontra o nó mais próximo de um ponto lat/lon em grafo projetado."""
    if ox is None:
        raise RuntimeError("OSMnx não está disponível para busca espacial no grafo.")

    transformer = transformer_wgs84_to_projected(projected_graph.graph["crs"])
    x, y = transformer.transform(point[1], point[0])
    return int(ox.distance.nearest_nodes(projected_graph, X=float(x), Y=float(y)))


def nearest_node_from_query(
    projected_graph: Any,
    query: str,
    fallback_latlon: LatLon | None = None,
) -> tuple[int, LatLon]:
    """Geocodifica uma consulta e retorna o nó mais próximo no grafo projetado."""
    point = geocode_point(query, fallback_latlon=fallback_latlon)
    return nearest_node_from_latlon(projected_graph, point), point


def node_latlon(graph: Any, node: int) -> LatLon:
    """Coordenada lat/lon de um nó em grafo não projetado."""
    return (float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"]))


def node_xy(projected_graph: Any, node: int) -> XY:
    """Coordenada projetada em metros de um nó."""
    return (float(projected_graph.nodes[node]["x"]), float(projected_graph.nodes[node]["y"]))


def graph_distance(graph: Any, origin_node: int, destination_node: int, weight: str = "length") -> float:
    """Calcula d_G por menor caminho ponderado."""
    return float(
        nx.shortest_path_length(graph, source=origin_node, target=destination_node, weight=weight)
    )


def shortest_route(
    graph: Any,
    origin_node: int,
    destination_node: int,
    weight: str = "length",
) -> list[int]:
    """Retorna a sequência de nós do menor caminho entre origem e destino."""
    return list(nx.shortest_path(graph, source=origin_node, target=destination_node, weight=weight))


def route_length(graph: Any, route: list[int], weight: str = "length") -> float:
    """Calcula o comprimento de uma rota explícita, tratando arestas paralelas."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            raise ValueError(f"Não há aresta entre {u} e {v}.")

        if isinstance(edge_data, dict) and all(isinstance(k, int) for k in edge_data):
            total += min(float(attrs.get(weight, 1.0)) for attrs in edge_data.values())
        else:
            total += float(edge_data.get(weight, 1.0))
    return total


def tortuosity_from_nodes(graph: Any, origin_node: int, destination_node: int) -> float:
    """Tortuosidade d_G/d_2 calculada diretamente de dois nós projetados."""
    from nao_e_so_reta.norms import lp_distance_xy

    euclidean = lp_distance_xy(node_xy(graph, origin_node), node_xy(graph, destination_node), 2.0)
    if euclidean <= 0:
        return math.nan
    return graph_distance(graph, origin_node, destination_node) / euclidean


def shortest_route_between_points(
    graph: Any,
    projected_graph: Any,
    origin: LatLon,
    destination: LatLon,
) -> RouteResult:
    """Calcula menor caminho entre os nós mais próximos de origem/destino."""
    u = nearest_node(graph, origin)
    v = nearest_node(graph, destination)

    origin_node_latlon = node_latlon(graph, u)
    destination_node_latlon = node_latlon(graph, v)
    origin_xy = node_xy(projected_graph, u)
    destination_xy = node_xy(projected_graph, v)

    try:
        route_nodes = nx.shortest_path(graph, u, v, weight="length")
        length_m = float(nx.shortest_path_length(graph, u, v, weight="length"))
    except nx.NetworkXNoPath:
        return RouteResult(
            origin_node=u,
            destination_node=v,
            origin_node_latlon=origin_node_latlon,
            destination_node_latlon=destination_node_latlon,
            origin_xy=origin_xy,
            destination_xy=destination_xy,
            route_nodes=[],
            route_latlon=[],
            length_m=None,
            error="Não há caminho viável entre estes pontos no grafo selecionado.",
        )

    route_latlon = [node_latlon(graph, n) for n in route_nodes]

    return RouteResult(
        origin_node=u,
        destination_node=v,
        origin_node_latlon=origin_node_latlon,
        destination_node_latlon=destination_node_latlon,
        origin_xy=origin_xy,
        destination_xy=destination_xy,
        route_nodes=list(map(int, route_nodes)),
        route_latlon=route_latlon,
        length_m=length_m,
    )
