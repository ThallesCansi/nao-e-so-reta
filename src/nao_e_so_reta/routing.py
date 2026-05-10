from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

try:
    import osmnx as ox
except ImportError:  # permite testar módulos puros sem instalar OSMnx
    ox = None

from nao_e_so_reta.config import LatLon, XY


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


def node_latlon(graph: Any, node: int) -> LatLon:
    """Coordenada lat/lon de um nó em grafo não projetado."""
    return (float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"]))


def node_xy(projected_graph: Any, node: int) -> XY:
    """Coordenada projetada em metros de um nó."""
    return (float(projected_graph.nodes[node]["x"]), float(projected_graph.nodes[node]["y"]))


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
