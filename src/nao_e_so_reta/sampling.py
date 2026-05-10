from __future__ import annotations

import random
from typing import Any

import networkx as nx

from nao_e_so_reta.config import XY
from nao_e_so_reta.routing import node_xy


def largest_weakly_connected_nodes(graph: Any) -> list[int]:
    """Retorna nós da maior componente fracamente conectada."""
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)

    largest = max(components, key=len)
    return list(largest)


def sample_node_pairs(
    graph: Any,
    *,
    n_pairs: int,
    seed: int = 42,
    nodes: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Amostra pares distintos de nós da rede."""
    rng = random.Random(seed)
    population = nodes if nodes is not None else largest_weakly_connected_nodes(graph)

    if len(population) < 2:
        raise ValueError("O grafo precisa ter ao menos dois nós.")

    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(1000, n_pairs * 30)

    while len(out) < n_pairs and attempts < max_attempts:
        u, v = rng.sample(population, 2)
        if u != v:
            out.append((int(u), int(v)))
        attempts += 1

    return out


def build_calibration_pairs(
    graph: Any,
    projected_graph: Any,
    *,
    n_pairs: int,
    seed: int = 42,
    cutoff_failures: int | None = None,
) -> list[tuple[XY, XY, float]]:
    """Amostra pares e calcula distância real no grafo.

    Retorna lista de (a_xy, b_xy, graph_distance_m), removendo pares sem caminho.
    """
    nodes = largest_weakly_connected_nodes(graph)
    candidate_pairs = sample_node_pairs(graph, n_pairs=n_pairs * 3, seed=seed, nodes=nodes)

    out: list[tuple[XY, XY, float]] = []
    failures = 0
    max_failures = cutoff_failures if cutoff_failures is not None else max(20, n_pairs)

    for u, v in candidate_pairs:
        if len(out) >= n_pairs:
            break

        try:
            d_graph = float(nx.shortest_path_length(graph, u, v, weight="length"))
        except nx.NetworkXNoPath:
            failures += 1
            if failures >= max_failures:
                break
            continue

        if d_graph <= 0:
            continue

        out.append((node_xy(projected_graph, u), node_xy(projected_graph, v), d_graph))

    if not out:
        raise ValueError("Não foi possível obter pares conectados para calibração.")

    return out
