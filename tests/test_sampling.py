from __future__ import annotations

import networkx as nx

from nao_e_so_reta.sampling import largest_weakly_connected_nodes, sample_node_pairs


def test_largest_weakly_connected_nodes_directed() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(10, 11)

    nodes = set(largest_weakly_connected_nodes(graph))
    assert nodes == {1, 2, 3}


def test_sample_node_pairs() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))

    pairs = sample_node_pairs(graph, n_pairs=4, seed=1, nodes=list(graph.nodes))
    assert len(pairs) == 4
    assert all(u != v for u, v in pairs)
