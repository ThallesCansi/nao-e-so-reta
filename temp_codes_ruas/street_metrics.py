"""
street_metrics.py
=================

Funções para comparar métricas normadas no plano com a distância intrínseca
em um grafo real de ruas baixado do OpenStreetMap via OSMnx.

Projeto-base:
    - Estudo pontual: Ciclo Básico Unicamp -> Terminal Barão Geraldo.
    - Estudo estatístico: 10.000 pares de vértices em Campinas/Barão Geraldo.

Convenção matemática:
    - d_G(u, v): menor caminho no grafo ponderado por comprimento de aresta.
    - d_p(u, v): distância L^p entre as coordenadas projetadas dos vértices.
    - d_inf(u, v): distância L^infinito.
    - tortuosidade: d_G(u, v) / d_2(u, v).

Observação importante:
    Nunca calcule d_p diretamente em latitude/longitude. Este módulo projeta o
    grafo para um CRS métrico antes de calcular distâncias euclidianas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import warnings

import numpy as np
import pandas as pd
import networkx as nx

try:
    import osmnx as ox
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "O pacote 'osmnx' é necessário. Instale com, por exemplo: "
        "conda install -c conda-forge osmnx geopandas networkx pandas numpy scipy matplotlib tqdm jupyter"
    ) from exc

try:
    from pyproj import CRS, Transformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "O pacote 'pyproj' é necessário. Instale com: conda install -c conda-forge pyproj"
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


DEFAULT_P_VALUES: tuple[float, ...] = (
    1.0,
    1.25,
    1.5,
    1.54,
    1.75,
    2.0,
    3.0,
    5.0,
    10.0,
    math.inf,
)


@dataclass(frozen=True)
class GeocodedPoint:
    """Ponto em latitude/longitude e, opcionalmente, o texto usado na busca."""

    lat: float
    lon: float
    query: str | None = None


@dataclass(frozen=True)
class SinglePairResult:
    """Resultado completo para um único par origem-destino."""

    origin_node: int
    destination_node: int
    route: list[int]
    graph_distance_m: float
    metrics: dict[str, float]
    tortuosity: float


# ---------------------------------------------------------------------------
# Compatibilidade com pequenas diferenças entre versões do OSMnx
# ---------------------------------------------------------------------------


def _project_graph(G: nx.MultiDiGraph | nx.MultiGraph, to_crs=None):
    """Projeta grafo usando API pública do OSMnx, com fallback para namespaces."""
    if hasattr(ox, "project_graph"):
        return ox.project_graph(G, to_crs=to_crs)
    if hasattr(ox, "projection") and hasattr(ox.projection, "project_graph"):
        return ox.projection.project_graph(G, to_crs=to_crs)
    raise AttributeError("Não encontrei função project_graph na instalação atual do OSMnx.")


def _to_undirected(G: nx.MultiDiGraph | nx.MultiGraph) -> nx.MultiGraph:
    """Converte grafo OSMnx para não direcionado."""
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        return ox.convert.to_undirected(G)
    if hasattr(ox, "utils_graph") and hasattr(ox.utils_graph, "get_undirected"):
        return ox.utils_graph.get_undirected(G)
    return G.to_undirected(as_view=False)


def _save_graphml(G: nx.MultiDiGraph | nx.MultiGraph, filepath: str | Path) -> None:
    """Salva grafo em GraphML usando API pública do OSMnx."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ox, "save_graphml"):
        ox.save_graphml(G, filepath=filepath)
    elif hasattr(ox, "io") and hasattr(ox.io, "save_graphml"):
        ox.io.save_graphml(G, filepath=filepath)
    else:
        raise AttributeError("Não encontrei função save_graphml na instalação atual do OSMnx.")


def _load_graphml(filepath: str | Path):
    """Carrega grafo GraphML usando API pública do OSMnx."""
    if hasattr(ox, "load_graphml"):
        return ox.load_graphml(filepath=filepath)
    if hasattr(ox, "io") and hasattr(ox.io, "load_graphml"):
        return ox.io.load_graphml(filepath=filepath)
    raise AttributeError("Não encontrei função load_graphml na instalação atual do OSMnx.")


# ---------------------------------------------------------------------------
# Grafo: download, carregamento, projeção e conectividade
# ---------------------------------------------------------------------------


def largest_connected_component(G: nx.MultiGraph) -> nx.MultiGraph:
    """
    Retorna a maior componente conexa de um grafo não direcionado.

    Isso evita distâncias infinitas ou exceções ao calcular menor caminho entre
    vértices pertencentes a componentes distintas.
    """
    if len(G) == 0:
        raise ValueError("O grafo está vazio.")

    if nx.is_connected(G):
        return G.copy()

    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def prepare_graph(
    G_raw: nx.MultiDiGraph | nx.MultiGraph,
    *,
    make_undirected: bool = True,
    keep_largest_component: bool = True,
    to_crs=None,
) -> nx.MultiGraph | nx.MultiDiGraph:
    """
    Projeta o grafo para CRS métrico, converte para não direcionado e filtra a
    maior componente conexa.

    Parameters
    ----------
    G_raw:
        Grafo baixado ou carregado pelo OSMnx.
    make_undirected:
        Se True, ignora direção das vias.
    keep_largest_component:
        Se True, mantém apenas a maior componente conexa.
    to_crs:
        CRS de destino. Se None, OSMnx escolhe uma projeção UTM apropriada.
    """
    G_proj = _project_graph(G_raw, to_crs=to_crs)

    if make_undirected:
        G_proj = _to_undirected(G_proj)
        if keep_largest_component:
            G_proj = largest_connected_component(G_proj)
    elif keep_largest_component:
        warnings.warn(
            "keep_largest_component=True foi solicitado em grafo direcionado. "
            "Para este projeto, recomenda-se make_undirected=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    return G_proj


def download_graph(
    *,
    place: str | dict | list[str | dict] | None = None,
    center_point: tuple[float, float] | None = None,
    dist: float | None = None,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
    custom_filter: str | None = None,
) -> nx.MultiDiGraph:
    """
    Baixa um grafo do OSMnx por nome de lugar ou por ponto central + raio.

    Use `place` para Campinas inteira quando o Nominatim reconhece o polígono.
    Use `center_point` + `dist` para um recorte local de Barão Geraldo quando o
    nome do bairro não resolver como polígono.

    Parameters
    ----------
    place:
        Nome ou consulta aceitos pelo geocoder do OSMnx/Nominatim.
    center_point:
        Par (lat, lon). Usado com `dist`, em metros.
    dist:
        Raio/alcance em metros ao usar `center_point`.
    network_type:
        Tipo de rede OSMnx. Para este projeto, o padrão é "drive".
    simplify:
        Se True, simplifica a topologia da rede.
    retain_all:
        Se False, OSMnx retém a maior componente fracamente conectada no grafo bruto.
    custom_filter:
        Filtro Overpass opcional.
    """
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
    to_crs=None,
) -> nx.MultiGraph | nx.MultiDiGraph:
    """
    Carrega um grafo bruto de GraphML ou baixa do OSMnx e salva localmente.

    O arquivo salvo é o grafo bruto retornado pelo OSMnx. A projeção métrica,
    conversão para não direcionado e filtragem de componente são refeitas a cada
    carregamento para manter a rotina transparente e reprodutível.
    """
    filepath = Path(filepath)

    if filepath.exists() and not force_download:
        G_raw = _load_graphml(filepath)
    else:
        G_raw = download_graph(
            place=place,
            center_point=center_point,
            dist=dist,
            network_type=network_type,
            simplify=simplify,
            retain_all=retain_all,
            custom_filter=custom_filter,
        )
        _save_graphml(G_raw, filepath)

    return prepare_graph(
        G_raw,
        make_undirected=make_undirected,
        keep_largest_component=keep_largest_component,
        to_crs=to_crs,
    )


# ---------------------------------------------------------------------------
# Geocodificação e vértices mais próximos
# ---------------------------------------------------------------------------


def geocode_point(query: str, fallback_latlon: tuple[float, float] | None = None) -> GeocodedPoint:
    """
    Geocodifica uma consulta textual e retorna latitude/longitude.

    Se a geocodificação falhar e `fallback_latlon` for informado, usa esse
    fallback. Isso é útil para pontos como prédios ou terminais, cujos nomes
    podem variar no OpenStreetMap/Nominatim.
    """
    try:
        lat, lon = ox.geocode(query)
        return GeocodedPoint(float(lat), float(lon), query=query)
    except Exception as exc:
        if fallback_latlon is None:
            raise RuntimeError(
                f"Falha ao geocodificar '{query}'. Informe fallback_latlon=(lat, lon)."
            ) from exc
        lat, lon = fallback_latlon
        warnings.warn(
            f"Falha ao geocodificar '{query}'. Usando fallback_latlon={fallback_latlon}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return GeocodedPoint(float(lat), float(lon), query=query)


def graph_crs(G: nx.Graph) -> CRS:
    """Retorna o CRS do grafo como objeto pyproj.CRS."""
    if "crs" not in G.graph:
        raise ValueError("O grafo não possui atributo G.graph['crs'].")
    return CRS.from_user_input(G.graph["crs"])


def project_latlon_to_graph_xy(G: nx.Graph, lat: float, lon: float) -> tuple[float, float]:
    """
    Converte latitude/longitude WGS84 para coordenadas x/y no CRS do grafo.

    Retorna (x, y), isto é, (easting, northing) ou equivalente no CRS projetado.
    """
    transformer = Transformer.from_crs("EPSG:4326", graph_crs(G), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def nearest_node_from_latlon(G: nx.Graph, lat: float, lon: float) -> int:
    """Encontra o vértice mais próximo de um ponto dado em latitude/longitude."""
    x, y = project_latlon_to_graph_xy(G, lat, lon)
    node = ox.distance.nearest_nodes(G, X=x, Y=y)
    return int(node)


def nearest_node_from_query(
    G: nx.Graph,
    query: str,
    fallback_latlon: tuple[float, float] | None = None,
) -> tuple[int, GeocodedPoint]:
    """Geocodifica uma consulta e retorna o vértice mais próximo no grafo."""
    point = geocode_point(query, fallback_latlon=fallback_latlon)
    node = nearest_node_from_latlon(G, point.lat, point.lon)
    return node, point


# ---------------------------------------------------------------------------
# Métricas Lp e distância intrínseca no grafo
# ---------------------------------------------------------------------------


def node_xy(G: nx.Graph, node: int) -> tuple[float, float]:
    """Retorna coordenadas projetadas (x, y) de um vértice."""
    data = G.nodes[node]
    return float(data["x"]), float(data["y"])


def lp_distance_xy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    p: float,
) -> float:
    """Calcula a distância L^p entre dois pontos do plano projetado."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    if math.isinf(p):
        return float(max(dx, dy))
    if p <= 0:
        raise ValueError("p deve ser positivo ou math.inf.")

    return float((dx**p + dy**p) ** (1.0 / p))


def lp_distance_nodes(G: nx.Graph, u: int, v: int, p: float) -> float:
    """Calcula d_p entre as coordenadas projetadas de dois vértices."""
    x1, y1 = node_xy(G, u)
    x2, y2 = node_xy(G, v)
    return lp_distance_xy(x1, y1, x2, y2, p)


def metric_name_from_p(p: float) -> str:
    """Nome estável de coluna para uma métrica Lp."""
    if math.isinf(p):
        return "Linf"
    if abs(p - round(p)) < 1e-12:
        return f"L{int(round(p))}"
    return "L" + str(p).replace(".", "_")


def metric_column_from_p(p: float) -> str:
    """Nome da coluna de distância, em metros, para p."""
    return f"d_{metric_name_from_p(p)}_m"


def graph_distance(G: nx.Graph, u: int, v: int, weight: str = "length") -> float:
    """Calcula d_G(u, v) por menor caminho ponderado."""
    return float(nx.shortest_path_length(G, source=u, target=v, weight=weight))


def shortest_route(G: nx.Graph, u: int, v: int, weight: str = "length") -> list[int]:
    """Retorna a sequência de vértices do menor caminho entre u e v."""
    return list(nx.shortest_path(G, source=u, target=v, weight=weight))


def route_length(G: nx.Graph, route: Sequence[int], weight: str = "length") -> float:
    """
    Calcula o comprimento de uma rota explícita.

    Em MultiGraph, quando há arestas paralelas entre dois vértices consecutivos,
    usa a menor aresta segundo `weight`.
    """
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            raise ValueError(f"Não há aresta entre {u} e {v}.")

        if isinstance(edge_data, dict) and all(isinstance(k, int) for k in edge_data.keys()):
            total += min(float(attrs.get(weight, 1.0)) for attrs in edge_data.values())
        else:
            total += float(edge_data.get(weight, 1.0))
    return total


# ---------------------------------------------------------------------------
# Caso pontual: origem -> destino
# ---------------------------------------------------------------------------


def compute_single_pair_result(
    G: nx.Graph,
    origin_node: int,
    destination_node: int,
    *,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    weight: str = "length",
) -> SinglePairResult:
    """Calcula d_G, rota e métricas Lp para um par de vértices."""
    route = shortest_route(G, origin_node, destination_node, weight=weight)
    d_g = route_length(G, route, weight=weight)

    metrics = {
        metric_column_from_p(p): lp_distance_nodes(G, origin_node, destination_node, p)
        for p in p_values
    }

    d2 = metrics.get(metric_column_from_p(2.0))
    if d2 is None:
        d2 = lp_distance_nodes(G, origin_node, destination_node, 2.0)

    tortuosity = float(d_g / d2) if d2 > 0 else math.nan

    return SinglePairResult(
        origin_node=origin_node,
        destination_node=destination_node,
        route=route,
        graph_distance_m=float(d_g),
        metrics=metrics,
        tortuosity=tortuosity,
    )


def compute_single_pair_from_queries(
    G: nx.Graph,
    origin_query: str,
    destination_query: str,
    *,
    origin_fallback_latlon: tuple[float, float] | None = None,
    destination_fallback_latlon: tuple[float, float] | None = None,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    weight: str = "length",
) -> tuple[SinglePairResult, GeocodedPoint, GeocodedPoint]:
    """
    Geocodifica origem/destino, encontra vértices próximos e calcula métricas.
    """
    origin_node, origin_point = nearest_node_from_query(
        G, origin_query, fallback_latlon=origin_fallback_latlon
    )
    destination_node, destination_point = nearest_node_from_query(
        G, destination_query, fallback_latlon=destination_fallback_latlon
    )
    result = compute_single_pair_result(
        G,
        origin_node,
        destination_node,
        p_values=p_values,
        weight=weight,
    )
    return result, origin_point, destination_point


def single_pair_to_dataframe(result: SinglePairResult) -> pd.DataFrame:
    """Converte resultado pontual para DataFrame de uma linha."""
    row = {
        "origin_node": result.origin_node,
        "destination_node": result.destination_node,
        "d_graph_m": result.graph_distance_m,
        "tortuosity_dG_dL2": result.tortuosity,
    }
    row.update(result.metrics)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Amostragem de pares e cálculo em lote
# ---------------------------------------------------------------------------


def sample_vertex_pairs(
    G: nx.Graph,
    *,
    n_pairs: int = 10_000,
    n_origins: int | None = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Amostra pares de vértices distintos.

    Por padrão, usa um número limitado de origens para acelerar o cálculo de
    menor caminho: roda-se Dijkstra uma vez por origem e avaliam-se vários
    destinos. Ainda assim, os destinos são sorteados aleatoriamente.

    Parameters
    ----------
    G:
        Grafo conectado.
    n_pairs:
        Número total de pares desejado.
    n_origins:
        Número de origens distintas. Se None, sorteia origens livremente para
        cada par, mas isso tende a deixar o cálculo bem mais lento.
    seed:
        Semente para reprodutibilidade.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs deve ser positivo.")

    nodes = np.array(list(G.nodes))
    if len(nodes) < 2:
        raise ValueError("O grafo precisa ter pelo menos dois vértices.")

    rng = np.random.default_rng(seed)

    if n_origins is None:
        origins = rng.choice(nodes, size=n_pairs, replace=True)
    else:
        n_origins = int(min(max(1, n_origins), len(nodes), n_pairs))
        chosen_origins = rng.choice(nodes, size=n_origins, replace=False)
        counts = np.full(n_origins, n_pairs // n_origins, dtype=int)
        counts[: n_pairs % n_origins] += 1
        origins = np.repeat(chosen_origins, counts)
        rng.shuffle(origins)

    targets = rng.choice(nodes, size=n_pairs, replace=True)

    # Garante u != v. Reamostra somente os conflitos.
    conflicts = targets == origins
    while conflicts.any():
        targets[conflicts] = rng.choice(nodes, size=int(conflicts.sum()), replace=True)
        conflicts = targets == origins

    return pd.DataFrame({"origin": origins.astype(object), "target": targets.astype(object)})


def _node_coordinate_arrays(G: nx.Graph, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coordenadas projetadas dos vértices de origem e destino."""
    x = nx.get_node_attributes(G, "x")
    y = nx.get_node_attributes(G, "y")

    out = pairs_df.copy()
    out["origin_x"] = out["origin"].map(x).astype(float)
    out["origin_y"] = out["origin"].map(y).astype(float)
    out["target_x"] = out["target"].map(x).astype(float)
    out["target_y"] = out["target"].map(y).astype(float)
    return out


def add_lp_columns(
    df: pd.DataFrame,
    *,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    origin_x: str = "origin_x",
    origin_y: str = "origin_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
) -> pd.DataFrame:
    """Adiciona colunas d_Lp_m a um DataFrame com coordenadas projetadas."""
    out = df.copy()

    dx = (out[origin_x] - out[target_x]).abs().to_numpy(dtype=float)
    dy = (out[origin_y] - out[target_y]).abs().to_numpy(dtype=float)

    for p in p_values:
        col = metric_column_from_p(p)
        if math.isinf(p):
            out[col] = np.maximum(dx, dy)
        else:
            out[col] = (dx**p + dy**p) ** (1.0 / p)

    return out


def add_graph_distances_grouped(
    G: nx.Graph,
    pairs_df: pd.DataFrame,
    *,
    origin_col: str = "origin",
    target_col: str = "target",
    weight: str = "length",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Adiciona coluna d_graph_m calculando Dijkstra uma vez por origem.

    Essa é a parte mais custosa do experimento estatístico. Agrupar por origem
    costuma ser muito mais rápido do que chamar shortest_path_length para cada
    par isoladamente.
    """
    out = pairs_df.copy()
    out["d_graph_m"] = np.nan

    grouped = out.groupby(origin_col, sort=False).groups
    iterator = grouped.items()
    if show_progress:
        iterator = tqdm(iterator, total=len(grouped), desc="Dijkstra por origem")

    for origin, idx in iterator:
        lengths = nx.single_source_dijkstra_path_length(G, source=origin, weight=weight)
        targets = out.loc[idx, target_col]
        out.loc[idx, "d_graph_m"] = [float(lengths.get(t, math.nan)) for t in targets]

    missing = out["d_graph_m"].isna().sum()
    if missing:
        warnings.warn(
            f"{missing} pares ficaram sem caminho no grafo. Eles serão mantidos com NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    return out


def compute_pair_metrics(
    G: nx.Graph,
    pairs_df: pd.DataFrame,
    *,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    weight: str = "length",
    show_progress: bool = True,
    drop_unreachable: bool = True,
) -> pd.DataFrame:
    """
    Calcula d_G e todas as distâncias Lp para uma tabela de pares de vértices.

    A tabela de entrada deve conter colunas `origin` e `target`.
    """
    required = {"origin", "target"}
    missing = required.difference(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df precisa conter as colunas {sorted(required)}. Faltam: {sorted(missing)}")

    out = _node_coordinate_arrays(G, pairs_df)
    out = add_graph_distances_grouped(
        G,
        out,
        weight=weight,
        show_progress=show_progress,
    )
    out = add_lp_columns(out, p_values=p_values)

    d2_col = metric_column_from_p(2.0)
    if d2_col not in out.columns:
        out = add_lp_columns(out, p_values=[2.0])

    out["tortuosity_dG_dL2"] = out["d_graph_m"] / out[d2_col]

    if drop_unreachable:
        out = out.dropna(subset=["d_graph_m"]).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Erros, tortuosidade e busca do melhor p
# ---------------------------------------------------------------------------


def summarize_metric_errors(
    results_df: pd.DataFrame,
    *,
    p_values: Sequence[float] = DEFAULT_P_VALUES,
    graph_col: str = "d_graph_m",
) -> pd.DataFrame:
    """
    Resume MAE, MAPE e razões para cada métrica Lp.

    MAPE usa d_G no denominador:
        mean(abs(d_G - d_p) / d_G) * 100.
    """
    if graph_col not in results_df.columns:
        raise ValueError(f"Coluna {graph_col!r} não encontrada.")

    rows = []
    d_g = results_df[graph_col].to_numpy(dtype=float)
    valid_graph = np.isfinite(d_g) & (d_g > 0)

    for p in p_values:
        col = metric_column_from_p(p)
        if col not in results_df.columns:
            continue

        d_m = results_df[col].to_numpy(dtype=float)
        valid = valid_graph & np.isfinite(d_m) & (d_m > 0)
        if not valid.any():
            continue

        errors = d_m[valid] - d_g[valid]
        abs_errors = np.abs(errors)
        ape = abs_errors / d_g[valid]
        ratio_graph_metric = d_g[valid] / d_m[valid]
        ratio_metric_graph = d_m[valid] / d_g[valid]

        rows.append(
            {
                "metric": metric_name_from_p(p),
                "p": np.inf if math.isinf(p) else float(p),
                "n": int(valid.sum()),
                "MAE_m": float(np.mean(abs_errors)),
                "median_AE_m": float(np.median(abs_errors)),
                "MAPE_percent": float(np.mean(ape) * 100.0),
                "median_APE_percent": float(np.median(ape) * 100.0),
                "bias_m": float(np.mean(errors)),
                "RMSE_m": float(np.sqrt(np.mean(errors**2))),
                "mean_d_metric_m": float(np.mean(d_m[valid])),
                "mean_d_graph_m": float(np.mean(d_g[valid])),
                "mean_dG_over_metric": float(np.mean(ratio_graph_metric)),
                "median_dG_over_metric": float(np.median(ratio_graph_metric)),
                "mean_metric_over_dG": float(np.mean(ratio_metric_graph)),
                "median_metric_over_dG": float(np.median(ratio_metric_graph)),
            }
        )

    return pd.DataFrame(rows).sort_values("MAPE_percent").reset_index(drop=True)


def summarize_tortuosity(
    results_df: pd.DataFrame,
    *,
    tortuosity_col: str = "tortuosity_dG_dL2",
) -> pd.Series:
    """Resumo estatístico da tortuosidade d_G/d_2."""
    if tortuosity_col not in results_df.columns:
        raise ValueError(f"Coluna {tortuosity_col!r} não encontrada.")

    s = results_df[tortuosity_col].replace([np.inf, -np.inf], np.nan).dropna()
    return pd.Series(
        {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }
    )


def p_grid(start: float = 1.0, stop: float = 10.0, step: float = 0.01) -> np.ndarray:
    """Cria uma grade fechada de valores de p."""
    if start <= 0 or stop < start or step <= 0:
        raise ValueError("Use 0 < start <= stop e step > 0.")
    n = int(round((stop - start) / step))
    return np.round(start + step * np.arange(n + 1), 10)


def evaluate_p_grid(
    results_df: pd.DataFrame,
    p_values_grid: Iterable[float],
    *,
    graph_col: str = "d_graph_m",
    origin_x: str = "origin_x",
    origin_y: str = "origin_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
) -> pd.DataFrame:
    """
    Avalia uma grade de p diretamente pelas coordenadas do DataFrame.

    Útil para buscar numericamente o p que minimiza MAE ou MAPE sem recalcular
    menores caminhos no grafo.
    """
    required = {graph_col, origin_x, origin_y, target_x, target_y}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Faltam colunas em results_df: {sorted(missing)}")

    d_g = results_df[graph_col].to_numpy(dtype=float)
    dx = (results_df[origin_x] - results_df[target_x]).abs().to_numpy(dtype=float)
    dy = (results_df[origin_y] - results_df[target_y]).abs().to_numpy(dtype=float)
    valid_graph = np.isfinite(d_g) & (d_g > 0)

    rows = []
    for p in p_values_grid:
        p = float(p)
        if p <= 0:
            continue
        d_p = (dx**p + dy**p) ** (1.0 / p)
        valid = valid_graph & np.isfinite(d_p) & (d_p > 0)
        if not valid.any():
            continue

        errors = d_p[valid] - d_g[valid]
        abs_errors = np.abs(errors)
        ape = abs_errors / d_g[valid]

        rows.append(
            {
                "p": p,
                "n": int(valid.sum()),
                "MAE_m": float(np.mean(abs_errors)),
                "MAPE_percent": float(np.mean(ape) * 100.0),
                "RMSE_m": float(np.sqrt(np.mean(errors**2))),
                "bias_m": float(np.mean(errors)),
            }
        )

    return pd.DataFrame(rows)


def best_p_by(
    grid_results: pd.DataFrame,
    criterion: str = "MAPE_percent",
) -> pd.Series:
    """Retorna a linha da grade com menor valor no critério escolhido."""
    if criterion not in grid_results.columns:
        raise ValueError(f"Critério {criterion!r} não está em grid_results.")
    if grid_results.empty:
        raise ValueError("grid_results está vazio.")
    return grid_results.loc[grid_results[criterion].idxmin()]


# ---------------------------------------------------------------------------
# Exportação e gráficos
# ---------------------------------------------------------------------------


def save_dataframe(df: pd.DataFrame, filepath: str | Path, *, index: bool = False) -> Path:
    """Salva DataFrame em CSV criando diretórios se necessário."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    return filepath


def plot_route(
    G: nx.Graph,
    route: Sequence[int],
    *,
    filepath: str | Path | None = None,
    show: bool = True,
    close: bool = False,
):
    """
    Plota uma rota no grafo usando OSMnx.

    Retorna (fig, ax). Se `filepath` for informado, salva a figura.
    """
    fig, ax = ox.plot_graph_route(
        G,
        list(route),
        node_size=0,
        route_linewidth=4,
        show=show,
        close=close,
    )
    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
    return fig, ax

def latex_label_from_metric_col(metric_col: str) -> str:
    """
    Converte nomes de colunas do DataFrame em labels LaTeX para gráficos.

    Exemplos:
        d_l1_m     -> $d_1$
        d_l2_m     -> $d_2$
        d_l1_54_m  -> $d_{1.54}$
        d_linf_m   -> $d_\infty$
    """
    if metric_col == "d_graph_m":
        return r"$d_G$"

    if metric_col == "d_linf_m":
        return r"$d_\infty$"

    if metric_col.startswith("d_l") and metric_col.endswith("_m"):
        p_str = metric_col.removeprefix("d_l").removesuffix("_m")
        p_str = p_str.replace("_", ".")

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

    metric_label = latex_label_from_metric_col(metric_col)
    print(f"Plotando {metric_label} vs d_G para {len(results_df)} pares de vértices...")

    ax.scatter(x, y, alpha=alpha, s=s)

    max_val = float(np.nanmax([x.max(), y.max()]))
    ax.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        linewidth=1,
        label=r"$y=x$",
    )

    ax.set_xlabel(rf"${metric_label}$ $(\mathrm{{m}})$", fontsize=12)
    ax.set_ylabel(r"$d_G$ $(\mathrm{m})$", fontsize=12)
    ax.set_title(rf"Comparação entre ${metric_label}$ e $d_G$", fontsize=13)

    ax.grid(True, alpha=0.3)
    ax.legend()

    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")

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
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
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
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
    return fig, ax