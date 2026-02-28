import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pyproj
import streamlit as st
from streamlit_folium import st_folium

LatLon = Tuple[float, float]  # (lat, lon)
XY = Tuple[float, float]      # (x, y) em metros (CRS projetado)


@dataclass(frozen=True)
class AppConfig:
    institution: str = "Ilum - Escola de Ciência"
    course: str = "Espaços Normados"

    # Dados e mapa
    default_place_name: str = "Barão Geraldo, Campinas, Brazil"
    default_network_type: str = "drive"  # drive | walk | bike
    map_center: LatLon = (-22.8184, -47.0647)  # Unicamp (aprox.)
    map_zoom: int = 14

    # Visualização
    default_p: float = 1.54
    n_superellipse_points: int = 140
    n_visual_curve_points: int = 70

    # Caminho padrão do grafo offline (GraphML)
    default_graph_path: str = "data/graph.graphml"


# ----------------------------
# Utilidades matemáticas
# ----------------------------

def minkowski_distance(dx: float, dy: float, p: float) -> float:
    """Distância Minkowski no plano (dx, dy >= 0)."""
    if p <= 0:
        raise ValueError("p deve ser > 0")
    return (dx**p + dy**p) ** (1.0 / p)


def superellipse_points_xy(center_xy: XY, radius: float, p: float, n_points: int) -> List[XY]:
    """Bola Minkowski no plano projetado: |x|^p + |y|^p = R^p."""
    cx, cy = center_xy
    if radius <= 0:
        return [(cx, cy)]

    pts: List[XY] = []
    step = 2 * math.pi / n_points
    for k in range(n_points + 1):
        t = k * step
        ct = math.cos(t)
        st_ = math.sin(t)

        # Parametrização de Lamé (superelipse)
        x = radius * math.copysign(abs(ct) ** (2.0 / p), ct)
        y = radius * math.copysign(abs(st_) ** (2.0 / p), st_)
        pts.append((cx + x, cy + y))

    return pts


def visual_minkowski_curve_xy(a_xy: XY, b_xy: XY, p: float, n_points: int) -> List[XY]:
    """
    Curva "visual" entre A e B (não é geodésica e não é rota real).

    Intuição: p→1 favorece "quebras" tipo L; p→2 tende a suavizar.
    Aqui, fazemos uma interpolação superelíptica apenas para ilustração.
    """
    ax, ay = a_xy
    bx, by = b_xy
    dx = bx - ax
    dy = by - ay

    t = np.linspace(0.0, 1.0, n_points)
    k_visual = max(0.5, float(p) * 2.5)

    x_norm = t
    y_norm = 1.0 - (1.0 - t**k_visual) ** (1.0 / k_visual)

    xs = ax + x_norm * dx
    ys = ay + y_norm * dy
    return list(zip(xs.tolist(), ys.tolist()))


def project_xy_to_latlon(points_xy: List[XY], transformer: pyproj.Transformer) -> List[LatLon]:
    """Converte (x,y) projetado -> (lat,lon) para desenhar no Folium."""
    out: List[LatLon] = []
    for x, y in points_xy:
        lon, lat = transformer.transform(x, y)
        out.append((lat, lon))
    return out


# ----------------------------
# Dados (OSMnx) e projeção
# ----------------------------

def _build_transformer_from_projected_graph(G_proj) -> pyproj.Transformer:
    crs_proj = pyproj.CRS.from_user_input(G_proj.graph["crs"])
    return pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)


@st.cache_resource(show_spinner=False)
def load_graph(place_name: str, network_type: str, graph_path: str):
    """
    Carrega grafo do OSM.

    Preferência:
    1) Se existir um arquivo GraphML em graph_path, carrega dele (mais rápido e reprodutível).
    2) Caso contrário, baixa do OpenStreetMap via OSMnx.

    Retorna: (G_latlon, G_proj_metros, transformer_proj_to_wgs84)
    """
    ox.settings.log_console = False
    ox.settings.use_cache = True

    gp = Path(graph_path)

    if gp.exists() and gp.is_file():
        G = ox.load_graphml(gp)
    else:
        # Se não houver arquivo offline, baixa do OSM
        G = ox.graph_from_place(place_name, network_type=network_type)

    G_proj = ox.project_graph(G)
    transformer = _build_transformer_from_projected_graph(G_proj)

    return G, G_proj, transformer


def nearest_node(G, point: LatLon) -> int:
    lat, lon = point
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))


def node_xy(G_proj, node: int) -> XY:
    return (float(G_proj.nodes[node]["x"]), float(G_proj.nodes[node]["y"]))


# ----------------------------
# Estado e cliques
# ----------------------------

def ensure_state():
    if "markers" not in st.session_state:
        st.session_state["markers"] = []
    if "last_click" not in st.session_state:
        st.session_state["last_click"] = None


def reset_markers():
    st.session_state["markers"] = []
    st.session_state["last_click"] = None


def handle_click(output: Dict) -> bool:
    """Atualiza markers a partir do clique do folium. Retorna True se mudou."""
    last = output.get("last_clicked")
    if not last:
        return False

    new_point = (float(last["lat"]), float(last["lng"]))

    if st.session_state.get("last_click") == new_point:
        return False
    st.session_state["last_click"] = new_point

    markers: List[LatLon] = st.session_state["markers"]

    if len(markers) >= 2:
        st.session_state["markers"] = [new_point]
    else:
        if not markers or markers[-1] != new_point:
            markers.append(new_point)
            st.session_state["markers"] = markers

    return True


# ----------------------------
# Cálculos e renderização
# ----------------------------

def compute_metrics(G, G_proj, origin: LatLon, dest: LatLon, p: float) -> Dict[str, Optional[float]]:
    """Calcula distância real (ruas) e distâncias Lp no plano projetado. Tudo em metros."""
    u = nearest_node(G, origin)
    v = nearest_node(G, dest)

    p1 = node_xy(G_proj, u)
    p2 = node_xy(G_proj, v)
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    d_euclid = math.hypot(dx, dy)
    d_manhat = dx + dy
    d_cheby = max(dx, dy)
    d_mink = minkowski_distance(dx, dy, p)

    dist_real: Optional[float]
    try:
        dist_real = float(nx.shortest_path_length(G, u, v, weight="length"))
    except nx.NetworkXNoPath:
        dist_real = None

    return {
        "dist_real": dist_real,
        "d_euclid": d_euclid,
        "d_manhat": d_manhat,
        "d_cheby": d_cheby,
        "d_mink": d_mink,
    }


def safe_percent_error(estimate: float, reference: Optional[float]) -> Optional[float]:
    if reference is None or reference <= 0:
        return None
    return ((estimate - reference) / reference) * 100.0


def add_markers_to_map(m: folium.Map, markers: List[LatLon]):
    for i, coords in enumerate(markers):
        color = "green" if i == 0 else "red"
        label = "Origem" if i == 0 else "Destino"
        folium.Marker(location=coords, icon=folium.Icon(color=color), tooltip=label).add_to(m)


def add_routes_and_theory_to_map(
    m: folium.Map,
    G,
    G_proj,
    transformer: pyproj.Transformer,
    origin: LatLon,
    dest: LatLon,
    p: float,
    cfg: AppConfig,
    show_euclid: bool,
    show_manhattan: bool,
    show_minkowski_ball: bool,
    show_minkowski_curve: bool,
) -> Tuple[Optional[List[LatLon]], Optional[str]]:
    """Desenha rota real e curvas teóricas no mapa. Retorna (rota_real, erro_msg)."""

    u = nearest_node(G, origin)
    v = nearest_node(G, dest)

    # Linhas teóricas diretas (em lat/lon)
    if show_euclid:
        folium.PolyLine([origin, dest], color="green", weight=2, dash_array="5, 10", tooltip="Euclidiana (L2)").add_to(m)

    if show_manhattan:
        corner = (origin[0], dest[1])
        folium.PolyLine(
            [origin, corner, dest],
            color="red",
            weight=3,
            opacity=0.6,
            dash_array="5, 10",
            tooltip="Manhattan (L1, visual)",
        ).add_to(m)
        folium.CircleMarker(location=corner, radius=3, color="red", fill=True).add_to(m)

    # Minkowski no plano projetado (metros) -> reprojeta para lat/lon
    if show_minkowski_ball or show_minkowski_curve:
        a_xy = node_xy(G_proj, u)
        b_xy = node_xy(G_proj, v)
        R = minkowski_distance(abs(a_xy[0] - b_xy[0]), abs(a_xy[1] - b_xy[1]), p)

        if show_minkowski_ball:
            ball_xy = superellipse_points_xy(a_xy, R, p=p, n_points=cfg.n_superellipse_points)
            ball_latlon = project_xy_to_latlon(ball_xy, transformer)
            folium.Polygon(
                locations=ball_latlon,
                color="orange",
                weight=3,
                fill=False,
                dash_array="5, 5",
                tooltip=f"Fronteira Minkowski (p={p:.2f})",
            ).add_to(m)

        if show_minkowski_curve:
            curve_xy = visual_minkowski_curve_xy(a_xy, b_xy, p=p, n_points=cfg.n_visual_curve_points)
            curve_latlon = project_xy_to_latlon(curve_xy, transformer)
            folium.PolyLine(
                locations=curve_latlon,
                color="purple",
                weight=4,
                opacity=0.85,
                tooltip=f"Curva visual Minkowski (p={p:.2f})",
            ).add_to(m)

    # Rota real no grafo
    try:
        route = nx.shortest_path(G, u, v, weight="length")
        route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
        folium.PolyLine(route_latlon, color="blue", weight=5, opacity=0.75, tooltip="Distância real (ruas)").add_to(m)
        return route_latlon, None
    except nx.NetworkXNoPath:
        return None, "Não há caminho viável entre estes pontos (grafo desconectado)."


def render_metrics_panel(metrics: Dict[str, Optional[float]], p: float):
    st.markdown("---")
    st.subheader("Comparação de métricas")

    dist_real = metrics["dist_real"]
    d_euclid = float(metrics["d_euclid"])
    d_manhat = float(metrics["d_manhat"])
    d_cheby = float(metrics["d_cheby"])
    d_mink = float(metrics["d_mink"])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Real (grafo)", "N/A" if dist_real is None else f"{dist_real:.0f} m")

    with col2:
        err = safe_percent_error(d_euclid, dist_real)
        st.metric("Euclidiana (L2)", f"{d_euclid:.0f} m", "N/A" if err is None else f"{err:.1f}%")

    with col3:
        err = safe_percent_error(d_manhat, dist_real)
        st.metric("Manhattan (L1)", f"{d_manhat:.0f} m", "N/A" if err is None else f"{err:.1f}%")

    with col4:
        err = safe_percent_error(d_mink, dist_real)
        st.metric(f"Minkowski (p={p:.2f})", f"{d_mink:.0f} m", "N/A" if err is None else f"{err:.1f}%")

    with col5:
        err = safe_percent_error(d_cheby, dist_real)
        st.metric("Chebyshev (L∞)", f"{d_cheby:.0f} m", "N/A" if err is None else f"{err:.1f}%")

    if dist_real is not None and d_euclid > 0:
        st.caption(f"Tortuosidade (ruas / L2): {dist_real / d_euclid:.2f}")


# ----------------------------
# App (Streamlit)
# ----------------------------

def main():
    cfg = AppConfig()

    st.set_page_config(layout="wide", page_title="Normas e Métricas em Redes Urbanas")
    st.title("Normas e Métricas em Redes Urbanas")

    with st.expander("Sobre o projeto", expanded=False):
        st.write(
            f"Este app foi construído como trabalho final da disciplina **{cfg.course}** em **{cfg.institution}**. "
            "A ideia é comparar a métrica **intrínseca** do grafo viário (menor caminho em ruas) "
            "com métricas **extrínsecas** induzidas por normas Lp no plano. "
            "Isso aparece em problemas práticos (planejamento urbano, logística, apps de navegação) e também "
            "em questões matemáticas (distorsão de embeddings, geometria métrica)."
        )
        st.write(
            "Como usar: clique no mapa para definir **origem** (verde) e **destino** (vermelho). "
            "A rota real (azul) é o menor caminho no grafo do OpenStreetMap. "
            "As curvas/linhas teóricas (verde/vermelho/laranja/roxo) são comparações visuais com L2, L1 e Minkowski."
        )

    ensure_state()

    # Sidebar: parâmetros e controles
    with st.sidebar:
        st.header("Parâmetros")
        p = st.slider("Parâmetro p (Minkowski)", 1.0, 4.0, float(cfg.default_p), 0.01)

        st.divider()
        st.subheader("Camadas no mapa")
        show_euclid = st.checkbox("Mostrar Euclidiana (L2)", value=True)
        show_manhattan = st.checkbox("Mostrar Manhattan (L1, visual)", value=True)
        show_minkowski_ball = st.checkbox("Mostrar fronteira Minkowski", value=True)
        show_minkowski_curve = st.checkbox("Mostrar curva visual Minkowski", value=True)

        st.divider()
        if st.button("Resetar origem/destino", use_container_width=True):
            reset_markers()
            st.rerun()

        st.caption(
            "Nota: a curva Minkowski exibida é uma ilustração no plano projetado (metros) e reprojetada para o mapa."
        )

        st.divider()
        with st.expander("Configuração de dados (avançado)"):
            graph_path = st.text_input(
                "Arquivo GraphML (opcional)",
                value=os.environ.get("GRAPH_PATH", cfg.default_graph_path),
                help=(
                    "Se este arquivo existir no repositório, o app carrega dele (mais rápido). "
                    "Se não existir, o app baixa do OpenStreetMap usando o 'Local'."
                ),
            )
            place_name = st.text_input("Local (consulta ao OSM)", value=cfg.default_place_name)
            network_type = st.selectbox("Tipo de rede", options=["drive", "walk", "bike"], index=["drive", "walk", "bike"].index(cfg.default_network_type))

            if st.button("Recarregar grafo", use_container_width=True):
                load_graph.clear()  # limpa cache
                reset_markers()
                st.rerun()

    # Carrega grafo
    with st.spinner("Carregando malha viária (na primeira vez pode demorar)..."):
        try:
            G, G_proj, transformer = load_graph(place_name, network_type, graph_path)
        except Exception as e:
            st.error("Falha ao carregar o grafo (offline/OSM).")
            st.exception(e)
            return

    # Mapa base
    m = folium.Map(location=list(cfg.map_center), zoom_start=cfg.map_zoom)

    markers: List[LatLon] = st.session_state["markers"]
    add_markers_to_map(m, markers)

    route_error = None
    if len(markers) == 2:
        origin, dest = markers
        _, route_error = add_routes_and_theory_to_map(
            m=m,
            G=G,
            G_proj=G_proj,
            transformer=transformer,
            origin=origin,
            dest=dest,
            p=p,
            cfg=cfg,
            show_euclid=show_euclid,
            show_manhattan=show_manhattan,
            show_minkowski_ball=show_minkowski_ball,
            show_minkowski_curve=show_minkowski_curve,
        )

    output = st_folium(m, width=1000, height=520)

    if route_error:
        st.warning(route_error)

    if handle_click(output):
        st.rerun()

    if len(st.session_state["markers"]) == 2:
        origin, dest = st.session_state["markers"]
        metrics = compute_metrics(G, G_proj, origin, dest, p=p)
        render_metrics_panel(metrics, p=p)


if __name__ == "__main__":
    main()
