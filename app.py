from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from nao_e_so_reta.analysis import (
    compare_norms_for_pair,
    tortuosity,
    calibration_curve,
    records_to_dicts,
)
from nao_e_so_reta.config import AppConfig, LatLon
from nao_e_so_reta.graph_io import load_graph_from_path_or_place
from nao_e_so_reta.norms import lp_distance_xy
from nao_e_so_reta.routing import shortest_route_between_points
from nao_e_so_reta.sampling import build_calibration_pairs
from nao_e_so_reta.ui_text import CALIBRATION_EXPLANATION, MATH_PANEL, PROJECT_EXPLANATION
from nao_e_so_reta.visualization import (
    add_legend,
    add_nearest_node_markers,
    add_point_markers,
    add_route_polyline,
    add_theoretical_layers,
    base_map,
)


@st.cache_resource(show_spinner=False)
def cached_graph(place_name: str, network_type: str, graph_path: str):
    return load_graph_from_path_or_place(
        place_name=place_name,
        network_type=network_type,
        graph_path=graph_path,
    )


def ensure_state() -> None:
    st.session_state.setdefault("markers", [])
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("calibration_df", None)


def reset_markers() -> None:
    st.session_state["markers"] = []
    st.session_state["last_click"] = None


def handle_click(output: dict) -> bool:
    last = output.get("last_clicked")
    if not last:
        return False

    point: LatLon = (float(last["lat"]), float(last["lng"]))

    if st.session_state.get("last_click") == point:
        return False

    st.session_state["last_click"] = point
    markers: list[LatLon] = st.session_state["markers"]

    if len(markers) >= 2:
        st.session_state["markers"] = [point]
    else:
        markers.append(point)
        st.session_state["markers"] = markers

    return True


def render_metric_table(comparisons) -> None:
    rows = []
    for item in comparisons:
        rows.append(
            {
                "Métrica": item.name,
                "Distância (m)": item.distance_m,
                "Erro absoluto (m)": item.absolute_error_m,
                "Erro relativo (%)": item.relative_error_pct,
                "Grafo / métrica": item.graph_over_metric,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Distância (m)": st.column_config.NumberColumn(format="%.1f"),
            "Erro absoluto (m)": st.column_config.NumberColumn(format="%.1f"),
            "Erro relativo (%)": st.column_config.NumberColumn(format="%.2f"),
            "Grafo / métrica": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def render_pair_analysis(route, p: float) -> None:
    st.subheader("Comparação de métricas para o par selecionado")

    comparisons = compare_norms_for_pair(
        route.origin_xy,
        route.destination_xy,
        graph_distance_m=route.length_m,
        p=p,
    )

    l2 = lp_distance_xy(route.origin_xy, route.destination_xy, p=2.0)
    tau = tortuosity(route.length_m, l2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distância real", "N/A" if route.length_m is None else f"{route.length_m:.0f} m")
    col2.metric("Euclidiana L2", f"{l2:.0f} m")
    col3.metric("Tortuosidade", "N/A" if tau is None else f"{tau:.3f}")
    col4.metric("Nós na rota", f"{len(route.route_nodes)}")

    render_metric_table(comparisons)

    with st.expander("Detalhes dos nós usados no cálculo"):
        st.write(
            {
                "origin_node": route.origin_node,
                "destination_node": route.destination_node,
                "origin_node_latlon": route.origin_node_latlon,
                "destination_node_latlon": route.destination_node_latlon,
                "origin_xy_m": route.origin_xy,
                "destination_xy_m": route.destination_xy,
            }
        )


def render_calibration_panel(graph, projected_graph, cfg: AppConfig) -> None:
    st.subheader("Calibração empírica de p")
    st.markdown(CALIBRATION_EXPLANATION)

    col1, col2, col3 = st.columns([1, 1, 1])
    n_samples = col1.slider(
        "Número de pares amostrados",
        min_value=20,
        max_value=cfg.calibration_max_samples,
        value=cfg.calibration_default_samples,
        step=20,
    )
    seed = col2.number_input("Semente", min_value=0, value=cfg.calibration_seed, step=1)
    criterion = col3.selectbox("Critério de melhor p", ["mape_pct", "rmse_m", "mae_m"], index=0)

    p_values = [round(1.0 + 0.05 * i, 2) for i in range(0, 61)]  # 1.00 até 4.00

    if st.button("Rodar calibração", use_container_width=True):
        with st.spinner("Amostrando pares e calculando menores caminhos..."):
            pairs = build_calibration_pairs(
                graph,
                projected_graph,
                n_pairs=int(n_samples),
                seed=int(seed),
            )
            records = calibration_curve(pairs, p_values=p_values)
            df = pd.DataFrame(records_to_dicts(records))
            st.session_state["calibration_df"] = df

    df = st.session_state.get("calibration_df")
    if df is None:
        return

    if criterion == "mape_pct":
        best_row = df.loc[df["MAPE_pct"].idxmin()]
        criterion_label = "MAPE"
    elif criterion == "rmse_m":
        best_row = df.loc[df["RMSE_m"].idxmin()]
        criterion_label = "RMSE"
    else:
        best_row = df.loc[df["MAE_m"].idxmin()]
        criterion_label = "MAE"

    st.success(
        f"Melhor p por {criterion_label}: p={best_row['p']:.2f}, "
        f"α={best_row['alpha']:.3f}, MAPE={best_row['MAPE_pct']:.2f}%."
    )

    chart_df = df.set_index("p")[["MAPE_pct", "RMSE_m", "MAE_m"]]
    st.line_chart(chart_df, use_container_width=True)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "p": st.column_config.NumberColumn(format="%.2f"),
            "alpha": st.column_config.NumberColumn(format="%.4f"),
            "MAE_m": st.column_config.NumberColumn(format="%.2f"),
            "RMSE_m": st.column_config.NumberColumn(format="%.2f"),
            "MAPE_pct": st.column_config.NumberColumn(format="%.2f"),
            "mean_distortion": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.download_button(
        "Baixar calibração em CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="calibracao_metricas_urbanas.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    cfg = AppConfig()
    st.set_page_config(layout="wide", page_title=cfg.project_name)
    ensure_state()

    st.title("Não é só reta — Laboratório de Métricas Urbanas")
    st.markdown(PROJECT_EXPLANATION)

    with st.sidebar:
        st.header("Parâmetros")

        p = st.slider(
            "Parâmetro p da norma Lp",
            cfg.p_min,
            cfg.p_max,
            float(cfg.default_p),
            cfg.p_step,
        )

        st.divider()
        st.subheader("Camadas")
        show_euclidean = st.checkbox("Euclidiana visual", value=True)
        show_manhattan = st.checkbox("Manhattan visual", value=True)
        show_minkowski_ball = st.checkbox("Bola Lp", value=True)
        show_minkowski_curve = st.checkbox("Curva didática Lp", value=True)

        st.divider()
        if st.button("Resetar origem/destino", use_container_width=True):
            reset_markers()
            st.rerun()

        st.divider()
        with st.expander("Dados e grafo", expanded=False):
            graph_path = st.text_input(
                "Arquivo GraphML",
                value=os.environ.get("GRAPH_PATH", cfg.default_graph_path),
            )
            place_name = st.text_input("Local OSM", value=cfg.default_place_name)
            network_type = st.selectbox(
                "Tipo de rede",
                options=["drive", "walk", "bike"],
                index=["drive", "walk", "bike"].index(cfg.default_network_type),
            )
            if st.button("Recarregar grafo", use_container_width=True):
                cached_graph.clear()
                reset_markers()
                st.rerun()

        with st.expander("Configuração atual"):
            st.json(asdict(cfg))

    with st.spinner("Carregando malha viária..."):
        try:
            graph, projected_graph, transformer, source = cached_graph(place_name, network_type, graph_path)
        except Exception as exc:
            st.error("Falha ao carregar o grafo.")
            st.exception(exc)
            return

    st.caption(f"Fonte do grafo: {source}")

    markers: list[LatLon] = st.session_state["markers"]
    map_obj = base_map(cfg.map_center, cfg.map_zoom)
    add_point_markers(map_obj, markers)

    route = None
    if len(markers) == 2:
        route = shortest_route_between_points(graph, projected_graph, markers[0], markers[1])
        add_nearest_node_markers(map_obj, route.origin_node_latlon, route.destination_node_latlon)

        if route.ok:
            add_route_polyline(map_obj, route.route_latlon)

        add_theoretical_layers(
            map_obj,
            origin_click=markers[0],
            destination_click=markers[1],
            origin_xy=route.origin_xy,
            destination_xy=route.destination_xy,
            transformer_projected_to_wgs84=transformer,
            p=p,
            show_euclidean=show_euclidean,
            show_manhattan=show_manhattan,
            show_minkowski_ball=show_minkowski_ball,
            show_minkowski_curve=show_minkowski_curve,
            n_superellipse_points=cfg.n_superellipse_points,
            n_visual_curve_points=cfg.n_visual_curve_points,
        )

    add_legend(
        map_obj,
        has_origin=len(markers) >= 1,
        has_destination=len(markers) >= 2,
        has_route=route is not None and route.ok,
        show_euclidean=show_euclidean,
        show_manhattan=show_manhattan,
        show_minkowski_ball=show_minkowski_ball,
        show_minkowski_curve=show_minkowski_curve,
    )

    output = st_folium(map_obj, width=None, height=560, returned_objects=["last_clicked"])

    if handle_click(output):
        st.rerun()

    if len(markers) < 2:
        st.info("Clique no mapa para definir origem e destino.")
    elif route is not None:
        if route.error:
            st.warning(route.error)
        render_pair_analysis(route, p=p)

    with st.expander("Matemática do projeto", expanded=False):
        st.markdown(MATH_PANEL)

    with st.expander("Experimento: calibrar p para a rede", expanded=False):
        render_calibration_panel(graph, projected_graph, cfg)


if __name__ == "__main__":
    main()
