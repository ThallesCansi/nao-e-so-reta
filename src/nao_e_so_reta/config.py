from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LatLon = tuple[float, float]  # (latitude, longitude)
XY = tuple[float, float]      # (x, y), em metros no CRS projetado
NetworkType = Literal["drive", "walk", "bike"]


@dataclass(frozen=True)
class AppConfig:
    """Configurações gerais do projeto."""

    project_name: str = "Não é só reta"
    course: str = "Espaços Normados"

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")

    place_case: str = "Barão Geraldo, Campinas, São Paulo, Brazil"
    place_stats: str = "Campinas, São Paulo, Brazil"
    network_type: NetworkType = "drive"
    make_undirected: bool = True

    default_graph_path: str = "data/graph.graphml"
    default_place_name: str = "Barão Geraldo, Campinas, São Paulo, Brazil"
    default_network_type: NetworkType = "drive"

    map_center: LatLon = (-22.8190, -47.0700)
    map_zoom: int = 14

    p_min: float = 1.0
    p_max: float = 5.0
    default_p: float = 2.0
    p_step: float = 0.05

    n_superellipse_points: int = 180
    n_visual_curve_points: int = 90

    calibration_default_samples: int = 100
    calibration_max_samples: int = 1000
    calibration_seed: int = 42

    barao_center: LatLon = (-22.8190, -47.0700)
    barao_dist_m: int = 6000

    origin_query: str = "Ciclo Básico I - CB - Rua Monteiro Lobato, 421 - Cidade Universitária, Campinas - SP, Brazil"
    destination_query: str = "Rua Alzira de Águiar Aranha - Jardim Santa Genebra II, Campinas - SP, Brazil"
    origin_fallback_latlon: LatLon = (-22.817132, -47.068200)
    destination_fallback_latlon: LatLon = (-22.830650, -47.079570)

    n_pairs: int = 10_000
    n_origins: int = 250
    seed: int = 42
