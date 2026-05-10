from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LatLon = tuple[float, float]  # (lat, lon)
XY = tuple[float, float]      # (x, y), em metros no CRS projetado
NetworkType = Literal["drive", "walk", "bike"]


@dataclass(frozen=True)
class AppConfig:
    """Configuração central do aplicativo."""

    institution: str = "Ilum - Escola de Ciência"
    course: str = "Espaços Normados"
    project_name: str = "Não é só reta"

    default_place_name: str = "Barão Geraldo, Campinas, Brazil"
    default_network_type: NetworkType = "drive"
    default_graph_path: str = "data/graph.graphml"

    map_center: LatLon = (-22.8184, -47.0647)
    map_zoom: int = 14

    default_p: float = 1.54
    p_min: float = 1.0
    p_max: float = 4.0
    p_step: float = 0.01

    n_superellipse_points: int = 180
    n_visual_curve_points: int = 90

    calibration_default_samples: int = 120
    calibration_max_samples: int = 1000
    calibration_seed: int = 42
