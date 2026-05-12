from __future__ import annotations

import pyproj

from nao_e_so_reta.config import XY, LatLon


def transformer_projected_to_wgs84(projected_crs: object) -> pyproj.Transformer:
    """Cria transformador do CRS projetado do grafo para WGS84."""
    crs_proj = pyproj.CRS.from_user_input(projected_crs)
    return pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)


def transformer_wgs84_to_projected(projected_crs: object) -> pyproj.Transformer:
    """Cria transformador de WGS84 para o CRS projetado do grafo."""
    crs_proj = pyproj.CRS.from_user_input(projected_crs)
    return pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)


def xy_to_latlon(point: XY, transformer: pyproj.Transformer) -> LatLon:
    """Converte (x, y) projetado para (lat, lon)."""
    lon, lat = transformer.transform(point[0], point[1])
    return (float(lat), float(lon))


def latlon_to_xy(point: LatLon, transformer: pyproj.Transformer) -> XY:
    """Converte (lat, lon) para (x, y) projetado."""
    lat, lon = point
    x, y = transformer.transform(lon, lat)
    return (float(x), float(y))


def points_xy_to_latlon(points: list[XY], transformer: pyproj.Transformer) -> list[LatLon]:
    """Converte lista de pontos projetados para latitude/longitude."""
    return [xy_to_latlon(p, transformer) for p in points]
