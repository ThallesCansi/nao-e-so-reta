from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nao_e_so_reta.graph_io import save_graphml_for_place  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa um grafo do OpenStreetMap via OSMnx e salva em GraphML."
    )
    parser.add_argument("--place", required=True, help="Consulta de lugar para o OSM/Nominatim.")
    parser.add_argument(
        "--network",
        default="drive",
        choices=["drive", "walk", "bike"],
        help="Tipo de rede viária.",
    )
    parser.add_argument(
        "--out",
        default="data/graph.graphml",
        help="Caminho de saída do GraphML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = save_graphml_for_place(
        place_name=args.place,
        network_type=args.network,
        output_path=args.out,
    )
    print(f"Grafo salvo em: {out}")


if __name__ == "__main__":
    main()
