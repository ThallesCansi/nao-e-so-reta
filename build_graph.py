"""Baixa um grafo do OpenStreetMap via OSMnx e salva em GraphML.

Uso:
  python scripts/build_graph.py --place "Barão Geraldo, Campinas, Brazil" --network drive --out data/graph.graphml

"""

import argparse
from pathlib import Path

import osmnx as ox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", required=True, help="Consulta de lugar para o OSM (Nominatim)")
    parser.add_argument("--network", default="drive", choices=["drive", "walk", "bike"], help="Tipo de rede")
    parser.add_argument("--out", default="data/graph.graphml", help="Caminho de saída (GraphML)")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ox.settings.log_console = True
    ox.settings.use_cache = True

    G = ox.graph_from_place(args.place, network_type=args.network)
    ox.save_graphml(G, out)
    print(f"Salvo em: {out}")


if __name__ == "__main__":
    main()
