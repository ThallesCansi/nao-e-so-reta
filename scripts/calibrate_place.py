from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nao_e_so_reta.analysis import calibration_curve, records_to_dicts  # noqa: E402
from nao_e_so_reta.graph_io import load_graph_from_path_or_place  # noqa: E402
from nao_e_so_reta.sampling import build_calibration_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra p em d_G ≈ alpha * ||x-y||_p para uma rede OSM."
    )
    parser.add_argument("--place", required=True, help="Lugar OSM/Nominatim.")
    parser.add_argument("--network", default="drive", choices=["drive", "walk", "bike"])
    parser.add_argument("--graph", default="data/graph.graphml", help="GraphML local opcional.")
    parser.add_argument("--samples", type=int, default=250, help="Número de pares válidos.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/calibration.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, projected, _, source = load_graph_from_path_or_place(
        place_name=args.place,
        network_type=args.network,
        graph_path=args.graph,
        log_console=True,
    )
    print(f"Fonte: {source}")

    pairs = build_calibration_pairs(
        graph,
        projected,
        n_pairs=args.samples,
        seed=args.seed,
    )

    p_values = [round(1.0 + 0.05 * i, 2) for i in range(0, 61)]
    records = calibration_curve(pairs, p_values=p_values)

    df = pd.DataFrame(records_to_dicts(records))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    best = df.loc[df["MAPE_pct"].idxmin()]
    print(
        f"Melhor p por MAPE: p={best['p']:.2f}, "
        f"alpha={best['alpha']:.4f}, MAPE={best['MAPE_pct']:.2f}%"
    )
    print(f"Resultado salvo em: {out}")


if __name__ == "__main__":
    main()
