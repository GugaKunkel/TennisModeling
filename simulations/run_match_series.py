#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulations.simulate_match import (
    DEFAULT_STATE_FIELDS,
    load_player_matrices,
    simulate_match,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run multiple simulated matches between two players and report win percentages."
    )
    parser.add_argument(
        "--player_a_dir",
        required=True,
        help="Directory containing player A matrices (matrix_result.csv, matrix_continue.csv).",
    )
    parser.add_argument(
        "--player_b_dir",
        required=True,
        help="Directory containing player B matrices (matrix_result.csv, matrix_continue.csv).",
    )
    parser.add_argument("--matches", type=int, default=100, help="Number of simulated matches.")
    parser.add_argument("--best_of", type=int, default=5, help="Best-of sets (3 or 5).")
    parser.add_argument(
        "--state_fields",
        default=",".join(DEFAULT_STATE_FIELDS),
        help="Comma-separated state fields matching the matrices.",
    )
    parser.add_argument(
        "--fallback_dir",
        help="Optional directory containing global matrices for backoff (matrix_result.csv, matrix_continue.csv).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional base RNG seed.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    state_fields = [f.strip() for f in args.state_fields.split(",") if f.strip()]

    player_a = Path(args.player_a_dir).name.replace("player_", "").replace("_", " ")
    player_b = Path(args.player_b_dir).name.replace("player_", "").replace("_", " ")

    mats = {
        player_a: load_player_matrices(Path(args.player_a_dir)),
        player_b: load_player_matrices(Path(args.player_b_dir)),
    }
    fallback = load_player_matrices(Path(args.fallback_dir)) if args.fallback_dir else None

    wins = {player_a: 0, player_b: 0}
    base_seed = args.seed
    for i in range(args.matches):
        seed_i = None if base_seed is None else base_seed + i
        winner, _scores = simulate_match(
            player_a=player_a,
            player_b=player_b,
            mats=mats,
            fallback=fallback,
            best_of=args.best_of,
            state_fields=state_fields,
            seed=seed_i,
        )
        wins[winner] += 1

    total = args.matches
    pct_a = wins[player_a] / total * 100
    pct_b = wins[player_b] / total * 100
    print(f"{player_a}: {wins[player_a]} wins ({pct_a:.1f}%)")
    print(f"{player_b}: {wins[player_b]} wins ({pct_b:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
