#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return False


def normalize_name(name: str) -> str:
    return str(name).replace("\xa0", " ").strip()


def load_rankings(path: Path | None) -> Dict[str, float]:
    if not path:
        return {}
    if not path.exists():
        print(f"Rankings file not found: {path}")
        return {}
    df = pd.read_csv(path)
    rank_cols = [c for c in ["atp_rank", "wta_rank", "elo_rank", "helo_rank", "celo_rank"] if c in df.columns]
    name_col = "player" if "player" in df.columns else ("name" if "name" in df.columns else None)
    if not name_col:
        return {}
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        ranks = [row[c] for c in rank_cols if pd.notna(row[c])]
        if not ranks:
            continue
        mapping[normalize_name(row[name_col])] = float(ranks[0])
    return mapping


def load_player_map(match_files: list[Path]) -> Dict[str, tuple[str, str]]:
    mapping: Dict[str, tuple[str, str]] = {}
    for path in match_files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["match_id", "Player 1", "Player 2"])
        except Exception:
            continue
        for _, row in df.iterrows():
            mid = str(row.get("match_id", "")).strip()
            if not mid or mid in mapping:
                continue
            p1 = normalize_name(row.get("Player 1", ""))
            p2 = normalize_name(row.get("Player 2", ""))
            mapping[mid] = (p1, p2)
    return mapping


def weight_for_row(row: pd.Series, player_map: Dict[str, tuple[str, str]], rank_map: Dict[str, float]) -> float:
    if not player_map or not rank_map:
        return 1.0
    match_id = row.get("match_id", None)
    if pd.isna(match_id):
        return 1.0
    players = player_map.get(str(match_id))
    if not players:
        return 1.0
    label = row.get("player_label", "")
    if label not in {"P1", "P2"}:
        return 1.0
    opp = players[1] if label == "P1" else players[0]
    rank = rank_map.get(normalize_name(opp))
    if rank is None:
        return 1.0
    if rank <= 10:
        return 0.1
    if rank <= 25:
        return 0.2
    if rank <= 50:
        return 0.4
    if rank <= 100:
        return 0.8
    if rank <= 200:
        return 1.0
    return 1.1


def build_serve_score_transitions(df: pd.DataFrame, player_map: Dict[str, tuple[str, str]], rank_map: Dict[str, float]) -> pd.DataFrame:
    """Only count terminal rows where the hitter is the server (server_flag == serve_side)."""
    subset = df[df["server_flag"] == "serve_side"]
    subset = subset[subset["is_terminal"].apply(coerce_bool)]
    subset = subset[["match_id", "player_label", "point_score", "point_score_after"]].dropna()
    if subset.empty:
        return subset
    subset = subset.rename(columns={"point_score": "state", "point_score_after": "new_state"})
    subset["weight"] = subset.apply(lambda r: weight_for_row(r, player_map, rank_map), axis=1)
    counts = subset.groupby(["state", "new_state"])["weight"].sum().reset_index(name="count")
    return counts


def build_score_matrix(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    states = sorted(transitions["state"].unique())
    next_states = sorted(transitions["new_state"].unique())
    counts = transitions.pivot_table(
        index="state", columns="new_state", values="count", aggfunc="sum", fill_value=0
    ).reindex(index=states, columns=next_states, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index.name = ""
    return probs


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.insert(0, "", out.index)
    out.to_csv(path, index=False)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute serving-only score transition matrices from state_transitions CSV."
    )
    parser.add_argument("--input", required=True, help="Path to state_transitions CSV.")
    parser.add_argument("--output_dir", default="serve_score_matrices", help="Output directory.")
    parser.add_argument(
        "--per_player",
        action="store_true",
        help="If set, build matrices for each player_name separately under output_dir/player_*.",
    )
    parser.add_argument(
        "--include_players",
        action="append",
        help="Optional player_name filter (can specify multiple). Only applied in per_player mode.",
    )
    parser.add_argument(
        "--rankings_csv",
        help="Optional rankings file (e.g., raw_data/atp_elo_ratings.csv) for opponent-strength weighting.",
    )
    parser.add_argument(
        "--matches_file",
        action="append",
        help="Optional matches CSVs (match_id, Player 1, Player 2) to infer opponents; can specify multiple.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    rank_map = load_rankings(Path(args.rankings_csv)) if args.rankings_csv else {}
    default_matches = [Path("raw_data/charting-m-matches.csv"), Path("raw_data/charting-w-matches.csv")]
    match_files = [Path(p) for p in args.matches_file] if args.matches_file else default_matches
    player_map = load_player_map(match_files)

    if args.per_player:
        if "player_name" not in df.columns:
            print("Input missing player_name column; cannot run per-player.")
            return 1
        players = df["player_name"].dropna().unique()
        if args.include_players:
            players = [p for p in players if p in args.include_players]
        if len(players) == 0:
            print("No players to process.")
            return 1
        for p in players:
            sub = df[(df["player_name"] == p)]
            transitions = build_serve_score_transitions(sub, player_map, rank_map)
            out_dir = Path(args.output_dir) / f"player_{p.replace(' ', '_')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            transitions.to_csv(out_dir / "serve_score_transitions.csv", index=False)
            matrix = build_score_matrix(transitions)
            write_matrix(matrix, out_dir / "serve_score_matrix.csv")
            print(f"Wrote serve score transitions ({len(transitions)} rows) and matrix to {out_dir}")
        return 0

    transitions = build_serve_score_transitions(df, player_map, rank_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(out_dir / "serve_score_transitions.csv", index=False)
    matrix = build_score_matrix(transitions)
    write_matrix(matrix, out_dir / "serve_score_matrix.csv")
    print(f"Wrote serve score transitions ({len(transitions)} rows) and matrix to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
