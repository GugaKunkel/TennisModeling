#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


BIN_EDGES = [
    (1, 10, "Top10"),
    (11, 25, "11_25"),
    (26, 50, "26_50"),
    (51, 100, "51_100"),
    (101, 200, "101_200"),
]
DEFAULT_BIN = "200_plus"


def coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y"}


def normalize_name(name: str) -> str:
    return str(name).replace("\xa0", " ").strip()


def classify_bin(rank: float | None) -> str:
    if rank is None or pd.isna(rank):
        return DEFAULT_BIN
    for lo, hi, label in BIN_EDGES:
        if lo <= rank <= hi:
            return label
    return DEFAULT_BIN


def load_rankings(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    name_col = "player" if "player" in df.columns else ("name" if "name" in df.columns else None)
    if not name_col:
        return {}
    rank_cols = [c for c in ("atp_rank", "wta_rank", "elo_rank", "helo_rank", "celo_rank") if c in df.columns]
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        ranks = [row[c] for c in rank_cols if pd.notna(row[c])]
        if not ranks:
            continue
        mapping[normalize_name(row[name_col])] = float(ranks[0])
    return mapping


def load_match_map(files: Iterable[Path]) -> Dict[str, tuple[str, str]]:
    mapping: Dict[str, tuple[str, str]] = {}
    for path in files:
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


def attach_bins(df: pd.DataFrame, rank_map: Dict[str, float], match_map: Dict[str, tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    df["player_name_norm"] = df["player_name"].apply(normalize_name)
    # Opponent lookup via match map and player_label
    opps = []
    for _, row in df.iterrows():
        mid = str(row.get("match_id", ""))
        players = match_map.get(mid, ("", ""))
        label = row.get("player_label", "")
        if label == "P1":
            opps.append(players[1])
        elif label == "P2":
            opps.append(players[0])
        else:
            opps.append("")
    df["opponent_name_norm"] = opps
    df["player_rank"] = df["player_name_norm"].map(rank_map)
    df["opponent_rank"] = df["opponent_name_norm"].map(rank_map)
    df["player_bin"] = df["player_rank"].apply(classify_bin)
    df["opponent_bin"] = df["opponent_rank"].apply(classify_bin)
    return df


def compute_result_matrix(df: pd.DataFrame) -> pd.DataFrame:
    term = df["is_terminal"].apply(coerce_bool)
    player_label = df["player_label"].fillna("")
    winner_label = df["winner_label"].fillna("")
    result = []
    for t, pl, wl in zip(term, player_label, winner_label):
        if not t:
            result.append("continue")
        else:
            if wl and wl == pl:
                result.append("win")
            elif wl in {"P1", "P2"} and wl != pl:
                result.append("loss")
            else:
                result.append("continue")
    df_local = df.copy()
    df_local["result_for_hitter"] = result
    states = sorted(df_local["state"].dropna().unique())
    result_cats = ["win", "loss", "continue"]
    counts = df_local.groupby(["state", "result_for_hitter"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=result_cats, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs = probs.reindex(states).fillna(0.0)
    probs.index.name = ""
    return probs


def compute_continue_matrix(df: pd.DataFrame) -> pd.DataFrame:
    non_term = df[df["is_terminal"].apply(coerce_bool) == False]
    states = sorted(non_term["state"].dropna().unique())
    next_states = sorted(non_term["new_state"].dropna().unique())
    if not states or not next_states:
        return pd.DataFrame(columns=[""] + next_states)
    counts = non_term.groupby(["state", "new_state"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=states, columns=next_states, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index.name = ""
    return probs


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.insert(0, "", out.index)
    out.to_csv(path, index=False)


def save_matrices(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute_result_matrix(df)
    cont = compute_continue_matrix(df)
    write_matrix(res, out_dir / "matrix_result.csv")
    write_matrix(cont, out_dir / "matrix_continue.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rank-bin-aware state transition matrices with fallbacks."
    )
    parser.add_argument("--input", required=True, help="Path to state_transitions CSV.")
    parser.add_argument("--rankings_csv", required=True, help="CSV with player rankings/Elo.")
    parser.add_argument(
        "--matches_file",
        action="append",
        help="Match CSVs containing match_id, Player 1, Player 2. Defaults to charting match files.",
    )
    parser.add_argument(
        "--output_dir",
        default="rank_bin_matrices",
        help="Output base directory for all matrices.",
    )
    parser.add_argument(
        "--bins",
        nargs="*",
        help="Optional explicit bin labels to keep (default uses predefined bins + 200_plus).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    rank_map = load_rankings(Path(args.rankings_csv))
    default_matches = [
        Path("raw_data/charting-m-matches.csv"),
        Path("raw_data/charting-w-matches.csv"),
        Path("data/charting-m-matches.csv"),
        Path("data/charting-w-matches.csv"),
    ]
    match_files = [Path(p) for p in args.matches_file] if args.matches_file else default_matches
    match_map = load_match_map(match_files)
    df = attach_bins(df, rank_map, match_map)

    bins = args.bins if args.bins else [label for _, _, label in BIN_EDGES] + [DEFAULT_BIN]
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Global everyone vs everyone
    save_matrices(df, out_base / "global")

    # Bin vs bin
    for hb in bins:
        for ob in bins:
            subset = df[(df["player_bin"] == hb) & (df["opponent_bin"] == ob)]
            if subset.empty:
                continue
            save_matrices(subset, out_base / "bin_vs_bin" / f"{hb}_vs_{ob}")

    # Bin vs all opponents
    for hb in bins:
        subset = df[df["player_bin"] == hb]
        if subset.empty:
            continue
        save_matrices(subset, out_base / "bin_vs_all" / hb)

    # Per player
    for player in sorted(df["player_name_norm"].dropna().unique()):
        subset = df[df["player_name_norm"] == player]
        if subset.empty:
            continue
        slug = player.replace(" ", "_")
        save_matrices(subset, out_base / "player" / f"player_{slug}")

    print(f"Wrote rank-bin matrices to {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
