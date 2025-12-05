import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable
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


def normalize_name(name: str) -> str:
    return str(name).replace("\xa0", " ").strip()

def coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y"}

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
        return ValueError("No player or name column found in rankings CSV.")
    rank_cols = [c for c in ("atp_rank", "wta_rank", "elo_rank", "helo_rank", "celo_rank") if c in df.columns]
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        ranks = [row[c] for c in rank_cols if c in row and pd.notna(row[c])]
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
            m_id = str(row.get("match_id", "")).strip()
            if not m_id or m_id in mapping:
                continue
            p1 = normalize_name(row.get("Player 1", ""))
            p2 = normalize_name(row.get("Player 2", ""))
            mapping[m_id] = (p1, p2)
    return mapping

def attach_bins(df: pd.DataFrame, rank_map: Dict[str, float], match_map: Dict[str, tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    df["player_name_norm"] = df["player_name"].apply(normalize_name)
    opps = []
    for _, row in df.iterrows():
        m_id = str(row.get("match_id", ""))
        players = match_map.get(m_id, ("", ""))
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

def build_score_transitions(df: pd.DataFrame, use_bins: bool = False) -> pd.DataFrame:
    base_cols = ["point_score", "point_score_after"]
    cols = base_cols + (["opponent_bin"] if use_bins else [])
    subset = df[
        (df["server_flag"] == "serve_side")
        & df["is_terminal"].apply(coerce_bool)
        & df["point_score"].notna()
        & df["point_score_after"].notna()
    ][cols]
    if subset.empty:
        return subset
    if use_bins:
        subset["state"] = subset.apply(
            lambda r: f"{r['point_score']}|opp_bin={r['opponent_bin']}", axis=1
        )
        subset["new_state"] = subset.apply(
            lambda r: f"{r['point_score_after']}|opp_bin={r['opponent_bin']}", axis=1
        )
    else:
        subset["state"] = subset["point_score"]
        subset["new_state"] = subset["point_score_after"]
    counts = subset.groupby(["state", "new_state"]).size().reset_index(name="count")
    return counts

def build_score_matrix(transitions: pd.DataFrame, all_states: Iterable[str] | None = None) -> pd.DataFrame:
    """Build a square probability matrix. If all_states is None, use union of observed states."""
    if all_states is None:
        states = sorted(set(transitions["state"].unique()) | set(transitions["new_state"].unique()))
    else:
        states = sorted(set(all_states))
    if not states:
        return pd.DataFrame()
    if transitions.empty:
        zero_df = pd.DataFrame(0.0, index=states, columns=states)
        zero_df.index.name = ""
        return zero_df

    counts = transitions.pivot_table(
        index="state", columns="new_state", values="count", aggfunc="sum", fill_value=0
    ).reindex(index=states, columns=states, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index.name = ""
    return probs

def write_matrix(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.insert(0, "", out.index)
    out.to_csv(path, index=False)

def save_matrices(df: pd.DataFrame, out_dir: Path, use_bins: bool, all_states: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    transitions = build_score_transitions(df, use_bins=use_bins)
    transitions.to_csv(out_dir / "score_transitions.csv", index=False)
    matrix = build_score_matrix(transitions, all_states=all_states)
    write_matrix(matrix, out_dir / "score_matrix.csv")

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute score transition matrices (serving only).")
    parser.add_argument("--input", required=True, help="Path to state_transitions CSV.")
    parser.add_argument("--rankings_csv", help="Path to rankings CSV. If provided, will attach rank bins.")
    parser.add_argument("--rank_col", default="atp_rank", help="Column name in rankings CSV to use for rank.")
    parser.add_argument("--output_dir", default="rank_bin_score_matrices", help="Base output directory.")
    parser.add_argument("--bins", nargs="*", help="Optional explicit bin labels to use. Defaults to predefined bins.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    use_bins = bool(args.rankings_csv)
    all_states: Iterable[str] | None = None
    if use_bins:
        rank_map = load_rankings(Path(args.rankings_csv))
        match_files = [
            Path("raw_data/charting-m-matches.csv"),
            Path("raw_data/charting-w-matches.csv"),
            Path("data/charting-m-matches.csv"),
            Path("data/charting-w-matches.csv"),
        ]
        match_map = load_match_map(match_files)
        df = attach_bins(df, rank_map, match_map)
        global_transitions = build_score_transitions(df, use_bins=True)
        all_states = sorted(set(global_transitions["state"].unique()) | set(global_transitions["new_state"].unique()))
    else:
        global_transitions = build_score_transitions(df, use_bins=False)
        all_states = sorted(set(global_transitions["state"].unique()) | set(global_transitions["new_state"].unique()))

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if use_bins:
        bins = args.bins if args.bins else [label for _, _, label in BIN_EDGES] + [DEFAULT_BIN]

        # Global
        save_matrices(df, out_base / "global", use_bins=True, all_states=all_states)

        # Bin vs bin
        for hb in bins:
            for ob in bins:
                sub = df[(df["player_bin"] == hb) & (df["opponent_bin"] == ob)]
                save_matrices(sub, out_base / "bin_vs_bin" / f"{hb}_vs_{ob}", use_bins=True, all_states=all_states)

        # Bin vs all
        for hb in bins:
            sub = df[df["player_bin"] == hb]
            save_matrices(sub, out_base / "bin_vs_all" / hb, use_bins=True, all_states=all_states)

        # Per player
        for player in sorted(df["player_name_norm"].dropna().unique()):
            sub = df[df["player_name_norm"] == player]
            save_matrices(sub, out_base / "player" / f"player_{player.replace(' ', '_')}", use_bins=True, all_states=all_states)
    else:
        # No bins: just global and per-player matrices with plain score states
        save_matrices(df, out_base / "global", use_bins=False, all_states=all_states)
        for player in sorted(df["player_name"].dropna().unique()):
            sub = df[df["player_name"] == player]
            save_matrices(sub, out_base / "player" / f"player_{player.replace(' ', '_')}", use_bins=False, all_states=all_states)

    print(f"Wrote score matrices to {out_base}")
    return 0
    

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
