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


def coerce_bool(val) -> bool:
    """Robustly convert common truthy/falsey representations to bool."""
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y"}

def normalize_name(name: str) -> str:
    """Standardize player names for consistent matching."""
    return str(name).replace("\xa0", " ").strip()

def classify_bin(rank: float | None) -> str:
    """Map numeric rank to a predefined bin label."""
    if rank is None or pd.isna(rank):
        return DEFAULT_BIN
    for lo, hi, label in BIN_EDGES:
        if lo <= rank <= hi:
            return label
    return DEFAULT_BIN

def load_rankings(path: Path) -> Dict[str, float]:
    """Read rankings CSV into name -> rank mapping."""
    df = pd.read_csv(path)
    name_col = "player" if "player" in df.columns else ("name" if "name" in df.columns else None)
    if not name_col:
        raise ValueError("No player or name column found in rankings CSV.")
    rank_cols = [c for c in ("atp_rank", "wta_rank", "elo_rank", "helo_rank", "celo_rank") if c in df.columns]
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        ranks = [row[c] for c in rank_cols if c in row and pd.notna(row[c])]
        if not ranks:
            continue
        mapping[normalize_name(row[name_col])] = float(ranks[0])
    return mapping

def load_match_map(files: Iterable[Path]) -> Dict[str, tuple[str, str]]:
    """Build match_id -> (P1, P2) map from MCP matches CSVs."""
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
    """Add player/opponent rank bins to the transitions frame."""
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


def compute_result_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """P(result_for_hitter | state) with result in {win, loss, continue}."""
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


def compute_continuation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """P(new_state | state, continue) for non-terminal transitions."""
    mask = df["is_terminal"].apply(coerce_bool) == False
    non_term = df[mask]

    states = sorted(non_term["state"].dropna().unique())
    next_states = sorted(non_term["new_state"].dropna().unique())

    if not states or not next_states:
        empty = pd.DataFrame(columns=[""] + next_states)
        return empty

    counts = non_term.groupby(["state", "new_state"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=states, columns=next_states, fill_value=0)

    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index.name = ""
    return probs


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    """Write wide matrix with blank header for row labels."""
    out = df.copy()
    out.insert(0, "", out.index)
    out.to_csv(path, index=False)


def save_matrices(df: pd.DataFrame, out_dir: Path) -> None:
    """Write result and continuation matrices to the output folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result_mat = compute_result_matrix(df)
    cont_mat = compute_continuation_matrix(df)
    write_matrix(result_mat, out_dir / "matrix_result.csv")
    write_matrix(cont_mat, out_dir / "matrix_continue.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute shot-level state transition matrices.")
    parser.add_argument("--input", required=True, help="Path to state_transitions.csv")
    parser.add_argument("--rankings_csv", help="Path to rankings CSV. If provided, will attach rank bins")
    parser.add_argument("--output_dir", default="state_matrices", help="Base directory for output matrices.")
    parser.add_argument("--bins", nargs="*", help="Optional explicit bin labels to use. Defaults to predefined bins.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input)
    use_bins = bool(args.rankings_csv)
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

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
        bins = args.bins if args.bins else [label for _, _, label in BIN_EDGES] + [DEFAULT_BIN]

        # Global
        save_matrices(df, out_base / "global")

        # Bin vs bin
        for hb in bins:
            for ob in bins:
                subset = df[(df["player_bin"] == hb) & (df["opponent_bin"] == ob)]
                if subset.empty:
                    continue
                save_matrices(subset, out_base / "bin_vs_bin" / f"{hb}_vs_{ob}")

        # Bin vs all
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
    else:
        # Global
        save_matrices(df, out_base / "global")
        # Per player
        for player in sorted(df["player_name"].dropna().unique()):
            subset = df[df["player_name"] == player]
            if subset.empty:
                continue
            slug = player.replace(" ", "_")
            save_matrices(subset, out_base / "player" / f"player_{slug}")

    print(f"Wrote state matrices to {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
