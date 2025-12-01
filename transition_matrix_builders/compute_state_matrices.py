from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def coerce_bool(val) -> bool:
    """Robustly convert common truthy/falsey representations to bool."""
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute state transition matrices.")
    parser.add_argument("--input", required=True, help="Path to state_transitions.csv")
    parser.add_argument("--output_dir", default=".", help="Directory to write matrices")
    parser.add_argument(
        "--include_players",
        action="append",
        help="Player name to include (can be specified multiple times)",
    )
    parser.add_argument(
        "--per_player",
        action="store_true",
        help="Generate matrices for each player individually (ignores --include_players).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    df = pd.read_csv(args.input)

    # Per-player mode: loop over unique player_name values.
    if args.per_player:
        players = sorted(df["player_name"].dropna().unique())
        if not players:
            print("No player_name values found.")
            return 1
        for p in players:
            sub = df[df["player_name"] == p]
            out_dir = Path(args.output_dir) / f"player_{p.replace(' ', '_')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            result_mat = compute_result_matrix(sub)
            cont_mat = compute_continuation_matrix(sub)
            write_matrix(result_mat, out_dir / "matrix_result.csv")
            write_matrix(cont_mat, out_dir / "matrix_continue.csv")
            print(f"Wrote matrices for {p} to {out_dir}")
        return 0

    # Single run (optionally filtered)
    if args.include_players:
        df = df[df["player_name"].isin(args.include_players)]
        if df.empty:
            print("No rows after filtering by player.")
            return 1

    result_mat = compute_result_matrix(df)
    cont_mat = compute_continuation_matrix(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(result_mat, out_dir / "matrix_result.csv")
    write_matrix(cont_mat, out_dir / "matrix_continue.csv")
    print(f"Wrote matrices to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
