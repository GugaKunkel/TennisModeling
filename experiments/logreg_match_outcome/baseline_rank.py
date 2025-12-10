"""
Simple baseline: predict the higher-ranked player (lower rank number) wins.
Uses the same holdout as other scripts: matches with tourney_date > 20250610.

Usage:
    python3 experiments/logreg_match_outcome/baseline_rank.py --matches data/atp_matches_2025.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def predict_higher_rank_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Predict winner by lower rank number on the holdout window."""
    df = df.copy()
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df[df["tourney_date"] > 20250610]

    # Predicted winner is whoever has the lower rank number; if missing ranks or tie, predict the listed winner.
    def predict_row(row) -> int:
        w_rank = pd.to_numeric(row.get("winner_rank"), errors="coerce")
        l_rank = pd.to_numeric(row.get("loser_rank"), errors="coerce")
        if pd.isna(w_rank) or pd.isna(l_rank):
            return 1  # fallback: assume listed winner
        if w_rank < l_rank:
            return 1
        if w_rank > l_rank:
            return 0
        return 1  # tie -> listed winner

    df["pred_label"] = df.apply(predict_row, axis=1)
    df["true_label"] = 1  # winner is labeled 1 in this canonical form
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: higher-ranked player wins.")
    parser.add_argument("--matches", nargs="+", type=Path, required=True, help="CSV file(s) with ATP matches.")
    args = parser.parse_args()

    frames = [pd.read_csv(p) for p in args.matches]
    df = pd.concat(frames, ignore_index=True)
    pred_df = predict_higher_rank_winner(df)

    if pred_df.empty:
        print("No matches with tourney_date > 20250610.")
        return

    accuracy = (pred_df["pred_label"] == pred_df["true_label"]).mean()
    print(f"Holdout matches: {len(pred_df)}")
    print(f"Baseline accuracy (higher rank wins): {accuracy:.3f}")


if __name__ == "__main__":
    main()
