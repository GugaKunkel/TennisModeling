"""
Baseline score predictor: always predicts the most common set score observed in historical data.
Uses the same holdout split as other scripts: tourney_date > 20250610 is test, <= is training.

Usage:
    python3 experiments/score_model_match_outcome/baseline_score.py --matches data/atp_matches_2025.csv
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_set_token(token: str) -> Optional[tuple[int, int]]:
    clean_token = token.replace("RET", "")
    # Strip tiebreak parentheses and contents, e.g., 7-6(6) -> 7-6
    clean_token = re.sub(r"\([^)]*\)", "", clean_token)
    parts = clean_token.split("-")
    if len(parts) != 2:
        return None
    try:
        def clean(x: str) -> int:
            digits = "".join(ch for ch in x if ch.isdigit())
            return int(digits) if digits else 0
        a, b = clean(parts[0]), clean(parts[1])
        return a, b
    except Exception:
        return None


def _parse_scoreline(score: str) -> list[tuple[int, int]]:
    if not isinstance(score, str):
        return []
    tokens = [t for t in score.replace("\xa0", " ").split() if t]
    sets: list[tuple[int, int]] = []
    for tok in tokens:
        if "RET" in tok.upper():
            continue
        parsed = _parse_set_token(tok)
        if not parsed:
            continue
        a, b = parsed
        if parsed and (a == 6 or b == 6 or a == 7 or b == 7):
            sets.append(parsed)
    return sets


def _score_distance_bin(distance: float | None) -> Optional[str]:
    if distance is None:
        return None
    if distance == 0:
        return "exact"
    if distance <= 1:
        return "<=1"
    if distance <= 2:
        return "<=2"
    if distance <= 3:
        return "<=3"
    if distance <= 4:
        return "<=4"
    return ">=5"


def _score_distance(actual_score: str, predicted_sets: list[str]) -> tuple[Optional[float], Optional[str], int]:
    """
    Compute per-set Manhattan/2 distance for sets where predicted winner matches actual winner.
    Returns (avg_distance, bin_label, sets_compared).
    """
    actual_sets = _parse_scoreline(actual_score)
    pred_sets: list[tuple[int, int]] = []
    for s in predicted_sets:
        try:
            a_str, b_str = s.split("-")
            pred_sets.append((int(a_str), int(b_str)))
        except Exception:
            continue

    if not actual_sets or not pred_sets:
        return None, None, 0
    compare_len = min(len(actual_sets), len(pred_sets))
    dists: list[float] = []
    for i in range(compare_len):
        a_a, a_b = actual_sets[i]
        p_a, p_b = pred_sets[i]
        # Only compare if predicted the correct set winner.
        if (a_a > a_b and p_a > p_b) or (a_b > a_a and p_b > p_a):
            dist = (abs(a_a - p_a) + abs(a_b - p_b)) / 2.0
            dists.append(dist)

    if not dists:
        return None, None, 0
    avg_dist = sum(dists)
    return avg_dist, _score_distance_bin(avg_dist), len(dists)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: predict the most common set score.")
    parser.add_argument("--matches", nargs="+", type=Path, required=True, help="CSV file(s) with ATP matches.")
    parser.add_argument("--cutoff", type=int, default=20250610, help="Cutoff tourney_date; > is test, <= is train.")
    parser.add_argument("--output", type=Path, help="Optional path to write baseline predictions.")
    args = parser.parse_args()

    frames = [pd.read_csv(p) for p in args.matches]
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")

    train = df[df["tourney_date"] <= args.cutoff]
    test = df[df["tourney_date"] > args.cutoff]

    # Count set score occurrences in train
    set_counts: dict[tuple[int, int], int] = {}
    for score in train["score"]:
        for s in _parse_scoreline(score):
            set_counts[s] = set_counts.get(s, 0) + 1
    if not set_counts:
        print("No set scores found in training data; cannot build baseline.")
        return
    mode_set, mode_count = max(set_counts.items(), key=lambda kv: kv[1])
    mode_str = f"{mode_set[0]}-{mode_set[1]}"

    preds = []
    for _, row in test.iterrows():
        best_of = int(row.get("best_of", 3))
        sets_to_win = best_of // 2 + 1
        scoreline_pred = [mode_str] * sets_to_win
        dist, dist_bin, sets_compared = _score_distance(row.get("score"), scoreline_pred)
        preds.append(
            {
                "player_a": row.get("winner_name"),
                "player_b": row.get("loser_name"),
                "best_of": best_of,
                "pred_scoreline": "|".join(scoreline_pred),
                "score_actual": row.get("score"),
                "score_distance": dist,
                "score_distance_bin": dist_bin,
                "score_sets_compared": sets_compared,
            }
        )

    pred_df = pd.DataFrame(preds)
    if args.output:
        pred_df.to_csv(args.output, index=False)
        print(f"Wrote baseline predictions to {args.output}")

    bins = pred_df["score_distance_bin"].value_counts(dropna=False).to_dict()
    skipped = bins.pop(None, 0) if None in bins else 0
    skipped += bins.pop(pd.NA, 0) if pd.NA in bins else 0
    valid = int(sum(bins.values()))

    print(f"Most common set score (train): {mode_str} (count={mode_count})")
    print(f"Holdout matches: {len(test)}")
    print(f"Score distance bins: {bins}")
    print(f"Score distances computed: {valid}; skipped: {skipped}")


if __name__ == "__main__":
    main()
