"""
Batch predict match win probabilities for a matches CSV using the score_model.

Example:
    python3 experiments/score_model_match_outcome/match_predict_all.py \
        --matches data/atp_matches_2025.csv \
        --output predictions.csv
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from score_model.predictor import (
    TransitionMatrix,
    load_transition_matrix,
    normaize_filename,
    predict_match,
)

@dataclass(frozen=True)
class PlayerMatrix:
    matrix: TransitionMatrix
    binned: bool

def rank_to_bin(rank: float | int | None) -> Optional[str]:
    """Map ATP rank to the bin labels used in score_rank matrices."""
    if pd.isna(rank):
        return "200_plus"  # fallback when rank is missing
    try:
        r = float(rank)
    except Exception:
        return "200_plus"
    if r <= 10:
        return "Top10"
    if r <= 25:
        return "11_25"
    if r <= 50:
        return "26_50"
    if r <= 100:
        return "51_100"
    if r <= 200:
        return "101_200"
    return "200_plus"


def load_player_matrix(
    player_name: str,
    base_rank: Path,
    base_unbinned: Path,
    prefer_binned: bool = True,
) -> PlayerMatrix:
    """Load a player's transition matrix, preferring binned score_rank then falling back to unbinned."""
    norm = (
        normaize_filename(player_name)
        .replace("'", "")
        .replace(".", "")
        .replace("-", "_")
    )
    binned_path = base_rank / "player" / f"player_{norm}" / "score_matrix.csv"
    if prefer_binned and binned_path.exists():
        return PlayerMatrix(load_transition_matrix(binned_path), True)

    unbinned_path = base_unbinned / "player" / f"player_{norm}" / "score_matrix.csv"
    if unbinned_path.exists():
        return PlayerMatrix(load_transition_matrix(unbinned_path), False)

    raise FileNotFoundError(f"No transition matrix found for '{player_name}' ")

def load_bin_vs_bin_matrix(server_bin: str, opp_bin: str, base_rank: Path) -> PlayerMatrix:
    """
    Load a rank-bin-vs-bin matrix (server perspective).
    """
    path = base_rank / "bin_vs_bin" / f"{server_bin}_vs_{opp_bin}" / "score_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Bin-vs-bin matrix not found at {path}")
    return PlayerMatrix(load_transition_matrix(path), True)


def predict_matches_for_file(
    matches_path: Path,
    base_rank: Path,
    base_unbinned: Path,
    prefer_binned: bool = True,
    use_bin_when_missing: bool = True,
) -> pd.DataFrame:
    """
    Produce predictions and include the ground-truth label (player_a is the listed winner).
    """
    df = pd.read_csv(matches_path)
    # Use the same time-based holdout as logreg: only matches after 20250610.
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df[df["tourney_date"] > 20250610]
    if df.empty:
        print("[warn] No matches with tourney_date > 20250610; nothing to predict.")
        return pd.DataFrame()
    results = []

    cache: dict[str, PlayerMatrix] = {}

    for _, row in df.iterrows():
        a_name = row.get("winner_name") or row.get("player_a_name")
        b_name = row.get("loser_name") or row.get("player_b_name")
        if not a_name or not b_name:
            continue

        a_bin = rank_to_bin(row.get("winner_rank"))
        b_bin = rank_to_bin(row.get("loser_rank"))

        if a_name not in cache:
            try:
                cache[a_name] = load_player_matrix(a_name, base_rank, base_unbinned, prefer_binned)
            except FileNotFoundError as exc:
                if use_bin_when_missing and prefer_binned and a_bin is not None and b_bin is not None:
                    try:
                        cache[a_name] = load_bin_vs_bin_matrix(a_bin, b_bin, base_rank)
                        print(f"[info] Using bin_vs_bin matrix for {a_name} ({a_bin} vs {b_bin})")
                    except FileNotFoundError:
                        print(f"[warn] {exc}")
                        continue
                else:
                    print(f"[warn] {exc}")
                    continue
        if b_name not in cache:
            try:
                cache[b_name] = load_player_matrix(b_name, base_rank, base_unbinned, prefer_binned)
            except FileNotFoundError as exc:
                if use_bin_when_missing and prefer_binned and a_bin is not None and b_bin is not None:
                    try:
                        cache[b_name] = load_bin_vs_bin_matrix(b_bin, a_bin, base_rank)
                        print(f"[info] Using bin_vs_bin matrix for {b_name} ({b_bin} vs {a_bin})")
                    except FileNotFoundError:
                        print(f"[warn] {exc}")
                        continue
                else:
                    print(f"[warn] {exc}")
                    continue

        a_matrix = cache[a_name]
        b_matrix = cache[b_name]

        # recompute bins if unbinned player matrix was used
        a_bin = rank_to_bin(row.get("winner_rank")) if a_matrix.binned else None
        b_bin = rank_to_bin(row.get("loser_rank")) if b_matrix.binned else None

        if a_matrix.binned and a_bin is None:
            print(f"[warn] Missing/unknown rank bin for {a_name}; skipping match.")
            continue
        if b_matrix.binned and b_bin is None:
            print(f"[warn] Missing/unknown rank bin for {b_name}; skipping match.")
            continue

        best_of = int(row.get("best_of", 3))

        try:
            pred = predict_match(
                player_a_matrix=a_matrix.matrix,
                player_a_bin=a_bin,
                player_b_matrix=b_matrix.matrix,
                player_b_bin=b_bin,
                server_first="A",
                best_of=best_of,
            )
        except Exception as exc:
            print(f"[warn] failed to predict {a_name} vs {b_name}: {exc}")
            continue

        results.append(
            {
                "player_a": a_name,
                "player_b": b_name,
                "player_a_bin": a_bin,
                "player_b_bin": b_bin,
                "best_of": best_of,
                "p_player_a_win": pred.p_match_win,
                "p_player_b_win": 1 - pred.p_match_win,
                "mle_scoreline": "|".join(pred.most_likely_scoreline),
                "mle_scoreline_prob": pred.scoreline_probability,
                "label": 1,  # player_a is the listed winner
            }
        )

    return pd.DataFrame(results)

def evaluate(preds: pd.DataFrame) -> dict[str, float]:
    if preds.empty:
        return {"accuracy": float("nan"), "log_loss": float("nan")}
    y_true = preds["label"].to_numpy()
    y_prob = preds["p_player_a_win"].to_numpy().clip(1e-9, 1 - 1e-9)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    ll = float(np.mean(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))))
    return {"accuracy": acc, "log_loss": ll}

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch match predictions using score_model.")
    parser.add_argument("--matches", type=Path, default=Path("data/atp_matches_2025.csv"), help="Path to matches CSV.")
    parser.add_argument(
        "--base-rank",
        type=Path,
        default=Path("transition_matrices/score_rank"),
        help="Base directory for binned score matrices (score_rank).",
    )
    parser.add_argument(
        "--base-unbinned",
        type=Path,
        default=Path("transition_matrices/score"),
        help="Base directory for unbinned score matrices (score).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write predictions CSV (if omitted, no file is written).",
    )
    parser.add_argument(
        "--use-bin-when-missing",
        action="store_true",
        dest="use_bin_when_missing",
        default=True,
        help="Use bin_vs_bin fallback when player matrix is missing (default: on).",
    )
    parser.add_argument(
        "--no-use-bin-when-missing",
        action="store_false",
        dest="use_bin_when_missing",
        help="Disable bin_vs_bin fallback when player matrix is missing.",
    )
    args = parser.parse_args()

    if not args.base_rank.exists() and not args.base_unbinned.exists():
        raise ValueError("At least one of --base-rank or --base-unbinned must exist.")
    if args.base_rank.exists() and args.base_unbinned.exists():
        raise ValueError("Only one of --base-rank or --base-unbinned should exist.")
    
    prefer_binned = args.base_rank.exists()
    preds = predict_matches_for_file(
        matches_path=args.matches,
        base_rank=args.base_rank,
        base_unbinned=args.base_unbinned,
        prefer_binned=prefer_binned,
        use_bin_when_missing=args.use_bin_when_missing,
    )
    metrics = evaluate(preds)

    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"Wrote predictions to {args.output}")
    print(
        f"accuracy={metrics['accuracy']:.3f}, "
        f"logloss={metrics['log_loss']:.3f}, "
        f"predictions={len(preds)}"
    )
    print(preds.head())


if __name__ == "__main__":
    main()
