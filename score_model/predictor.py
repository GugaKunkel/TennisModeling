from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple
import argparse
import numpy as np
import pandas as pd

from .game_absorption import game_win_probability
from .set_match import SetOutcome, compute_set_outcome


@dataclass(frozen=True)
class MatchPrediction:
    most_likely_scoreline: List[str]
    scoreline_probability: float
    set_outcome: SetOutcome

@dataclass(frozen=True)
class TransitionMatrix:
    states: list[str]
    index_for_state: Mapping[str, int]
    matrix: np.ndarray
    absorbing: set[int]
    transient: set[int]

    def restrict_to(self, opp_bin: str) -> TransitionMatrix:
        """Return a sub-matrix containing only states for a given opponent bin."""
        opp_suffix = f"opp_bin={opp_bin}"
        keep_indices = [i for i, label in enumerate(self.states) if opp_suffix in label]
        if keep_indices is None:
            raise ValueError(f"No states found for opp_bin={opp_bin}")

        sub_states = [self.states[i] for i in keep_indices]
        sub_index_map = {s: i for i, s in enumerate(sub_states)}
        sub_matrix = self.matrix[np.ix_(keep_indices, keep_indices)]

        sub_absorbing = {sub_index_map[s] for s in sub_states if _is_absorbing_state(s)}
        sub_transient = set(range(len(sub_states))) - sub_absorbing

        return TransitionMatrix(
                states=sub_states,
                index_for_state=sub_index_map,
                matrix=sub_matrix,
                absorbing=sub_absorbing,
                transient=sub_transient,
            )

def _is_absorbing_state(label: str) -> bool:
    return label.startswith("game_server") or label.startswith("game_returner") or label.startswith("set_") or label.startswith("match_")

def load_transition_matrix(path: Path) -> TransitionMatrix:
    """Load a square CSV transition matrix into a TransitionMatrix."""
    df = pd.read_csv(path)
    row_states = df.iloc[:, 0].astype(str).tolist()
    col_states = [str(c) for c in df.columns[1:]]

    if len(row_states) != len(col_states):
        raise ValueError("Transition matrix must be square: row/column counts differ.")
    if set(col_states) != set(row_states):
        missing = set(row_states) - set(col_states)
        extra = set(col_states) - set(row_states)
        raise ValueError(f"Transition matrix states mismatch. Missing: {missing}, Extra: {extra}")

    matrix_df = df.iloc[:, 1:].copy()
    matrix_df.index = row_states
    matrix_df.columns = col_states
    if col_states != row_states:
        print("Reordering transition matrix columns to match rows...")
        matrix_df = matrix_df.reindex(columns=row_states)

    states = row_states
    matrix_values = matrix_df.to_numpy(dtype=float)

    index_for_state = {state: i for i, state in enumerate(states)}
    absorbing = {i for i, s in enumerate(states) if _is_absorbing_state(s)}
    transient = set(range(len(states))) - absorbing

    return TransitionMatrix(
        states=states,
        index_for_state=index_for_state,
        matrix=matrix_values,
        absorbing=absorbing,
        transient=transient,
    )

def predict_match(
    player_a_matrix: TransitionMatrix,
    player_a_bin: str | None,
    player_b_matrix: TransitionMatrix,
    player_b_bin: str | None,
    server_first: str = "A",
    best_of: int = 5,
) -> MatchPrediction:
    if player_a_bin is None or player_b_bin is None:
        player_a_gp = game_win_probability(player_a_matrix)
        player_b_gp = game_win_probability(player_b_matrix)
    else:
        player_a_gp = game_win_probability(player_a_matrix, opp_bin=player_b_bin)
        player_b_gp = game_win_probability(player_b_matrix, opp_bin=player_a_bin)

    set_outcome = compute_set_outcome(
                    a_hold=player_a_gp.p_server_wins,
                    b_hold=player_b_gp.p_server_wins,
                    server_first=server_first
                )
    
    ml_scoreline, ml_prob = most_likely_scoreline(set_outcome)
    return MatchPrediction(
        most_likely_scoreline=[f"{a}-{b}" for a, b in ml_scoreline],
        scoreline_probability=ml_prob,
        set_outcome=set_outcome,
    )


def most_likely_scoreline(set_outcome: SetOutcome) -> Tuple[List[Tuple[int, int]], float]:
    """Return the most likely match scoreline (sequence of set scores)."""
    a_scores = [(score, p) for score, p in set_outcome.score_probs.items() if score[0] > score[1]]
    b_scores = [(score, p) for score, p in set_outcome.score_probs.items() if score[1] > score[0]]

    def best_pair(scores):
        best = ([], 0.0)
        for s1, p1 in scores:
            for s2, p2 in scores:
                prob = p1 * p2
                if prob > best[1]:
                    best = ([s1, s2], prob)
        return best

    def best_two_one(win_scores, lose_scores, winner: str):
        best = ([], 0.0)
        for s_win1, p_win1 in win_scores:
            for s_win2, p_win2 in win_scores:
                for s_lose, p_lose in lose_scores:
                    # valid orders: winner loses one set before clinching
                    seqs = (
                        [s_win1, s_lose, s_win2],  # W, L, W
                        [s_lose, s_win1, s_win2],  # L, W, W
                    )
                    for seq in seqs:
                        prob = p_win1 * p_win2 * p_lose
                        if prob > best[1]:
                            best = (seq, prob)
        return best

    best_20_seq, best_20_prob = best_pair(a_scores)
    best_02_seq, best_02_prob = best_pair(b_scores)
    best_21_seq, best_21_prob = best_two_one(a_scores, b_scores, "A")
    best_12_seq, best_12_prob = best_two_one(b_scores, a_scores, "B")

    candidates = [
        (best_20_seq, best_20_prob),
        (best_02_seq, best_02_prob),
        (best_21_seq, best_21_prob),
        (best_12_seq, best_12_prob),
    ]
    ml_seq, ml_prob = max(candidates, key=lambda x: x[1])
    return ml_seq, ml_prob


def cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Match predictor using score transition matrices.")
    parser.add_argument("--player-a", required=True, help="Name of Player A.")
    parser.add_argument("--player-a-bin", required=True, help="Rank bin of Player A.")
    parser.add_argument("--player-b", required=True, help="Name of Player B.")
    parser.add_argument("--player-b-bin", required=True, help="Rank bin of Player B.")
    parser.add_argument("--server-first", choices=["A", "B"], default="A", help="Who serves first (A or B).")
    parser.add_argument("--base-path", required=True, type=Path, help="Path to directory containing transition matrices.")
    args = parser.parse_args(argv)

    tm_a = load_transition_matrix(args.base_path / f"player/player_{args.player_a_bin}/score_matrix.csv")
    tm_b = load_transition_matrix(args.base_path / f"player/player_{args.player_b_bin}/score_matrix.csv")
    pred = predict_match(player_a_matrix=tm_a,
                player_a_bin=args.player_a_bin,
                player_b_matrix=tm_b,
                player_b_bin=args.player_b_bin,
                server_first=args.server_first
            )

    print(f"Matchup: {args.player_a} (A) vs {args.player_b} (B)")
    print(f"player_a_bin={args.player_a_bin}, player_b_bin={args.player_b_bin}, server_first={args.server_first}")
    print(f"P(A wins match): {pred.p_match_win:.3f}")
    print(f"Most likely scoreline: {pred.most_likely_scoreline} (p={pred.scoreline_probability:.3f})")


if __name__ == "__main__":
    cli()
