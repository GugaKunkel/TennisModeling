from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping
import sys
import argparse
import numpy as np
import pandas as pd

# Allow running as a script or as part of the score_model package.
if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from score_model.game_absorption import game_win_probability
    from score_model.set_match import MatchDPResult, SetOutcome, compute_match_dp, compute_set_outcome
else:
    from .game_absorption import game_win_probability
    from .set_match import MatchDPResult, SetOutcome, compute_match_dp, compute_set_outcome


@dataclass(frozen=True)
class MatchPrediction:
    p_match_win: float
    most_likely_scoreline: List[str]
    scoreline_probability: float
    set_outcome_A_first: SetOutcome
    set_outcome_B_first: SetOutcome

@dataclass(frozen=True)
class TransitionMatrix:
    states: list[str]
    index_for_state: Mapping[str, int]
    matrix: np.ndarray
    absorbing: set[int]
    transient: set[int]

    def restrict_to(self, opp_bin: str) -> tuple["TransitionMatrix", str]:
        """Return a sub-matrix containing only states for a given opponent bin (server perspective)."""
        opp_suffix = f"opp_bin={opp_bin}".lower()
        keep_indices = [i for i, label in enumerate(self.states) if opp_suffix in label.lower()]
        if not keep_indices:
            raise ValueError(f"No states found for opp_bin={opp_bin}.")

        # Preserve the exact bin text from the label to construct start/absorbing labels consistently.
        sample_label = self.states[keep_indices[0]]
        chosen_bin = sample_label.split("opp_bin=")[-1]

        sub_states = [self.states[i] for i in keep_indices]
        sub_index_map = {s: i for i, s in enumerate(sub_states)}
        sub_matrix = self.matrix[np.ix_(keep_indices, keep_indices)]

        sub_absorbing = {sub_index_map[s] for s in sub_states if _is_absorbing_state(s)}
        sub_transient = set(range(len(sub_states))) - sub_absorbing

        return (
            TransitionMatrix(
                states=sub_states,
                index_for_state=sub_index_map,
                matrix=sub_matrix,
                absorbing=sub_absorbing,
                transient=sub_transient,
            ),
            chosen_bin,
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
        a_serving = game_win_probability(player_a_matrix)
        b_serving = game_win_probability(player_b_matrix)
    else:
        a_serving = game_win_probability(player_a_matrix, opp_bin=player_b_bin)
        b_serving = game_win_probability(player_b_matrix, opp_bin=player_a_bin)

    # Build set outcomes for both serving orders.
    set_outcome_A_first = compute_set_outcome(
        a_hold=a_serving.p_server_wins,
        b_hold=b_serving.p_server_wins,
        server_first="A",
    )
    set_outcome_B_first = compute_set_outcome(
        a_hold=a_serving.p_server_wins,
        b_hold=b_serving.p_server_wins,
        server_first="B",
    )

    match_result: MatchDPResult = compute_match_dp(
        set_A_first=set_outcome_A_first,
        set_B_first=set_outcome_B_first,
        best_of=best_of,
        initial_server=server_first,
    )

    ml_scoreline = [f"{a}-{b}" for a, b in match_result.mle_scoreline]
    return MatchPrediction(
        p_match_win=match_result.p_win,
        most_likely_scoreline=ml_scoreline,
        scoreline_probability=match_result.mle_probability,
        set_outcome_A_first=set_outcome_A_first,
        set_outcome_B_first=set_outcome_B_first,
    )

def normaize_filename(name: str) -> str:
    return name.strip().replace(" ", "_")


def _debug_default_args() -> List[str]:
    """Defaults so VS Code debugging works without manual args."""
    repo_root = Path(__file__).resolve().parent.parent
    base_path = repo_root / "transition_matrices" / "score_rank"
    return [
        "--player-a",
        "Jannik Sinner",
        "--player-a-bin",
        "Top10",
        "--player-b",
        "Richard Gasquet",
        "--player-b-bin",
        "101_200",
        "--server-first",
        "A",
        "--best-of",
        "5",
        "--base-path",
        str(base_path),
    ]

def cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Match predictor using score transition matrices.")
    parser.add_argument("--player-a", required=True, help="Name of Player A.")
    parser.add_argument("--player-a-bin", help="Rank bin of Player A.")
    parser.add_argument("--player-b", required=True, help="Name of Player B.")
    parser.add_argument("--player-b-bin", help="Rank bin of Player B.")
    parser.add_argument("--server-first", choices=["A", "B"], default="A", help="Who serves first (A or B).")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of match length (must be odd).")
    parser.add_argument("--base-path", required=True, type=Path, help="Path to directory containing transition matrices.")

    argv_list = argv if argv is not None else sys.argv[1:]
    if not argv_list:
        argv_list = _debug_default_args()
        print("No args provided; using debug defaults for VS Code debugging:")
        print("  ", " ".join(argv_list))

    args = parser.parse_args(argv_list)

    pa_path_name = normaize_filename(args.player_a)
    pb_path_name = normaize_filename(args.player_b)
    tm_a = load_transition_matrix(args.base_path / f"player/player_{pa_path_name}/score_matrix.csv")
    tm_b = load_transition_matrix(args.base_path / f"player/player_{pb_path_name}/score_matrix.csv")
    pred = predict_match(player_a_matrix=tm_a,
                player_a_bin=args.player_a_bin,
                player_b_matrix=tm_b,
                player_b_bin=args.player_b_bin,
                server_first=args.server_first,
                best_of=args.best_of,
            )

    print(f"Matchup: {args.player_a} (A) vs {args.player_b} (B)")
    print(f"player_a_bin={args.player_a_bin}, player_b_bin={args.player_b_bin}, server_first={args.server_first}, best_of={args.best_of}")
    print(f"P(A wins match): {pred.p_match_win:.3f}")
    print(f"Most likely scoreline: {pred.most_likely_scoreline} (p={pred.scoreline_probability:.3f})")


if __name__ == "__main__":
    cli()
