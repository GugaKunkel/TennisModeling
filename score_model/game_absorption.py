from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .predictor import TransitionMatrix


@dataclass(frozen=True)
class GameProbabilities:
    opp_bin: str | None
    p_server_wins: float
    p_server_loses: float

def absorption_from_state(matrix: TransitionMatrix, start_label: str) -> dict[str, float]:
    """Return absorption probabilities for each absorbing state label from start_label."""
    if start_label not in matrix.index_for_state:
        raise ValueError(f"Start state '{start_label}' not found in transition matrix.")

    absorbing_idx = sorted(matrix.absorbing)
    transient_idx = sorted(matrix.transient)

    if not absorbing_idx:
        raise ValueError("No absorbing states found in transition matrix.")

    # Build Q and R submatrices
    Q = matrix.matrix[np.ix_(transient_idx, transient_idx)]
    R = matrix.matrix[np.ix_(transient_idx, absorbing_idx)]

    # Fundamental matrix N = (I - Q)^-1
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    B = N @ R  # absorption probabilities from each transient state

    start_transient_pos = transient_idx.index(matrix.index_for_state[start_label])
    start_row = B[start_transient_pos]

    return {matrix.states[absorbing_idx[i]]: float(start_row[i]) for i in range(len(absorbing_idx))}


def game_win_probability(transition_matrix: TransitionMatrix, opp_bin: str | None) -> GameProbabilities:
    """Probability the (current) server wins the game"""
    if opp_bin:
        sub = transition_matrix.restrict_to(opp_bin=opp_bin)
        start_label = f"0-0|opp_bin={opp_bin}"
        absorb_probs = absorption_from_state(sub, start_label)
        p_server_wins = absorb_probs.get(f"game_server|opp_bin={opp_bin}", 0.0)
        p_server_loses = absorb_probs.get(f"game_returner|opp_bin={opp_bin}", 0.0)
        return GameProbabilities(opp_bin=opp_bin, p_server_wins=p_server_wins, p_server_loses=p_server_loses)
    else:
        absorb_probs = absorption_from_state(transition_matrix, "0-0")
        p_server_wins = absorb_probs.get("game_server", 0.0)
        p_server_loses = absorb_probs.get("game_returner", 0.0)
        if not np.isclose(p_server_wins + p_server_loses, 1.0):
            raise ValueError("Absorption probabilities do not sum to 1.0")
        return GameProbabilities(opp_bin=None, p_server_wins=p_server_wins, p_server_loses=p_server_loses)


__all__ = ["GameProbabilities", "absorption_from_state", "game_win_probability"]
