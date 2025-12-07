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
    used_bin: str | None

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


def _available_bins(matrix: TransitionMatrix) -> set[str]:
    """Collect available opponent bins from state labels (may be empty for unbinned)."""
    bins: set[str] = set()
    for label in matrix.states:
        if "opp_bin=" in label:
            bins.add(label.split("opp_bin=")[-1])
    return bins


def game_win_probability(transition_matrix: TransitionMatrix, opp_bin: str | None = None) -> GameProbabilities:
    """
    Probability the (current) server wins a game vs the given opponent bin.
    If opp_bin is omitted and only one bin exists in the matrix, that bin is used.
    For unbinned matrices, opp_bin can be omitted.
    """
    bins = _available_bins(transition_matrix)
    if bins:
        if opp_bin is None:
            if len(bins) != 1:
                raise ValueError(f"opp_bin must be provided; available bins: {sorted(bins)}")
            opp_bin = next(iter(bins))
        sub, used_bin = transition_matrix.restrict_to(opp_bin=opp_bin)
        start_label = f"0-0|opp_bin={used_bin}"
        server_label = f"game_server|opp_bin={used_bin}"
        returner_label = f"game_returner|opp_bin={used_bin}"
    else:
        # Unbinned matrix: use as-is.
        sub, used_bin = transition_matrix.restrict_to(opp_bin=None)
        start_label = "0-0"
        server_label = "game_server"
        returner_label = "game_returner"

    absorb_probs = absorption_from_state(sub, start_label)
    p_server_wins = absorb_probs.get(server_label, 0.0)
    p_server_loses = absorb_probs.get(returner_label, 0.0)
    return GameProbabilities(opp_bin=opp_bin if bins else "unbinned", p_server_wins=p_server_wins, p_server_loses=p_server_loses, used_bin=used_bin)


__all__ = ["GameProbabilities", "absorption_from_state", "game_win_probability"]
