from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple
from .tiebreak import point_prob_from_hold, tiebreak_win_probability


@dataclass(frozen=True)
class GameWinProbs:
    p_hold: float  # Player A holding serve
    p_break: float  # Player A breaking opponent serve

@dataclass(frozen=True)
class SetOutcome:
    score_probs: Dict[Tuple[int, int], float]
    p_win: float

def _toggle(server: str) -> str:
    return "B" if server == "A" else "A"

def compute_set_outcome(a_hold: float, b_hold: float, server_first: str = "A") -> SetOutcome:
    """Return full set score distribution and win probability for Player A."""
    if server_first not in {"A", "B"}:
        raise ValueError("server_first must be 'A' or 'B'")

    # Derive point-level estimates to compute tiebreak win probability
    p_point_serve_a = point_prob_from_hold(a_hold)
    p_point_serve_b = point_prob_from_hold(b_hold)
    p_tb = tiebreak_win_probability(
        p_a_point_on_serve=p_point_serve_a,
        p_b_point_on_serve=p_point_serve_b,
        server_first=server_first,
    )

    @lru_cache(maxsize=None)
    def dp(a_games: int, b_games: int, next_server: str) -> float:
        # terminal conditions (tiebreak at 6-6 handled separately)
        if a_games >= 6 and a_games - b_games >= 2:
            return 1.0
        if b_games >= 6 and b_games - a_games >= 2:
            return 0.0
        if a_games == 6 and b_games == 6:
            return p_tb

        p_win_game = a_hold if next_server == "A" else b_hold
        win = p_win_game * dp(a_games + 1, b_games, _toggle(next_server))
        lose = (1 - p_win_game) * dp(a_games, b_games + 1, _toggle(next_server))
        return win + lose

    p_set_win = dp(0, 0, server_first)

    # Enumerate score distribution via forward DP of states -> probability mass.
    score_probs: Dict[Tuple[int, int], float] = {}
    frontier: Dict[Tuple[int, int, str], float] = {(0, 0, server_first): 1.0}

    while frontier:
        new_frontier: Dict[Tuple[int, int, str], float] = {}
        for (a_games, b_games, next_server), prob in list(frontier.items()):
            # terminal
            if a_games >= 6 and a_games - b_games >= 2:
                score_probs[(a_games, b_games)] = score_probs.get((a_games, b_games), 0.0) + prob
                continue
            if b_games >= 6 and b_games - a_games >= 2:
                score_probs[(a_games, b_games)] = score_probs.get((a_games, b_games), 0.0) + prob
                continue
            if a_games == 6 and b_games == 6:
                score_probs[(7, 6)] = score_probs.get((7, 6), 0.0) + prob * p_tb
                score_probs[(6, 7)] = score_probs.get((6, 7), 0.0) + prob * (1 - p_tb)
                continue

            p_win_game = a_hold if next_server == "A" else b_hold
            p_lose_game = 1 - p_win_game
            next_server_swapped = _toggle(next_server)

            win_key = (a_games + 1, b_games, next_server_swapped)
            lose_key = (a_games, b_games + 1, next_server_swapped)
            new_frontier[win_key] = new_frontier.get(win_key, 0.0) + prob * p_win_game
            new_frontier[lose_key] = new_frontier.get(lose_key, 0.0) + prob * p_lose_game

        frontier = new_frontier

    return SetOutcome(score_probs=score_probs, p_win=p_set_win)


__all__ = ["GameWinProbs", "SetOutcome", "MatchOutcome", "compute_set_outcome"]
