from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple
from .tiebreak import point_prob_from_hold, tiebreak_win_probability


@dataclass(frozen=True)
class GameWinProbs:
    p_hold: float  # Player A holding serve
    p_break: float  # Player A breaking opponent serve

@dataclass(frozen=True)
class SetOutcome:
    score_probs: Dict[Tuple[int, int], float]
    p_win: float

@dataclass(frozen=True)
class MatchOutcome:
    p_win: float
    scoreline_probs: Dict[Tuple[int, int], float]

@dataclass(frozen=True)
class MatchDPResult:
    p_win: float
    mle_scoreline: List[Tuple[int, int]]
    mle_probability: float


def _toggle(server: str) -> str:
    return "B" if server == "A" else "A"


def _next_set_server(current_server: str, a_games: int, b_games: int) -> str:
    total_games = a_games + b_games
    if total_games % 2 == 0:
        return current_server
    return _toggle(current_server)

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


def compute_match_dp(
    set_A_first: SetOutcome,
    set_B_first: SetOutcome,
    best_of: int,
    initial_server: str = "A",
) -> MatchDPResult:
    """
    Dynamic-programming match predictor using set-level distributions.
    State: (sets_A, sets_B, server_first) where server_first is who serves first in the next set.
    """
    if best_of % 2 != 1:
        raise ValueError("best_of must be odd (1, 3, 5, 7, ...)")
    if initial_server not in {"A", "B"}:
        raise ValueError("initial_server must be 'A' or 'B'")

    sets_to_win = best_of // 2 + 1

    @lru_cache(maxsize=None)
    def dp_win_prob(sets_a: int, sets_b: int, server_first: str) -> float:
        if sets_a >= sets_to_win:
            return 1.0
        if sets_b >= sets_to_win:
            return 0.0

        set_outcome = set_A_first if server_first == "A" else set_B_first
        total = 0.0
        for (a_games, b_games), p_set in set_outcome.score_probs.items():
            if p_set == 0:
                continue
            next_server = _next_set_server(server_first, a_games, b_games)
            total += p_set * dp_win_prob(
                sets_a + (1 if a_games > b_games else 0),
                sets_b + (1 if b_games > a_games else 0),
                next_server,
            )
        return total

    @lru_cache(maxsize=None)
    def dp_mle(sets_a: int, sets_b: int, server_first: str) -> Tuple[float, List[Tuple[int, int]]]:
        if sets_a >= sets_to_win:
            return 1.0, []
        if sets_b >= sets_to_win:
            return 1.0, []

        set_outcome = set_A_first if server_first == "A" else set_B_first
        best_prob: float = -1.0
        best_seq: List[Tuple[int, int]] = []

        for (a_games, b_games), p_set in set_outcome.score_probs.items():
            if p_set == 0:
                continue
            next_server = _next_set_server(server_first, a_games, b_games)
            future_prob, future_seq = dp_mle(
                sets_a + (1 if a_games > b_games else 0),
                sets_b + (1 if b_games > a_games else 0),
                next_server,
            )
            path_prob = p_set * future_prob
            if path_prob > best_prob:
                best_prob = path_prob
                best_seq = [(a_games, b_games), *future_seq]

        if best_prob < 0:
            return 0.0, []
        return best_prob, best_seq

    p_win = dp_win_prob(0, 0, initial_server)
    mle_prob, mle_seq = dp_mle(0, 0, initial_server)
    return MatchDPResult(p_win=p_win, mle_scoreline=mle_seq, mle_probability=mle_prob)


__all__ = ["GameWinProbs", "SetOutcome", "MatchOutcome", "MatchDPResult", "compute_set_outcome", "compute_match_dp"]
