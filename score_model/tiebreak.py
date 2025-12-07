from functools import lru_cache
import numpy as np


def hold_probability_from_point(p: float) -> float:
    """Game hold probability (advantage scoring) given point win prob p on serve."""
    q = 1 - p
    return p**4 * (1 + 4 * q + 10 * q**2 / (1 - 2 * p * q))


def point_prob_from_hold(target_hold: float, *, tol: float = 1e-6) -> float:
    """Invert hold probability -> point probability via binary search."""
    lo, hi = 0.01, 0.99
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        hold = hold_probability_from_point(mid)
        if abs(hold - target_hold) < tol:
            return mid
        if hold < target_hold:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tiebreak_win_probability(p_a_point_on_serve: float, p_b_point_on_serve: float, server_first: str = "A") -> float:
    """Probability player A wins a 7-point tiebreak (win by 2)."""
    if server_first not in {"A", "B"}:
        raise ValueError("server_first must be 'A' or 'B'")

    @lru_cache(maxsize=None)
    def rec(a_points: int, b_points: int) -> float:
        # terminal
        if a_points >= 7 and a_points - b_points >= 2:
            return 1.0
        if b_points >= 7 and b_points - a_points >= 2:
            return 0.0

        # Once both at or beyond 6, use the deuce-cycle closed form.
        if a_points >= 6 and b_points >= 6:
            diff = a_points - b_points
            offset = (a_points + b_points) % 4
            return extended_tiebreak_prob(
                diff,
                offset,
                p_a_point_on_serve,
                1-p_b_point_on_serve,
                server_first
            )

        point_idx = a_points + b_points
        srv = server_for_point(point_idx, server_first)
        p_win_point = p_a_point_on_serve if srv == "A" else 1 - p_b_point_on_serve

        win = p_win_point * rec(a_points + 1, b_points)
        lose = (1 - p_win_point) * rec(a_points, b_points + 1)
        return win + lose

    return rec(0, 0)

def extended_tiebreak_prob(
    diff: int,
    offset: int,
    p_point_on_serve: float,
    p_point_on_return: float,
    server_first: str,
) -> float:
    """Probability to win from deuce territory (scores >= 6-6) with periodic serve order."""
    if abs(diff) > 1:
        return 1.0 if diff >= 2 else 0.0

    serve_for_offset = [server_for_point(i, server_first) for i in range(4)]

    # Build transient states (-1,0,1) x offsets 0..3
    states = []
    for d in (-1, 0, 1):
        for off in range(4):
            states.append((d, off))
    state_idx = {s: i for i, s in enumerate(states)}

    n = len(states)
    absorbing_win = n
    absorbing_lose = n + 1
    P = np.zeros((n + 2, n + 2))

    for i, (d, off) in enumerate(states):
        srv = serve_for_offset[off]
        p_win_point = p_point_on_serve if srv == "A" else p_point_on_return
        p_lose_point = 1 - p_win_point

        next_off = (off + 1) % 4

        d_win = d + 1
        d_lose = d - 1

        if d_win >= 2:
            P[i, absorbing_win] += p_win_point
        else:
            P[i, state_idx[(d_win, next_off)]] += p_win_point

        if d_lose <= -2:
            P[i, absorbing_lose] += p_lose_point
        else:
            P[i, state_idx[(d_lose, next_off)]] += p_lose_point

    # Absorbing rows
    P[absorbing_win, absorbing_win] = 1.0
    P[absorbing_lose, absorbing_lose] = 1.0

    transient_indices = list(range(n))
    Q = P[np.ix_(transient_indices, transient_indices)]
    R = P[np.ix_(transient_indices, [absorbing_win, absorbing_lose])]
    N = np.linalg.inv(np.eye(len(Q)) - Q)
    B = N @ R

    start = state_idx[(diff, offset % 4)]
    return float(B[start, 0])


def server_for_point(point_idx: int, server_first: str) -> str:
    """Return which player serves a given tiebreak point (0-indexed)."""
    if point_idx == 0:
        return server_first
    block = (point_idx - 1) // 2
    return "B" if (block % 2 == 0) == (server_first == "A") else "A"


__all__ = ["point_prob_from_hold", "hold_probability_from_point", "tiebreak_win_probability"]
