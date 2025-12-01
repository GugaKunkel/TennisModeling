#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


POINT_MAP = {0: "0", 1: "15", 2: "30", 3: "40"}


def load_score_matrix(path: Path) -> pd.DataFrame:
    """Load a serve score matrix (rows: score before point, cols: score after point)."""
    df = pd.read_csv(path, index_col=0)
    df.fillna(0.0, inplace=True)
    return df


def coerce_probs(row: pd.Series) -> Tuple[np.ndarray, list[str]]:
    labels = list(row.index)
    probs = row.to_numpy(dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.zeros(len(labels)), labels
    return probs / total, labels


def score_label(server_pts: int, return_pts: int, tiebreak: bool = False) -> str:
    if tiebreak:
        return f"TB_{server_pts}-{return_pts}"
    if server_pts >= 3 and return_pts >= 3:
        if server_pts == return_pts:
            return "40-40"
        if server_pts == return_pts + 1:
            return "Ad-In"
        if return_pts == server_pts + 1:
            return "Ad-Out"
    return f"{POINT_MAP.get(server_pts, '40')}-{POINT_MAP.get(return_pts, '40')}"


def sample_next_score(
    current_score: str, matrix: pd.DataFrame, rng: np.random.Generator
) -> str | None:
    if current_score not in matrix.index:
        return None
    probs, labels = coerce_probs(matrix.loc[current_score])
    if probs.sum() <= 0:
        return None
    return rng.choice(labels, p=probs)


def simulate_game_points(
    server_matrix: pd.DataFrame, rng: np.random.Generator
) -> str:
    """Simulate one service game; return 'server' or 'returner' winner."""
    s_pts = r_pts = 0
    while True:
        label = score_label(s_pts, r_pts, tiebreak=False)
        next_score = sample_next_score(label, server_matrix, rng)
        if next_score is None:
            # fallback: 50/50 point
            winner = rng.choice(["server", "returner"])
        elif next_score == "game_server":
            return "server"
        elif next_score == "game_returner":
            return "returner"
        else:
            # derive winner by comparing progression
            # simple heuristic: if label unchanged, flip a coin
            winner = None
            if next_score.startswith("Ad-In"):
                winner = "server"
            elif next_score.startswith("Ad-Out"):
                winner = "returner"
            elif next_score == "40-40" and (label == "Ad-In" or label == "Ad-Out"):
                winner = "returner" if label == "Ad-In" else "server"
            if winner is None:
                winner = rng.choice(["server", "returner"])
        if winner == "server":
            s_pts += 1
        else:
            r_pts += 1
        if s_pts >= 4 or r_pts >= 4:
            if abs(s_pts - r_pts) >= 2:
                return "server" if s_pts > r_pts else "returner"


def simulate_tiebreak_points(
    server_first: str,
    returner_first: str,
    matrices: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> str:
    s_pts = r_pts = 0
    points_played = 0
    server = server_first
    returner = returner_first
    while True:
        # tiebreak serve rotation: first point by server_first, then every two points alternate
        if points_played == 0:
            point_server = server_first
            point_returner = returner_first
        else:
            block = (points_played - 1) // 2
            if block % 2 == 0:
                point_server, point_returner = returner_first, server_first
            else:
                point_server, point_returner = server_first, returner_first
        label = score_label(s_pts, r_pts, tiebreak=True)
        server_matrix = matrices.get(point_server)
        next_score = sample_next_score(label, server_matrix, rng) if server_matrix is not None else None
        winner = None
        if next_score is None:
            winner = rng.choice(["server", "returner"])
        elif next_score.startswith("TB_"):
            # No terminal markers; infer by comparing increments
            tb_vals = next_score.replace("TB_", "").split("-")
            try:
                n_s, n_r = int(tb_vals[0]), int(tb_vals[1])
                if n_s == s_pts + 1 and n_r == r_pts:
                    winner = "server"
                elif n_r == r_pts + 1 and n_s == s_pts:
                    winner = "returner"
            except Exception:
                winner = None
        if winner is None:
            winner = rng.choice(["server", "returner"])
        if winner == "server":
            s_pts += 1
        else:
            r_pts += 1
        points_played += 1
        if (s_pts >= 7 or r_pts >= 7) and abs(s_pts - r_pts) >= 2:
            return server if s_pts > r_pts else returner


def simulate_set(
    player_a: str,
    player_b: str,
    matrices: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> Tuple[str, Tuple[int, int]]:
    games_a = games_b = 0
    server = player_a
    returner = player_b
    while True:
        server_matrix = matrices.get(server)
        winner = simulate_game_points(server_matrix, rng) if server_matrix is not None else rng.choice(["server", "returner"])
        if winner == "server":
            if server == player_a:
                games_a += 1
            else:
                games_b += 1
        else:
            if server == player_a:
                games_b += 1
            else:
                games_a += 1
        if (games_a >= 6 or games_b >= 6) and abs(games_a - games_b) >= 2:
            return (player_a if games_a > games_b else player_b), (games_a, games_b)
        if games_a == 6 and games_b == 6:
            tb_winner = simulate_tiebreak_points(server, returner, matrices, rng)
            if tb_winner == player_a:
                games_a += 1
            else:
                games_b += 1
            return (tb_winner), (games_a, games_b)
        server, returner = returner, server  # alternate servers


def simulate_match(
    player_a: str,
    player_b: str,
    matrices: Dict[str, pd.DataFrame],
    best_of: int = 3,
    seed: int | None = None,
) -> Tuple[str, list[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    sets_to_win = best_of // 2 + 1
    sets_won = {player_a: 0, player_b: 0}
    set_scores: list[Tuple[int, int]] = []
    server = player_a
    returner = player_b
    while max(sets_won.values()) < sets_to_win:
        set_winner, score = simulate_set(server, returner, matrices, rng)
        set_scores.append(score if set_winner == server else score[::-1])
        sets_won[set_winner] += 1
        server, returner = returner, server  # alternate who serves first next set
    winner = player_a if sets_won[player_a] > sets_won[player_b] else player_b
    return winner, set_scores


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate matches using serve-only score matrices.")
    parser.add_argument("--player_a_dir", required=True, help="Directory with player A serve_score_matrix.csv")
    parser.add_argument("--player_b_dir", required=True, help="Directory with player B serve_score_matrix.csv")
    parser.add_argument("--matches", type=int, default=1, help="Number of matches to simulate.")
    parser.add_argument("--best_of", type=int, default=3, help="Best-of sets (3 or 5).")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    player_a = Path(args.player_a_dir).name.replace("player_", "").replace("_", " ")
    player_b = Path(args.player_b_dir).name.replace("player_", "").replace("_", " ")

    matrices = {
        player_a: load_score_matrix(Path(args.player_a_dir) / "serve_score_matrix.csv"),
        player_b: load_score_matrix(Path(args.player_b_dir) / "serve_score_matrix.csv"),
    }

    wins = {player_a: 0, player_b: 0}
    base_seed = args.seed
    for i in range(args.matches):
        seed_i = None if base_seed is None else base_seed + i
        winner, _ = simulate_match(
            player_a=player_a,
            player_b=player_b,
            matrices=matrices,
            best_of=args.best_of,
            seed=seed_i,
        )
        wins[winner] += 1

    total = args.matches
    pct_a = wins[player_a] / total * 100
    pct_b = wins[player_b] / total * 100
    print(f"{player_a}: {wins[player_a]} wins ({pct_a:.1f}%)")
    print(f"{player_b}: {wins[player_b]} wins ({pct_b:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
