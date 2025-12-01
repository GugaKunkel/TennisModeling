#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np
import pandas as pd


DEFAULT_STATE_FIELDS = [
    "server_flag",
    "prev_family",
    "prev_direction",
    "rally_bin",
    "point_score",
    "point_start_serve",
]


def load_matrix(path: Path) -> pd.DataFrame:
    """Load a wide matrix CSV (blank header column for row labels)."""
    df = pd.read_csv(path, index_col=0)
    df.fillna(0.0, inplace=True)
    return df


@dataclass
class PlayerMatrices:
    result: pd.DataFrame  # rows: state, cols: win/loss/continue
    cont: pd.DataFrame  # rows: state, cols: new_state


def load_player_matrices(player_dir: Path) -> PlayerMatrices:
    return PlayerMatrices(
        result=load_matrix(player_dir / "matrix_result.csv"),
        cont=load_matrix(player_dir / "matrix_continue.csv"),
    )


def coerce_probs(row: pd.Series, fallback: str | None = None) -> Tuple[np.ndarray, list[str]]:
    """Return (probs, labels); if row missing or sums to 0, use fallback."""
    labels = list(row.index)
    if row.empty or row.sum() <= 0:
        probs = np.zeros(len(labels))
        if fallback is not None and fallback in labels:
            probs[labels.index(fallback)] = 1.0
        return probs, labels
    probs = row.to_numpy(dtype=float)
    total = probs.sum()
    if total <= 0:
        if fallback is not None and fallback in labels:
            probs = np.zeros(len(labels))
            probs[labels.index(fallback)] = 1.0
        return probs, labels
    return probs / total, labels


def sample_result(
    state: str,
    mats: PlayerMatrices,
    rng: np.random.Generator,
    fallback: PlayerMatrices | None = None,
) -> str:
    if state in mats.result.index:
        row = mats.result.loc[state]
        if row.sum() > 0:
            probs, labels = coerce_probs(row, fallback="continue")
            return rng.choice(labels, p=probs)
    if fallback and state in fallback.result.index:
        row = fallback.result.loc[state]
        probs, labels = coerce_probs(row, fallback="continue")
        return rng.choice(labels, p=probs)
    return "continue"


def sample_next_state(
    state: str,
    mats: PlayerMatrices,
    rng: np.random.Generator,
    fallback: PlayerMatrices | None = None,
) -> str | None:
    if state in mats.cont.index:
        row = mats.cont.loc[state]
        if row.sum() > 0:
            probs, labels = coerce_probs(row)
            if probs.sum() > 0:
                return rng.choice(labels, p=probs)
    if fallback and state in fallback.cont.index:
        row = fallback.cont.loc[state]
        probs, labels = coerce_probs(row)
        if probs.sum() > 0:
            return rng.choice(labels, p=probs)
    return None


def format_state(state_fields: list[str], context: dict[str, str]) -> str:
    parts = []
    for key in state_fields:
        if key == "player_label":
            parts.append(context["player_label"])
        elif key == "server_flag":
            parts.append(context["server_flag"])
        elif key == "prev_family":
            parts.append(context["prev_family"])
        elif key == "prev_direction":
            parts.append(f"dir={context['prev_direction']}")
        elif key == "rally_bin":
            parts.append(f"rbin={context['rally_bin']}")
        elif key == "point_score":
            parts.append(f"score={context['point_score']}")
        elif key == "point_start_serve":
            parts.append(f"start_srv={context['point_start_serve']}")
        elif key == "rally_index":
            parts.append(str(context["rally_index"]))
        else:
            raise ValueError(f"Unsupported state field '{key}'")
    return "|".join(parts)


def tennis_score_label(server_pts: int, return_pts: int) -> str:
    mapping = {0: "0", 1: "15", 2: "30", 3: "40"}
    if server_pts >= 3 and return_pts >= 3:
        if server_pts == return_pts:
            return "40-40"
        if server_pts == return_pts + 1:
            return "Ad-In"
        if return_pts == server_pts + 1:
            return "Ad-Out"
    return f"{mapping.get(server_pts, '40')}-{mapping.get(return_pts, '40')}"


def simulate_point(
    server: str,
    returner: str,
    mats: Dict[str, PlayerMatrices],
    fallback: PlayerMatrices | None,
    state_fields: list[str],
    rng: np.random.Generator,
    start_score: str = "0-0",
    max_shots: int = 200,
) -> str:
    """Simulate a single point; return winner name."""
    state = format_state(
        state_fields,
        {
            "player_label": "P1",
            "server_flag": "serve_side",
            "prev_family": "BOS",
            "prev_direction": "0",
            "rally_bin": "0",
            "rally_index": 0,
            "point_score": start_score,
            "point_start_serve": "1st",
        },
    )
    hitter = server
    opponent = returner
    rally_index = 0
    prev_family = "BOS"
    prev_dir = "0"
    while max_shots > 0:
        max_shots -= 1
        mats_hitter = mats.get(hitter)
        if mats_hitter is None:
            return hitter  # fallback
        result = sample_result(state, mats_hitter, rng, fallback)
        if result == "win":
            return hitter
        if result == "loss":
            return opponent
        # continue
        next_state = sample_next_state(state, mats_hitter, rng, fallback)
        if not next_state:
            # no continuation info; pick winner at random
            return rng.choice([hitter, opponent])
        state = next_state
        prev_family = ""  # already encoded
        prev_dir = ""  # already encoded
        hitter, opponent = opponent, hitter
        rally_index += 1
    return rng.choice([server, returner])


def simulate_game(
    server: str,
    returner: str,
    mats: Dict[str, PlayerMatrices],
    fallback: PlayerMatrices | None,
    state_fields: list[str],
    rng: np.random.Generator,
) -> str:
    server_pts = 0
    returner_pts = 0
    while True:
        score_label = tennis_score_label(server_pts, returner_pts)
        winner = simulate_point(server, returner, mats, fallback, state_fields, rng, start_score=score_label)
        if winner == server:
            server_pts += 1
        else:
            returner_pts += 1
        if server_pts >= 4 or returner_pts >= 4:
            if abs(server_pts - returner_pts) >= 2:
                return server if server_pts > returner_pts else returner


def simulate_tiebreak(
    server: str,
    returner: str,
    mats: Dict[str, PlayerMatrices],
    fallback: PlayerMatrices | None,
    state_fields: list[str],
    rng: np.random.Generator,
) -> str:
    server_pts = 0
    returner_pts = 0
    points_played = 0
    while True:
        # Determine server for this point: first point by initial server, then switch after 1, then every 2 points.
        if points_played == 0:
            point_server = server
            point_returner = returner
        else:
            block = (points_played - 1) // 2
            if block % 2 == 0:
                point_server, point_returner = returner, server
            else:
                point_server, point_returner = server, returner

        score_label = f"TB_{server_pts}-{returner_pts}"
        winner = simulate_point(
            point_server, point_returner, mats, fallback, state_fields, rng, start_score=score_label
        )
        points_played += 1
        if winner == server:
            server_pts += 1
        else:
            returner_pts += 1
        if (server_pts >= 7 or returner_pts >= 7) and abs(server_pts - returner_pts) >= 2:
            return server if server_pts > returner_pts else returner


def simulate_set(
    server_first: str,
    returner_first: str,
    mats: Dict[str, PlayerMatrices],
    fallback: PlayerMatrices | None,
    state_fields: list[str],
    rng: np.random.Generator,
) -> Tuple[str, Tuple[int, int]]:
    server_games = 0
    returner_games = 0
    server = server_first
    returner = returner_first
    while True:
        winner = simulate_game(server, returner, mats, fallback, state_fields, rng)
        if winner == server:
            server_games += 1
        else:
            returner_games += 1
        if (server_games >= 6 or returner_games >= 6) and abs(server_games - returner_games) >= 2:
            return (server if server_games > returner_games else returner), (server_games, returner_games)
        if server_games == 6 and returner_games == 6:
            tb_winner = simulate_tiebreak(server, returner, mats, fallback, state_fields, rng)
            if tb_winner == server:
                server_games += 1
            else:
                returner_games += 1
            return (tb_winner), (server_games, returner_games)
        server, returner = returner, server  # alternate servers


def simulate_match(
    player_a: str,
    player_b: str,
    mats: Dict[str, PlayerMatrices],
    fallback: PlayerMatrices | None = None,
    best_of: int = 3,
    state_fields: list[str] | None = None,
    seed: int | None = None,
) -> Tuple[str, list[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    state_fields = state_fields or DEFAULT_STATE_FIELDS
    sets_to_win = best_of // 2 + 1
    sets_won = {player_a: 0, player_b: 0}
    set_scores: list[Tuple[int, int]] = []
    server = player_a
    returner = player_b
    while max(sets_won.values()) < sets_to_win:
        set_winner, score = simulate_set(server, returner, mats, fallback, state_fields, rng)
        set_scores.append(score if set_winner == server else score[::-1])
        sets_won[set_winner] += 1
        server, returner = returner, server  # alternate starting server each set
    winner = max(sets_won, key=sets_won.get)
    return winner, set_scores


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a tennis match using transition matrices.")
    parser.add_argument("--player_a_dir", required=True, help="Directory containing player A matrices.")
    parser.add_argument("--player_b_dir", required=True, help="Directory containing player B matrices.")
    parser.add_argument("--best_of", type=int, default=3, help="Number of sets (3 or 5).")
    parser.add_argument(
        "--state_fields",
        default=",".join(DEFAULT_STATE_FIELDS),
        help="Comma-separated state fields matching the matrices (default uses server_flag,...).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument(
        "--fallback_dir",
        help="Optional directory containing global matrices (matrix_result.csv and matrix_continue.csv) for backoff.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    state_fields = [f.strip() for f in args.state_fields.split(",") if f.strip()]

    player_a = Path(args.player_a_dir).name.replace("player_", "").replace("_", " ")
    player_b = Path(args.player_b_dir).name.replace("player_", "").replace("_", " ")

    mats = {
        player_a: load_player_matrices(Path(args.player_a_dir)),
        player_b: load_player_matrices(Path(args.player_b_dir)),
    }

    fallback = load_player_matrices(Path(args.fallback_dir)) if args.fallback_dir else None

    winner, set_scores = simulate_match(
        player_a=player_a,
        player_b=player_b,
        mats=mats,
        fallback=fallback,
        best_of=args.best_of,
        state_fields=state_fields,
        seed=args.seed,
    )
    readable_scores = " ".join([f"{s[0]}-{s[1]}" for s in set_scores])
    print(f"Winner: {winner} ({readable_scores})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
