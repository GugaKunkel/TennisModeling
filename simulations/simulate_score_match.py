from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def load_score_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.fillna(0.0, inplace=True)
    return df

def matrix_is_binned(matrix: pd.DataFrame) -> bool:
    return any("|opp_bin=" in idx for idx in matrix.index)

def state_label(score: str, opp_bin: str | None, use_bin: bool) -> str:
    return f"{score}|opp_bin={opp_bin}" if use_bin and opp_bin else score

def coerce_probs(row: pd.Series) -> Tuple[np.ndarray, list[str]]:
    labels = list(row.index)
    probs = row.to_numpy(dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.zeros(len(labels)), labels
    return probs / total, labels

def merge_matrices(
    primary: pd.DataFrame | None, fallbacks: list[pd.DataFrame | None], label: str | None = None
) -> pd.DataFrame:
    candidates = [m for m in [primary, *fallbacks] if m is not None and not m.empty]
    if not candidates:
        return pd.DataFrame()
    all_states = sorted(set().union(*[set(m.index) for m in candidates]))
    all_cols = sorted(set().union(*[set(m.columns) for m in candidates]))
    merged = pd.DataFrame(0.0, index=all_states, columns=all_cols)
    for state in all_states:
        for idx, m in enumerate(candidates):
            if state in m.index and m.loc[state].sum() > 0:
                row = m.loc[state].reindex(all_cols).fillna(0.0)
                total = row.sum()
                if total > 0:
                    merged.loc[state] = row / total
                    # if primary is None or m is not primary:
                    #     source = "primary" if m is primary else f"fallback_{idx}"
                    #     who = f" [{label}]" if label else ""
                    #     print(f"Filling state '{state}' from {source}{who}")
                break
    merged.index.name = ""
    return merged

def sample_next_state(state: str, matrix: pd.DataFrame, rng: np.random.Generator) -> str | None:
    if state not in matrix.index:
        return None
    probs, labels = coerce_probs(matrix.loc[state])
    if probs.sum() <= 0:
        return None
    return rng.choice(labels, p=probs)

def is_game_terminal(new_state: str) -> Tuple[bool, str | None]:
    if new_state.startswith("game_server"):
        return True, "server"
    if new_state.startswith("game_returner"):
        return True, "returner"
    return False, None

def simulate_game(server_matrix: pd.DataFrame, rng: np.random.Generator, opp_bin: str | None, use_bin: bool) -> str:
    state = state_label("0-0", opp_bin, use_bin)
    safety = 200
    while safety > 0:
        safety -= 1
        next_state = sample_next_state(state, server_matrix, rng)
        if next_state is None:
            print(f"Warning: missing transition for state '{state}', falling back to coin flip.")
            return rng.choice(["server", "returner"])
        terminal, winner = is_game_terminal(next_state)
        if terminal:
            return winner
        state = next_state
    return rng.choice(["server", "returner"])

def simulate_tiebreak(
    server_matrix: pd.DataFrame,
    returner_matrix: pd.DataFrame,
    rng: np.random.Generator,
    opp_bin: str | None,
    use_bin: bool,
) -> str:
    s_pts = r_pts = 0
    points_played = 0
    while True:
        if points_played == 0:
            mat = server_matrix
            label = state_label("TB_0-0", opp_bin, use_bin)
        else:
            block = (points_played - 1) // 2
            if block % 2 == 0:
                mat = returner_matrix
                label = state_label(f"TB_{r_pts}-{s_pts}", opp_bin, use_bin)
            else:
                mat = server_matrix
                label = state_label(f"TB_{s_pts}-{r_pts}", opp_bin, use_bin)
        next_state = sample_next_state(label, mat, rng)
        if next_state is None:
            winner = rng.choice(["server", "returner"])
        else:
            if next_state.startswith("TB_"):
                try:
                    a, b = next_state.replace("TB_", "").split("-")
                    n_s, n_r = int(a), int(b)
                    if n_s == s_pts + 1 and n_r == r_pts:
                        winner = "server"
                    elif n_r == r_pts + 1 and n_s == s_pts:
                        winner = "returner"
                    else:
                        winner = rng.choice(["server", "returner"])
                except Exception:
                    winner = rng.choice(["server", "returner"])
            else:
                term, w = is_game_terminal(next_state)
                winner = w if term else rng.choice(["server", "returner"])
        if winner == "server":
            s_pts += 1
        else:
            r_pts += 1
        points_played += 1
        if (s_pts >= 7 or r_pts >= 7) and abs(s_pts - r_pts) >= 2:
            return "server" if s_pts > r_pts else "returner"

def simulate_set(
    server_a_matrix: pd.DataFrame,
    server_b_matrix: pd.DataFrame,
    rng: np.random.Generator,
    a_bin: str | None,
    b_bin: str | None,
    use_bin: bool,
) -> Tuple[str, Tuple[int, int]]:
    games_a = games_b = 0
    server = "A"
    while True:
        if server == "A":
            winner = simulate_game(server_a_matrix, rng, b_bin, use_bin)
            if winner == "server":
                games_a += 1
            else:
                games_b += 1
            server = "B"
        else:
            winner = simulate_game(server_b_matrix, rng, a_bin, use_bin)
            if winner == "server":
                games_b += 1
            else:
                games_a += 1
            server = "A"

        if (games_a >= 6 or games_b >= 6) and abs(games_a - games_b) >= 2:
            return ("A" if games_a > games_b else "B"), (games_a, games_b)
        if games_a == 6 and games_b == 6:
            if server == "A":
                tb_winner = simulate_tiebreak(server_a_matrix, server_b_matrix, rng, b_bin, use_bin)
            else:
                tb_winner = simulate_tiebreak(server_b_matrix, server_a_matrix, rng, a_bin, use_bin)
            if tb_winner == "server":
                if server == "A":
                    games_a += 1
                    return "A", (7, 6)
                else:
                    games_b += 1
                    return "B", (6, 7)
            else:
                if server == "A":
                    games_b += 1
                    return "B", (6, 7)
                else:
                    games_a += 1
                    return "A", (7, 6)

def simulate_match(
    server_a_matrix: pd.DataFrame,
    server_b_matrix: pd.DataFrame,
    a_bin: str | None = None,
    b_bin: str | None = None,
    best_of: int = 5,
    seed: int | None = None,
) -> Tuple[str, list[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    use_bin = matrix_is_binned(server_a_matrix) or matrix_is_binned(server_b_matrix)
    sets_needed = best_of // 2 + 1
    sets_a = sets_b = 0
    scores: list[Tuple[int, int]] = []
    while max(sets_a, sets_b) < sets_needed:
        set_winner, score = simulate_set(server_a_matrix, server_b_matrix, rng, a_bin, b_bin, use_bin)
        if set_winner == "A":
            sets_a += 1
            scores.append(score)
        else:
            sets_b += 1
            scores.append((score[1], score[0]))
    return ("A" if sets_a > sets_b else "B"), scores

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate matches using score matrices.")
    parser.add_argument("--server_a_matrix", required=True, help="Score matrix CSV for player A serve.")
    parser.add_argument("--server_b_matrix", required=True, help="Score matrix CSV for player B serve.")
    parser.add_argument("--player_a_bin", help="Rank bin label for player A.")
    parser.add_argument("--player_b_bin", help="Rank bin label for player B.")
    parser.add_argument("--matches", type=int, default=1, help="Number of matches.")
    parser.add_argument("--best_of", type=int, default=5, help="Number of sets in match (3 or 5).")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    parser.add_argument(
        "--base_dir",
        help="Optional base directory containing bin_vs_bin/bin_vs_all/global score matrices for fallbacks.",
    )
    return parser.parse_args(argv)

def friendly_player_name(path: Path) -> str:
    stem = path.stem.replace("_filled_score_matrix", "").replace("_", " ").replace("-", " ")
    if stem.lower() == "score matrix":
        stem = path.parent.name.replace("player_", "").replace("_", " ").replace("-", " ")
    return stem

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    primary_a = load_score_matrix(Path(args.server_a_matrix))
    primary_b = load_score_matrix(Path(args.server_b_matrix))
    player_a = friendly_player_name(Path(args.server_a_matrix))
    player_b = friendly_player_name(Path(args.server_b_matrix))

    if args.base_dir:
        base = Path(args.base_dir)
        def load_fb(h_bin: str | None, o_bin: str | None):
            if h_bin and o_bin:
                bin_bin = load_score_matrix(base / "bin_vs_bin" / f"{h_bin}_vs_{o_bin}" / "score_matrix.csv") if (base / "bin_vs_bin" / f"{h_bin}_vs_{o_bin}" / "score_matrix.csv").exists() else None
            else:
                bin_bin = None
            bin_all = load_score_matrix(base / "bin_vs_all" / h_bin / "score_matrix.csv") if h_bin and (base / "bin_vs_all" / h_bin / "score_matrix.csv").exists() else None
            glob = load_score_matrix(base / "global" / "score_matrix.csv") if (base / "global" / "score_matrix.csv").exists() else None
            return [bin_bin, bin_all, glob]

        mat_a = merge_matrices(primary_a, load_fb(args.player_a_bin, args.player_b_bin), label=player_a)
        mat_b = merge_matrices(primary_b, load_fb(args.player_b_bin, args.player_a_bin), label=player_b)
    else:
        mat_a = primary_a
        mat_b = primary_b
    base_seed = args.seed
    wins = {player_a: 0, player_b: 0}
    for i in range(args.matches):
        seed_i = None if base_seed is None else base_seed + i
        winner, scores = simulate_match(
            mat_a,
            mat_b,
            a_bin=args.player_a_bin,
            b_bin=args.player_b_bin,
            best_of=args.best_of,
            seed=seed_i,
        )
        winner_name = player_a if winner == "A" else player_b
        wins[winner_name] += 1
    total = args.matches
    pct_a = wins[player_a] / total * 100
    pct_b = wins[player_b] / total * 100
    print(f"{player_a}: {wins[player_a]} wins ({pct_a:.1f}%)")
    print(f"{player_b}: {wins[player_b]} wins ({pct_b:.1f}%)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
