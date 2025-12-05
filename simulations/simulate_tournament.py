import sys
import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from simulations.simulate_score_match import load_score_matrix, merge_matrices
from simulations.simulate_score_match import simulate_match as score_sim
from simulations.simulate_match import simulate_match as state_sim

BIN_EDGES = [
    (1, 10, "Top10"),
    (11, 25, "11_25"),
    (26, 50, "26_50"),
    (51, 100, "51_100"),
    (101, 200, "101_200"),
]
DEFAULT_BIN = "200_plus"

@dataclass
class MatchSpec:
    player1: str
    player2: str
    best_of: int = 5

def normalize_name(name: str) -> str:
    """Loosen name matching to handle hyphens, apostrophes, underscores, extra tokens."""
    cleaned = (
        name.replace("-", " ")
        .replace("_", " ")
        .replace("'", "")
        .replace(".", " ")
        .replace("\xa0", " ")
    )
    return " ".join(cleaned.split()).lower()


def load_bracket(path: Path) -> List[MatchSpec]:
    """Load a round-1 bracket from JSON (list of dicts)."""
    data = json.loads(Path(path).read_text())
    specs: List[MatchSpec] = []
    for item in data:
        specs.append(
            MatchSpec(
                player1=item["player1"],
                player2=item["player2"],
                best_of=item.get("best_of", 5),
            )
        )
    return specs


def render_bracket(summary_df: pd.DataFrame) -> str:
    """
    Build a simple ASCII bracket from the summary file (one row per match).
    Expects columns: round_number, player1, player2, winner, winner_pct.
    """
    lines: List[str] = []
    for rnd in sorted(summary_df["round_number"].unique()):
        lines.append(f"Round {rnd}")
        round_rows = summary_df[summary_df["round_number"] == rnd]
        for _, row in round_rows.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            w = row["winner"]
            pct = row.get("winner_pct", np.nan)
            pct_str = f" ({pct:.0%})" if pd.notna(pct) else ""
            lines.append(f"  {p1} ──────┐")
            lines.append(f"            ├── {w}{pct_str}")
            lines.append(f"  {p2} ──────┘")
        lines.append("")  # spacer
    return "\n".join(lines)


def classify_bin(rank: float | None) -> str:
    if rank is None or pd.isna(rank):
        return DEFAULT_BIN
    for lo, hi, label in BIN_EDGES:
        if lo <= rank <= hi:
            return label
    return DEFAULT_BIN


def load_rankings(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    name_col = "player" if "player" in df.columns else ("name" if "name" in df.columns else None)
    if not name_col:
        raise ValueError("Rankings CSV must have a 'player' or 'name' column.")
    rank_col = None
    for c in ("atp_rank", "wta_rank", "rank", "elo_rank"):
        if c in df.columns:
            rank_col = c
            break
    if not rank_col:
        raise ValueError("Rankings CSV must have a rank column (atp_rank/wta_rank/rank/elo_rank).")
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        nm = str(row[name_col]).replace("\xa0", " ").strip()
        key = normalize_name(nm)
        if pd.notna(row[rank_col]):
            mapping[key] = float(row[rank_col])
    return mapping


class Tournament:
    def __init__(
        self,
        tournament_id: str,
        round1_matches: Iterable[MatchSpec],
        simulator: Callable[[str, str, int, Optional[str], Optional[str]], Tuple[str, Tuple[int, int]]],
        player_bins: Optional[Dict[str, str]] = None,
    ):
        self.tournament_id = tournament_id
        self.simulator = simulator
        self.round1_matches = list(round1_matches)
        self.player_bins = player_bins or {}

    def run(
        self,
        simulations_per_match: int = 1,
        detail_out: Optional[Path] = None,
        summary_out: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the tournament, returning (detail_df, summary_df)."""
        detail_rows = []
        summary_rows = []
        current_round_matches = self.round1_matches
        round_number = 1
        match_counter = 1

        while len(current_round_matches) >= 1:
            next_round: List[MatchSpec] = []
            for m in current_round_matches:
                wins = {m.player1: 0, m.player2: 0}
                a_bin = self.player_bins.get(m.player1)
                b_bin = self.player_bins.get(m.player2)
                for sim_idx in range(simulations_per_match):
                    winner, score = self.simulator(
                        m.player1, m.player2, best_of=m.best_of, a_bin=a_bin, b_bin=b_bin
                    )
                    wins[winner] = wins.get(winner, 0) + 1
                    detail_rows.append(
                        {
                            "tournament_id": self.tournament_id,
                            "round_number": round_number,
                            "match_id": f"R{round_number}_M{match_counter}",
                            "player1": m.player1,
                            "player2": m.player2,
                            "sim_number": sim_idx,
                            "score1": score[0],
                            "score2": score[1],
                            "winner": winner,
                        }
                    )
                # choose winner by most wins (tie -> player1)
                winner = m.player1 if wins[m.player1] >= wins[m.player2] else m.player2
                winner_pct = wins[winner] / simulations_per_match if simulations_per_match > 0 else 0.0
                summary_rows.append(
                    {
                        "tournament_id": self.tournament_id,
                        "round_number": round_number,
                        "match_id": f"R{round_number}_M{match_counter}",
                        "player1": m.player1,
                        "player2": m.player2,
                        "winner": winner,
                        "winner_pct": winner_pct,
                        "sims": simulations_per_match,
                    }
                )
                match_counter += 1
                next_round.append(MatchSpec(player1=winner, player2=None, best_of=m.best_of))

            # pair winners for next round
            paired: List[MatchSpec] = []
            winners = [m.player1 for m in next_round if m.player1]
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    paired.append(MatchSpec(winners[i], winners[i + 1], best_of=5))
            current_round_matches = paired
            round_number += 1

            if len(current_round_matches) == 0:
                break

        detail_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows)
        if detail_out:
            detail_out.parent.mkdir(parents=True, exist_ok=True)
            detail_df.to_csv(detail_out, index=False)
        if summary_out:
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_out, index=False)
        return detail_df, summary_df


# --- Score-based adapter ---
def make_score_simulator(
    base_dir: Path, player_bins: Optional[Dict[str, str]] = None
) -> Callable[[str, str, int, Optional[str], Optional[str]], Tuple[str, Tuple[int, int]]]:
    """
    Build a simulator that wraps simulate_score_match.simulate_match using serve matrices
    located under base_dir. Expects filenames either in player folders or *_filled_score_matrix.csv.
    Also merges the same fallbacks as simulate_score_match.main: primary -> bin_vs_bin -> bin_vs_all -> global
    (all within base_dir).
    """

    cache: dict[str, pd.DataFrame] = {}

    def slugify(name: str) -> str:
        slug = name.replace("-", " ").replace("'", "").replace(".", " ")
        slug = "_".join(slug.split())
        return slug

    def load_player_matrix(name: str) -> pd.DataFrame:
        if name in cache:
            return cache[name]
        slug = slugify(name)
        cand1 = base_dir / f"{slug}_filled_score_matrix.csv"
        cand2 = base_dir / "player" / f"player_{slug}" / "score_matrix.csv"
        if cand1.exists():
            mat = load_score_matrix(cand1)
        elif cand2.exists():
            mat = load_score_matrix(cand2)
        else:
            raise FileNotFoundError(f"Could not find score matrix for {name}")
        cache[name] = mat
        return mat

    merged_cache: dict[tuple[str, Optional[str], Optional[str]], pd.DataFrame] = {}

    def adapter(p1: str, p2: str, best_of: int = 5, a_bin: Optional[str] = None, b_bin: Optional[str] = None) -> Tuple[str, Tuple[int, int]]:

        def load_fb(h_bin: str | None, o_bin: str | None):
            bin_bin = None
            bin_all = None
            glob = None
            if h_bin and o_bin:
                bb = base_dir / "bin_vs_bin" / f"{h_bin}_vs_{o_bin}" / "score_matrix.csv"
                if bb.exists():
                    bin_bin = load_score_matrix(bb)
            if h_bin:
                ba = base_dir / "bin_vs_all" / h_bin / "score_matrix.csv"
                if ba.exists():
                    bin_all = load_score_matrix(ba)
            g = base_dir / "global" / "score_matrix.csv"
            if g.exists():
                glob = load_score_matrix(g)
            return [bin_bin, bin_all, glob]

        # Merge fallbacks so simulate_match receives a fully merged matrix (bin-aware if bins provided)
        h_bin = a_bin if a_bin is not None else (player_bins.get(p1) if player_bins else None)
        o_bin = b_bin if b_bin is not None else (player_bins.get(p2) if player_bins else None)
        key_a = (p1, h_bin, o_bin)
        key_b = (p2, o_bin, h_bin)

        if key_a in merged_cache:
            mat_a = merged_cache[key_a]
        else:
            print(f"Merging matrices for {p1} with base {base_dir} (bin {h_bin})")
            mat_a = merge_matrices(load_player_matrix(p1), load_fb(h_bin, o_bin), label=p1)
            merged_cache[key_a] = mat_a

        if key_b in merged_cache:
            mat_b = merged_cache[key_b]
        else:
            print(f"Merging matrices for {p2} with base {base_dir} (bin {o_bin})")
            mat_b = merge_matrices(load_player_matrix(p2), load_fb(o_bin, h_bin), label=p2)
            merged_cache[key_b] = mat_b

        winner, scores = score_sim(mat_a, mat_b, a_bin=h_bin, b_bin=o_bin, best_of=best_of)
        # scores is list of set tuples; convert to total games for output
        total = (sum(s[0] for s in scores), sum(s[1] for s in scores))
        return winner, total

    return adapter


# --- State-based adapter ---
def make_state_simulator(
    sim_fn: Callable[[str, str, int], Tuple[str, list]]
) -> Callable[[str, str, int, Optional[str], Optional[str]], Tuple[str, Tuple[int, int]]]:
    """
    Adapter stub for a state/shot-level simulator. Provide a callable `sim_fn`
    that returns (winner_name, list_of_set_scores) where each set score is (games_a, games_b).
    """

    def adapter(
        p1: str, p2: str, best_of: int = 5, a_bin: Optional[str] = None, b_bin: Optional[str] = None
    ) -> Tuple[str, Tuple[int, int]]:
        winner, scores = sim_fn(p1, p2, best_of=best_of)
        total = (sum(s[0] for s in scores), sum(s[1] for s in scores))
        return winner, total

    return adapter


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tournament simulator (score or state based).")
    parser.add_argument("--bracket", required=True, help="Path to JSON bracket file.")
    parser.add_argument("--mode", choices=["score", "state"], default="score", help="Simulation mode.")
    parser.add_argument("--base_dir", required=True, help="Base directory for transition matrices.")
    parser.add_argument("--simulations", type=int, default=1, help="Number of simulations per match.")
    parser.add_argument("--best_of", type=int, default=5, help="Default best-of sets if bracket entries omit best_of.")
    parser.add_argument("--tournament_id", default="tourney1", help="Identifier for outputs.")
    parser.add_argument("--detail_out", default="tournament_detail.csv", help="Path for detailed CSV.")
    parser.add_argument("--summary_out", default="tournament_summary.csv", help="Path for summary CSV.")
    parser.add_argument("--bracket_out", default="tournament_bracket.txt", help="Path for ASCII bracket output.")
    parser.add_argument("--rankings_csv", help="CSV containing player ranks; enables rank-bin fallbacks.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bracket_specs = load_bracket(Path(args.bracket))
    # apply default best_of where missing
    round1 = [MatchSpec(m.player1, m.player2, m.best_of if m.best_of else args.best_of) for m in bracket_specs]

    player_bins: Dict[str, str] = {}
    if args.rankings_csv:
        rank_map = load_rankings(Path(args.rankings_csv))
        # build bins for any player mentioned in bracket
        names = set()
        for m in round1:
            names.add(m.player1)
            names.add(m.player2)
        for name in names:
            player_bins[name] = classify_bin(rank_map.get(normalize_name(name)))
    if args.mode == "score":
        sim = make_score_simulator(Path(args.base_dir), player_bins=player_bins if player_bins else None)
    else:
        sim = make_state_simulator(state_sim)

    tourney = Tournament(args.tournament_id, round1, simulator=sim, player_bins=player_bins)
    detail_df, summary_df = tourney.run(
        simulations_per_match=args.simulations,
        detail_out=Path(args.detail_out),
        summary_out=Path(args.summary_out),
    )
    bracket_txt = render_bracket(summary_df)
    Path(args.bracket_out).write_text(bracket_txt)
    print(f"Wrote detail -> {args.detail_out}")
    print(f"Wrote summary -> {args.summary_out}")
    print(f"Wrote bracket -> {args.bracket_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
