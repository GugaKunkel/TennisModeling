import tempfile
import unittest
from pathlib import Path
import pandas as pd
from transition_matrix_builders import compute_score_matrices as cs

class ComputeScoreMatricesTests(unittest.TestCase):
    def _write_csv(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def test_no_rankings_produces_global_and_player_matrices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            transitions_path = base / "state_transitions.csv"

            transitions = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "player_name": "Alice",
                        "player_label": "P1",
                        "server_flag": "serve_side",
                        "is_terminal": True,
                        "point_score": "0-0",
                        "point_score_after": "15-0",
                    },
                    {
                        "match_id": "m1",
                        "player_name": "Alice",
                        "player_label": "P1",
                        "server_flag": "serve_side",
                        "is_terminal": True,
                        "point_score": "15-0",
                        "point_score_after": "30-0",
                    },
                ]
            )
            self._write_csv(transitions_path, transitions)

            out_dir = base / "out"
            cs.main(
                [
                    "--input",
                    str(transitions_path),
                    "--output_dir",
                    str(out_dir),
                ]
            )

            global_matrix = pd.read_csv(out_dir / "global" / "score_matrix.csv")
            player_matrix = pd.read_csv(out_dir / "player" / "player_Alice" / "score_matrix.csv")

            self.assertTrue(all("opp_bin=" not in s for s in global_matrix.iloc[:, 0]))
            self.assertTrue(all("opp_bin=" not in s for s in player_matrix.iloc[:, 0]))

            row = player_matrix[player_matrix.iloc[:, 0] == "0-0"]
            self.assertEqual(row.shape[0], 1)
            self.assertAlmostEqual(float(row["15-0"]), 1.0)

    def test_rankings_enable_bin_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            transitions_path = base / "state_transitions.csv"
            rankings_path = base / "ranks.csv"
            matches_path = base / "matches.csv"

            transitions = pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "player_name": "Alice",
                        "player_label": "P1",
                        "server_flag": "serve_side",
                        "is_terminal": True,
                        "point_score": "0-0",
                        "point_score_after": "15-0",
                    },
                    {
                        "match_id": "m1",
                        "player_name": "Bob",
                        "player_label": "P2",
                        "server_flag": "serve_side",
                        "is_terminal": True,
                        "point_score": "0-0",
                        "point_score_after": "0-15",
                    },
                ]
            )
            ranks = pd.DataFrame(
                [
                    {"player": "Alice", "atp_rank": 5},
                    {"player": "Bob", "atp_rank": 120},
                ]
            )
            matches = pd.DataFrame([{"match_id": "m1", "Player 1": "Alice", "Player 2": "Bob"}])

            self._write_csv(transitions_path, transitions)
            self._write_csv(rankings_path, ranks)
            self._write_csv(matches_path, matches)

            out_dir = base / "out_bins"
            cs.main(
                [
                    "--input",
                    str(transitions_path),
                    "--output_dir",
                    str(out_dir),
                    "--rankings_csv",
                    str(rankings_path),
                ]
            )

            bin_dir = out_dir / "bin_vs_all" / "Top10"
            self.assertTrue((bin_dir / "score_matrix.csv").exists())
            bin_matrix = pd.read_csv(bin_dir / "score_matrix.csv")
            self.assertTrue(any("opp_bin=" in s for s in bin_matrix.iloc[:, 0]))

            player_matrix = pd.read_csv(out_dir / "player" / "player_Alice" / "score_matrix.csv")
            self.assertTrue(any("opp_bin=" in s for s in player_matrix.iloc[:, 0]))

    def test_build_score_transitions_plain_and_binned(self):
        df = pd.DataFrame(
            [
                {
                    "server_flag": "serve_side",
                    "is_terminal": True,
                    "point_score": "0-0",
                    "point_score_after": "15-0",
                    "opponent_bin": "Top10",
                },
                {
                    "server_flag": "return_side",
                    "is_terminal": True,
                    "point_score": "0-0",
                    "point_score_after": "0-15",
                    "opponent_bin": "Top10",
                },
            ]
        )
        trans_plain = cs.build_score_transitions(df, use_bins=False)
        self.assertEqual(len(trans_plain), 1)
        self.assertEqual(trans_plain.iloc[0]["state"], "0-0")
        self.assertEqual(trans_plain.iloc[0]["new_state"], "15-0")

        trans_bin = cs.build_score_transitions(df, use_bins=True)
        self.assertEqual(len(trans_bin), 1)
        self.assertEqual(trans_bin.iloc[0]["state"], "0-0|opp_bin=Top10")
        self.assertEqual(trans_bin.iloc[0]["new_state"], "15-0|opp_bin=Top10")

    def test_build_score_matrix(self):
        transitions = pd.DataFrame(
            [
                {"state": "0-0", "new_state": "15-0", "count": 2},
                {"state": "0-0", "new_state": "0-15", "count": 1},
            ]
        )
        mat = cs.build_score_matrix(transitions)
        self.assertIn("0-0", mat.index)
        self.assertAlmostEqual(mat.loc["0-0"]["15-0"], 2 / 3)
        self.assertAlmostEqual(mat.loc["0-0"]["0-15"], 1 / 3)

    def test_load_match_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            matches_df = pd.DataFrame(
                [
                    {"match_id": "m1", "Player 1": "Alice", "Player 2": "Bob"},
                    {"match_id": "m2", "Player 1": "Cara", "Player 2": "Dan"},
                ]
            )
            matches_path = base / "matches.csv"
            self._write_csv(matches_path, matches_df)
            mapping = cs.load_match_map([matches_path])
            self.assertEqual(mapping["m1"], ("Alice", "Bob"))
            self.assertEqual(mapping["m2"], ("Cara", "Dan"))


if __name__ == "__main__":
    unittest.main()
