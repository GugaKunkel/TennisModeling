import tempfile
import unittest
from pathlib import Path
import pandas as pd
from transition_matrix_builders import compute_state_matrices as csm


class TestComputeStateMatrices(unittest.TestCase):
    def test_coerce_bool_handles_various_inputs(self):
        self.assertTrue(csm.coerce_bool(True))
        self.assertTrue(csm.coerce_bool(1))
        self.assertTrue(csm.coerce_bool("Yes"))
        self.assertFalse(csm.coerce_bool(False))
        self.assertFalse(csm.coerce_bool(0))
        self.assertFalse(csm.coerce_bool("no"))
        self.assertFalse(csm.coerce_bool(None))

    def test_normalize_and_classify_bin(self):
        self.assertEqual(csm.normalize_name(" A\xa0B "), "A B")
        self.assertEqual(csm.classify_bin(5), "Top10")
        self.assertEqual(csm.classify_bin(25), "11_25")
        self.assertEqual(csm.classify_bin(75), "51_100")
        self.assertEqual(csm.classify_bin(None), csm.DEFAULT_BIN)

    def test_load_rankings_and_match_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranks = Path(tmp) / "ranks.csv"
            pd.DataFrame({"player": ["A", "B"], "atp_rank": [3, 130]}).to_csv(ranks, index=False)
            mapping = csm.load_rankings(ranks)
            self.assertEqual(mapping["A"], 3.0)
            matches = Path(tmp) / "matches.csv"
            pd.DataFrame({"match_id": ["m1"], "Player 1": ["A"], "Player 2": ["B"]}).to_csv(
                matches, index=False
            )
            mm = csm.load_match_map([matches])
            self.assertEqual(mm["m1"], ("A", "B"))

    def test_compute_result_matrix_counts(self):
        df = pd.DataFrame(
            {
                "state": ["S", "S", "T"],
                "is_terminal": [True, True, False],
                "player_label": ["P1", "P1", "P2"],
                "winner_label": ["P1", "P2", ""],
            }
        )
        res = csm.compute_result_matrix(df)
        self.assertEqual(set(res.index), {"S", "T"})
        self.assertAlmostEqual(res.loc["S", "win"], 0.5)
        self.assertAlmostEqual(res.loc["S", "loss"], 0.5)
        self.assertAlmostEqual(res.loc["T", "continue"], 1.0)

    def test_compute_continuation_matrix_probs(self):
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B"],
                "new_state": ["X", "Y", "Y"],
                "is_terminal": [False, False, False],
            }
        )
        cont = csm.compute_continuation_matrix(df)
        self.assertEqual(set(cont.index), {"A", "B"})
        self.assertEqual(set(cont.columns), {"X", "Y"})
        self.assertAlmostEqual(cont.loc["A", "X"], 0.5)
        self.assertAlmostEqual(cont.loc["A", "Y"], 0.5)
        self.assertAlmostEqual(cont.loc["B", "Y"], 1.0)

    def test_attach_bins_assigns_expected_labels(self):
        rank_map = {"Alice": 8.0, "Bob": 120.0, "Cara": 35.0}
        match_map = {"m1": ("Alice", "Bob"), "m2": ("Cara", "Alice")}
        df = pd.DataFrame(
            {
                "match_id": ["m1", "m2"],
                "player_label": ["P1", "P2"],
                "player_name": ["Alice", "Bob"],
            }
        )
        out = csm.attach_bins(df, rank_map, match_map)
        self.assertEqual(list(out["player_bin"]), ["Top10", "101_200"])
        self.assertEqual(list(out["opponent_bin"]), ["101_200", "26_50"])

    def test_write_matrix_includes_row_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"win": [1.0], "loss": [0.0]}, index=["state1"])
            out_path = Path(tmp) / "mat.csv"
            csm.write_matrix(df, out_path)
            loaded = pd.read_csv(out_path)
            self.assertTrue(loaded.columns[0].startswith("Unnamed"))
            self.assertEqual(loaded.iloc[0, 0], "state1")

    def test_save_matrices_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "state": ["S", "S"],
                    "new_state": ["R", "R"],
                    "is_terminal": [False, True],
                    "player_label": ["P1", "P2"],
                    "winner_label": ["P1", "P1"],
                    "player_name": ["A", "B"],
                }
            )
            out_dir = Path(tmp) / "out"
            csm.save_matrices(df, out_dir)
            self.assertTrue((out_dir / "matrix_result.csv").exists())
            self.assertTrue((out_dir / "matrix_continue.csv").exists())

    def test_main_without_bins_writes_global_and_players(self):
        with tempfile.TemporaryDirectory() as tmp:
            transitions = Path(tmp) / "transitions.csv"
            df = pd.DataFrame(
                {
                    "state": ["S", "S", "T"],
                    "new_state": ["R", "R", "U"],
                    "is_terminal": [False, True, False],
                    "player_label": ["P1", "P2", "P2"],
                    "winner_label": ["P1", "P1", ""],
                    "player_name": ["Alice", "Bob", "Bob"],
                }
            )
            df.to_csv(transitions, index=False)
            out_dir = Path(tmp) / "out"
            rc = csm.main(
                [
                    "--input",
                    str(transitions),
                    "--output_dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "global" / "matrix_result.csv").exists())
            self.assertTrue((out_dir / "global" / "matrix_continue.csv").exists())
            self.assertTrue((out_dir / "player" / "player_Alice" / "matrix_result.csv").exists())

    def test_main_with_bins_creates_global_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            transitions = Path(tmp) / "transitions.csv"
            df = pd.DataFrame(
                {
                    "match_id": ["m1"],
                    "state": ["S"],
                    "new_state": ["R"],
                    "is_terminal": [False],
                    "player_label": ["P1"],
                    "winner_label": [""],
                    "player_name": ["Alice"],
                }
            )
            df.to_csv(transitions, index=False)
            rankings = Path(tmp) / "ranks.csv"
            pd.DataFrame({"player": ["Alice"], "atp_rank": [4]}).to_csv(rankings, index=False)
            out_dir = Path(tmp) / "out_bins"
            rc = csm.main(
                [
                    "--input",
                    str(transitions),
                    "--rankings_csv",
                    str(rankings),
                    "--output_dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "global" / "matrix_result.csv").exists())
            self.assertTrue((out_dir / "global" / "matrix_continue.csv").exists())

if __name__ == "__main__":
    unittest.main()
