import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
from simulations import simulate_score_match as ssm


def make_matrix(mapping: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = sorted(mapping.keys())
    cols = sorted({c for row in mapping.values() for c in row})
    data = []
    for r in rows:
        data.append([mapping[r].get(c, 0.0) for c in cols])
    df = pd.DataFrame(data, index=rows, columns=cols)
    return df


class TestSimulateScoreMatch(unittest.TestCase):
    def test_load_score_matrix_fills_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            pd.DataFrame({"": ["s1"], "n1": [np.nan], "n2": [1.0]}).to_csv(path, index=False)
            df = ssm.load_score_matrix(path)
            self.assertEqual(df.index.tolist(), ["s1"])
            self.assertEqual(df.loc["s1", "n1"], 0.0)

    def test_state_label_and_binned_detection(self):
        self.assertFalse(ssm.matrix_is_binned(make_matrix({"0-0": {"game_server": 1.0}})))
        self.assertEqual(ssm.state_label("0-0", "Top10", use_bin=True), "0-0|opp_bin=Top10")
        self.assertEqual(ssm.state_label("0-0", None, use_bin=True), "0-0")

    def test_coerce_probs(self):
        probs, labels = ssm.coerce_probs(pd.Series({"a": 0.0, "b": 0.0}))
        self.assertTrue(np.allclose(probs, [0.0, 0.0]))
        probs, labels = ssm.coerce_probs(pd.Series({"a": 1.0, "b": 1.0}))
        self.assertTrue(np.allclose(probs, [0.5, 0.5]))
        self.assertEqual(labels, ["a", "b"])

    def test_merge_matrices_prefers_primary_and_fallback(self):
        primary = make_matrix({"s": {"x": 1.0}})
        fallback = make_matrix({"s": {"y": 1.0}, "t": {"z": 1.0}})
        merged = ssm.merge_matrices(primary, [fallback])
        self.assertAlmostEqual(merged.loc["s", "x"], 1.0)
        self.assertAlmostEqual(merged.loc["t", "z"], 1.0)

    def test_sample_next_state_and_game_terminal(self):
        rng = np.random.default_rng(0)
        mat = make_matrix({"s": {"a": 0.3, "b": 0.7}})
        ns = ssm.sample_next_state("s", mat, rng)
        self.assertIn(ns, {"a", "b"})
        self.assertEqual(ssm.sample_next_state("missing", mat, rng), None)
        self.assertEqual(ssm.is_game_terminal("game_server_win")[0], True)
        self.assertEqual(ssm.is_game_terminal("noop")[0], False)

    def test_simulate_game_terminates_on_missing(self):
        rng = np.random.default_rng(1)
        mat = make_matrix({"0-0": {"non_term": 0.0}})
        winner = ssm.simulate_game(mat, rng, opp_bin=None, use_bin=False)
        self.assertIn(winner, {"server", "returner"})

    def test_simulate_game_uses_terminal(self):
        rng = np.random.default_rng(1)
        mat = make_matrix({"0-0": {"game_server": 1.0}})
        winner = ssm.simulate_game(mat, rng, opp_bin=None, use_bin=False)
        self.assertEqual(winner, "server")

    def test_simulate_tiebreak_with_mocked_sample(self):
        rng = np.random.default_rng(0)
        mat = make_matrix({"TB_0-0": {"game_server": 1.0}})
        with mock.patch.object(ssm, "sample_next_state", return_value="game_server"):
            winner = ssm.simulate_tiebreak(mat, mat, rng, opp_bin=None, use_bin=False)
            self.assertEqual(winner, "server")

    def test_simulate_set_no_tiebreak(self):
        rng = np.random.default_rng(0)
        mat_a = make_matrix({"0-0": {"game_server": 1.0}})
        mat_b = make_matrix({"0-0": {"game_returner": 1.0}})
        winner, score = ssm.simulate_set(mat_a, mat_b, rng, a_bin=None, b_bin=None, use_bin=False)
        self.assertEqual(winner, "A")
        self.assertEqual(score, (6, 0))

    def test_simulate_match_best_of_one(self):
        mat_a = make_matrix({"0-0": {"game_server": 1.0}})
        mat_b = make_matrix({"0-0": {"game_returner": 1.0}})
        winner, scores = ssm.simulate_match(mat_a, mat_b, best_of=1, seed=0)
        self.assertEqual(winner, "A")
        self.assertEqual(len(scores), 1)

    def test_main_runs_with_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            # simple matrices with immediate terminal results
            mat_df = pd.DataFrame({"": ["0-0"], "game_server": [1.0]})
            a_path = Path(tmp) / "player_A.csv"
            b_path = Path(tmp) / "player_B.csv"
            mat_df.to_csv(a_path, index=False)
            mat_df.to_csv(b_path, index=False)
            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                rc = ssm.main(
                    [
                        "--server_a_matrix",
                        str(a_path),
                        "--server_b_matrix",
                        str(b_path),
                        "--matches",
                        "2",
                        "--best_of",
                        "1",
                        "--seed",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("player A", out.getvalue())


if __name__ == "__main__":
    unittest.main()
