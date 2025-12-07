import unittest

import numpy as np

from score_model.game_absorption import game_win_probability
from score_model.predictor import TransitionMatrix


def make_simple_matrix(p_server_wins: float) -> TransitionMatrix:
    states = ["0-0", "game_server", "game_returner"]
    matrix = np.array(
        [
            [0.0, p_server_wins, 1 - p_server_wins],  # from 0-0 go directly to terminal
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return TransitionMatrix(
        states=states,
        index_for_state={s: i for i, s in enumerate(states)},
        matrix=matrix,
        absorbing={1, 2},
        transient={0},
    )


class TestGameAbsorptionUnbinned(unittest.TestCase):
    def test_unbinned_game_win_probability(self):
        tm = make_simple_matrix(0.6)
        res = game_win_probability(tm)
        self.assertAlmostEqual(res.p_server_wins, 0.6)
        self.assertIsNone(res.used_bin)
        self.assertEqual(res.opp_bin, "unbinned")


if __name__ == "__main__":
    unittest.main()
