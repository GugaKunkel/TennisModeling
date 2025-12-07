import unittest

from score_model.set_match import compute_set_outcome


class TestComputeSetOutcome(unittest.TestCase):
    def test_stronger_server_increases_set_win_prob(self):
        # A holds 54.5%, B holds 24.9% (from example matrices)
        outcome = compute_set_outcome(a_hold=0.545, b_hold=0.249, server_first="A")
        self.assertGreater(outcome.p_win, 0.5)

        # If A is weaker than B, win probability should drop below 0.5
        weaker_outcome = compute_set_outcome(a_hold=0.545, b_hold=0.70, server_first="A")
        self.assertLess(weaker_outcome.p_win, 0.5)

    def test_server_first_does_not_flip_fair_match(self):
        # Symmetric holds -> fair set regardless of who serves first
        even_a = compute_set_outcome(a_hold=0.7, b_hold=0.7, server_first="A")
        even_b = compute_set_outcome(a_hold=0.7, b_hold=0.7, server_first="B")
        self.assertAlmostEqual(even_a.p_win, 0.5, places=6)
        self.assertAlmostEqual(even_b.p_win, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
