import tempfile
import unittest
from pathlib import Path
import pandas as pd
import build_state_transitions as bst


class BuildStateTransitionsUnitTests(unittest.TestCase):
    def test_serialize_state_defaults_and_player_field(self):
        ctx = {
            "player_label": "P1",
            "server_flag": "serve_side",
            "prev_family": "BOS",
            "prev_direction": "0",
            "rally_bin": "0",
            "rally_index": 0,
            "point_score": "0-0",
            "point_start_serve": "1st",
        }
        default_state = bst.serialize_state(bst.DEFAULT_STATE_FIELDS, ctx)
        self.assertEqual(default_state, "serve_side|BOS|dir=0|rbin=0|score=0-0|start_srv=1st")
        with_player = bst.serialize_state(
            ["player_label", "server_flag", "prev_family", "prev_direction"], ctx
        )
        self.assertEqual(with_player, "P1|serve_side|BOS|dir=0")

    def test_score_helpers(self):
        self.assertEqual(bst.canonical_point_score("15-30", True), "15-30")
        self.assertEqual(bst.canonical_point_score("0-15", False), "15-0")
        self.assertEqual(bst.canonical_point_score("7-6", True), "TB_7-6")
        self.assertEqual(bst.advance_point_score("40-0", True), "game_server")
        self.assertEqual(bst.advance_point_score("40-30", False), "40-40")
        self.assertEqual(bst.advance_point_score("Ad-Out", True), "40-40")

    def test_load_player_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            df = pd.DataFrame(
                [
                    {"match_id": "mA", "Player 1": "Alice", "Player 2": "Bob"},
                    {"match_id": "mB", "Player 1": "Cara", "Player 2": "Dan"},
                ]
            )
            path = base / "charting-m-matches.csv"
            df.to_csv(path, index=False)
            mapping = bst.load_player_map([base])
            self.assertEqual(mapping["mA"], ("Alice", "Bob"))
            self.assertEqual(mapping["mB"], ("Cara", "Dan"))

    def test_collect_input_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "a.xlsm").write_text("")
            (base / "b.xlsx").write_text("")
            (base / "ignore.txt").write_text("")
            files = bst.collect_input_files([str(base)])
            names = sorted(f.name for f in files)
            self.assertEqual(names, ["a.xlsm", "b.xlsx"])

    def test_transitions_for_point_serializes_state_fields(self):
        row = pd.Series(
            {
                "match_id": "m1",
                "Pt": 1,
                "Pts": "0-0",
                "Svr": 1,
                "1st": "6#",
                "2nd": "",
            }
        )
        state_fields = ["player_label", "server_flag", "prev_family", "prev_direction"]
        rows = bst.transitions_for_point(
            row=row,
            match_id="m1",
            player1="Alice",
            player2="Bob",
            state_fields=state_fields,
        )
        self.assertEqual(len(rows), 1)
        r0 = rows[0]
        self.assertEqual(r0["player_name"], "Alice")
        self.assertTrue(r0["is_terminal"])
        self.assertEqual(r0["state"], "P1|serve_side|BOS|dir=0")
        self.assertEqual(r0["new_state"].split("|")[0], "P2")  # next player label present

    def test_process_file_uses_player_map(self):
        points_df = pd.DataFrame(
            [
                {
                    "match_id": "match-1",
                    "Pt": 1,
                    "Pts": "0-0",
                    "Svr": 1,
                    "1st": "6#",
                    "2nd": "",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            points_path = Path(tmpdir) / "points.csv"
            points_df.to_csv(points_path, index=False)
            player_map = {"match-1": ("Player One", "Player Two")}
            rows = bst.process_file(points_path, bst.DEFAULT_STATE_FIELDS, player_map)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["player_name"], "Player One")
            self.assertEqual(rows[0]["player_label"], "P1")
            self.assertEqual(rows[0]["state"], "serve_side|BOS|dir=0|rbin=0|score=0-0|start_srv=1st")
            self.assertTrue(rows[0]["is_terminal"])

    def test_second_serve_after_fault(self):
        row = pd.Series(
            {
                "match_id": "m2",
                "Pt": 1,
                "Pts": "0-0",
                "Svr": 1,
                "1st": "6d",
                "2nd": "4f2n@",
            }
        )
        rows = bst.transitions_for_point(
            row=row,
            match_id="m2",
            player1="Alice",
            player2="Bob",
            state_fields=bst.DEFAULT_STATE_FIELDS,
        )
        self.assertEqual(len(rows), 3)  # fault + second serve + finisher
        self.assertEqual(rows[0]["outcome_raw"], "fault")
        self.assertEqual(rows[1]["shot_type"], "serve")
        self.assertTrue(rows[-1]["is_terminal"])

    def test_transitions_handle_parse_error(self):
        bad_row = pd.Series(
            {
                "match_id": "m3",
                "Pt": 1,
                "Pts": "0-0",
                "Svr": 1,
                "1st": "$$$",  # invalid rally string
                "2nd": "",
            }
        )
        rows = bst.transitions_for_point(
            bad_row, match_id="m3", player1="X", player2="Y", state_fields=bst.DEFAULT_STATE_FIELDS
        )
        self.assertEqual(rows, [])

    def test_terminal_and_non_terminal_transitions(self):
        row = pd.Series(
            {
                "match_id": "m4",
                "Pt": 1,
                "Pts": "0-0",
                "Svr": 1,
                "1st": "6f3",
                "2nd": "",
            }
        )
        rows = bst.transitions_for_point(
            row, match_id="m4", player1="P1", player2="P2", state_fields=bst.DEFAULT_STATE_FIELDS
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["is_terminal"])
        self.assertFalse(rows[1]["is_terminal"])

class BuildStateTransitionsDataTests(unittest.TestCase):
    def test_server_assignment_for_specific_match(self):
        match_id = "20250610-M-ITF_Martos-Q2-Preston_Stearns-Alejandro_Lopez_Escribano"
        points_path = Path("data/charting-m-points-2020s.csv")
        matches_path = Path("data/charting-m-matches.csv")

        if not points_path.exists() or not matches_path.exists():
            self.skipTest("Required charting files not available.")

        points_df = pd.read_csv(points_path)
        matches_df = pd.read_csv(matches_path)

        row = points_df[points_df["match_id"] == match_id].iloc[0]
        match_row = matches_df[matches_df["match_id"] == match_id].iloc[0]

        player1 = str(match_row["Player 1"])
        player2 = str(match_row["Player 2"])

        transitions = bst.transitions_for_point(
            row=row,
            match_id=match_id,
            player1=player1,
            player2=player2,
            state_fields=bst.DEFAULT_STATE_FIELDS,
        )

        self.assertTrue(transitions, "Expected transitions for the point.")
        first = transitions[0]
        self.assertEqual(first["player_label"], "P1")
        self.assertEqual(first["player_name"], "Preston Stearns")
        self.assertEqual(first["server_flag"], "serve_side")
        self.assertEqual(first["point_score"], "0-0")
        self.assertIn("score=0-0", first["state"])


if __name__ == "__main__":
    unittest.main()
