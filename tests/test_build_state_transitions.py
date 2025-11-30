import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_state_transitions import (
    DEFAULT_STATE_FIELDS,
    advance_point_score,
    canonical_point_score,
    collect_input_files,
    load_player_map,
    process_file,
    serialize_state,
    transitions_for_point,
)


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
        default_state = serialize_state(DEFAULT_STATE_FIELDS, ctx)
        self.assertEqual(default_state, "serve_side|BOS|dir=0|rbin=0|score=0-0|start_srv=1st")
        with_player = serialize_state(
            ["player_label", "server_flag", "prev_family", "prev_direction"], ctx
        )
        self.assertEqual(with_player, "P1|serve_side|BOS|dir=0")

    def test_score_helpers(self):
        self.assertEqual(canonical_point_score("15-30", True), "15-30")
        self.assertEqual(canonical_point_score("0-15", False), "15-0")
        self.assertEqual(canonical_point_score("7-6", True), "TB_7-6")
        self.assertEqual(advance_point_score("40-0", True), "game_server")
        self.assertEqual(advance_point_score("40-30", False), "40-40")
        self.assertEqual(advance_point_score("Ad-Out", True), "40-40")

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
            mapping = load_player_map([base])
            self.assertEqual(mapping["mA"], ("Alice", "Bob"))
            self.assertEqual(mapping["mB"], ("Cara", "Dan"))

    def test_collect_input_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "a.xlsm").write_text("")
            (base / "b.xlsx").write_text("")
            (base / "ignore.txt").write_text("")
            files = collect_input_files([str(base)])
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
        rows = transitions_for_point(
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
        # Build a tiny points CSV on disk
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
            rows = process_file(points_path, DEFAULT_STATE_FIELDS, player_map)
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
        rows = transitions_for_point(
            row=row,
            match_id="m2",
            player1="Alice",
            player2="Bob",
            state_fields=DEFAULT_STATE_FIELDS,
        )
        # Expect fault + second serve + rally finisher
        self.assertEqual(len(rows), 3)
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
        rows = transitions_for_point(
            bad_row, match_id="m3", player1="X", player2="Y", state_fields=DEFAULT_STATE_FIELDS
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
        rows = transitions_for_point(
            row, match_id="m4", player1="P1", player2="P2", state_fields=DEFAULT_STATE_FIELDS
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["is_terminal"])
        self.assertFalse(rows[1]["is_terminal"])


if __name__ == "__main__":
    unittest.main()
