import csv
from pathlib import Path
import unittest
from data_prep.data_parser import ParseError, parse_point, parse_rally


DATA_DIR = Path("raw_data")
POINT_FILES = [
    "charting-m-points-to-2009.csv",
    "charting-m-points-2010s.csv",
    "charting-m-points-2020s.csv",
    "charting-w-points-to-2009.csv",
    "charting-w-points-2010s.csv",
    "charting-w-points-2020s.csv",
]


class ParseRallyUnitTests(unittest.TestCase):
    def test_parse_unreturnable_serve(self):
        shots = parse_rally("6#")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertEqual(serve["code"], "serve")
        self.assertEqual(serve["shot_type"], "serve")
        self.assertEqual(serve["direction_code"], "6")
        self.assertEqual(serve["direction"], "down the T")
        self.assertIsNone(serve["depth_code"])
        self.assertIsNone(serve["depth"])
        self.assertEqual(serve["outcome"], "service winner (unreturnable)")
        self.assertTrue(serve["terminal"])

    def test_parse_rally_sequence_with_return_depth(self):
        shots = parse_rally("4b27f3n@")
        self.assertEqual(len(shots), 3)
        serve, ret, final = shots
        self.assertEqual(serve["shot_type"], "serve")
        self.assertTrue(ret["shot_type"].startswith("backhand"))
        self.assertEqual(ret["direction_code"], "2")
        self.assertEqual(ret["depth_code"], "7")
        self.assertEqual(ret["depth"], "return landed inside service boxes")
        self.assertEqual(final["outcome"], "unforced error")
        self.assertEqual(final["error"], "net")
        self.assertTrue(final["terminal"])

    def test_parse_return_depth_and_winner(self):
        shots = parse_rally("4+b27v1*")
        self.assertEqual(len(shots), 3)
        serve, ret, finisher = shots
        self.assertTrue(any("serve-and-volley" in mod for mod in serve["modifiers"]))
        self.assertEqual(ret["depth_code"], "7")
        self.assertEqual(ret["depth"], "return landed inside service boxes")
        self.assertEqual(finisher["shot_type"], "forehand volley")
        self.assertEqual(finisher["outcome"], "winner")

    def test_parse_fault_with_let_and_metadata_stop(self):
        shots = parse_rally("c6e+N43")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertIn("1 let before serve", serve["modifiers"])
        self.assertEqual(serve["error"], "unknown fault")
        self.assertIn("serve-and-volley", serve["modifiers"][1])
        self.assertEqual(serve["outcome"], "fault")
        self.assertFalse(serve["terminal"])

    def test_special_events_and_time_violation(self):
        awarded = parse_rally("S")
        self.assertTrue(awarded[0]["outcome"].startswith("point awarded"))
        self.assertTrue(awarded[0]["terminal"])

        awarded = parse_rally("R")
        self.assertTrue(awarded[0]["outcome"].startswith("point awarded"))
        self.assertTrue(awarded[0]["terminal"])

        tv = parse_rally("V")
        self.assertEqual(tv[0]["shot_type"], "time violation")
        self.assertFalse(tv[0]["terminal"])

        penalty = parse_rally("P")
        self.assertEqual(penalty[0]["shot_type"], "administrative decision")
        self.assertTrue(penalty[0]["terminal"])

        penalty = parse_rally("Q")
        self.assertEqual(penalty[0]["shot_type"], "administrative decision")
        self.assertTrue(penalty[0]["terminal"])

    def test_invalid_codes_raise(self):
        with self.assertRaises(ParseError):
            parse_rally("$$$")
            parse_rally("4f3z")
            parse_rally("!@#")

    def test_missing_serve_direction_creates_unknown_serve(self):
        shots = parse_rally("u2n#")
        self.assertEqual(len(shots), 2)
        serve, rally = shots
        self.assertEqual(serve["direction_code"], "0")
        self.assertEqual(rally["code"], "u")

    def test_fault_without_direction_letter_only(self):
        shots = parse_rally("e")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertEqual(serve["error"], "unknown fault")
        self.assertEqual(serve["direction_code"], "0")

    def test_extra_numeric_digits_preserved(self):
        shots = parse_rally("4f329f1*")
        self.assertEqual(len(shots), 3)
        _, mid, finisher = shots
        self.assertEqual(mid["code"], "f")
        self.assertEqual(mid["direction_code"], "3")
        self.assertIsNone(mid["depth_code"])
        self.assertTrue(any(tag.startswith("legacy numeric tag") for tag in mid["modifiers"]))
        self.assertEqual(finisher["outcome"], "winner")
        self.assertEqual(finisher["shot_type"], "forehand groundstroke")
        self.assertEqual(finisher["direction_code"], "1")
        self.assertTrue(finisher["terminal"])

    def test_double_fault_flags_and_trailing_markers(self):
        shots = parse_rally("6d@")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertEqual(serve["outcome"], "double fault (unforced)")
        self.assertTrue(serve["terminal"])

    def test_trailing_let_after_direction(self):
        shots = parse_rally("5c*")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertTrue(any("let serve" in mod for mod in serve["modifiers"]))
        self.assertEqual(serve["outcome"], "ace")

    def test_digit_direction_falls_back_to_unknown(self):
        shots = parse_rally("9d")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertEqual(serve["direction_code"], "0")

    def test_sentence_returns_note(self):
        shots = parse_rally("Play suspended- resumed next day at 550 PM")
        self.assertEqual(len(shots), 1)
        note = shots[0]
        self.assertEqual(note["shot_type"], "unparsed note")
        self.assertTrue(note["terminal"])

    def test_numeric_prefix_before_shot(self):
        shots = parse_rally("438b+1n@")
        self.assertEqual(len(shots), 2)
        serve, rally = shots
        self.assertEqual(serve["shot_type"], "serve")
        self.assertTrue(any(mod.startswith("legacy numeric prefix 38") for mod in rally["modifiers"]))
        self.assertEqual(rally["outcome"], "unforced error")
        self.assertTrue(rally["terminal"])

    def test_unknown_serve_marker_treated_as_unknown_direction(self):
        shots = parse_rally("?s39b*")
        self.assertEqual(shots[0]["direction_code"], "0")
        self.assertEqual(shots[-1]["outcome"], "winner")

    def test_ace_without_direction_code(self):
        shots = parse_rally("&*")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertEqual(serve["outcome"], "ace")
        self.assertTrue(serve["terminal"])

    def test_net_cord_modifier_with_c(self):
        shots = parse_rally("5+f3c@")
        self.assertEqual(len(shots), 2)
        rally = shots[1]
        self.assertIn("clipped the net cord (old notation)", rally["modifiers"])
        self.assertEqual(rally["outcome"], "unforced error")

        shots = parse_rally("5+f3;@")
        rally = shots[1]
        self.assertIn("clipped the net cord", rally["modifiers"])
        self.assertEqual(rally["outcome"], "unforced error")
    
    def test_own_charted_rallies(self):
        shots = parse_rally("c6f!3x#")
        self.assertEqual(len(shots), 2)
        serve, rally = shots
        self.assertIn("1 let before serve", serve["modifiers"])
        self.assertEqual(rally["outcome"], "forced error") 

    def test_stop_marker_halts_parsing(self):
        shots = parse_rally("4f3Nf1")
        self.assertEqual(len(shots), 2)  # serve + first rally shot only
        self.assertEqual(shots[-1]["code"], "f")
        self.assertFalse(shots[-1]["terminal"])

    def test_challenge_stops_point(self):
        serve_only = parse_rally("6C")
        self.assertTrue(serve_only[0]["terminal"])
        self.assertIn("challenge", serve_only[0]["outcome"])

        rally_challenge = parse_rally("4f2C")
        self.assertEqual(len(rally_challenge), 2)
        self.assertTrue(rally_challenge[-1]["terminal"])
        self.assertIn("challenge", rally_challenge[-1]["outcome"])

    def test_multiple_lets_before_serve(self):
        shots = parse_rally("cc6f3")
        self.assertEqual(shots[0]["shot_type"], "serve")
        self.assertIn("2 lets before serve", shots[0]["modifiers"])
        self.assertEqual(shots[1]["code"], "f")
        self.assertEqual(shots[1]["direction_code"], "3")
    
    # Don't know if it should be treated as returner shank or extra fault marker yet
    def test_fault_suffix_markers_and_digits(self):
        shots = parse_rally("6d!2")
        self.assertEqual(len(shots), 1)
        serve = shots[0]
        self.assertIn("extra fault marker !", serve["modifiers"])
        self.assertTrue(any(tag.endswith("2") for tag in serve["modifiers"]))
        self.assertEqual(serve["outcome"], "fault")
        self.assertFalse(serve["terminal"])

    def test_chained_fault_digits_preserved(self):
        shots = parse_rally("!3x#")
        self.assertTrue(any("legacy numeric tag 3" in m for m in shots[0]["modifiers"]))
        self.assertTrue(shots[-1]["terminal"])


class ParseRallyDatasetSmokeTests(unittest.TestCase):
    """Dataset-driven smoke tests to ensure no crashes on real CSV rows."""

    MAX_ROWS_PER_FILE = 400  # keep tests fast while covering diverse codes

    def test_first_rows_of_point_files_parse(self):
        failures = []
        for filename in POINT_FILES:
            path = DATA_DIR / filename
            if not path.exists():
                continue
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=2):
                    if idx - 1 > self.MAX_ROWS_PER_FILE:
                        break
                    server = (row.get("Svr") or "").strip()
                    server_one_starts = server != "2"
                    for col in ("1st", "2nd"):
                        rally = (row.get(col) or "").strip()
                        if not rally:
                            continue
                        try:
                            parse_rally(rally, player1_starts=server_one_starts)
                        except ParseError as exc:
                            failures.append((filename, idx, col, rally, str(exc)))
                            if len(failures) >= 5:
                                break
                    if failures:
                        break
            if failures:
                break

        self.assertFalse(
            failures,
            msg="\n".join(
                f"{fname}:{line} [{col}] '{rally}' -> {err}" for fname, line, col, rally, err in failures
            ),
        )


class ParsePointIntegrationTests(unittest.TestCase):
    def test_first_serve_ace_skips_second(self):
        strokes = parse_point(first_serve="6*", second_serve="4b3", server_is_player1=True)
        self.assertEqual(len(strokes), 1)
        self.assertEqual(strokes[0].shot_type, "serve")
        self.assertEqual(strokes[0].outcome, "ace")
        self.assertTrue(strokes[0].terminal)

    def test_first_fault_uses_second_and_offsets_indices(self):
        strokes = parse_point(first_serve="6d", second_serve="4f2n@", server_is_player1=True)
        self.assertEqual(len(strokes), 3)  # first serve fault + second serve + rally finisher
        self.assertEqual([s.stroke_idx for s in strokes], [0, 1, 2])
        self.assertEqual(strokes[0].outcome, "fault")
        self.assertEqual(strokes[1].shot_type, "serve")
        self.assertTrue(strokes[-1].terminal)

    def test_first_fault_second_fault_stops_point(self):
        strokes = parse_point(first_serve="6d", second_serve="4d@", server_is_player1=True)
        self.assertEqual(len(strokes), 2)
        self.assertEqual(strokes[0].outcome, "fault")
        self.assertEqual(strokes[1].outcome, "double fault (unforced)")
        self.assertTrue(strokes[1].terminal)

    def test_empty_first_serve_uses_second_only(self):
        strokes = parse_point(first_serve="", second_serve="4f3", server_is_player1=False)
        self.assertEqual(len(strokes), 2)  # serve + return
        self.assertEqual(strokes[0].shot_type, "serve")
        self.assertEqual(strokes[0].player_to_hit, 2)  # server is player 2 in this case
        self.assertFalse(strokes[-1].terminal)

    def test_time_violation_then_second_serve(self):
        strokes = parse_point(first_serve="V", second_serve="4f2", server_is_player1=True)
        self.assertEqual(strokes[0].shot_type, "time violation")
        self.assertFalse(strokes[0].terminal)
        self.assertEqual(strokes[1].shot_type, "serve")


if __name__ == "__main__":
    unittest.main()
