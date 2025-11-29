from __future__ import annotations
from dataclasses import dataclass, asdict, field
import re


class ParseError(ValueError):
    """Raised when a rally string cannot be parsed according to MatchChart rules."""


SERVE_DIRECTIONS = {
    "4": "out wide",
    "5": "body",
    "6": "down the T",
    "0": "unknown serve direction",
}

SHOT_DIRECTIONS = {
    "0": "unknown direction",
    "1": "to a right-handed forehand / left-handed backhand",
    "2": "down the middle",
    "3": "to a right-handed backhand / left-handed forehand",
}

RETURN_DEPTH = {
    "7": "return landed inside service boxes",
    "8": "return beyond service line but short of baseline",
    "9": "return near the baseline",
    "0": "return depth unknown",
}

SERVE_FAULT_TYPES = {
    "n": "net",
    "w": "wide",
    "d": "deep",
    "x": "wide and deep",
    "g": "foot fault",
    "!": "shank",
    "e": "unknown fault",
}

ERROR_TYPES = {
    "n": "net",
    "w": "wide",
    "d": "deep",
    "x": "wide and deep",
    "!": "shank",
    "e": "unknown",
}

SHOT_TYPES = {
    "f": "forehand groundstroke",
    "b": "backhand groundstroke",
    "r": "forehand slice",
    "s": "backhand slice",
    "v": "forehand volley",
    "z": "backhand volley",
    "o": "overhead smash",
    "p": "backhand overhead smash",
    "u": "forehand drop shot",
    "y": "backhand drop shot",
    "l": "forehand lob",
    "m": "backhand lob",
    "h": "forehand half-volley",
    "i": "backhand half-volley",
    "j": "forehand swinging volley",
    "k": "backhand swinging volley",
    "t": "trick shot",
    "q": "unknown shot",
}

MODIFIER_MAP = {
    "+": "approach / serve-and-volley attempt",
    "-": "taken near the net",
    "=": "taken near the baseline",
    ";": "clipped the net cord",
    "c": "clipped the net cord (old notation)",
    "^": "drop/stop volley",
}

FORCE_MAP = {"#": "forced error", "@": "unforced error"}

SPECIAL_POINT_CODES = {
    "S": "point awarded to server (missed point)",
    "R": "point awarded to returner (missed point)",
    "P": "point penalty against server",
    "Q": "point penalty against returner",
}

# Chars that occasionaly show up that we can safely ignore when parsing rally strings.
IGNORABLE_CHARS = {" ", "&", "_", "(", ")", "\t"}

# Rare cases where parsing should stop early because even with
# ignoring these chars, the rest of the rally would not make sense.
STOP_PARSING_CHARS = {"N"}


@dataclass
class Stroke:
    stroke_idx: int
    player_to_hit: int
    code: str
    shot_type: str
    direction_code: str | None = None
    direction: str | None = None
    depth_code: str | None = None
    depth: str | None = None
    modifiers: list[str] = field(default_factory=list)
    error: str | None = None
    outcome: str | None = None
    terminal: bool = False
    shot_family: str | None = None
    outcome_type: str | None = None
    point_winner: int | None = None


def parse_rally(rally_str: str, player1_starts: bool = True) -> list[dict]:
    """
    Parse a MatchChart point string (first- or second-serve column) into a list of shots.
    """
    parser = _RallyParser(rally_str, player1_starts)
    strokes = parser.parse()
    _annotate_strokes(strokes)
    return [asdict(stroke) for stroke in strokes]


# def parse_rally_events(rally_str: str, player1_starts: bool = True) -> list[Stroke]:
#     """
#     Parse a rally string but return Stroke instances for downstream processing.
#     """
#     parser = _RallyParser(rally_str, player1_starts)
#     strokes = parser.parse()
#     _annotate_strokes(strokes)
#     return strokes


# def parse_point(first_serve: str, second_serve: str, server_is_player1: bool = True) -> list[Stroke]:
#     """
#     Parse a full point using the first- and second-serve strings from MatchChartingProject data.

#     The second-serve string is only used when the first serve does not already contain
#     a terminal outcome and ends with a fault / lost first serve.
#     """
#     strokes: list[Stroke] = []
#     first = parse_rally_events(first_serve, player1_starts=server_is_player1)
#     strokes.extend(first)

#     needs_second = False
#     if not first:
#         needs_second = True
#     else:
#         last = first[-1]
#         needs_second = not last.terminal and (last.outcome in (None, "fault") or last.shot_type == "time violation")

#     if second_serve and needs_second:
#         second = parse_rally_events(second_serve, player1_starts=server_is_player1)
#         offset = len(strokes)
#         for idx, stroke in enumerate(second):
#             stroke.stroke_idx = offset + idx
#         strokes.extend(second)

#     _annotate_strokes(strokes)
#     return strokes


def _annotate_strokes(strokes: list[Stroke]) -> None:
    """Attach derived fields that downstream consumers need (family, outcome type, winner)."""
    for stroke in strokes:
        stroke.shot_family = _map_shot_family(stroke)
        stroke.outcome_type = _normalize_outcome(stroke)
        stroke.point_winner = _infer_point_winner(stroke)
        if stroke.direction_code is None:
            stroke.direction_code = "0"


def _map_shot_family(stroke: Stroke) -> str:
    code = (stroke.code or "").lower()
    if stroke.shot_type == "serve" or code == "serve":
        return "SERVE"
    if code in {"v", "z", "h", "i", "j", "k"}:
        return "VOLLEY"
    if code in {"f"}:
        return "FH"
    if code in {"b"}:
        return "BH"
    if code in {"r", "s"}:
        return "SLICE"
    if code in {"o", "p"}:
        return "SMASH"
    if code in {"u", "y", "l", "m"}:
        return "DROP_LOB"
    return "OTHER"


def _normalize_outcome(stroke: Stroke) -> str:
    if stroke.outcome is None:
        return "in_play"
    lowered = stroke.outcome.lower()
    if "ace" in lowered:
        return "ace"
    if "service winner" in lowered:
        return "service_winner"
    if lowered == "winner":
        return "winner"
    if "double fault" in lowered:
        return "double_fault"
    if lowered == "fault":
        return "fault"
    if lowered == "forced error":
        return "forced_error"
    if lowered == "unforced error":
        return "unforced_error"
    if lowered.startswith("point awarded"):
        return "administrative"
    if "challenge" in lowered:
        return "challenge_stop"
    return "other"


def _infer_point_winner(stroke: Stroke) -> int | None:
    if not stroke.terminal:
        return None
    outcome_type = _normalize_outcome(stroke)
    if outcome_type in {"winner", "service_winner", "ace"}:
        return stroke.player_to_hit
    if outcome_type in {"forced_error", "unforced_error", "double_fault"}:
        return 2 if stroke.player_to_hit == 1 else 1
    return None


class _RallyParser:
    def __init__(self, rally_str: str, player1_starts: bool) -> None:
        self.raw = (rally_str or "").strip()
        self.player = 1 if player1_starts else 2
        self.strokes: list[Stroke] = []
        self.idx = 0
        self.pos = 0

    def parse(self) -> list[Stroke]:
        if not self.raw:
            return []

        if self._looks_like_free_text():
            return [self._free_text_stroke()]

        if len(self.raw) == 1 and self.raw in SPECIAL_POINT_CODES:
            return [self._special_point_stroke(self.raw)]

        if self.raw == "V":
            return [self._time_violation()]

        need_serve = True
        while self.pos < len(self.raw):
            if self._skip_noise():
                continue

            if need_serve:
                result = self._parse_serve()
                if result == "stop":
                    break
                if result == "again":
                    continue
                need_serve = False
                continue

            shot = self._parse_shot()
            if shot.terminal:
                break

        return self.strokes

    def _special_point_stroke(self, code: str) -> Stroke:
        outcome = SPECIAL_POINT_CODES[code]
        return Stroke(
            stroke_idx=0,
            player_to_hit=self.player,
            code=code,
            shot_type="administrative decision",
            outcome=outcome,
            terminal=True,
        )

    def _free_text_stroke(self) -> Stroke:
        return Stroke(
            stroke_idx=0,
            player_to_hit=self.player,
            code="note",
            shot_type="unparsed note",
            outcome=self.raw,
            terminal=True,
        )

    def _time_violation(self) -> Stroke:
        return Stroke(
            stroke_idx=0,
            player_to_hit=self.player,
            code="V",
            shot_type="time violation",
            outcome="time violation – first serve lost",
            terminal=False,
        )

    def _parse_serve(self) -> None:
        lets = 0
        while self.pos < len(self.raw) and self._peek() == "c":
            lets += 1
            self.pos += 1

        if self.pos >= len(self.raw):
            return "stop"

        dir_code = self._peek()
        direction_code = dir_code if dir_code in SERVE_DIRECTIONS else "0"
        direction = SERVE_DIRECTIONS.get(dir_code, SERVE_DIRECTIONS["0"])

        if dir_code in {"*", "#"}:
            stroke = Stroke(
                stroke_idx=self.idx,
                player_to_hit=self.player,
                code="serve",
                shot_type="serve",
                direction_code="0",
                direction=SERVE_DIRECTIONS["0"],
            )
            self.idx += 1
            self.pos += 1
            stroke.outcome = "ace" if dir_code == "*" else "service winner (unreturnable)"
            stroke.terminal = True
            self.strokes.append(stroke)
            return "stop"

        if dir_code in SERVE_DIRECTIONS:
            self.pos += 1
        elif dir_code.isdigit():
            self.pos += 1
        else:
            lowered = dir_code.lower()
            if lowered in SERVE_FAULT_TYPES:
                stroke = Stroke(
                    stroke_idx=self.idx,
                    player_to_hit=self.player,
                    code="serve",
                    shot_type="serve",
                    direction_code="0",
                    direction=SERVE_DIRECTIONS["0"],
                    error=SERVE_FAULT_TYPES[lowered],
                    outcome="fault",
                )
                self.idx += 1
                self.pos += 1
                self._consume_modifiers(stroke, skip_chars={"c"})
                self._apply_fault_flag(stroke)
                self._consume_fault_suffix(stroke)
                self.strokes.append(stroke)
                if stroke.terminal:
                    return "stop"
                return "again"

            if dir_code.isalpha():
                stroke = Stroke(
                    stroke_idx=self.idx,
                    player_to_hit=self.player,
                    code="serve",
                    shot_type="serve",
                    direction_code="0",
                    direction=SERVE_DIRECTIONS["0"],
                )
                self.idx += 1
                self.strokes.append(stroke)
                self._toggle_player()
                return "rally"

            stroke = Stroke(
                stroke_idx=self.idx,
                player_to_hit=self.player,
                code="serve",
                shot_type="serve",
                direction_code="0",
                direction=SERVE_DIRECTIONS["0"],
                modifiers=[f"unknown serve marker '{dir_code}'"],
            )
            self.idx += 1
            self.pos += 1
            self._consume_modifiers(stroke, skip_chars={"c"})
            self.strokes.append(stroke)
            self._toggle_player()
            return "rally"

        stroke = Stroke(
            stroke_idx=self.idx,
            player_to_hit=self.player,
            code="serve",
            shot_type="serve",
            direction_code=direction_code,
            direction=direction,
        )
        self.idx += 1

        if lets:
            stroke.modifiers.append(f"{lets} let{'s' if lets > 1 else ''} before serve")

        self._consume_modifiers(stroke, skip_chars={"c"})

        while self.pos < len(self.raw) and self._peek() == "c":
            stroke.modifiers.append("let serve")
            self.pos += 1

        if self.pos >= len(self.raw):
            self.strokes.append(stroke)
            return

        if self.pos >= len(self.raw):
            raise ParseError("Sequence ended unexpectedly while parsing rally shots.")
        if self.pos >= len(self.raw):
            raise ParseError("Sequence ended unexpectedly while parsing rally shots.")
        ch = self._peek()
        lowered_fault = ch.lower()
        if lowered_fault in SERVE_FAULT_TYPES:
            stroke.error = SERVE_FAULT_TYPES[lowered_fault]
            stroke.outcome = "fault"
            self.pos += 1
            self._consume_modifiers(stroke)
            self._apply_fault_flag(stroke)
            self._consume_fault_suffix(stroke)
            self.strokes.append(stroke)
            if stroke.terminal:
                return "stop"
            return "again"

        if ch == "*":
            stroke.outcome = "ace"
            stroke.terminal = True
            self.pos += 1
            self.strokes.append(stroke)
            return "stop"

        if ch == "#":
            stroke.outcome = "service winner (unreturnable)"
            stroke.terminal = True
            self.pos += 1
            self.strokes.append(stroke)
            return "stop"

        if ch == "C":
            stroke.outcome = "stopped for unsuccessful challenge"
            stroke.terminal = True
            self.pos += 1
            self.strokes.append(stroke)
            return "stop"

        self.strokes.append(stroke)
        self._toggle_player()
        return "rally"

    def _parse_shot(self) -> Stroke:
        prefix_digits: list[str] = []
        while self.pos < len(self.raw) and self._peek().isdigit():
            prefix_digits.append(self._peek())
            self.pos += 1

        if self.pos >= len(self.raw):
            raise ParseError("Sequence ended unexpectedly while parsing rally shots.")

        ch = self._peek()
        if ch in STOP_PARSING_CHARS:
            self.pos = len(self.raw)
            raise ParseError("Unexpected early stop marker while parsing shots.")

        stroke_code: str | None = None
        stroke_type: str | None = None

        if ch.isalpha():
            lower = ch.lower()
            stroke_code = lower
            stroke_type = SHOT_TYPES.get(lower, "unknown shot")
            self.pos += 1
        else:
            if ch in ("*", "#", "@", "C"):
                stroke = Stroke(
                    stroke_idx=self.idx,
                    player_to_hit=self.player,
                    code="?",
                    shot_type="unknown shot",
                )
                if prefix_digits:
                    stroke.modifiers.append(f"legacy numeric prefix {''.join(prefix_digits)}")
                self.idx += 1
                if ch in FORCE_MAP:
                    stroke.outcome = FORCE_MAP[ch]
                elif ch == "*":
                    stroke.outcome = "winner"
                elif ch == "C":
                    stroke.outcome = "stopped for unsuccessful challenge"
                stroke.terminal = True
                self.pos += 1
                self.strokes.append(stroke)
                self._toggle_player()
                return stroke

            if prefix_digits:
                stroke = Stroke(
                    stroke_idx=self.idx,
                    player_to_hit=self.player,
                    code="?",
                    shot_type="unknown shot",
                    modifiers=[f"legacy numeric prefix {''.join(prefix_digits)}"],
                )
                stroke.modifiers.append(f"unknown leading character '{ch}'")
                self.idx += 1
                self.pos += 1
                self.strokes.append(stroke)
                self._toggle_player()
                return stroke

            raise ParseError(f"Unexpected character '{ch}' while parsing rally shots.")

        stroke = Stroke(
            stroke_idx=self.idx,
            player_to_hit=self.player,
            code=stroke_code,
            shot_type=stroke_type,
        )
        self.idx += 1

        if prefix_digits:
            stroke.modifiers.append(f"legacy numeric prefix {''.join(prefix_digits)}")

        direction_set = False
        depth_set = False

        while self.pos < len(self.raw):
            if self._skip_noise():
                break

            if self.pos >= len(self.raw):
                break

            ch = self._peek()

            if ch in STOP_PARSING_CHARS:
                self.pos = len(self.raw)
                break

            if ch in MODIFIER_MAP:
                stroke.modifiers.append(MODIFIER_MAP[ch])
                self.pos += 1
                continue

            if ch.isdigit():
                if not direction_set and ch in SHOT_DIRECTIONS:
                    stroke.direction_code = ch
                    stroke.direction = SHOT_DIRECTIONS[ch]
                    direction_set = True
                    self.pos += 1
                    continue

                if not depth_set and ch in RETURN_DEPTH:
                    stroke.depth_code = ch
                    stroke.depth = RETURN_DEPTH[ch]
                    depth_set = True
                    self.pos += 1
                    continue

                extra_digits = []
                while self.pos < len(self.raw) and self._peek().isdigit():
                    extra_digits.append(self._peek())
                    self.pos += 1
                if extra_digits:
                    stroke.modifiers.append(f"legacy numeric tag {''.join(extra_digits)}")
                    continue
                break

            lowered = ch.lower()
            if lowered in ERROR_TYPES:
                stroke.error = ERROR_TYPES[lowered]
                self.pos += 1
                continue

            if ch in FORCE_MAP:
                stroke.outcome = FORCE_MAP[ch]
                stroke.terminal = True
                self.pos += 1
                break

            if ch == "*":
                stroke.outcome = "winner"
                stroke.terminal = True
                self.pos += 1
                break

            if ch == "C":
                stroke.outcome = "stopped for unsuccessful challenge"
                stroke.terminal = True
                self.pos += 1
                break

            break

        self.strokes.append(stroke)
        self._toggle_player()
        return stroke

    def _consume_modifiers(self, stroke: Stroke, skip_chars: set[str] | None = None) -> None:
        while self.pos < len(self.raw):
            ch = self._peek()
            if skip_chars and ch in skip_chars:
                return
            if ch in MODIFIER_MAP:
                stroke.modifiers.append(MODIFIER_MAP[ch])
                self.pos += 1
                continue
            break

    def _apply_fault_flag(self, stroke: Stroke) -> None:
        if self.pos < len(self.raw):
            ch = self._peek()
            if ch in FORCE_MAP:
                if ch == "@":
                    stroke.outcome = "double fault (unforced)"
                elif ch == "#":
                    stroke.outcome = "double fault (forced)" # If you see this there was a charting mistake
                else:
                    stroke.outcome = FORCE_MAP[ch]
                stroke.terminal = True
                self.pos += 1

    def _consume_fault_suffix(self, stroke: Stroke) -> None:
        consumed_digit = False
        while self.pos < len(self.raw):
            ch = self._peek()
            lowered = ch.lower()
            if not consumed_digit and lowered in SERVE_FAULT_TYPES:
                stroke.modifiers.append(f"extra fault marker {lowered}")
                self.pos += 1
                continue
            if ch.isdigit():
                consumed_digit = True
                extras = []
                while self.pos < len(self.raw) and self._peek().isdigit():
                    extras.append(self._peek())
                    self.pos += 1
                if extras:
                    stroke.modifiers.append(f"legacy numeric tag {''.join(extras)}")
                    continue
            break

    def _toggle_player(self) -> None:
        self.player = 2 if self.player == 1 else 1

    def _skip_noise(self) -> bool:
        while self.pos < len(self.raw):
            ch = self._peek()
            if ch in STOP_PARSING_CHARS:
                self.pos = len(self.raw)
                return True
            if ch in IGNORABLE_CHARS:
                self.pos += 1
                continue
            break
        return False

    def _peek(self) -> str:
        return self.raw[self.pos]

    def _looks_like_free_text(self) -> bool:
        if not self.raw:
            return False
        if not any(ch.isspace() for ch in self.raw):
            return False
        long_words = re.findall(r"[A-Za-z]{2,}", self.raw)
        return bool(long_words)
