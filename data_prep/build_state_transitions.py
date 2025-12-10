import argparse
import sys
from pathlib import Path
import pandas as pd
from data_prep.data_parser import ParseError, parse_point

DEFAULT_STATE_FIELDS = [
    "server_flag",
    "prev_family",
    "prev_direction",
    "rally_bin",
    "point_score",
    "point_start_serve",
]

def collect_input_files(paths: list[str]) -> list[Path]:
    """Expand a mix of files/dirs into the list of point tables to process."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.xlsm")))
            files.extend(sorted(path.glob("*.xlsx")))
        elif path.suffix.lower() in {".xlsm", ".xlsx", ".csv"}:
            files.append(path)
    return files


def load_player_map(base_dirs: list[Path]) -> dict[str, tuple[str, str]]:
    """Load match_id -> (Player1, Player2) from match CSVs if present."""
    mapping: dict[str, tuple[str, str]] = {}
    match_files = ["charting-m-matches.csv", "charting-w-matches.csv"]
    for base in base_dirs:
        for name in match_files:
            path = base / name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["match_id", "Player 1", "Player 2"])
            except ValueError:
                continue
            for _, row in df.iterrows():
                mid = str(row.get("match_id", "")).strip()
                if not mid or mid in mapping:
                    continue
                p1 = str(row.get("Player 1", "")).strip()
                p2 = str(row.get("Player 2", "")).strip()
                mapping[mid] = (p1, p2)
    return mapping


def load_points_table(path: Path) -> pd.DataFrame:
    """Load the MCP points sheet (CSV or Points tab in XLSX/XLSM)."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    try:
        return pd.read_excel(path, sheet_name="Points", engine="openpyxl")
    except ValueError:
        # Fall back to the first sheet that looks like the points tab.
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for name, df in sheets.items():
            if "point" in name.lower():
                return df
        # Otherwise just return the first sheet.
        return next(iter(sheets.values()))


def _coerce_bool(val) -> bool:
    """Robust bool coercion used on MCP columns."""
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y"}


def canonical_point_score(raw_score: str | float | None, is_tiebreak: bool = False) -> str:
    """Normalize raw Pts strings, assuming score is listed server-first in the source data."""
    if raw_score is None or (isinstance(raw_score, float) and pd.isna(raw_score)):
        return "TB_0-0" if is_tiebreak else "0-0"

    score_str = str(raw_score).strip()
    if not score_str:
        return "TB_0-0" if is_tiebreak else "0-0"

    parts = score_str.split("-")
    if len(parts) != 2:
        return score_str

    left, right = parts[0].strip(), parts[1].strip()
    normal_tokens = {"0", "15", "30", "40", "AD", "A"}
    left_up, right_up = left.upper(), right.upper()

    if is_tiebreak or left_up not in normal_tokens or right_up not in normal_tokens:
        try:
            l_val = int(float(left))
            r_val = int(float(right))
        except ValueError:
            return score_str
        return f"TB_{l_val}-{r_val}"

    server_token = left_up
    return_token = right_up

    if server_token in {"AD", "A"}:
        return "Ad-In"
    if return_token in {"AD", "A"}:
        return "Ad-Out"

    return f"{server_token}-{return_token}"

def _parse_tb_score(score: str) -> tuple[int, int] | None:
    if not isinstance(score, str) or not score.startswith("TB_"):
        return None
    try:
        raw = score.replace("TB_", "")
        left, right = raw.split("-")
        return int(left), int(right)
    except Exception:
        return None

def _tb_initial_server_is_p1(current_server_is_p1: bool, point_index: int) -> bool:
    """Infer initial tiebreak server given current server at point_index."""
    if point_index == 0:
        return current_server_is_p1
    block = (point_index - 1) // 2
    return current_server_is_p1 if block % 2 == 1 else not current_server_is_p1

def _tb_server_for_index(initial_server_is_p1: bool, point_index: int) -> bool:
    """Server (is P1) for a tiebreak point index (0-based)."""
    if point_index == 0:
        return initial_server_is_p1
    block = (point_index - 1) // 2
    return initial_server_is_p1 if block % 2 == 1 else not initial_server_is_p1

def advance_point_score(score: str, winner_is_server: bool) -> str:
    """Return the server-first score after awarding a point to the winner."""
    if score.startswith("TB_"):
        raw = score.replace("TB_", "")
        try:
            server_pts_str, ret_pts_str = raw.split("-")
            server_pts = int(server_pts_str)
            ret_pts = int(ret_pts_str)
        except Exception:
            return score
        if winner_is_server:
            server_pts += 1
        else:
            ret_pts += 1
        return f"TB_{server_pts}-{ret_pts}"

    if score == "Ad-In":
        return "game_server" if winner_is_server else "40-40"
    if score == "Ad-Out":
        return "40-40" if winner_is_server else "game_returner"

    try:
        server_raw, ret_raw = score.split("-")
    except ValueError:
        return score

    def next_token(token: str) -> str:
        order = ["0", "15", "30", "40", "Ad"]
        try:
            idx = order.index(token)
        except ValueError:
            return token
        return "Ad" if idx == len(order) - 2 else order[min(idx + 1, len(order) - 1)]

    server = server_raw
    ret = ret_raw

    if winner_is_server:
        if server == "40" and ret not in {"40", "Ad"}:
            return "game_server"
        if server == "Ad":
            return "game_server"
        server = next_token(server)
        if server == "Ad":
            return "Ad-In"
    else:
        if ret == "40" and server not in {"40", "Ad"}:
            return "game_returner"
        if ret == "Ad":
            return "game_returner"
        ret = next_token(ret)
        if ret == "Ad":
            return "Ad-Out"

    return f"{server}-{ret}"


def rally_bin(rally_index: int) -> str:
    """Bucket rally length for state serialization."""
    if rally_index <= 0:
        return "0"
    if rally_index == 1:
        return "1"
    if rally_index == 2:
        return "2"
    if rally_index in (3, 4):
        return "3_4"
    return "5_plus"

def serialize_state(field_order: list[str], context: dict[str, str]) -> str:
    """
    Serialize a Markov state based on a configurable set of fields.

    Supported field keys:
        - player_label
        - server_flag
        - prev_family
        - prev_direction (serialized as dir=<val>)
        - rally_bin (serialized as rbin=<val>)
        - point_score (serialized as score=<val>)
        - point_start_serve (serialized as start_srv=<val>)
        - rally_index (raw integer)
    """

    parts: list[str] = []
    for key in field_order:
        if key == "player_label":
            parts.append(context["player_label"])
        elif key == "server_flag":
            parts.append(context["server_flag"])
        elif key == "prev_family":
            parts.append(context["prev_family"])
        elif key == "prev_direction":
            parts.append(f"dir={context['prev_direction']}")
        elif key == "rally_bin":
            parts.append(f"rbin={context['rally_bin']}")
        elif key == "point_score":
            parts.append(f"score={context['point_score']}")
        elif key == "point_start_serve":
            parts.append(f"start_srv={context['point_start_serve']}")
        elif key == "rally_index":
            parts.append(str(context["rally_index"]))
        else:
            raise ValueError(f"Unsupported state field '{key}'")
    return "|".join(parts)


def infer_players(match_id: str) -> tuple[str, str]:
    """Heuristic fallback for player names when match metadata is missing."""
    parts = match_id.split("-")
    if len(parts) >= 2:
        return parts[-2].replace("_", " "), parts[-1].replace("_", " ")
    return "Player1", "Player2"


def clean_code(value: str | float | None) -> str:
    """Normalize serve/shot codes by stripping NaNs and whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def transitions_for_point(
    row: pd.Series,
    match_id: str,
    player1: str,
    player2: str,
    state_fields: list[str],
) -> list[dict]:
    """Convert one MCP point row into a list of state transitions (one per shot)."""
    server_val = row.get("Svr")
    try:
        server = int(server_val)
    except Exception:
        server = 1

    first_code = clean_code(row.get("1st"))
    second_code = clean_code(row.get("2nd"))
    start_serve_label = "2nd" if not first_code and second_code else "1st"

    try:
        strokes = parse_point(first_code, second_code, server_is_player1=(server == 1))
    except ParseError as exc:
        print(f"Skipping point {row.get('Pt')} in {match_id}: parse error -> {exc}")
        return []

    if not strokes:
        return []

    point_id = row.get("Pt")
    raw_score = row.get("Pts")
    tb_flag = _coerce_bool(row.get("TbSet"))

    def is_tb_row(r: pd.Series) -> bool:
        if not tb_flag:
            return False
        try:
            gm1 = int(float(r.get("Gm1")))
            gm2 = int(float(r.get("Gm2")))
        except Exception:
            return False
        return gm1 == 6 and gm2 == 6

    is_tb = is_tb_row(row)
    point_score = canonical_point_score(raw_score, is_tiebreak=is_tb)
    point_winner_raw = row.get("PtWinner")
    point_winner = int(point_winner_raw) if pd.notna(point_winner_raw) else None

    transitions: list[dict] = []
    prev_family = "BOS"
    prev_dir = "0"
    serve_in_started = False
    shots_after_serve = 0

    for idx, shot in enumerate(strokes):
        current_prev_family = prev_family
        current_prev_dir = prev_dir
        player_label = "P1" if shot.player_to_hit == 1 else "P2"
        player_name = player1 if player_label == "P1" else player2
        is_server = shot.player_to_hit == server

        rally_index = 0 if not serve_in_started else shots_after_serve + 1
        context = {
            "player_label": player_label,
            "server_flag": "serve_side" if is_server else "return_side",
            "prev_family": current_prev_family,
            "prev_direction": current_prev_dir,
            "rally_index": rally_index,
            "rally_bin": rally_bin(rally_index),
            "point_score": point_score,
            "point_start_serve": start_serve_label,
        }
        state = serialize_state(state_fields, context)
        winner_id = shot.point_winner or point_winner
        winner_label = None
        winner_name = None
        if winner_id in (1, 2) and shot.terminal:
            winner_label = "P1" if winner_id == 1 else "P2"
            winner_name = player1 if winner_id == 1 else player2

        point_score_after = point_score
        if shot.terminal and winner_id in (1, 2):
            point_score_after = advance_point_score(point_score, winner_is_server=(winner_id == server))
            if is_tb:
                before_tb = _parse_tb_score(point_score)
                after_tb = _parse_tb_score(point_score_after)
                if before_tb and after_tb:
                    point_idx = before_tb[0] + before_tb[1]
                    init_srv_is_p1 = _tb_initial_server_is_p1(server == 1, point_idx)
                    next_srv_is_p1 = _tb_server_for_index(init_srv_is_p1, point_idx + 1)
                    if next_srv_is_p1 != (server == 1):
                        a, b = after_tb
                        point_score_after = f"TB_{b}-{a}"

        next_serve_started = serve_in_started
        next_shots_after_serve = shots_after_serve
        if not serve_in_started and shot.shot_family == "SERVE" and shot.outcome_type not in {"fault", "double_fault"}:
            next_serve_started = True
        if next_serve_started and shot.shot_family != "SERVE":
            next_shots_after_serve += 1

        new_prev_family = shot.shot_family or current_prev_family
        new_prev_dir = shot.direction_code or "0"

        if idx + 1 < len(strokes):
            next_player_id = strokes[idx + 1].player_to_hit
        else:
            next_player_id = 2 if shot.player_to_hit == 1 else 1
        next_player_label = "P1" if next_player_id == 1 else "P2"
        next_is_server = next_player_id == server
        next_rally_index = 0 if not next_serve_started else next_shots_after_serve + 1
        next_context = {
            "player_label": next_player_label,
            "server_flag": "serve_side" if next_is_server else "return_side",
            "prev_family": new_prev_family,
            "prev_direction": new_prev_dir,
            "rally_index": next_rally_index,
            "rally_bin": rally_bin(next_rally_index),
            "point_score": point_score_after,
            "point_start_serve": start_serve_label,
        }
        new_state = serialize_state(state_fields, next_context)

        transitions.append(
            {
                "match_id": match_id,
                "point_id": point_id,
                "shot_index": idx,
                "state_fields": ",".join(state_fields),
                "player_label": player_label,
                "player_name": player_name,
                "state": state,
                "new_state": new_state,
                "server_flag": "serve_side" if is_server else "return_side",
                "prev_shot_family": current_prev_family,
                "prev_direction": current_prev_dir,
                "rally_index": rally_index,
                "rally_bin": rally_bin(rally_index),
                "point_score": point_score,
                "point_score_after": point_score_after,
                "point_start_serve": start_serve_label,
                "shot_code": shot.code,
                "shot_type": shot.shot_type,
                "shot_family": shot.shot_family,
                "direction_code": shot.direction_code,
                "depth_code": shot.depth_code,
                "modifiers": ";".join(shot.modifiers) if shot.modifiers else "",
                "outcome_raw": shot.outcome,
                "point_outcome": shot.outcome_type,
                "error_detail": shot.error,
                "is_terminal": bool(shot.terminal),
                "winner_player": winner_name,
                "winner_label": winner_label,
            }
        )

        prev_family = new_prev_family
        prev_dir = new_prev_dir
        serve_in_started = next_serve_started
        shots_after_serve = next_shots_after_serve
        point_score = point_score_after if shot.terminal else point_score

    return transitions


def process_file(path: Path, state_fields: list[str], player_map: dict[str, tuple[str, str]]) -> list[dict]:
    """Process a single point-table file into serialized transitions."""
    df = load_points_table(path)
    df.columns = [c.strip() for c in df.columns]

    rows: list[dict] = []
    player_cache: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        match_id_val = row.get("match_id", None)
        match_id = (
            str(match_id_val)
            if match_id_val is not None and not pd.isna(match_id_val)
            else path.stem
        )
        if match_id not in player_cache:
            player_cache[match_id] = player_map.get(match_id, infer_players(match_id))
        player1, player2 = player_cache[match_id]

        rows.extend(transitions_for_point(row, match_id, player1, player2, state_fields))
    return rows


def main(argv: list[str]) -> int:
    """CLI entry: read MCP exports, emit state_transitions CSV."""
    parser = argparse.ArgumentParser(description="Build shot-level state transitions from MCP files.")
    parser.add_argument("paths", nargs="+", help="Paths to .xlsm/.xlsx/.csv files or directories containing them.")
    parser.add_argument("--output", default="state_transitions.csv", help="Output CSV file.")
    parser.add_argument(
        "--state-fields",
        default=",".join(DEFAULT_STATE_FIELDS),
        help=(
            "Comma-separated list controlling the state serialization. "
            "Supported: player_label, server_flag, prev_family, prev_direction, rally_bin, rally_index, "
            "point_score, point_start_serve. Default uses server_flag,prev_family,prev_direction,"
            "rally_bin,point_score,point_start_serve."
        ),
    )
    args = parser.parse_args(argv)

    state_fields = [field.strip() for field in args.state_fields.split(",") if field.strip()]
    if not state_fields:
        print("No state fields specified.")
        return 1

    files = collect_input_files(args.paths)
    if not files:
        print("No input files found.")
        return 1

    base_dirs = {Path("raw_data"), Path("data")}
    base_dirs.update({p.parent for p in files})
    player_map = load_player_map(list(base_dirs))

    all_rows: list[dict] = []
    for path in files:
        print(f"Processing {path} ...")
        all_rows.extend(process_file(path, state_fields, player_map))

    if not all_rows:
        print("No transitions generated.")
        return 1

    out_path = Path(args.output)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Wrote {len(all_rows)} transitions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
