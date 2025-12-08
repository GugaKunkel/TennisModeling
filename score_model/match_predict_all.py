"""
Compatibility wrapper to run match predictions via `python -m score_model.match_predict_all`.
Delegates to experiments/score_model_match_outcome/match_predict_all.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent.parent / "experiments" / "score_model_match_outcome" / "match_predict_all.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
