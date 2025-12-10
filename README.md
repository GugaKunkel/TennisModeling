# TennisModeling
The Goal of this project is to model tennis as a Markov process at different levels. Other ways of modeling tennis will offer strong prediction accuracy but often fail to explain why someone will win or lose a tennis match. By modeling as a Markov process the goal is to keep prediction accuracy while alse being able to pull insights on how players actually win matches. Due to how the game of tennis is set up we can choose many different levels at which to model the game as a Markov process. Two main levels of focus in this repo are the point/score level and the shot/state level.

The data used for this project comes from Jeff Sackmanns Match Charting Project (MCP) which provised point and shot level data along with player rankings and from the TML-Database (TMLD) which provides overall match data. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/ae4bf5a5-0032-4dd2-9bf3-64f785af384b" width="50%">
</p>

## Repository map
- `data/`: Holds csv data pulled from the MCP and TMLD
- `data_prep/`: Holds data parsing code and is used to build shot level `state_transitions.csv` file
- `parsed_data/`: Output folder for csvs of processed data such as the `state_transitions.csv` file
- `transition_matrix_builders/`: Holds find to build score-levela and shot-level transition matrices. Pulls from `state_transitions.csv`.
- `transition_matrices/`: Output folder for built transition matrices.
- `score_model/`: Match and score predicton code. Contains (`predictor.py`) file plus helpers for games/sets/matches.
- `simulations/`: Monte-Carlo simulation code for score matrices (`simulate_score_match.py`), shot/state matrices (`simulate_match.py`), and tournaments (`simulate_tournament.py`).
- `experiments/`: Evaluation scripts for score-model along with other models such as logistic regression.
- `tests/`: pytest coverage for parsers, matrix builders, and simulators.

## Setup
- Python 3.10+ recommended.
- Install deps (numpy, pandas, scikit-learn, openpyxl, pytest) into a virtual env:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install numpy pandas scikit-learn openpyxl pytest
  ```
- Data: place MCP downloads under `data/` or `raw_data/` (expects files like `charting-m-matches.csv`, point-level XLSX/XLSM exports).

## Data prep pipeline
### 1) Build state transitions from MCP point logs  
   ```bash
   python data_prep/build_state_transitions.py path/to/files_or_dirs --output state_transitions.csv \
     --state-fields server_flag,prev_family,prev_direction,rally_bin,point_score,point_start_serve
   ```
   Args:
   - `paths` (positional, one or more): files or dirs containing `.xlsm`, `.xlsx`, or `.csv` points tables.
   - `--output`: CSV to write (default `state_transitions.csv`).
   - `--state-fields`: comma list of fields encoded into state labels; defaults to `server_flag,prev_family,prev_direction,rally_bin,point_score,point_start_serve`. Changing fields alters matrix dimensionality.

### 2) Build shot/state matrices (for shot-level simulation)  
   ```bash
   python transition_matrix_builders/compute_state_matrices.py --input state_transitions.csv \
     --rankings_csv data/atp_rankings.csv --output_dir transition_matrices/state
   ```
   Args:
   - `--input` (required): `state_transitions.csv` from step 1.
   - `--rankings_csv`: attaches player/opponent rank bins; enables bin-specific matrices.
   - `--output_dir`: base folder for outputs (default `state_matrices`).
   - `--bins`: optional explicit bin labels; defaults to predefined ATP rank bins.
   Output: per-player, bin-vs-bin, bin-vs-all, and global `matrix_result.csv`/`matrix_continue.csv`.

### 3) Build score matrices (for score-model / fast sim)  
   ```bash
   python transition_matrix_builders/compute_score_matrices.py --input state_transitions.csv \
     --rankings_csv data/atp_rankings.csv --output_dir transition_matrices/score_rank
   ```
   Args:
   - `--input` (required): `state_transitions.csv`.
   - `--rankings_csv`: if provided, builds opponent-rank-binned matrices; otherwise unbinned.
   - `--rank_col`: rank column name in the rankings CSV (default `atp_rank`).
   - `--output_dir`: base folder (default `rank_bin_score_matrices`).
   - `--bins`: override bin labels.
   Output mirrors the shot-level builder: global, bin-vs-bin, bin-vs-all, per-player `score_matrix.csv` and `tb_matrix.csv`.

## Running Score Model for single match
- To predict a single matchup outcome and scoreline from score matrices:
  ```bash
  python -m score_model.predictor \
    --player-a "Jannik Sinner" --player-a-bin Top10 \
    --player-b "Richard Gasquet" --player-b-bin 101_200 \
    --server-first A --best-of 5 \
    --base-path transition_matrices/score_rank
  ```
  Args:
  - `--player-a` / `--player-b` (required): player names (match folder names under `base-path/player/player_*`).
  - `--player-a-bin` / `--player-b-bin`: opponent rank bins if matrices are binned; omit for unbinned.
  - `--server-first`: `A` or `B` initial server (default `A`).
  - `--best-of`: odd integer sets (default 5).
  - `--base-path`: directory containing `player/*/score_matrix.csv`.
  Outputs match win probability and most likely set scoreline.

## Testing Score Model prediciton accuracy
- Batch predictions for a matches CSV:
  ```bash
  python experiments/score_model_match_outcome/match_predict_all.py \
    --matches data/atp_matches_2025.csv \
    --base-rank transition_matrices/score_rank \
    --output predictions.csv
  ```
  Args:
  - `--matches`: input matches CSV (expects winner/loser columns; holdout uses `tourney_date > 20250610`).
  - `--base-rank`: binned score matrices root (mutually exclusive with `--base-unbinned`).
  - `--base-unbinned`: unbinned score matrices root (use when no bins).
  - `--output`: optional predictions CSV path.
  - `--use-bin-when-missing` / `--no-use-bin-when-missing`: toggle fallback to bin-vs-bin matrices when a player file is missing (default on).
  Metrics (accuracy/log-loss/score-distance) are printed; the script is also runnable as `python -m score_model.match_predict_all`.

## Running Monte Carlo simulations

### 1) Simulating single match with score matricies:
  ```bash
  python simulations/simulate_score_match.py \
    --server_a_matrix transition_matrices/score_rank/player/player_Jannik_Sinner/score_matrix.csv \
    --server_b_matrix transition_matrices/score_rank/player/player_Richard_Gasquet/score_matrix.csv \
    --player_a_bin Top10 --player_b_bin 101_200 \
    --best_of 5 --matches 100 --seed 42 \
    --base_dir transition_matrices/score_rank
  ```
  Args:
  - `--server_a_matrix` / `--server_b_matrix` (required): CSV paths for each player’s serve matrix.
  - `--player_a_bin` / `--player_b_bin`: rank bins when using binned matrices.
  - `--matches`: number of simulated matches (default 1).
  - `--best_of`: 3 or 5 (default 5).
  - `--seed`: base RNG seed (increments per match when set).
  - `--base_dir`: optional root containing fallbacks (`bin_vs_bin`, `bin_vs_all`, `global`); merged with primaries.

### 2) Simulating single match with shot matricies:
  ```bash
  python simulations/simulate_match.py \
    --player_a_dir transition_matrices/state/player/player_Jannik_Sinner \
    --player_b_dir transition_matrices/state/player/player_Richard_Gasquet \
    --best_of 3 \
    --state_fields server_flag,prev_family,prev_direction,rally_bin,point_score,point_start_serve \
    --seed 123 \
    --fallback_dir transition_matrices/state/global
  ```
  Args:
  - `--player_a_dir` / `--player_b_dir` (required): folders with `matrix_result.csv` and `matrix_continue.csv`.
  - `--best_of`: 3 or 5 (default 3).
  - `--state_fields`: comma list matching matrix state encoding (default mirrors data prep defaults).
  - `--seed`: RNG seed.
  - `--fallback_dir`: global matrices to backfill missing states.

### 3) Simulating entire tournament:
  ```bash
  python simulations/simulate_tournament.py \
    --bracket simulations/french_open_2025.json \
    --mode score \
    --base_dir transition_matrices/score_rank \
    --simulations 50 \
    --best_of 5 \
    --tournament_id FO25 \
    --detail_out runs/fo_detail.csv \
    --summary_out runs/fo_summary.csv \
    --bracket_out runs/fo_bracket.txt \
    --rankings_csv data/fo_ranks.csv
  ```
  Args:
  - `--bracket` (required): JSON list of `{player1, player2, best_of?}`.
  - `--mode`: `score` (uses score matrices + bin fallbacks) or `state` (shot-level adapter).
  - `--base_dir`: root containing matrices (score or state, matching `--mode`).
  - `--simulations`: sims per match (default 1).
  - `--best_of`: default sets when bracket entries omit it (default 5).
  - `--tournament_id`: label for outputs.
  - `--detail_out` / `--summary_out` / `--bracket_out`: output paths for per-sim detail, per-match summary, ASCII bracket.
  - `--rankings_csv`: optional ranks to auto-assign bins for score-mode fallbacks.

## Baselines and experiments
- Logistic baselines:
  ```bash
  python experiments/logreg_match_outcome/run_logreg.py --train data/atp_matches_2025.csv --seed 42
  ```
  Args: `--train` (one or more match CSVs, required), `--seed` (default 42). Uses time split `tourney_date > 20250610` as holdout; trains three feature sets.

- Rank-only baseline:
  ```bash
  python experiments/logreg_match_outcome/baseline_rank.py --matches data/atp_matches_2025.csv
  ```
  Args: `--matches` (one or more CSVs, required). Predicts higher-ranked player wins on the holdout window.

- Scoreline baseline:
  ```bash
  python experiments/score_model_match_outcome/baseline_score.py \
    --matches data/atp_matches_2025.csv \
    --cutoff 20250610 \
    --output baseline_scores.csv
  ```
  Args: `--matches` (required), `--cutoff` date separating train/test (default `20250610`), `--output` optional CSV.

## Unit Testing
- Run the test suite:
  ```bash
  pytest
  ```
- Notable tests live in `tests/` (parsers, matrix builders, score-model math, simulators).

## Notes and tips
- Transition matrix file names are normalized versions of player names (`player_<Name_With_Underscores>`). The scripts do the normalization for inputs you pass via CLI.
- When using binned matrices, both opponent bins must be provided (or inferred from ranks); unbinned matrices should omit bin flags.
- Score-level methods (`score_model/` and `simulate_score_match.py`) are faster and need only serve-side matrices; shot-level simulation uses both continuation and result matrices.
