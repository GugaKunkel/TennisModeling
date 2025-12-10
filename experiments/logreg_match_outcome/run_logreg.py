"""Train two logistic regression baselines for match outcome prediction.
    python3 experiments/logreg_match_outcome/run_logreg.py \
        --train data/atp_matches_2025.csv \
        --holdout-fraction 0.2
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS_BASE = [
    "rank_diff",
]

NUMERIC_COLS_EXTRA = [
    "player_a_age",
    "player_b_age",
    "age_diff",
    "player_a_ht",
    "player_b_ht",
    "ht_diff",
    "rank_points_diff",
    "player_a_seeded",
    "player_b_seeded",
]

NUMERIC_COLS_SERVE = [
    "player_a_ace",
    "player_b_ace",
    "player_a_1stInPct",
    "player_b_1stInPct",
]

CATEGORICAL_COLS_EXTRA = [
    "surface",
    "indoor",
]


@dataclass
class ModelSpec:
    name: str
    numeric_cols: list[str]
    categorical_cols: list[str]


MODEL_SPECS = [
    ModelSpec("A_names_and_ranks", NUMERIC_COLS_BASE, []),
    ModelSpec(
        "B_rich_features",
        NUMERIC_COLS_BASE + NUMERIC_COLS_EXTRA,
        CATEGORICAL_COLS_EXTRA,
    ),
    ModelSpec(
        "C_rich_features_with_serve",
        NUMERIC_COLS_BASE + NUMERIC_COLS_EXTRA + NUMERIC_COLS_SERVE,
        CATEGORICAL_COLS_EXTRA,
    ),
]


def load_matches(paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate one or more match CSVs."""
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def canonicalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Create a symmetric view so label is not leaked by player ordering."""
    df = df.copy()

    def _ratio(num, denom):
        num = pd.to_numeric(num, errors="coerce")
        denom = pd.to_numeric(denom, errors="coerce").replace({0: pd.NA})
        return num / denom

    def ordered_row(row) -> Tuple[str, str, int]:
        w, l = row["winner_name"], row["loser_name"]
        if not w or not l:
            return w, l, 1  # default order
        if w <= l:
            return w, l, 1
        return l, w, 0

    ordered = df.apply(ordered_row, axis=1, result_type="expand")
    df["player_a_name"], df["player_b_name"], labels = ordered[0], ordered[1], ordered[2]
    df["label"] = labels.astype(int)

    # Swap numeric attributes when we swapped players
    swap_mask = df["label"] == 0
    for base in ["rank", "rank_points", "age", "ht", "seed"]:
        win_col, lose_col = f"winner_{base}", f"loser_{base}"
        a_col, b_col = f"player_a_{base}", f"player_b_{base}"
        df[a_col] = df[win_col]
        df[b_col] = df[lose_col]
        df.loc[swap_mask, [a_col, b_col]] = df.loc[swap_mask, [b_col, a_col]].values

    # Serve metrics (aces direct counts; 1st serve/1st won as ratios).
    df["player_a_ace"] = pd.to_numeric(df["w_ace"], errors="coerce")
    df["player_b_ace"] = pd.to_numeric(df["l_ace"], errors="coerce")
    df.loc[swap_mask, ["player_a_ace", "player_b_ace"]] = df.loc[swap_mask, ["player_b_ace", "player_a_ace"]].values

    df["player_a_1stInPct"] = _ratio(df["w_1stIn"], df["w_svpt"])
    df["player_b_1stInPct"] = _ratio(df["l_1stIn"], df["l_svpt"])
    df.loc[swap_mask, ["player_a_1stInPct", "player_b_1stInPct"]] = df.loc[
        swap_mask, ["player_b_1stInPct", "player_a_1stInPct"]
    ].values

    df["player_a_1stWonPct"] = _ratio(df["w_1stWon"], df["w_1stIn"])
    df["player_b_1stWonPct"] = _ratio(df["l_1stWon"], df["l_1stIn"])
    df.loc[swap_mask, ["player_a_1stWonPct", "player_b_1stWonPct"]] = df.loc[
        swap_mask, ["player_b_1stWonPct", "player_a_1stWonPct"]
    ].values

    for col in ["player_a_1stInPct", "player_b_1stInPct", "player_a_1stWonPct", "player_b_1stWonPct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label 1 means player_a was the original winner
    return df


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive numeric diff columns and seed flags."""
    df = df.copy()
    # Ensure numeric dtypes
    for col in [
        "player_a_rank",
        "player_b_rank",
        "player_a_rank_points",
        "player_b_rank_points",
        "player_a_age",
        "player_b_age",
        "player_a_ht",
        "player_b_ht",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rank_diff"] = df["player_a_rank"] - df["player_b_rank"]
    df["rank_points_diff"] = df["player_a_rank_points"] - df["player_b_rank_points"]
    df["age_diff"] = df["player_a_age"] - df["player_b_age"]
    df["ht_diff"] = df["player_a_ht"] - df["player_b_ht"]
    df["player_a_seeded"] = df["player_a_seed"].notna() & (df["player_a_seed"].astype(str) != "")
    df["player_b_seeded"] = df["player_b_seed"].notna() & (df["player_b_seed"].astype(str) != "")
    df["player_a_seed"] = pd.to_numeric(df["player_a_seed"], errors="coerce")
    df["player_b_seed"] = pd.to_numeric(df["player_b_seed"], errors="coerce")
    return df


def build_preprocessor(spec: ModelSpec) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, spec.numeric_cols),
            ("cat", categorical, spec.categorical_cols),
        ]
    )


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, spec: ModelSpec, seed: int) -> dict:
    preprocessor = build_preprocessor(spec)
    model = LogisticRegression(max_iter=400, n_jobs=1)
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    clf.fit(train_df[spec.numeric_cols + spec.categorical_cols], train_df["label"])
    probs = clf.predict_proba(test_df[spec.numeric_cols + spec.categorical_cols])[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "spec": spec.name,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "accuracy": accuracy_score(test_df["label"], preds),
        "log_loss": log_loss(test_df["label"], probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic baselines for match prediction.")
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="CSV file(s) with ATP matches.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = load_matches(args.train)
    df = canonicalize_matches(df)
    df = add_feature_columns(df)

    # Time-based split: anything after 20250610 is test/holdout.
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    test_df = df[df["tourney_date"] > 20250610]
    train_df = df[~(df["tourney_date"] > 20250610)]

    if test_df.empty:
        raise ValueError("No rows found with tourney_date > 20250610 for holdout test set.")

    results = [train_and_eval(train_df, test_df, spec, args.seed) for spec in MODEL_SPECS]

    print("\nResults")
    for res in results:
        print(
            f"{res['spec']}: acc={res['accuracy']:.3f}, logloss={res['log_loss']:.3f}, "
            f"train={res['train_size']}, holdout={res['test_size']}"
        )

if __name__ == "__main__":
    main()
