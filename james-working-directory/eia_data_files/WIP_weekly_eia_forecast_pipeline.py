#!/usr/bin/env python3
"""Robust Local Modeling Pipeline for Weekly EIA Storage Change Prediction

Requirements:
    pandas, numpy, scikit-learn, lightgbm, joblib,
    matplotlib, tensorflow (optional), colorlog, tabulate, optuna

Assumes files in CWD:
    final_<region>_df.csv (regions: east, midwest, mountain,
    pacific, south_central, total)
    final_eia_changes.csv

Outputs:
    - Inline summary bar chart with annotated MAE values
    - Printed summary tables via tabulate
"""

import glob
import logging
import os

import colorlog
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tabulate import tabulate

# Optional NeuralNet
try:
    from tensorflow import keras
    from tensorflow.keras import layers

    tf_available = True
except ImportError:
    tf_available = False

# === Logging Setup ===
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s [%(name)s]:%(reset)s %(message)s",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    ),
)
logger = colorlog.getLogger("eia_pipeline")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Constants
REGIONS = [
    "east",
    "midwest",
    "mountain",
    "pacific",
    "south_central",
    "total",
]
DATE_COL = "date"
WEEKLY_FREQ = "W-FRI"
LAG = 4


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to snake_case."""
    logger.debug("Normalizing columns")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace("[^0-9a-z_]+", "_", regex=True).str.replace("_+", "_", regex=True).str.strip("_")
    return df


def load_data() -> tuple[dict, pd.DataFrame, dict]:
    """Load regional data and EIA changes, return data dict, eia df, map."""
    logger.info("Loading data")
    data = {}
    for region in REGIONS:
        fname = f"final_{region}_df.csv"
        logger.debug(f"Reading {fname}")
        if not os.path.exists(fname):
            logger.error(f"Missing file: {fname}")
            raise FileNotFoundError(fname)
        df = pd.read_csv(fname, parse_dates=[DATE_COL])
        df = normalize_columns(df)
        df.set_index(DATE_COL, inplace=True)
        df = df.resample(WEEKLY_FREQ).mean()
        df.interpolate(inplace=True)
        df.ffill(inplace=True)
        logger.info(f"{region}: {df.shape[0]} weeks, {df.shape[1]} features")
        data[region] = df

    eia_file = "final_eia_changes.csv"
    logger.debug(f"Reading {eia_file}")
    if not os.path.exists(eia_file):
        logger.error(f"Missing file: {eia_file}")
        raise FileNotFoundError(eia_file)
    eia = pd.read_csv(eia_file, parse_dates=["week_ending"])
    eia = normalize_columns(eia)
    eia.set_index("week_ending", inplace=True)
    eia = eia.asfreq(WEEKLY_FREQ)
    logger.info(f"EIA: {eia.shape[0]} weeks, cols={len(eia.columns)}")

    target_map = {}
    cols = eia.columns.tolist()
    for region in REGIONS:
        if region == "total":
            cand = [c for c in cols if "total" in c]
        else:
            cand = [c for c in cols if region in c]
        if not cand:
            logger.critical(f"No EIA col for region: {region}")
            raise KeyError(region)
        target_map[region] = sorted(cand, key=len)[0]
        logger.debug(f"{region} -> {target_map[region]}")
    return data, eia, target_map


def make_features(
    df: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create lag features and month for time series."""
    logger.debug("Creating features")
    df = df.copy()
    df["target"] = target.reindex(df.index)
    for lag in range(1, LAG + 1):
        df[f"target_lag_{lag}"] = df["target"].shift(lag)
    df["month"] = df.index.month
    df.dropna(inplace=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    logger.debug(f"X shape: {X.shape}, y length: {len(y)}")
    return X, y


def build_nn(dim: int):
    """Build a simple feedforward neural network."""
    logger.debug(f"Building NN dim={dim}")
    model = keras.Sequential([
        layers.Input(shape=(dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def evaluate_region(
    region: str,
    df: pd.DataFrame,
    eia: pd.DataFrame,
    tmap: dict,
    tune: bool = False,
) -> pd.DataFrame:
    """Train and evaluate models for one region, with optional LGBM tuning."""
    logger.info(f"Evaluate region {region}")
    col = tmap[region]
    X, y = make_features(df, eia[col])
    if len(X) <= LAG:
        logger.warning(f"{region}: not enough data ({len(X)})")
        return pd.DataFrame()
    X_tr, X_te = X[:-LAG], X[-LAG:]
    y_tr, y_te = y[:-LAG], y[-LAG:]

    if tune:

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "l1",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 5, 128),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
            cv = TimeSeriesSplit(n_splits=5)
            callbacks = [lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
            cv_results = lgb.cv(
                params,
                lgb.Dataset(X_tr, label=y_tr),
                folds=cv,
                callbacks=callbacks,
            )
            # LightGBM cv returns 'valid l1-mean' for MAE on validation folds
            return cv_results["valid l1-mean"][-1]

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        logger.info(f"Best LGBM params for {region}: {best_params}")
    else:
        best_params = {"n_estimators": 100, "random_state": 42}

    results = []
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LGBM": lgb.LGBMRegressor(**best_params),
    }
    for name, m in models.items():
        logger.debug(f"{region}: train {name}")
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        logger.info(f"{region} {name} MAE={mae:.3f}")
        results.append((name, mae, y_te.iloc[-1], preds[-1], preds[-2]))
        joblib.dump(m, f"{region}_{name}.pkl")

    if tf_available:
        nn = build_nn(X_tr.shape[1])
        logger.debug(f"{region}: train NeuralNet")
        nn.fit(X_tr, y_tr, epochs=500, batch_size=8, verbose=0)
        pnn = nn.predict(X_te).flatten()
        mae_nn = mean_absolute_error(y_te, pnn)
        logger.info(f"{region} NeuralNet MAE={mae_nn:.3f}")
        results.append(("NeuralNet", mae_nn, y_te.iloc[-1], pnn[-1], pnn[-2]))

    return pd.DataFrame(results, columns=["Model", "MAE", "Actual", "Next", "Prior"])


def main():
    """Run full pipeline: load, evaluate, plot, print."""
    logger.info("Pipeline start")
    for pat in ("*.pkl", "*.h5"):
        for f in glob.glob(pat):
            os.remove(f)

    data, eia, tmap = load_data()
    summaries = {}
    for region, df in data.items():
        do_tune = region == "south_central"
        summaries[region] = evaluate_region(region, df, eia, tmap, tune=do_tune)

    # MAE bar chart
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for ax, region in zip(axes, REGIONS, strict=False):
        df_s = summaries[region]
        if not df_s.empty:
            bars = ax.bar(df_s["Model"], df_s["MAE"], color="skyblue")
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_title(region.title(), fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.suptitle("Weekly EIA Storage Change Prediction MAE", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Actual vs Predicted
    def plot_predictions(summaries):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        axes = axes.flatten()
        for ax, region in zip(axes, REGIONS, strict=False):
            df_s = summaries[region]
            if df_s.empty:
                continue
            ax.scatter(df_s["Model"], df_s["Actual"], label="Actual", marker="o")
            ax.scatter(df_s["Model"], df_s["Next"], label="Predicted", marker="x")
            for i, row in df_s.iterrows():
                ax.text(i, row["Next"], f"{row['Next']:.1f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{region.title()} Next vs Actual")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2)
        plt.suptitle("Weekly EIA Storage Change Prediction: Actual vs Predicted", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    plot_predictions(summaries)

    # tables
    for region in REGIONS:
        df_s = summaries[region]
        if df_s.empty:
            continue
        logger.info(f"Table for {region}")
        print(f"\n=== {region.title()} ===")
        print(tabulate(df_s, headers="keys", tablefmt="github", floatfmt=".2f"))
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
