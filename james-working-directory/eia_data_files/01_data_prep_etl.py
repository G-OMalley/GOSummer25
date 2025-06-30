#!/usr/bin/env python3
"""ETL Data Preparation Script for EIA Regional Forecasting

This script performs the following:
  1. Automatically detect raw CSV/Excel files in the current working directory.
  2. Preprocess and pivot datasets (Fundamentals, Weather, PowerGen, EIA changes).
  3. Normalize all column names to snake_case for consistency.
  4. Validate that each dataset meets expected rules (columns, date formats, no duplicates).
  5. Construct regional DataFrames by merging relevant features.
  6. Apply final cleaning steps (date cutoff, interpolation, index setting, frequency checks).
  7. Validate target features in EIA changes.
  8. Save cleaned DataFrames back into the same directory with consistent date formatting.

Usage:
    Place this script in the directory containing your raw data files and run:
        python data_prep_etl.py

Requirements:
    pandas, numpy, openpyxl, dateparser
"""

import concurrent.futures
import logging
import os
import sys
from typing import Dict, List

import dateparser  # for robust date parsing
import numpy as np
import pandas as pd


# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = configure_logging()

# Constants
REQUIRED_COLUMNS = {
    "fundy.csv": ["date", "item", "value"],
    "weather.csv": ["date", "city_title"],
    "eiachanges.csv": ["week_ending"],
}
TARGET_FEATURE_MAP_CHECK = {
    "east_region": "final_east_df",
    "midwest_region": "final_midwest_df",
    "mountain_region": "final_mountain_df",
    "pacific_region": "final_pacific_df",
    "south_central_region": "final_south_central_df",
    "total_lower_48": "final_total_df",
}
REGIONS = ["total", "east", "midwest", "mountain", "pacific", "south_central"]
CUT_OFF_DATE = pd.to_datetime("2018-07-07")
DATE_OUTPUT_FORMAT = "%Y-%m-%d"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case, alphanumeric and underscores only."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace("[^0-9a-z_]+", "_", regex=True).str.replace("_+", "_", regex=True).str.strip("_")
    return df


def parse_date_str(s: str):
    """Parse a single date string using dateparser."""
    try:
        return dateparser.parse(s)
    except Exception:
        return pd.NaT


def validate_columns(name: str, df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Dataset '{name}' missing columns: {missing}")
        raise ValueError(f"Missing {missing} in {name}")
    logger.debug(f"Columns validated for {name}")


def load_csv(filepath: str, parse_dates=None) -> pd.DataFrame:
    name = os.path.basename(filepath).lower()
    logger.debug(f"Loading CSV: {name}")
    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        logger.error(f"Error loading {name}: {e}")
        raise
    df = normalize_columns(df)
    if parse_dates:
        for col in parse_dates:
            col_norm = col.strip().lower().replace(" ", "_")
            if col_norm not in df.columns:
                logger.error(f"Missing date column '{col}' in {name}")
                raise KeyError(f"Missing date column {col} in {name}")
            values = df[col_norm].fillna("").tolist()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                parsed = list(executor.map(parse_date_str, values))
            df[col_norm] = pd.to_datetime(parsed)
            nulls = df[col_norm].isnull().sum()
            if nulls:
                logger.warning(f"{nulls} null dates after parsing in {name}:{col_norm}")
    if name in REQUIRED_COLUMNS:
        validate_columns(name, df, REQUIRED_COLUMNS[name])
    return df


def load_excel_sheets(filepath: str) -> Dict[str, pd.DataFrame]:
    name = os.path.basename(filepath).lower()
    logger.debug(f"Loading Excel: {name}")
    try:
        sheets = pd.read_excel(filepath, sheet_name=None, dtype=str)
    except Exception as e:
        logger.error(f"Error loading {name}: {e}")
        raise
    valid = {}
    for sname, df in sheets.items():
        df = normalize_columns(df)
        # Detect date column
        date_col = next((c for c in df.columns if "date" in c), None)
        if not date_col:
            for c in df.columns:
                vals = df[c].fillna("").tolist()
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    parsed = list(executor.map(parse_date_str, vals))
                ratio = pd.Series(parsed).notnull().mean()
                if ratio > 0.8:
                    date_col = c
                    df[c] = parsed
                    break
        if not date_col:
            logger.warning(f"Skipping sheet '{sname}': no date column")
            continue
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        nulls = df["date"].isnull().sum()
        if nulls > 0:
            logger.warning(
                f"Dropping {nulls} rows with invalid dates in sheet '{sname}'",
            )
            df = df[df["date"].notnull()]
        df = df.drop_duplicates(subset=["date"])
        valid[sname.strip().lower().replace(" ", "_")] = df
    if not valid:
        logger.error(f"No valid date-based sheets in {name}")
        raise ValueError(f"No date-based sheets in {name}")
    return valid


def pivot_fundy(fundy: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pivoting Fundy...")
    df = fundy.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isnull().any():
        raise ValueError("Invalid dates in fundy.csv")
    wide = df.pivot(index="date", columns="item", values="value").reset_index()
    wide = normalize_columns(wide)
    return wide.drop_duplicates(subset=["date"])


def pivot_weather(weather: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pivoting Weather...")
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isnull().any():
        raise ValueError("Invalid dates in weather.csv")
    if "city_symbol" in df.columns:
        df = df.drop(columns=["city_symbol"])
    melt = df.melt(
        id_vars=["date", "city_title"],
        var_name="metric",
        value_name="value",
    )
    melt["feature"] = melt["city_title"] + " - " + melt["metric"]
    wide = melt.pivot(index="date", columns="feature", values="value").reset_index()
    wide = normalize_columns(wide)
    return wide.drop_duplicates(subset=["date"])


def process_powergen(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    logger.info("Processing PowerGen sheets...")
    out = {}
    for name, df in sheets.items():
        df = normalize_columns(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop_duplicates(subset=["date"])
        rename = {c: f"{name}_" + c for c in df.columns if c != "date"}
        out[name] = df.rename(columns=rename)
    return out


def merge_and_clean(
    fundy_wide: pd.DataFrame,
    weather_wide: pd.DataFrame,
    powergen: Dict[str, pd.DataFrame],
    eia_changes: pd.DataFrame,
    city_map: pd.DataFrame = None,
):
    logger.info("Merging and cleaning regional DataFrames")
    base_dates = pd.DataFrame({"date": sorted(fundy_wide["date"].unique())})
    regional = {r: base_dates.copy() for r in REGIONS}

    # --- Fundy merge ---
    conus = [c for c in fundy_wide.columns if c.startswith("conus") or c == "gom_prod"]
    fmap = {
        "total": fundy_wide.columns.tolist(),
        "east": conus + [c for c in fundy_wide.columns if "northeast" in c or "southeast" in c],
        "midwest": conus + [c for c in fundy_wide.columns if "midwest" in c],
        "mountain": conus + [c for c in fundy_wide.columns if "rockies" in c],
        "south_central": conus + [c for c in fundy_wide.columns if "south_central" in c],
        "pacific": conus + [c for c in fundy_wide.columns if "west" in c],
    }
    for r, cols in fmap.items():
        cols = [c for c in cols if c != "date" and c in fundy_wide.columns]
        df_sub = fundy_wide[["date"] + cols]
        regional[r] = regional[r].merge(df_sub, on="date", how="left")

    # --- Weather merge ---
    for r in REGIONS:
        if city_map is not None and "city_title" in city_map.columns and "region" in city_map.columns:
            mapping = city_map.set_index("city_title")["region"].to_dict()
            cols = [c for c in weather_wide.columns if c.split(" - ")[0] in mapping and mapping[c.split(" - ")[0]] == r]
        else:
            cols = [c for c in weather_wide.columns if r in c]
        regional[r] = regional[r].merge(
            weather_wide[["date"] + cols],
            on="date",
            how="left",
        )

    # --- PowerGen merge ---
    pgmap = {
        "CAISO Generation": ["pacific", "total"],
        "ERCOT Supply": ["south_central", "total"],
        "MISO Generation": ["midwest", "south_central", "total"],
        "ISONE Generation": ["east", "total"],
        "NYISO Generation": ["east", "total"],
        "PJM Generation": ["east", "total"],
        "SPP Generation": ["midwest", "south_central", "total"],
    }
    for sheet, df_pg in powergen.items():
        targets = pgmap.get(sheet, [])
        for r in targets:
            cols = [c for c in df_pg.columns if c != "date"]
            regional[r] = regional[r].merge(
                df_pg[["date"] + cols],
                on="date",
                how="left",
            )

    # --- Final cleaning & save ---
    for r, df in regional.items():
        df = df[df["date"] >= CUT_OFF_DATE].copy()
        df.sort_values("date", inplace=True)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].interpolate(limit=1)
        df[num_cols] = df[num_cols].ffill().bfill()
        if df[num_cols].isnull().sum().any():
            logger.warning(f"Remaining NaNs in {r}")
        df.set_index("date", inplace=True)
        if pd.infer_freq(df.index) != "D":
            logger.warning(f"{r} freq not daily")
        out_file = f"final_{r}_df.csv"
        df.to_csv(out_file, date_format=DATE_OUTPUT_FORMAT)
        logger.info(f"Saved {out_file}")

    # --- EIAchanges validation & save ---
    eia = normalize_columns(eia_changes)
    eia["week_ending"] = pd.to_datetime(eia["week_ending"], errors="coerce")
    if eia["week_ending"].isnull().any():
        raise ValueError("Invalid week_ending dates in EIAchanges.csv")
    eia = eia.set_index("week_ending")
    if pd.infer_freq(eia.index) != "W-FRI":
        try:
            eia = eia.asfreq("W-FRI")
        except Exception:
            logger.warning("Could not set EIAchanges freq to W-FRI")
    missing = [t for t in TARGET_FEATURE_MAP_CHECK if t not in eia.columns]
    if missing:
        raise ValueError(f"Missing targets in EIAchanges: {missing}")
    eia.to_csv("final_eia_changes.csv", date_format=DATE_OUTPUT_FORMAT)
    logger.info("Saved final_eia_changes.csv")


def main():
    base = os.getcwd()
    logger.info(f"Working dir: {base}")
    fundy = load_csv(os.path.join(base, "Fundy.csv"), parse_dates=["Date"])
    weather = load_csv(os.path.join(base, "WEATHER.csv"), parse_dates=["Date"])
    eia_changes = load_csv(
        os.path.join(base, "EIAchanges.csv"),
        parse_dates=["Week ending"],
    )
    try:
        city_map = load_csv(os.path.join(base, "city_eia_mapping.csv"))
    except FileNotFoundError:
        city_map = None
        logger.info("city_eia_mapping.csv not found, skipping")
    power_sheets = load_excel_sheets(os.path.join(base, "PowerGen.xlsx"))

    fw = pivot_fundy(fundy)
    ww = pivot_weather(weather)
    pg = process_powergen(power_sheets)

    merge_and_clean(fw, ww, pg, eia_changes, city_map)
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
