# UpdateCriterionNuclear.py

import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ==============================================================================
#  CONFIGURATION CONSTANTS
# ==============================================================================
# Using constants makes the script easier to read and maintain.

# --- File Paths ---
# pathlib is a modern way to handle file paths robustly.
SCRIPT_DIR = Path(__file__).parent
INFO_DIR = SCRIPT_DIR.parent / "INFO"
# MODIFIED: Added 'CriterionInfo' to the path to look inside the correct sub-folder
PAIRS_FILE_PATH = SCRIPT_DIR / "CriterionInfo" / "NuclearPairs.csv"
HIST_OUTPUT_PATH = INFO_DIR / "CriterionNuclearHist.csv"
FORECAST_OUTPUT_PATH = INFO_DIR / "CriterionNuclearForecast.csv"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- SQL Queries ---
# Placing complex SQL in constants keeps the main logic clean.
HISTORICAL_DATA_QUERY = text("""
    SELECT DISTINCT
        dat.date,
        dat.load_date,
        dat.reactor_name,
        dat.operational_percent::float,
        dat.operational_percent::float / 100 * map.net_summer_capacity AS summer_generation,
        dat.operational_percent::float / 100 * map.nameplate_capacity AS nameplate_generation,
        map.state,
        reg.eia_ng_regions
    FROM power.nrc_raw_data AS dat
    LEFT JOIN power.nrc_mappings AS map ON map.nrc_reactor_name = dat.reactor_name
    LEFT JOIN pipelines.regions AS reg ON reg.state_abb = map.state
    LEFT JOIN misc.date_master AS wk ON wk.date = dat.date
    WHERE dat.date >= :start_date
""")

FORECAST_DATA_QUERY = text("SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers(:tickers)")


# ==============================================================================
#  CORE FUNCTIONS
# ==============================================================================

def get_db_engine():
    """Establishes a secure database connection and returns a SQLAlchemy engine."""
    print("INFO: Attempting to connect to database...")
    try:
        dotenv_path = SCRIPT_DIR / ".env"
        if not dotenv_path.exists():
            raise FileNotFoundError(f".env file not found at {dotenv_path}")
            
        load_dotenv(dotenv_path=dotenv_path, override=True)
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        if not db_user or not db_password:
            raise ValueError("DB_USER or DB_PASSWORD not found in environment.")

        conn_url = f"postgresql+psycopg2://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(
            conn_url, connect_args={"sslmode": "require", "connect_timeout": 10}
        )
        
        # Verify connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("INFO: Database engine created and connection confirmed.")
        return engine

    except (FileNotFoundError, ValueError) as e:
        print(f"CRITICAL: Configuration error - {e}")
    except Exception as e:
        print(f"CRITICAL: Database connection failed. Error: {e}")
    return None


def process_historical_data(engine: create_engine, output_path: Path):
    """Fetches, processes, and saves historical nuclear generation data."""
    print("\n--- Processing Historical Nuclear Data ---")
    
    start_date_for_fetch = pd.to_datetime("2015-01-01")
    existing_df = None
    if output_path.exists():
        try:
            print(f"INFO: Reading existing file: {output_path.name}")
            existing_df = pd.read_csv(output_path, parse_dates=["Date"])
            # Set a safe lookback window for incremental updates
            max_date_in_file = existing_df["Date"].max()
            start_date_for_fetch = max_date_in_file - timedelta(days=90)
            print(f"INFO: Fetching data since {start_date_for_fetch:%Y-%m-%d} for incremental update.")
        except (FileNotFoundError, KeyError):
            print(f"INFO: No valid existing file found. Fetching all data since {start_date_for_fetch:%Y-%m-%d}.")
            existing_df = None

    print("INFO: Querying database for historical data...")
    new_data = pd.read_sql(HISTORICAL_DATA_QUERY, engine, params={"start_date": start_date_for_fetch})
    print(f"INFO: Fetched {len(new_data)} rows from the database.")

    if new_data.empty:
        print("WARNING: No new historical data found. Skipping update.")
        return

    # --- Data Transformation ---
    new_data["date"] = pd.to_datetime(new_data["date"])
    
    # Define and apply seasonal generation logic
    is_summer = new_data["date"].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
    seasonal_generation = new_data["summer_generation"].where(is_summer, new_data["nameplate_generation"])
    new_data["Value"] = (new_data["operational_percent"] / 100) * seasonal_generation

    # Shape the data into the final narrow format
    new_data.rename(columns={
            "load_date": "Date", "reactor_name": "Item",
            "state": "State", "eia_ng_regions": "EIA Region"
        }, inplace=True
    )
    final_cols = ["Date", "Item", "Value", "State", "EIA Region"]
    formatted_new_df = new_data[final_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- Combine and Save ---
    if existing_df is not None:
        old_data_to_keep = existing_df[pd.to_datetime(existing_df["Date"]) < start_date_for_fetch]
        final_df = pd.concat([old_data_to_keep, formatted_new_df], ignore_index=True)
    else:
        final_df = formatted_new_df

    final_df.sort_values(by=["Item", "Date"], inplace=True, ascending=[True, False])
    final_df.drop_duplicates(subset=["Date", "Item"], keep="first", inplace=True)
    
    final_df.to_csv(output_path, index=False, date_format="%m/%d/%Y")
    print(f"SUCCESS: Historical data saved to {output_path.name}")


def process_forecast_data(engine: create_engine, mapping_df: pd.DataFrame, output_path: Path):
    """Fetches, processes, and saves forecast nuclear generation data."""
    print("\n--- Processing Forecast Nuclear Data ---")
    
    existing_df = None
    if output_path.exists():
        try:
            print(f"INFO: Reading existing forecast file: {output_path.name}")
            existing_df = pd.read_csv(output_path, parse_dates=["Date"])
        except (FileNotFoundError, KeyError):
            existing_df = None
            print("INFO: No valid existing forecast file found. Will build from scratch.")

    forecast_tickers = tuple(mapping_df["Forecast Ticker"].dropna().unique())
    if not forecast_tickers:
        print("INFO: No forecast tickers found in mapping file.")
        return

    print("INFO: Querying database for forecast data...")
    new_data = pd.read_sql(FORECAST_DATA_QUERY, engine, params={"tickers": ",".join(forecast_tickers)})
    
    if new_data.empty:
        print("WARNING: No new forecast data returned from the database.")
        return

    # --- Data Transformation ---
    forecast_map = mapping_df.set_index("Forecast Ticker")["Item"].to_dict()
    new_data["date"] = pd.to_datetime(new_data["date"])
    new_data["Item"] = new_data["ticker"].map(forecast_map)
    new_data.rename(columns={"date": "Date", "value": "Value"}, inplace=True)
    formatted_new_df = new_data[["Date", "Item", "Value"]].copy()

    # --- Combine and Save ---
    if existing_df is not None:
        cutoff_date = datetime.now() - timedelta(days=30)
        old_data_to_keep = existing_df[existing_df["Date"] < pd.to_datetime(cutoff_date.date())]
        final_df = pd.concat([old_data_to_keep, formatted_new_df], ignore_index=True)
    else:
        final_df = formatted_new_df
    
    final_df.sort_values(by=["Item", "Date"], inplace=True)
    final_df.drop_duplicates(subset=["Date", "Item"], keep="last", inplace=True)
    final_df.to_csv(output_path, index=False, date_format="%m/%d/%Y")
    print(f"SUCCESS: Forecast data saved to {output_path.name}")


def main():
    """Orchestrates the entire nuclear data update process."""
    print("--- Starting Nuclear Data Update Process ---")
    engine = get_db_engine()
    if not engine:
        print("Halting process due to database connection failure.")
        return

    try:
        INFO_DIR.mkdir(exist_ok=True) # Ensure the output directory exists
        print(f"INFO: Loading mapping file from {PAIRS_FILE_PATH.name}...")
        pairs_df = pd.read_csv(PAIRS_FILE_PATH)

        process_historical_data(engine, HIST_OUTPUT_PATH)
        process_forecast_data(engine, pairs_df, FORECAST_OUTPUT_PATH)

    except FileNotFoundError as e:
        print(f"CRITICAL: Input file not found. Error: {e}")
    except Exception as e:
        print(f"CRITICAL: An unexpected error occurred in the main process.")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed. Process finished.")


if __name__ == "__main__":
    main()
