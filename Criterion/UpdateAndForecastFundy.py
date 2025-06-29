# UpdateAndForecastFundy.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback

# --- Load Environment Variables ---
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG: script_dir is: {script_dir}")
dotenv_path = os.path.join(script_dir, '.env')
print(f"DEBUG: dotenv_path is: {dotenv_path}")
print(f"DEBUG: Does dotenv_path exist? {os.path.exists(dotenv_path)}")

# MODIFIED LINE: Added override=True
load_dotenv(dotenv_path=dotenv_path, override=True)
print(f"DEBUG: dotenv_path {dotenv_path} loaded (Override enabled)") # Added confirmation

# --- Database Connection Details ---
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
print(f"DEBUG: DB_USER from env is: {DB_USER}") # Kept this for confirmation

DB_HOST = 'dda.criterionrsch.com'
DB_PORT = 443
DB_NAME = 'production'

if not DB_USER or not DB_PASSWORD or DB_USER == 'your_username': # Added check for placeholder
    print(f"ERROR: Database credentials not found or are placeholders. Tried loading .env from: {dotenv_path}")
    print(f"   DB_USER found: {DB_USER}")
    exit()

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Global Configuration ---
START_DATE_FROM_SCRATCH_ACTUALS = pd.to_datetime("2015-01-01")

def fetch_data_from_db(engine, ticker_string):
    """Fetches data from the database for a given ticker string."""
    if pd.isna(ticker_string) or str(ticker_string).strip() == "":
        print(f"   Skipping DB query due to empty or NaN ticker.")
        return pd.DataFrame()

    query = f"SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers('{str(ticker_string)}')"
    print(f"Executing DB query for: {str(ticker_string)[:100]}..." + ("..." if len(str(ticker_string)) > 100 else ""))
    try:
        db_df = pd.read_sql_query(query, engine)
        if 'date' in db_df.columns:
            db_df['date'] = pd.to_datetime(db_df['date'], errors='coerce')
        if 'value' in db_df.columns:
            db_df['value'] = pd.to_numeric(db_df['value'], errors='coerce')
        # The original script had 'region_name' being used later.
        # Ensure it's handled if it comes from the DB.
        # For now, the core issue is connection.
        db_df.dropna(subset=['date', 'value'], inplace=True) # Assuming ticker column will exist if data is valid
        return db_df
    except Exception as e:
        print(f"Error fetching data for ticker {str(ticker_string)[:50]}...: {e}")
        return pd.DataFrame()

def get_region_for_item(item_name, item_to_region_map):
    """Assigns a region based on item name, prioritizing map from fetched data."""
    if item_name in item_to_region_map:
        return item_to_region_map[item_name]
    item_name_lower = str(item_name).lower()
    if "northeast" in item_name_lower or "ne - balance" in item_name_lower: return "Northeast"
    if "midwest" in item_name_lower or "mid west - balance" in item_name_lower: return "Midwest"
    if "southcentral" in item_name_lower or "south central - balance" in item_name_lower: return "SouthCentral"
    if "southeast" in item_name_lower or "south east - balance" in item_name_lower: return "SouthEast"
    if "rockies" in item_name_lower: return "Rockies"
    if "west" in item_name_lower and "west[" not in item_name_lower: return "West"
    if "conus" in item_name_lower: return "Conus"
    if 'l48' in item_name_lower or 'us' in item_name_lower: return "Conus"
    return "Unknown"

def process_fundy_data(script_mode, engine, mapping_df_param, output_csv_name_param):
    """
    Core logic to process data for either ACTUALS or FORECASTS.
    script_mode: "ACTUALS" or "FORECAST"
    engine: SQLAlchemy engine instance
    mapping_df_param: DataFrame containing the ticker mappings for the current run
    output_csv_name_param: The base name of the output CSV file (e.g., "Fundy.csv" or "CriterionExtra.csv")
    """

    # Determine settings based on script_mode
    if script_mode == "ACTUALS":
        output_csv_name = output_csv_name_param
        ticker_column_in_mapping = 'History Ticker'
        print_process_type = "Actuals/Historical Update"
        current_start_date_from_scratch = START_DATE_FROM_SCRATCH_ACTUALS
    elif script_mode == "FORECAST":
        name_part, ext_part = os.path.splitext(output_csv_name_param)
        output_csv_name = f"{name_part}Forecast{ext_part}"
        ticker_column_in_mapping = 'Forecast Ticker'
        print_process_type = "Forecast Generation"
        current_start_date_from_scratch = None
    else:
        print(f"ERROR: Invalid script_mode '{script_mode}' in process_fundy_data.")
        return

    output_csv_path = os.path.join(script_dir, '../INFO', output_csv_name)
    output_file_was_created_from_scratch = False

    print(f"\n--- Starting {output_csv_name} {print_process_type} ---")

    try:
        print(f"Loading existing {output_csv_name} (if any) from: {output_csv_path}")
        try:
            df_original_output = pd.read_csv(output_csv_path)
            if 'Date' in df_original_output.columns:
                df_original_output['Date'] = pd.to_datetime(df_original_output['Date'], errors='coerce')
            else:
                df_original_output['Date'] = pd.Series(dtype='datetime64[ns]')
        except FileNotFoundError:
            print(f"WARNING: {output_csv_name} not found at {output_csv_path}.")
            if script_mode == "ACTUALS" and current_start_date_from_scratch:
                print(f"Will create it from scratch starting {current_start_date_from_scratch.strftime('%Y-%m-%d')}.")
            else:
                print("Will create it from scratch with all available data for this mode.")
            df_original_output = pd.DataFrame(columns=['Date', 'item', 'value', 'region'])
            df_original_output['Date'] = pd.Series(dtype='datetime64[ns]')
            output_file_was_created_from_scratch = True

        for col in ['item', 'value', 'region']:
            if col not in df_original_output.columns:
                df_original_output[col] = None
        df_original_output.dropna(subset=['Date', 'item', 'value'], how='any', inplace=True)

        if not {'Item', ticker_column_in_mapping}.issubset(mapping_df_param.columns):
            print(f"ERROR: Current mapping file must contain 'Item' and '{ticker_column_in_mapping}' columns for {script_mode} mode.")
            return

        item_to_ticker_map = mapping_df_param.set_index('Item')[ticker_column_in_mapping].to_dict()

        updated_item_data_list = []
        processed_items_from_mapping = set()

        for item_name, ticker_to_use in item_to_ticker_map.items():
            processed_items_from_mapping.add(item_name)

            if pd.isna(ticker_to_use) or str(ticker_to_use).strip() == "":
                print(f"\nSkipping item '{item_name}' due to missing or empty '{ticker_column_in_mapping}'.")
                if item_name in df_original_output['item'].unique() and script_mode == "ACTUALS":
                    updated_item_data_list.append(df_original_output[df_original_output['item'] == item_name])
                continue

            print(f"\nProcessing item: {item_name} using ticker: {ticker_to_use}")
            db_data_df = fetch_data_from_db(engine, ticker_to_use)

            if db_data_df.empty or 'date' not in db_data_df.columns or 'value' not in db_data_df.columns:
                print(f"   No valid data returned from DB for ticker '{ticker_to_use}'.")
                if item_name in df_original_output['item'].unique() and script_mode == "ACTUALS" and not output_file_was_created_from_scratch:
                    print(f"   Preserving existing actuals data for '{item_name}'.")
                    updated_item_data_list.append(df_original_output[df_original_output['item'] == item_name])
                continue
            
            # Assuming 'region_name' might come from the DB or needs to be derived.
            # The original fetch_data_from_db only explicitly parses 'date' and 'value'.
            # The original process_fundy_data uses db_data_df[['date', 'value', 'region_name']]
            # This implies 'region_name' is expected from fetch_data_from_db's result.
            # For now, we ensure 'region_name' exists, defaulting to None if not in columns from DB.
            if 'region_name' not in db_data_df.columns:
                db_data_df['region_name'] = None 

            db_data_df_processed = db_data_df[['date', 'value', 'region_name']].copy()
            db_data_df_processed.rename(columns={'date': 'Date', 'value': 'value', 'region_name': 'region'}, inplace=True)
            db_data_df_processed['item'] = item_name
            db_data_df_processed['value'] = pd.to_numeric(db_data_df_processed['value'], errors='coerce')
            db_data_df_processed.dropna(subset=['Date', 'value'], inplace=True)

            item_final_data = pd.DataFrame()

            if script_mode == "ACTUALS":
                if output_file_was_created_from_scratch:
                    db_data_df_processed = db_data_df_processed[db_data_df_processed['Date'] >= current_start_date_from_scratch]
                    item_final_data = db_data_df_processed
                    if not item_final_data.empty:
                        print(f"   Building actuals from scratch. Added {len(item_final_data)} rows for '{item_name}' from {current_start_date_from_scratch.strftime('%Y-%m-%d')}.")
                else:
                    current_item_original_data = df_original_output[df_original_output['item'] == item_name]
                    if not current_item_original_data.empty:
                        max_date_in_original = current_item_original_data['Date'].max()
                        print(f"   Most current date in {output_csv_name} for '{item_name}': {max_date_in_original.strftime('%Y-%m-%d')}")
                        sixty_days_prior = max_date_in_original - timedelta(days=60)

                        original_data_to_keep = current_item_original_data[current_item_original_data['Date'] < sixty_days_prior]

                        if max_date_in_original.tzinfo is not None and db_data_df_processed['Date'].dt.tz is None:
                            db_data_df_processed['Date'] = db_data_df_processed['Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(max_date_in_original.tzinfo)
                        elif max_date_in_original.tzinfo is None and db_data_df_processed['Date'].dt.tz is not None:
                            db_data_df_processed['Date'] = db_data_df_processed['Date'].dt.tz_convert(None)

                        db_data_for_update = db_data_df_processed[db_data_df_processed['Date'] >= sixty_days_prior]
                        item_final_data = pd.concat([original_data_to_keep, db_data_for_update], ignore_index=True)
                    else:
                        print(f"   Item '{item_name}' not found in original {output_csv_name}. Adding all fetched actuals data from {current_start_date_from_scratch.strftime('%Y-%m-%d')}.")
                        item_final_data = db_data_df_processed[db_data_df_processed['Date'] >= current_start_date_from_scratch]
            else: # FORECAST mode
                item_final_data = db_data_df_processed
                if not item_final_data.empty:
                    print(f"   Fetched {len(item_final_data)} forecast rows for '{item_name}'.")

            if not item_final_data.empty:
                updated_item_data_list.append(item_final_data)

        if script_mode == "ACTUALS":
            for item_name_orig in df_original_output['item'].unique():
                if item_name_orig not in processed_items_from_mapping:
                    print(f"\nItem '{item_name_orig}' from {output_csv_name} was not in mapping file. Preserving its original data.")
                    updated_item_data_list.append(df_original_output[df_original_output['item'] == item_name_orig])

        if not updated_item_data_list:
            print("No data available after fetch phase. Output file will be empty or reflect this.")
            df_after_fetch = pd.DataFrame(columns=['Date', 'item', 'value', 'region'])
        else:
            df_after_fetch = pd.concat(updated_item_data_list, ignore_index=True)

        df_after_fetch.drop_duplicates(subset=['item', 'Date'], keep='last', inplace=True)
        df_after_fetch.sort_values(by=['item', 'Date'], inplace=True)
        df_after_fetch['value'] = pd.to_numeric(df_after_fetch['value'], errors='coerce')

        # Create item_to_region_map from actually fetched/preserved data that has a region
        item_to_region_map_from_fetched = df_after_fetch[df_after_fetch['region'].notna() & (df_after_fetch['region'] != '')][['item', 'region']].drop_duplicates('item').set_index('item')['region'].to_dict()


        print("\n--- Completed Fetch and Update of Base Series ---")
        print("\n--- Starting Calculation of Derived Series ---")

        if df_after_fetch.empty or not df_after_fetch['value'].notna().any():
            print("Pivoting skipped: DataFrame is empty or 'value' column contains all NaNs after fetching.")
            final_df_long = df_after_fetch
        else:
            df_after_fetch['Date'] = pd.to_datetime(df_after_fetch['Date'])
            df_after_fetch['item'] = df_after_fetch['item'].astype(str)
            df_pivot = df_after_fetch.pivot_table(index='Date', columns='item', values='value')

            def get_col(df_pivot_local, col_name):
                if col_name in df_pivot_local.columns:
                    return df_pivot_local[col_name].fillna(0)
                else:
                    print(f"     Warning: Column '{str(col_name)}' not found for calculation. Assuming 0.")
                    return pd.Series(0, index=df_pivot_local.index, name=col_name)

            print("Calculating West region aggregates...")
            df_pivot['West - Ind'] = get_col(df_pivot, 'West[PNW] - Ind') + get_col(df_pivot, 'West[CA] - Ind')
            df_pivot['West - ResCom'] = get_col(df_pivot, 'West[PNW] - ResCom') + get_col(df_pivot, 'West[CA] - ResCom')
            df_pivot['West - Power'] = get_col(df_pivot, 'West[PNW] - Power') + get_col(df_pivot, 'West[CA] - Power')

            print("Calculating regional balances...")
            regions_for_balance = ["Northeast", "Midwest", "SouthCentral", "SouthEast", "Rockies", "West"]

            for region_prefix in regions_for_balance:
                prod_col_name = f"{region_prefix} - Prod"
                ind_col_name = f"{region_prefix} - Ind"
                rescom_col_name = f"{region_prefix} - ResCom"
                power_col_name = f"{region_prefix} - Power"
                balance_col_name = f"{region_prefix} - Balance"

                df_pivot[balance_col_name] = get_col(df_pivot, prod_col_name) - \
                                             (get_col(df_pivot, ind_col_name) + \
                                              get_col(df_pivot, rescom_col_name) + \
                                              get_col(df_pivot, power_col_name))
                print(f"   Calculated: {balance_col_name}")

            print("Calculating CONUS - Balance...")
            conus_supply = get_col(df_pivot, 'CONUS - LNGimp') + \
                           get_col(df_pivot, 'CONUS - CADimp') + \
                           get_col(df_pivot, 'CONUS - Prod')
            conus_demand_elements = [
                'CONUS - Ind', 'CONUS - ResCom', 'CONUS - Power',
                'CONUS - L&P', 'CONUS - P\'loss',
                'CONUS - MexExp', 'CONUS - LNGexp'
            ]
            conus_demand = sum(get_col(df_pivot, col) for col in conus_demand_elements)
            df_pivot['CONUS - Balance'] = conus_supply - conus_demand
            print("   Calculated: CONUS - Balance")

            final_df_long = df_pivot.reset_index().melt(id_vars='Date', var_name='item', value_name='value')
            # Apply region mapping
            final_df_long['region'] = final_df_long['item'].apply(lambda x: get_region_for_item(x, item_to_region_map_from_fetched))
            final_df_long.dropna(subset=['value'], inplace=True)

        final_df_long.sort_values(by=['item', 'Date'], inplace=True)

        print(f"\nSaving updated data to: {output_csv_path}")
        if 'Date' in final_df_long.columns and not final_df_long.empty:
            final_df_long['Date'] = pd.to_datetime(final_df_long['Date']).dt.strftime('%m/%d/%Y')

        output_columns = ['Date', 'item', 'value', 'region']
        for col in output_columns:
            if col not in final_df_long.columns:
                final_df_long[col] = pd.NA

        final_df_long = final_df_long.reindex(columns=output_columns)

        final_df_long.to_csv(output_csv_path, index=False)
        print(f"{output_csv_name} {print_process_type} completed.")

    except Exception as e:
        print(f"An unexpected error occurred during {print_process_type} for {output_csv_name}: {e}")
        traceback.print_exc()


def main():
    engine = None
    try:
        engine = create_engine(DATABASE_URL)

        # --- Process for "Fundy" files ---
        fundy_mapping_file_name = 'database_tables_list.csv' 
        # MODIFIED: Added 'CriterionInfo' to the path to look inside the correct sub-folder
        fundy_mapping_path = os.path.join(script_dir, 'CriterionInfo', fundy_mapping_file_name)
        print(f"Loading ticker mapping file for Fundy: {fundy_mapping_path}")
        try:
            fundy_mapping_df = pd.read_csv(fundy_mapping_path)
            if 'Item' not in fundy_mapping_df.columns:
                print(f"ERROR: Fundy mapping file {fundy_mapping_path} must contain at least 'Item' column. Skipping Fundy processing.")
            else:
                process_fundy_data(
                    script_mode="ACTUALS",
                    engine=engine,
                    mapping_df_param=fundy_mapping_df,
                    output_csv_name_param='Fundy.csv'
                )
                process_fundy_data(
                    script_mode="FORECAST",
                    engine=engine,
                    mapping_df_param=fundy_mapping_df,
                    output_csv_name_param='Fundy.csv'
                )
        except FileNotFoundError:
            print(f"ERROR: Fundy mapping file not found at {fundy_mapping_path}. Skipping Fundy processing.")
        except Exception as e:
            print(f"ERROR: Could not load Fundy mapping file. Details: {e}. Skipping Fundy processing.")


        # --- Process for "CriterionExtra" files ---
        extra_mapping_file_name = 'CriterionExtra_tables_list.csv'
        # MODIFIED: Added 'CriterionInfo' to the path to look inside the correct sub-folder
        extra_mapping_path = os.path.join(script_dir, 'CriterionInfo', extra_mapping_file_name)
        print(f"\nLoading ticker mapping file for CriterionExtra: {extra_mapping_path}")
        try:
            extra_mapping_df = pd.read_csv(extra_mapping_path)
            if 'Item' not in extra_mapping_df.columns:
                print(f"ERROR: CriterionExtra mapping file {extra_mapping_path} must contain at least 'Item' column. Skipping CriterionExtra processing.")
            else:
                process_fundy_data(
                    script_mode="ACTUALS",
                    engine=engine,
                    mapping_df_param=extra_mapping_df,
                    output_csv_name_param='CriterionExtra.csv'
                )
                process_fundy_data(
                    script_mode="FORECAST",
                    engine=engine,
                    mapping_df_param=extra_mapping_df,
                    output_csv_name_param='CriterionExtra.csv'
                )
        except FileNotFoundError:
            print(f"ERROR: CriterionExtra mapping file not found at {extra_mapping_path}. Skipping CriterionExtra processing.")
        except Exception as e:
            print(f"ERROR: Could not load CriterionExtra mapping file. Details: {e}. Skipping CriterionExtra processing.")

    except Exception as e:
        print(f"A critical error occurred in main execution: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed (if it was opened).")

if __name__ == '__main__':
    main()