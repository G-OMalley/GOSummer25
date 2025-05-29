# ProcessNuclearData.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta 
import traceback

# --- Load Environment Variables ---
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Database Connection Details ---
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = 'dda.criterionrsch.com'
DB_PORT = 443
DB_NAME = 'production'

if not DB_USER or not DB_PASSWORD:
    print(f"ERROR: Database credentials not found. Tried loading .env from: {dotenv_path}")
    exit()

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Global Configuration ---
START_DATE_FROM_SCRATCH_ACTUALS = pd.to_datetime("2015-01-01")
NUCLEAR_MAPPING_FILE_NAME = 'NuclearPairs.csv' # Located in Criterion folder
nuclear_mapping_path = os.path.join(script_dir, NUCLEAR_MAPPING_FILE_NAME)

def fetch_data_from_db(engine, ticker_string_list):
    """Fetches data from the database for a list of ticker strings."""
    if not ticker_string_list:
        print("  No tickers provided to fetch_data_from_db.")
        return pd.DataFrame()
    
    # Ensure all tickers are strings and filter out any potential NaNs/empty strings
    valid_tickers = [str(t).strip() for t in ticker_string_list if pd.notna(t) and str(t).strip()]
    if not valid_tickers:
        print("  No valid tickers after cleaning the list.")
        return pd.DataFrame()

    tickers_for_query = ",".join(valid_tickers)
    query = f"SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers('{tickers_for_query}')"
    print(f"Executing DB query for {len(valid_tickers)} tickers, starting with: {valid_tickers[0][:100]}..." + ("..." if len(valid_tickers[0]) > 100 else ""))
    
    try:
        db_df = pd.read_sql_query(query, engine)
        if 'date' in db_df.columns:
            db_df['date'] = pd.to_datetime(db_df['date'], errors='coerce')
        if 'value' in db_df.columns:
            db_df['value'] = pd.to_numeric(db_df['value'], errors='coerce')
        # The 'ticker' column from DB response will be the individual unit ticker
        db_df.dropna(subset=['date', 'value', 'ticker'], inplace=True)
        return db_df
    except Exception as e:
        print(f"Error fetching data for tickers (first few: {valid_tickers[:3]}...): {e}")
        return pd.DataFrame()

def process_nuclear_plant_data(script_mode, engine, nuclear_mapping_df):
    """
    Processes nuclear data: fetches unit data, aggregates to plant level, and saves.
    script_mode: "ACTUALS" or "FORECAST"
    engine: SQLAlchemy engine instance
    nuclear_mapping_df: DataFrame from NuclearPairs.csv
    """
    
    if script_mode == "ACTUALS":
        output_csv_name = 'Nuclear.csv'
        ticker_suffix = '.A'
        print_process_type = "Nuclear Actuals"
        current_start_date_from_scratch = START_DATE_FROM_SCRATCH_ACTUALS
    elif script_mode == "FORECAST":
        output_csv_name = 'NuclearForecast.csv'
        ticker_suffix = '.F'
        print_process_type = "Nuclear Forecasts"
        current_start_date_from_scratch = None # Forecasts take all available data
    else:
        print(f"ERROR: Invalid script_mode '{script_mode}' in process_nuclear_plant_data.")
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
            df_original_output = pd.DataFrame(columns=['Date', 'item', 'value', 'region', 'state']) # Added state
            df_original_output['Date'] = pd.Series(dtype='datetime64[ns]')
            output_file_was_created_from_scratch = True
            
        for col in ['item', 'value', 'region', 'state']: 
            if col not in df_original_output.columns:
                df_original_output[col] = None 
        df_original_output.dropna(subset=['Date', 'item', 'value'], how='any', inplace=True) # Key columns for logic

        # Assuming NuclearPairs.csv has columns: 'Ticker', 'Name', 'Region', 'State'
        # Let's rename them for clarity if needed, or use them directly.
        # For this script, let's assume standard names based on user description:
        # Column 1: 'Unit_Ticker_Raw', Column 2: 'Plant_Name', Column 3: 'Region', Column 4: 'State'
        # The user said "tickers in the first column", "Name in column 2", "Region", "State"
        # We'll need to ensure these are the actual column names in NuclearPairs.csv
        # For now, I'll use placeholder names and the user can adjust if their CSV has different headers.
        # Let's assume the CSV has headers: Ticker, Name, Region, State
        
        if not {'Ticker', 'Name', 'Region', 'State'}.issubset(nuclear_mapping_df.columns):
            print(f"ERROR: Nuclear mapping file {NUCLEAR_MAPPING_FILE_NAME} must contain 'Ticker', 'Name', 'Region', 'State' columns.")
            return

        all_plant_data_to_combine = []
        
        # Group by Plant Name to process each plant
        for plant_name, group in nuclear_mapping_df.groupby('Name'):
            print(f"\nProcessing Plant: {plant_name}")
            
            unit_tickers_for_plant = group['Ticker'][group['Ticker'].str.upper().str.endswith(ticker_suffix, na=False)].tolist()
            
            if not unit_tickers_for_plant:
                print(f"  No {script_mode} tickers found for plant '{plant_name}'. Skipping.")
                # If plant was in original output (actuals mode), preserve its data
                if script_mode == "ACTUALS" and plant_name in df_original_output['item'].unique():
                     all_plant_data_to_combine.append(df_original_output[df_original_output['item'] == plant_name])
                continue

            # Fetch data for all units of this plant in one go
            all_units_data_df = fetch_data_from_db(engine, unit_tickers_for_plant)

            if all_units_data_df.empty:
                print(f"  No data returned from DB for any units of plant '{plant_name}'.")
                if script_mode == "ACTUALS" and plant_name in df_original_output['item'].unique() and not output_file_was_created_from_scratch:
                    print(f"  Preserving existing actuals data for plant '{plant_name}'.")
                    all_plant_data_to_combine.append(df_original_output[df_original_output['item'] == plant_name])
                continue

            # Aggregate (sum) values by date for the current plant
            # The 'ticker' column in all_units_data_df refers to individual unit tickers
            # We are grouping by 'date' to sum up all units for that plant on that date
            aggregated_plant_data_df = all_units_data_df.groupby('date')['value'].sum().reset_index()
            aggregated_plant_data_df.rename(columns={'date': 'Date', 'value': 'value'}, inplace=True)
            
            # Add plant name, region, and state
            # Take region/state from the first unit in the group (assuming it's consistent for the plant)
            aggregated_plant_data_df['item'] = plant_name
            aggregated_plant_data_df['region'] = group['Region'].iloc[0] 
            aggregated_plant_data_df['state'] = group['State'].iloc[0]
            
            # Apply date logic (60-day overwrite for actuals, or build from scratch)
            item_final_data = pd.DataFrame() 

            if script_mode == "ACTUALS":
                if output_file_was_created_from_scratch:
                    aggregated_plant_data_df = aggregated_plant_data_df[aggregated_plant_data_df['Date'] >= current_start_date_from_scratch]
                    item_final_data = aggregated_plant_data_df
                    if not item_final_data.empty:
                         print(f"  Building actuals from scratch. Added {len(item_final_data)} rows for plant '{plant_name}' from {current_start_date_from_scratch.strftime('%Y-%m-%d')}.")
                else: 
                    current_plant_original_data = df_original_output[df_original_output['item'] == plant_name]
                    if not current_plant_original_data.empty:
                        max_date_in_original = current_plant_original_data['Date'].max()
                        print(f"  Most current date in {output_csv_name} for plant '{plant_name}': {max_date_in_original.strftime('%Y-%m-%d')}")
                        sixty_days_prior = max_date_in_original - timedelta(days=60)
                        
                        original_data_to_keep = current_plant_original_data[current_plant_original_data['Date'] < sixty_days_prior]
                        
                        # Timezone handling for date comparison
                        if max_date_in_original.tzinfo is not None and aggregated_plant_data_df['Date'].dt.tz is None:
                            aggregated_plant_data_df['Date'] = aggregated_plant_data_df['Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(max_date_in_original.tzinfo)
                        elif max_date_in_original.tzinfo is None and aggregated_plant_data_df['Date'].dt.tz is not None:
                            aggregated_plant_data_df['Date'] = aggregated_plant_data_df['Date'].dt.tz_convert(None)

                        db_data_for_update = aggregated_plant_data_df[aggregated_plant_data_df['Date'] >= sixty_days_prior]
                        item_final_data = pd.concat([original_data_to_keep, db_data_for_update], ignore_index=True)
                    else: 
                        print(f"  Plant '{plant_name}' not found in original {output_csv_name}. Adding all fetched actuals data from {current_start_date_from_scratch.strftime('%Y-%m-%d')}.")
                        item_final_data = aggregated_plant_data_df[aggregated_plant_data_df['Date'] >= current_start_date_from_scratch]
            else: # FORECAST mode - take all fetched data
                item_final_data = aggregated_plant_data_df
                if not item_final_data.empty:
                    print(f"  Fetched and aggregated {len(item_final_data)} forecast rows for plant '{plant_name}'.")
            
            if not item_final_data.empty:
                all_plant_data_to_combine.append(item_final_data)
        
        # Preserve data for plants in original actuals file that were not in nuclear_mapping_df
        if script_mode == "ACTUALS":
            for plant_name_orig in df_original_output['item'].unique():
                if plant_name_orig not in nuclear_mapping_df['Name'].unique():
                    print(f"\nPlant '{plant_name_orig}' from {output_csv_name} was not in {NUCLEAR_MAPPING_FILE_NAME}. Preserving its original data.")
                    all_plant_data_to_combine.append(df_original_output[df_original_output['item'] == plant_name_orig])

        if not all_plant_data_to_combine:
            print("No data available after processing all plants. Output file will be empty or reflect this.")
            final_df_long = pd.DataFrame(columns=['Date', 'item', 'value', 'region', 'state']) 
        else:
            final_df_long = pd.concat(all_plant_data_to_combine, ignore_index=True)
        
        final_df_long.drop_duplicates(subset=['item', 'Date'], keep='last', inplace=True)
        final_df_long.sort_values(by=['item', 'Date'], inplace=True)
        final_df_long['value'] = pd.to_numeric(final_df_long['value'], errors='coerce')
        final_df_long.dropna(subset=['value'], inplace=True) # Remove rows where value is NaN after all processing

        print(f"\nSaving updated data to: {output_csv_path}")
        if 'Date' in final_df_long.columns and not final_df_long.empty:
             final_df_long['Date'] = pd.to_datetime(final_df_long['Date']).dt.strftime('%m/%d/%Y')
        
        output_columns = ['Date', 'item', 'value', 'region', 'state']
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
        
        print(f"Loading nuclear mapping file: {nuclear_mapping_path}")
        try:
            nuclear_mapping_df_global = pd.read_csv(nuclear_mapping_path)
            # Validate essential columns based on user's description
            # Column 1: Ticker, Column 2: Name, Column 3: Region, Column 4: State
            # Assuming the CSV has headers: Ticker, Name, Region, State
            expected_nuclear_cols = ['Ticker', 'Name', 'Region', 'State']
            if not all(col in nuclear_mapping_df_global.columns for col in expected_nuclear_cols):
                 print(f"ERROR: Nuclear mapping file {nuclear_mapping_path} must contain columns: {', '.join(expected_nuclear_cols)}. Please check headers.")
                 return
        except FileNotFoundError:
            print(f"ERROR: Nuclear mapping file not found at {nuclear_mapping_path}. Cannot proceed.")
            return
        except Exception as e:
            print(f"ERROR: Could not load nuclear mapping file. Details: {e}")
            return

        # --- Process Nuclear ACTUALS ---
        process_nuclear_plant_data(
            script_mode="ACTUALS",
            engine=engine,
            nuclear_mapping_df=nuclear_mapping_df_global
        )
        
        # --- Process Nuclear FORECASTS ---
        process_nuclear_plant_data(
            script_mode="FORECAST",
            engine=engine,
            nuclear_mapping_df=nuclear_mapping_df_global
        )

    except Exception as e:
        print(f"A critical error occurred in main execution: {e}")
        traceback.print_exc()
    finally:
        if engine: 
            engine.dispose()
            print("\nDatabase connection closed (if it was opened).")

if __name__ == '__main__':
    main()