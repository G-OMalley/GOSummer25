# UpdateFundy.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta # Correctly import timedelta
import traceback

# --- Load Environment Variables ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes .env file is in the Criterion folder (same directory as this script)
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

# --- File Paths ---
# User's mapping file is in the Criterion folder (same as the script)
MAPPING_FILE_NAME = 'database_tables_list.csv' # User confirmed this is their adjusted mapping file
mapping_file_path = os.path.join(script_dir, MAPPING_FILE_NAME)

# Fundy.csv is in the INFO folder, which is a sibling of Criterion
fundy_csv_path = os.path.join(script_dir, '../INFO/Fundy.csv')
START_DATE_FROM_SCRATCH = pd.to_datetime("2015-01-01")


def fetch_data_from_db(engine, ticker_string):
    """Fetches data from the database for a given ticker string."""
    query = f"SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers('{ticker_string}')"
    print(f"Executing DB query for: {ticker_string[:100]}..." + ("..." if len(ticker_string) > 100 else ""))
    try:
        db_df = pd.read_sql_query(query, engine)
        if 'date' in db_df.columns:
            db_df['date'] = pd.to_datetime(db_df['date'], errors='coerce')
        if 'value' in db_df.columns:
            db_df['value'] = pd.to_numeric(db_df['value'], errors='coerce')
        # Drop rows where essential data (date or value) could not be parsed or is missing
        db_df.dropna(subset=['date', 'value'], inplace=True)
        return db_df
    except Exception as e:
        print(f"Error fetching data for tickers {ticker_string[:50]}...: {e}")
        # traceback.print_exc() # Uncomment for detailed error during fetching
        return pd.DataFrame()

def get_region_for_item(item_name, item_to_region_map):
    """
    Assigns a region. First checks a pre-populated map from base items, 
    then applies rules for calculated items.
    """
    # Check if the item (could be base or calculated) is in the map from original data
    if item_name in item_to_region_map:
        return item_to_region_map[item_name]

    # Rules for calculated items if not found above
    item_name_lower = str(item_name).lower() 
    if "northeast" in item_name_lower or "ne - balance" in item_name_lower : 
        return "Northeast"
    if "midwest" in item_name_lower or "mid west - balance" in item_name_lower:
        return "Midwest"
    if "southcentral" in item_name_lower or "south central - balance" in item_name_lower :
        return "SouthCentral"
    if "southeast" in item_name_lower or "south east - balance" in item_name_lower: 
        return "Southeast" # This should correctly match "SouthEast - Balance" when lowercased
    if "rockies" in item_name_lower:
        return "Rockies"
    if "west" in item_name_lower and "west[" not in item_name_lower : 
        return "West"
    if "conus" in item_name_lower:
        return "Conus"
    if 'l48' in item_name_lower or 'us' in item_name_lower :
        return "Conus"
    return "Unknown"


def main():
    print("Starting Fundy.csv update process...")
    engine = None 
    fundy_was_created_from_scratch = False

    try:
        # --- Step 1: Read Input Files ---
        print(f"Loading Fundy.csv from: {fundy_csv_path}")
        try:
            fundy_df_original = pd.read_csv(fundy_csv_path)
            if 'Date' in fundy_df_original.columns:
                 fundy_df_original['Date'] = pd.to_datetime(fundy_df_original['Date'], errors='coerce')
            else:
                fundy_df_original['Date'] = pd.Series(dtype='datetime64[ns]')
        except FileNotFoundError:
            print(f"WARNING: Fundy.csv not found at {fundy_csv_path}. Will create it from scratch starting {START_DATE_FROM_SCRATCH.strftime('%Y-%m-%d')}.")
            fundy_df_original = pd.DataFrame(columns=['Date', 'item', 'value', 'region'])
            fundy_df_original['Date'] = pd.Series(dtype='datetime64[ns]') # Ensure Date column is datetime
            fundy_was_created_from_scratch = True
            
        # Ensure essential columns exist
        for col in ['item', 'value', 'region']: 
            if col not in fundy_df_original.columns:
                fundy_df_original[col] = None 
        
        fundy_df_original.dropna(subset=['Date', 'item', 'value'], how='any', inplace=True)

        print(f"Loading ticker mapping file from: {mapping_file_path}")
        mapping_df = pd.read_csv(mapping_file_path)
        if not {'Item', 'History Ticker'}.issubset(mapping_df.columns):
            print(f"ERROR: Mapping file {mapping_file_path} must contain 'Item' and 'History Ticker' columns.")
            return
        
        item_to_history_ticker_map = mapping_df.set_index('Item')['History Ticker'].to_dict()

        engine = create_engine(DATABASE_URL)
        
        print("\nConnecting to database for fetching base series...")
        updated_item_data_list = []
        processed_items_from_mapping = set() 

        for item_name, history_ticker in item_to_history_ticker_map.items():
            processed_items_from_mapping.add(item_name) 

            if pd.isna(history_ticker) or str(history_ticker).strip() == "":
                print(f"\nSkipping item '{item_name}' due to missing or empty History Ticker in mapping file.")
                if item_name in fundy_df_original['item'].unique():
                    updated_item_data_list.append(fundy_df_original[fundy_df_original['item'] == item_name])
                continue
            
            print(f"\nProcessing item: {item_name} using ticker: {history_ticker}")
            db_data_df = fetch_data_from_db(engine, str(history_ticker))

            if db_data_df.empty or 'date' not in db_data_df.columns or 'value' not in db_data_df.columns:
                print(f"  No valid data returned from DB for ticker '{history_ticker}'.")
                if item_name in fundy_df_original['item'].unique() and not fundy_was_created_from_scratch:
                    print(f"  Preserving existing data for '{item_name}' from Fundy.csv.")
                    updated_item_data_list.append(fundy_df_original[fundy_df_original['item'] == item_name])
                continue
            
            db_data_df_processed = db_data_df[['date', 'value', 'region_name']].copy()
            db_data_df_processed.rename(columns={'date': 'Date', 'value': 'value', 'region_name': 'region'}, inplace=True)
            db_data_df_processed['item'] = item_name
            db_data_df_processed['value'] = pd.to_numeric(db_data_df_processed['value'], errors='coerce')
            db_data_df_processed.dropna(subset=['Date', 'value'], inplace=True) 

            if fundy_was_created_from_scratch:
                # If building from scratch, only take data from START_DATE_FROM_SCRATCH
                db_data_df_processed = db_data_df_processed[db_data_df_processed['Date'] >= START_DATE_FROM_SCRATCH]
                item_final_data = db_data_df_processed
                if not item_final_data.empty:
                     print(f"  Building from scratch. Added {len(item_final_data)} rows for '{item_name}' from {START_DATE_FROM_SCRATCH.strftime('%Y-%m-%d')}.")
            else:
                current_item_original_fundy_data = fundy_df_original[fundy_df_original['item'] == item_name]
                if not current_item_original_fundy_data.empty:
                    max_date_in_fundy = current_item_original_fundy_data['Date'].max()
                    print(f"  Most current date in Fundy.csv for '{item_name}': {max_date_in_fundy.strftime('%Y-%m-%d')}")
                    sixty_days_prior = max_date_in_fundy - timedelta(days=60)
                    
                    fundy_data_to_keep = current_item_original_fundy_data[current_item_original_fundy_data['Date'] < sixty_days_prior]
                    
                    if max_date_in_fundy.tzinfo is not None and db_data_df_processed['Date'].dt.tz is None:
                        db_data_df_processed['Date'] = db_data_df_processed['Date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(max_date_in_fundy.tzinfo)
                    elif max_date_in_fundy.tzinfo is None and db_data_df_processed['Date'].dt.tz is not None:
                        db_data_df_processed['Date'] = db_data_df_processed['Date'].dt.tz_convert(None)

                    db_data_for_update = db_data_df_processed[db_data_df_processed['Date'] >= sixty_days_prior]
                    item_final_data = pd.concat([fundy_data_to_keep, db_data_for_update], ignore_index=True)
                else:
                    print(f"  Item '{item_name}' not found in original Fundy.csv (but Fundy.csv existed). Adding all fetched data.")
                    # Apply start date if fundy existed but item was new and not building from scratch
                    item_final_data = db_data_df_processed[db_data_df_processed['Date'] >= START_DATE_FROM_SCRATCH] 
            
            updated_item_data_list.append(item_final_data)

        for item_name in fundy_df_original['item'].unique():
            if item_name not in processed_items_from_mapping: 
                print(f"\nItem '{item_name}' from Fundy.csv was not in mapping file or had no valid ticker. Preserving its original data.")
                updated_item_data_list.append(fundy_df_original[fundy_df_original['item'] == item_name])
        
        if not updated_item_data_list:
            print("No data available after fetch phase. Fundy.csv will reflect this.")
            fundy_df_after_fetch = pd.DataFrame(columns=['Date', 'item', 'value', 'region']) 
        else:
            fundy_df_after_fetch = pd.concat(updated_item_data_list, ignore_index=True)
        
        fundy_df_after_fetch.drop_duplicates(subset=['item', 'Date'], keep='last', inplace=True)
        fundy_df_after_fetch.sort_values(by=['item', 'Date'], inplace=True)
        fundy_df_after_fetch['value'] = pd.to_numeric(fundy_df_after_fetch['value'], errors='coerce')

        # Create a map of item to its original/fetched region before pivot
        item_to_region_map_from_fetched = fundy_df_after_fetch[fundy_df_after_fetch['region'].notna()][['item', 'region']].drop_duplicates('item').set_index('item')['region'].to_dict()

        print("\n--- Completed Fetch and Update of Base Series ---")
        print("\n--- Starting Calculation of Derived Series ---")
        
        if fundy_df_after_fetch.empty or not fundy_df_after_fetch['value'].notna().any():
             print("Pivoting skipped: DataFrame is empty or 'value' column contains all NaNs after fetching.")
             final_fundy_df_long = fundy_df_after_fetch 
        else:
            fundy_df_after_fetch['Date'] = pd.to_datetime(fundy_df_after_fetch['Date'])
            fundy_df_after_fetch['item'] = fundy_df_after_fetch['item'].astype(str) 
            fundy_pivot = fundy_df_after_fetch.pivot_table(index='Date', columns='item', values='value')

            def get_col(df_pivot, col_name):
                if col_name in df_pivot.columns:
                    return df_pivot[col_name].fillna(0)
                else:
                    print(f"    Warning: Column '{str(col_name)}' not found for calculation. Assuming 0.")
                    return pd.Series(0, index=df_pivot.index, name=col_name) 

            print("Calculating West region aggregates...")
            fundy_pivot['West - Ind'] = get_col(fundy_pivot, 'West[PNW] - Ind') + get_col(fundy_pivot, 'West[CA] - Ind')
            fundy_pivot['West - ResCom'] = get_col(fundy_pivot, 'West[PNW] - ResCom') + get_col(fundy_pivot, 'West[CA] - ResCom')
            fundy_pivot['West - Power'] = get_col(fundy_pivot, 'West[PNW] - Power') + get_col(fundy_pivot, 'West[CA] - Power')
            
            print("Calculating regional balances...")
            # Corrected capitalization for "SouthEast" to match item names from mapping
            regions_for_balance = ["Northeast", "Midwest", "SouthCentral", "SouthEast", "Rockies", "West"]
            
            for region_prefix in regions_for_balance:
                prod_col_name = f"{region_prefix} - Prod"
                ind_col_name = f"{region_prefix} - Ind"
                rescom_col_name = f"{region_prefix} - ResCom"
                power_col_name = f"{region_prefix} - Power"
                balance_col_name = f"{region_prefix} - Balance"
                
                fundy_pivot[balance_col_name] = get_col(fundy_pivot, prod_col_name) - \
                                        (get_col(fundy_pivot, ind_col_name) + \
                                            get_col(fundy_pivot, rescom_col_name) + \
                                            get_col(fundy_pivot, power_col_name))
                print(f"  Calculated: {balance_col_name}")

            print("Calculating CONUS - Balance...")
            conus_supply = get_col(fundy_pivot, 'CONUS - LNGimp') + \
                        get_col(fundy_pivot, 'CONUS - CADimp') + \
                        get_col(fundy_pivot, 'CONUS - Prod')
            conus_demand_elements = [
                'CONUS - Ind', 'CONUS - ResCom', 'CONUS - Power', 
                'CONUS - L&P', 'CONUS - P\'loss', 
                'CONUS - MexExp', 'CONUS - LNGexp'
            ]
            conus_demand = sum(get_col(fundy_pivot, col) for col in conus_demand_elements)
            fundy_pivot['CONUS - Balance'] = conus_supply - conus_demand
            print("  Calculated: CONUS - Balance")

            final_fundy_df_long = fundy_pivot.reset_index().melt(id_vars='Date', var_name='item', value_name='value')
            
            final_fundy_df_long['region'] = final_fundy_df_long['item'].apply(lambda x: get_region_for_item(x, item_to_region_map_from_fetched))
            
            final_fundy_df_long.dropna(subset=['value'], inplace=True) 
        
        final_fundy_df_long.sort_values(by=['item', 'Date'], inplace=True)
        
        print(f"\nSaving updated data to: {fundy_csv_path}")
        if 'Date' in final_fundy_df_long.columns and not final_fundy_df_long.empty:
             final_fundy_df_long['Date'] = pd.to_datetime(final_fundy_df_long['Date']).dt.strftime('%m/%d/%Y')
        
        output_columns = ['Date', 'item', 'value', 'region']
        for col in output_columns:
            if col not in final_fundy_df_long.columns:
                final_fundy_df_long[col] = None 

        final_fundy_df_long = final_fundy_df_long.reindex(columns=output_columns)

        final_fundy_df_long.to_csv(fundy_csv_path, index=False)
        print("Fundy.csv update process completed.")

    except FileNotFoundError as e:
        print(f"ERROR: File not found. Details: {e}")
        traceback.print_exc()
    except KeyError as e:
        print(f"ERROR: A required column was not found in one of the CSV files. Details: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine: 
            engine.dispose()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    main()