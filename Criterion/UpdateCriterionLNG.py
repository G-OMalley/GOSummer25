import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

def process_lng_data(engine, data_type, tickers, description_source, output_file_path):
    """
    Generic function to fetch, process, and save LNG data.
    - data_type: 'HISTORICAL' or 'FORECAST'
    - tickers: A list of ticker strings to query.
    - description_source: Either the string 'DB' to use database descriptions, or a
      dictionary to map tickers to hard-coded descriptions.
    - output_file_path: The full path to the output CSV file.
    """
    print(f"\n{'='*20} Processing {data_type} LNG Data {'='*20}")
    
    TICKERS_STRING = ",".join(tickers)
    
    # --- Determine Fetch Strategy based on Data Type ---
    start_date_for_fetch = pd.to_datetime('2015-01-01')
    existing_df = None

    if data_type == 'HISTORICAL':
        try:
            print(f"Checking for existing file at: {output_file_path}")
            existing_df = pd.read_csv(output_file_path, parse_dates=['Date'])
            if existing_df.empty:
                print("Existing file is empty. Performing a full data fetch.")
                existing_df = None
            else:
                max_date_in_file = existing_df['Date'].max()
                start_date_for_fetch = max_date_in_file - timedelta(days=60)
                print(f"Existing file found. Last date: {max_date_in_file.strftime('%Y-%m-%d')}.")
                print(f"Incremental update: fetching data from {start_date_for_fetch.strftime('%Y-%m-%d')}.")
        except FileNotFoundError:
            print("No existing file found. Performing a full data fetch.")
        except Exception as e:
            print(f"Could not read existing file due to error. Performing full fetch. Error: {e}")
    elif data_type == 'FORECAST':
        print(f"Forecast mode: Overwriting {os.path.basename(output_file_path)} completely.")
        start_date_for_fetch = pd.to_datetime(datetime.now().date()) # From today
        
    # --- Data Fetching ---
    sql_query = text("SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers(:tickers)")
    print(f"Executing query for tickers: {TICKERS_STRING}")
    
    db_data_df = pd.read_sql_query(sql_query, engine, params={'tickers': TICKERS_STRING})

    if db_data_df.empty:
        print(f"No {data_type.lower()} data returned from the database for the specified tickers.")
        return
    print(f"Successfully fetched {len(db_data_df)} rows of {data_type.lower()} data from database.")

    # --- Data Validation and Processing ---
    if 'series_desc' in db_data_df.columns:
        db_data_df.rename(columns={'series_desc': 'series_description'}, inplace=True)
        
    # Apply descriptions based on the specified source
    if description_source == 'DB':
        if 'series_description' not in db_data_df.columns:
             print("ERROR: 'series_description' column not found in database results for historical data.")
             return
        db_data_df['Item'] = db_data_df['series_description']
    elif isinstance(description_source, dict):
        db_data_df['Item'] = db_data_df['ticker'].map(description_source)
    else:
        print("ERROR: Invalid description_source provided.")
        return

    if 'state_name' not in db_data_df.columns:
        print("Warning: 'state_name' not returned from DB. Column will be blank.")
        db_data_df['state_name'] = 'Unknown'
            
    db_data_df['date'] = pd.to_datetime(db_data_df['date'])

    # --- Filter by Date Range ---
    if data_type == 'FORECAST':
        end_date_for_fetch = start_date_for_fetch + timedelta(days=60)
        print(f"Filtering forecast data between {start_date_for_fetch.strftime('%Y-%m-%d')} and {end_date_for_fetch.strftime('%Y-%m-%d')}.")
        new_data_df = db_data_df[
            (db_data_df['date'] >= start_date_for_fetch) & 
            (db_data_df['date'] <= end_date_for_fetch)
        ].copy()
    else: # HISTORICAL
        new_data_df = db_data_df[db_data_df['date'] >= start_date_for_fetch].copy()

    # --- Combine old and new data (for Historical only) ---
    if data_type == 'HISTORICAL' and existing_df is not None:
        old_data_to_keep = existing_df[existing_df['Date'] < start_date_for_fetch]
        new_data_df.rename(columns={'date': 'Date', 'value': 'Value'}, inplace=True)
        final_df = pd.concat([old_data_to_keep, new_data_df], ignore_index=True)
        print(f"\nCombined {len(old_data_to_keep)} old rows with {len(new_data_df)} new/updated rows.")
    else:
        new_data_df.rename(columns={'date': 'Date', 'value': 'Value'}, inplace=True)
        final_df = new_data_df
        print(f"\nProcessing {data_type} dataset from scratch.")

    # --- Final Formatting and Save ---
    final_df.sort_values(by=['Item', 'Date'], inplace=True)
    final_df.drop_duplicates(subset=['Date', 'Item'], keep='last', inplace=True)
    
    final_output_cols = ['Date', 'Item', 'Value', 'state_name']
    final_df = final_df.reindex(columns=final_output_cols)
    
    final_df.to_csv(output_file_path, index=False, date_format='%m/%d/%Y')
    print(f"\nData successfully processed and saved to:\n{output_file_path}")


def main():
    """
    Main function to orchestrate fetching both historical and forecast LNG data.
    """
    # --- Load Environment Variables ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')

    if not os.path.exists(dotenv_path):
        print(f"ERROR: .env file not found at {dotenv_path}. Please create it with your DB_USER and DB_PASSWORD.")
        return

    load_dotenv(dotenv_path=dotenv_path, override=True)

    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')

    if not DB_USER or not DB_PASSWORD:
        print("ERROR: Database credentials not found in .env file.")
        return

    # --- Configuration ---
    DB_HOST = 'dda.criterionrsch.com'
    DB_PORT = 443
    DB_NAME = 'production'
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'INFO'))
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Tickers and Descriptions ---
    hist_tickers = [
        'PLAG.LNGEXP.SUM.CALCP.A', 'PLAG.LNGEXP.SUM.SPL.A', 'PLAG.LNGEXP.SUM.CCL.A',
        'PLAG.LNGEXP.SUM.COVE.A', 'PLAG.LNGEXP.SUM.ELBA.A', 'PLAG.LNGEXP.SUM.CAMER.A',
        'PLAG.LNGEXP.SUM.FLNG.A', 'PLAG.LNGEXP.SUM.PLQ.A'
    ]
    
    forecast_tickers_map = {
        'PLAG.LNGEXP.CAMER.MAINTCAP.F': 'Cameron LNG Feed Gas',
        'PLAG.LNGEXP.CCL.MAINTCAP.F': 'Corpus Christi LNG Feed Gas',
        'PLAG.LNGEXP.FLNG.MAINTCAP.F': 'Freeport LNG Feed Gas',
        'PLAG.LNGEXP.SPL.MAINTCAP.F': 'Sabine Pass LNG Feed Gas',
        'PLAG.LNGEXP.CALCP.MAINTCAP.F': 'Calcasieu Pass LNG Feed Gas',
        'PLAG.LNGEXP.ELBA.MAINTCAP.F': 'Elba Island LNG Feed Gas',
        'PLAG.LNGEXP.PLQ.MAINTCAP.F': 'US LNG Exports - Plaquemines LNG',
        'PLAG.LNGEXP.COVE.MAINTCAP.F': 'Cove Point LNG Feed Gas'
    }

    engine = None
    try:
        connection_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_url, connect_args={'sslmode': 'require', 'connect_timeout': 10})
        print("Successfully created database engine.")

        # --- Process Historical Data ---
        hist_output_path = os.path.join(output_dir, "CriterionLNGHist.csv")
        process_lng_data(engine, 'HISTORICAL', hist_tickers, 'DB', hist_output_path)

        # --- Process Forecast Data ---
        forecast_output_path = os.path.join(output_dir, "CriterionLNGForecast.csv")
        process_lng_data(engine, 'FORECAST', list(forecast_tickers_map.keys()), forecast_tickers_map, forecast_output_path)

    except Exception as e:
        print(f"An error occurred during the main process: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    main()
