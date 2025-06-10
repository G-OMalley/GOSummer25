import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

def calculate_and_save_daily_storage_change():
    """
    Calculates the daily net storage change for each facility.
    If the output CSV exists, it performs an incremental update for the last 60 days.
    Otherwise, it fetches all data from 2015-01-01.
    The value saved is (SUM(scheduled_quantity * rec_del_sign)) * -1.
    """
    # --- Load Environment Variables ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env') 

    load_dotenv(dotenv_path=dotenv_path, override=True)

    # --- Database Connection Details ---
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = 'dda.criterionrsch.com'
    DB_PORT = 443
    DB_NAME = 'production'

    if not DB_USER or not DB_PASSWORD:
        print(f"ERROR: Database credentials not found. Please check .env file at {dotenv_path}")
        return

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None

    # --- Configuration ---
    output_dir_path = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO"
    output_csv_filename = "CriterionStorageChange.csv"
    output_csv_path = os.path.join(output_dir_path, output_csv_filename)
    
    START_DATE_FULL_FETCH = pd.to_datetime('2015-01-01')
    
    # --- Check for existing data and determine fetch strategy ---
    existing_df = None
    start_date_for_fetch = START_DATE_FULL_FETCH
    
    os.makedirs(output_dir_path, exist_ok=True)

    try:
        print(f"Checking for existing file at: {output_csv_path}")
        existing_df = pd.read_csv(output_csv_path, parse_dates=['eff_gas_day'])
        if existing_df.empty:
            print("Existing file is empty. Performing a full data fetch.")
            existing_df = None # Treat as if file doesn't exist
        else:
            max_date_in_file = existing_df['eff_gas_day'].max()
            start_date_for_fetch = max_date_in_file - timedelta(days=60)
            print(f"Existing file found. Last date is {max_date_in_file.strftime('%Y-%m-%d')}.")
            print(f"Incremental update: fetching new data from {start_date_for_fetch.strftime('%Y-%m-%d')}.")
    except FileNotFoundError:
        print("No existing file found. Performing a full data fetch from scratch.")
    except Exception as e:
        print(f"Could not read existing file. Performing full fetch. Error: {e}")
        existing_df = None
        
    try:
        engine = create_engine(DATABASE_URL, connect_args={'sslmode': 'require'})
        print(f"\nSuccessfully created SQLAlchemy engine.")

        # --- Data Fetching ---
        # The query now covers the entire range from the determined start date to today
        today_str = datetime.now().strftime('%Y-%m-%d')
        start_date_str = start_date_for_fetch.strftime('%Y-%m-%d')
        
        print(f"\nFetching and processing data for period: {start_date_str} to {today_str}...")

        sql_query = text("""
        SELECT
            meta.storage_name,
            noms.eff_gas_day,
            SUM(noms.scheduled_quantity * meta.rec_del_sign) AS intermediate_daily_net_flow
        FROM
            pipelines.nomination_points noms
        INNER JOIN
            pipelines.metadata meta ON meta.metadata_id = noms.metadata_id
        WHERE
            meta.storage_calc_flag = 'T'
            AND meta.category_short = 'Storage'
            AND meta.sub_category_desc = 'Daily Flows'
            AND (meta.ticker LIKE '%.7' OR meta.ticker LIKE '%.8') 
            AND noms.eff_gas_day BETWEEN :start_date AND :end_date
        GROUP BY
            meta.storage_name,
            noms.eff_gas_day
        """)
        
        with engine.connect() as connection:
            new_data_df = pd.read_sql_query(sql_query, connection, params={'start_date': start_date_str, 'end_date': today_str})
        
        if new_data_df.empty:
            print(f"No new data found for period {start_date_str} to {today_str}.")
            if existing_df is None: # If no new data and no old data, nothing to do.
                return
            final_df = existing_df # If there's old data, just save that back.
        else:
            print(f"Fetched {len(new_data_df)} aggregated rows for the update period.")
            
            # --- Data Processing and Combination ---
            new_data_df['daily_storage_change'] = new_data_df['intermediate_daily_net_flow'] * -1
            new_data_df['eff_gas_day'] = pd.to_datetime(new_data_df['eff_gas_day'])
            
            # Keep only the necessary columns
            new_data_df = new_data_df[['storage_name', 'eff_gas_day', 'daily_storage_change']]
            
            if existing_df is not None:
                old_data_to_keep = existing_df[existing_df['eff_gas_day'] < start_date_for_fetch]
                final_df = pd.concat([old_data_to_keep, new_data_df], ignore_index=True)
                print(f"Combined {len(old_data_to_keep)} old rows with {len(new_data_df)} new/updated rows.")
            else:
                final_df = new_data_df
                print("Processing full dataset from scratch.")

        # Sort data and remove duplicates
        final_df.sort_values(by=['storage_name', 'eff_gas_day'], inplace=True)
        final_df.drop_duplicates(subset=['storage_name', 'eff_gas_day'], keep='last', inplace=True)

        # Save to CSV
        final_df.to_csv(output_csv_path, index=False, date_format='%Y-%m-%d')
        print(f"\nData successfully processed and saved to:\n{output_csv_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    calculate_and_save_daily_storage_change()