# UpdateCriterionStorage.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime

def calculate_and_save_daily_storage_change():
    """
    Calculates the daily net storage change for each facility from 2015-01-01
    through today, applying the confirmed logic, and saves it to a CSV file.
    The value saved is (SUM(scheduled_quantity * rec_del_sign)) * -1.
    This means net withdrawals will be positive, net injections will be negative in the CSV.
    """
    # --- Load Environment Variables ---
    # Script determines .env path relative to its own location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env') 

    load_success = load_dotenv(dotenv_path=dotenv_path, override=True)

    if load_success:
        print(f"Loaded .env file from: {os.path.abspath(dotenv_path)} (Override enabled)")
    else:
        print(f"Warning: .env file not found at {os.path.abspath(dotenv_path)}. Relying on pre-set environment variables.")

    # --- Database Connection Details ---
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = 'dda.criterionrsch.com'
    DB_PORT = 443
    DB_NAME = 'production'

    print(f"Attempting to use DB_USER: {DB_USER}")

    if not DB_USER or not DB_PASSWORD or DB_USER == 'your_username':
        print(f"ERROR: Database credentials (DB_USER, DB_PASSWORD) not found or are placeholders.")
        print(f"Please ensure a .env file with correct credentials exists in: {os.path.abspath(dotenv_path)}")
        return

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None

    # --- Define Parameters ---
    # Output path is absolute as specified by user
    output_dir_path = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO"
    output_csv_filename = "CriterionStorageChange.csv" # Singular "Change"
    output_csv_path = os.path.join(output_dir_path, output_csv_filename)
    
    all_data_chunks = []
    
    start_process_year = 2015
    current_datetime = datetime.now() # Current date, e.g., June 1, 2025
    end_process_year = current_datetime.year

    try:
        connect_args = {'sslmode': 'require'}
        engine = create_engine(DATABASE_URL, connect_args=connect_args)
        print(f"\nSuccessfully created SQLAlchemy engine for: {DB_HOST}/{DB_NAME} as user {DB_USER} with SSL require")

        for year in range(start_process_year, end_process_year + 1):
            year_start_date_str = f"{year}-01-01"
            if year < end_process_year:
                year_end_date_str = f"{year}-12-31"
            else:
                year_end_date_str = current_datetime.strftime('%Y-%m-%d') # Up to current date for the current year
            
            print(f"\nFetching and processing data for period: {year_start_date_str} to {year_end_date_str}...")

            sql_query_chunk = f"""
            SELECT
                meta.storage_name,
                noms.eff_gas_day,
                SUM(noms.scheduled_quantity * meta.rec_del_sign) AS intermediate_daily_net_flow
                -- This sum represents (sum of positive injections) + (sum of negative withdrawals)
            FROM
                pipelines.nomination_points noms
            INNER JOIN
                pipelines.metadata meta ON meta.metadata_id = noms.metadata_id
            WHERE
                meta.storage_calc_flag = 'T'
                AND meta.category_short = 'Storage'
                AND meta.sub_category_desc = 'Daily Flows'
                AND (meta.ticker LIKE '%.7' OR meta.ticker LIKE '%.8') 
                AND noms.eff_gas_day BETWEEN '{year_start_date_str}' AND '{year_end_date_str}'
            GROUP BY
                meta.storage_name,
                noms.eff_gas_day
            ORDER BY
                meta.storage_name,
                noms.eff_gas_day;
            """
            
            with engine.connect() as connection:
                df_chunk = pd.read_sql_query(text(sql_query_chunk), connection)
            
            if not df_chunk.empty:
                print(f"Fetched {len(df_chunk)} aggregated rows for period {year_start_date_str} to {year_end_date_str}.")
                all_data_chunks.append(df_chunk)
            else:
                print(f"No data found for period {year_start_date_str} to {year_end_date_str}.")
        
        if not all_data_chunks:
            print(f"\nNo daily net storage change data found from {start_process_year}-01-01 to {current_datetime.strftime('%Y-%m-%d')} matching the criteria.")
            return

        # Concatenate all fetched chunks
        df_aggregated_flows = pd.concat(all_data_chunks, ignore_index=True)
        print(f"\nSuccessfully fetched and combined a total of {len(df_aggregated_flows)} rows.")
        
        # Apply the final step: multiply by -1 to get the daily_storage_change
        df_aggregated_flows['daily_storage_change'] = df_aggregated_flows['intermediate_daily_net_flow'] * -1
        
        # Prepare final DataFrame for CSV
        final_df = df_aggregated_flows[['storage_name', 'eff_gas_day', 'daily_storage_change']].copy()
        
        if 'eff_gas_day' in final_df.columns:
            final_df['eff_gas_day'] = pd.to_datetime(final_df['eff_gas_day'])
        
        # Sort data
        final_df.sort_values(by=['storage_name', 'eff_gas_day'], inplace=True)

        # Ensure the output directory exists
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            print(f"Created directory: {output_dir_path}")

        # Save to CSV
        try:
            final_df.to_csv(output_csv_path, index=False, date_format='%Y-%m-%d') # Using ISO date format for CSV
            print(f"Data saved to: {output_csv_path}")
        except Exception as e_csv:
            print(f"Error saving data to CSV {output_csv_path}: {e_csv}")
        
        print("\nFirst 5 rows of final daily storage changes:")
        print(final_df.head())
        if len(final_df) > 5:
            print("\nLast 5 rows of final daily storage changes:")
            print(final_df.tail())

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        if "SerializationFailure" in str(e) or "conflict with recovery" in str(e):
             print("\nADVICE: The 'SerializationFailure' error may require running during off-peak hours or smaller chunks (e.g., monthly).")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    script_dir_for_main = os.path.dirname(os.path.abspath(__file__))
    env_in_script_dir = os.path.join(script_dir_for_main, '.env')
    if not os.path.exists(env_in_script_dir):
        print(f"WARNING: .env file not found at {env_in_script_dir}! ")
        print("If you haven't, please create it with your DB_USER and DB_PASSWORD before running.")
            
    calculate_and_save_daily_storage_change()