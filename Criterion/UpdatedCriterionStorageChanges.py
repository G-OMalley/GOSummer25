import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

def get_storage_changes_last_n_days(days=5):
    """
    Fetches daily storage changes for the last N days and saves them to a CSV file.
    """
    # --- Load Environment Variables ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    
    if not os.path.exists(dotenv_path):
        dotenv_path = '.env'

    load_success = load_dotenv(dotenv_path=dotenv_path, override=True)
    
    if load_success:
        print(f"Loaded .env file from: {dotenv_path} (Override enabled)")
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Relying on environment variables being pre-set.")

    # --- Database Connection Details ---
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = 'dda.criterionrsch.com'
    DB_PORT = 443
    DB_NAME = 'production'

    print(f"Attempting to use DB_USER: {DB_USER}") 

    if not DB_USER or not DB_PASSWORD or DB_USER == 'your_username': 
        print(f"ERROR: Database credentials (DB_USER, DB_PASSWORD) not found or are placeholders in environment variables.")
        print(f"Please ensure they are correctly set in your .env file: {dotenv_path}")
        return

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None

    try:
        engine = create_engine(DATABASE_URL)
        print(f"Attempting to connect to database: {DB_HOST}/{DB_NAME} as user {DB_USER}")

        # --- Calculate Start Date for Query ---
        today = datetime.now()
        start_date = today - timedelta(days=days)
        start_date_str = start_date.strftime('%Y-%m-%d')
        print(f"Fetching data from {start_date_str} onwards (last {days} days).")

        # --- SQL Query (Modified to try meta.name for location identifier and ordering) ---
        sql_query = f"""
        SELECT distinct
            meta.name,  -- Trying 'meta.name'
            meta.state_name,
            meta.zone_name,
            meta.category_short,
            meta.sub_category_desc,
            meta.rec_del_sign,
            prod.cycle_id,
            prod.cycle_desc,
            prod.hourly_cycle_id,
            prod.eff_gas_day,
            dat.weekthu as eff_gas_day_thu,
            prod.end_eff_gas_day,
            prod.design_capacity,
            prod.operating_capacity,
            prod.scheduled_quantity,
            prod.scheduled_quantity * meta.rec_del_sign * -1 as schqty_sign_ad,
            prod.operationally_available,
            reg.region_name,
            reg.eia_ng_regions
        FROM pipelines.nomination_points prod
        INNER JOIN pipelines.metadata meta on meta.metadata_id = prod.metadata_id
        INNER JOIN misc.date_master dat on dat.date = prod.eff_gas_day
        LEFT JOIN pipelines.regions reg on reg.state_name = meta.state_name
        WHERE 
            meta.category_short = 'Storage' 
            AND meta.sub_category_desc IN ('Daily Flows') 
            AND prod.eff_gas_day >= '{start_date_str}'
        ORDER BY prod.eff_gas_day DESC, meta.name; -- Ordering by 'meta.name'
        """

        print("\nExecuting SQL query...")
        with engine.connect() as connection:
            df_storage_changes = pd.read_sql_query(text(sql_query), connection)
        
        if df_storage_changes.empty:
            print(f"No storage change data found for the last {days} days.")
        else:
            print(f"Successfully fetched {len(df_storage_changes)} rows of storage change data.")

            output_dir = os.path.join(script_dir, 'INFO')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            
            output_file_name = 'CriterionStorageChanges.csv'
            output_file_path = os.path.join(output_dir, output_file_name)

            df_storage_changes.to_csv(output_file_path, index=False)
            print(f"Data saved successfully to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("Database connection closed.")

if __name__ == '__main__':
    if not os.path.exists('.env'):
        print("WARNING: .env file not found! Attempting to create a dummy .env file.")
        print("IMPORTANT: Please populate this .env file with your actual database credentials.")
        with open('.env', 'w') as f:
            f.write("DB_USER=your_username\n")
            f.write("DB_PASSWORD=your_password\n")
            
    number_of_days_to_fetch = 5 
    get_storage_changes_last_n_days(days=number_of_days_to_fetch)
