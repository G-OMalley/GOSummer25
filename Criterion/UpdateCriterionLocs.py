# fetch_criterion_locs_final.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback

def fetch_and_save_locations():
    """
    Fetches location data using a specific SQL query and saves it to a CSV file.
    """
    # --- Load Environment Variables ---
    # Assumes .env file is in the same directory as the script.
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
        return # Changed from exit() to return for better control if used as a module

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None

    # SQL Query provided by the user
    sql_query = """
    SELECT  
        pipeline_name,
        tsp,
        tsp_short,
        ticker,
        loc_name,
        loc,
        loc_qti_id,
        loc_qti_short,
        loc_purp_desc,
        rec_del_sign,
        loc_zone,
        category_short,
        sub_category_desc,
        sub_category_2_desc,
        country_name,
        state_name,
        state_abb,
        province_name,
        offshore_block_name,
        county_name,
        latitude,
        longitude,
        connecting_pipeline,
        connecting_entity,
        storage_name,
        storage_calc_flag,
        units
    FROM pipelines.metadata
    GROUP BY 
        pipeline_name, tsp, tsp_short, ticker, loc_name, loc, loc_qti_id,
        loc_qti_short, loc_purp_desc, rec_del_sign, loc_zone, category_short,
        sub_category_desc, sub_category_2_desc, country_name, state_name,
        state_abb, province_name, offshore_block_name, county_name, latitude,
        longitude, connecting_pipeline, connecting_entity, storage_name,
        storage_calc_flag, units
    ORDER BY 
        pipeline_name, ticker, loc_name, loc_qti_id;
    """

    try:
        connect_args = {'sslmode': 'require'}
        engine = create_engine(DATABASE_URL, connect_args=connect_args)
        
        print(f"\nSuccessfully created SQLAlchemy engine for: {DB_HOST}/{DB_NAME} as user {DB_USER} with SSL require")
        print("\nExecuting SQL query to fetch location data...")
        
        with engine.connect() as connection:
            df_locations = pd.read_sql_query(text(sql_query), connection)
        
        if df_locations.empty:
            print(f"No data returned from the query for table 'pipelines.metadata'.")
        else:
            print(f"Successfully fetched {len(df_locations)} rows.")

            # --- Save to CSV ---
            # Output will be in the same directory as the script.
            output_file_name = 'CriterionLOCS.csv'
            output_file_path = os.path.join(script_dir, output_file_name)

            df_locations.to_csv(output_file_path, index=False)
            print(f"Data saved successfully to: {output_file_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    # Ensure a .env file exists, or create a dummy one if it doesn't.
    # User should replace dummy credentials with actual ones.
    if not os.path.exists('.env'):
        script_dir_for_main = os.path.dirname(os.path.abspath(__file__))
        env_in_script_dir = os.path.join(script_dir_for_main, '.env')
        print(f"WARNING: .env file not found at {env_in_script_dir}! ")
        print("If you haven't, please create it with your DB_USER and DB_PASSWORD.")
        # Example of creating a dummy if truly needed, but user should manage this.
        # with open(env_in_script_dir, 'w') as f:
        #     f.write("DB_USER=your_username\n")
        #     f.write("DB_PASSWORD=your_password\n")
        # print(f"A dummy .env was conceptualized; please ensure it's correct at {env_in_script_dir}")
            
    fetch_and_save_locations()