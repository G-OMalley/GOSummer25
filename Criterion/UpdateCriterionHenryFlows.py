import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

def get_db_engine():
    """Our standard, confirmed-working connection function."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(script_dir, '.env')
        load_dotenv(dotenv_path=dotenv_path, override=True)
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        if not db_user or not db_password: return None
        database_url = f"postgresql+psycopg2://{db_user}:{db_password}@dda.criterionrsch.com:443/production"
        engine = create_engine(database_url, connect_args={'sslmode': 'require', 'connect_timeout': 10})
        with engine.connect() as conn: conn.execute(text("SELECT 1"))
        print("INFO: Database engine created and connection confirmed.")
        return engine
    except Exception as e:
        print(f"FAILED to create database engine. Error: {e}")
        return None

def main():
    """Orchestrates the entire Henry Hub flows update process."""
    print("\n--- Starting Henry Hub Flows Update Process ---")
    engine = get_db_engine()
    if not engine:
        print("Halting process due to database connection failure.")
        return

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        locs_file_path = os.path.join(script_dir, '..', 'INFO', 'locs_list.csv')
        output_file_path = os.path.join(script_dir, '..', 'INFO', 'CriterionHenryFlows.csv')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        print(f"INFO: Loading tickers from {os.path.basename(locs_file_path)}...")
        locs_df = pd.read_csv(locs_file_path)
        henry_locs_df = locs_df[locs_df['market_component'].str.strip().str.lower() == 'henry']
        henry_tickers = tuple(henry_locs_df['ticker'].dropna().unique())
        if not henry_tickers:
            print("ERROR: No 'Henry' tickers found in locs_list.csv.")
            return

        with engine.connect() as connection:
            print(f"INFO: Finding metadata_ids for {len(henry_tickers)} Henry tickers...")
            id_query = text("SELECT metadata_id FROM pipelines.metadata WHERE ticker IN :tickers")
            metadata_ids_df = pd.read_sql_query(id_query, connection, params={'tickers': henry_tickers})
            metadata_ids = tuple(metadata_ids_df['metadata_id'].unique())
            if not metadata_ids:
                print("ERROR: Could not find matching metadata_ids for the tickers in your file.")
                return
            print(f"INFO: Found {len(metadata_ids)} unique metadata_ids to query for flow data.")

            start_date_for_fetch = pd.to_datetime('2015-01-01')
            existing_df = None
            try:
                print(f"INFO: Checking for existing file at {os.path.basename(output_file_path)}...")
                existing_df = pd.read_csv(output_file_path, parse_dates=['Date'])
                if not existing_df.empty:
                    max_date_in_file = existing_df['Date'].max()
                    start_date_for_fetch = max_date_in_file - timedelta(days=45)
                    print(f"INFO: Existing file found. Fetching data since {start_date_for_fetch.strftime('%Y-%m-%d')}.")
            except FileNotFoundError:
                print(f"INFO: No existing file. Fetching all data since {start_date_for_fetch.strftime('%Y-%m-%d')}.")

            print("INFO: Querying database for daily flow data...")
            flows_query = text("""
                SELECT
                    noms.eff_gas_day,
                    meta.loc_name,
                    noms.scheduled_quantity,
                    noms.operationally_available 
                FROM pipelines.nomination_points AS noms
                JOIN pipelines.metadata AS meta ON noms.metadata_id = meta.metadata_id
                WHERE noms.metadata_id IN :ids AND noms.eff_gas_day >= :start_date
            """)
            new_data_df = pd.read_sql_query(flows_query, connection, params={'ids': metadata_ids, 'start_date': start_date_for_fetch})
            print(f"INFO: Fetched {len(new_data_df)} new/updated rows from the database.")

        if new_data_df.empty and existing_df is None:
            print("WARNING: No data was found for the specified locations and date range.")
            return

        new_data_df.rename(columns={
            'eff_gas_day': 'Date',
            'scheduled_quantity': 'Scheduled Quantity',
            'operationally_available': 'Operationally Available'
        }, inplace=True)

        final_df = None
        if existing_df is not None and not existing_df.empty:
            old_data_to_keep = existing_df[existing_df['Date'] < start_date_for_fetch]
            final_df = pd.concat([old_data_to_keep, new_data_df], ignore_index=True)
            print(f"INFO: Merged {len(old_data_to_keep)} old rows with {len(new_data_df)} new rows.")
        else:
            final_df = new_data_df

        final_df['Date'] = pd.to_datetime(final_df['Date'])
        
        output_columns = ['Date', 'loc_name', 'Scheduled Quantity', 'Operationally Available']
        output_df = final_df[output_columns].copy()
        
        output_df.sort_values(by=['loc_name', 'Date'], inplace=True)
        output_df.drop_duplicates(subset=['Date', 'loc_name'], keep='last', inplace=True)

        output_df.to_csv(output_file_path, index=False, date_format='%m/%d/%Y')
        
        # --- FIXED PRINT STATEMENTS ---
        print(f"\nSUCCESS: Data processed and saved to:\n{output_file_path}")

    except Exception as e:
        # --- FIXED PRINT STATEMENTS ---
        print(f"\nERROR: AN UNEXPECTED ERROR OCCURRED: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed. Process finished.")

if __name__ == '__main__':
    main()