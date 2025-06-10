import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
import numpy as np

def update_criterion_locs():
    """
    Intelligently updates the local locs_list.csv with the latest data from
    the Criterion database. It adds new locations, updates existing ones while
    preserving manual entries, and flags orphaned records.
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

    # --- File Path Configuration ---
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'INFO'))
    output_csv_path = os.path.join(output_dir, "locs_list.csv")
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Columns ---
    # Columns to be fetched from the database
    db_columns = [
        "loc_name", "loc", "pipeline_name", "loc_zone", "category_short",
        "sub_category_desc", "sub_category_2_desc", "state_name", "county_name",
        "connecting_pipeline", "storage_name", "ticker" # Added ticker
    ]
    # Define the final order of columns in the output file
    final_column_order = [
        "loc_name", "loc", "pipeline_name", "loc_zone", "category_short",
        "sub_category_desc", "sub_category_2_desc", "state_name", "county_name",
        "connecting_pipeline", "storage_name", "market_component", "ticker"
    ]
    # Define a unique key for merging records
    merge_key = 'loc'

    try:
        # --- 1. Load Existing Local Data ---
        try:
            print(f"Loading existing data from: {output_csv_path}")
            existing_df = pd.read_csv(output_csv_path, encoding='latin1')
            existing_df[merge_key] = existing_df[merge_key].astype(str)
        except FileNotFoundError:
            print("Local 'locs_list.csv' not found. Will create a new one.")
            existing_df = pd.DataFrame(columns=final_column_order)

        # --- Handle schema drift by ensuring all expected columns exist ---
        for col in final_column_order:
            if col not in existing_df.columns:
                print(f"Adding missing column '{col}' to the local data to match the new schema.")
                existing_df[col] = np.nan

        # --- 2. Fetch Live Data from Database ---
        engine = create_engine(DATABASE_URL, connect_args={'sslmode': 'require'})
        print("\nConnecting to database to fetch live location data...")

        sql_query = text(f"""
            SELECT DISTINCT {', '.join(db_columns)}
            FROM pipelines.metadata
            WHERE country_name = 'United States'
        """)
        
        with engine.connect() as connection:
            db_df = pd.read_sql_query(sql_query, connection)
        print(f"Successfully fetched {len(db_df)} unique US locations from the database.")
        db_df[merge_key] = db_df[merge_key].astype(str)

        # --- 3. Merge and Compare Data ---
        merged_df = pd.merge(
            existing_df,
            db_df,
            on=merge_key,
            how='outer',
            suffixes=('_old', '_new'),
            indicator=True
        )

        updates_found = []
        new_records = []
        orphaned_records = []

        print("\nComparing local data with database data...")
        for index, row in merged_df.iterrows():
            if row['_merge'] == 'left_only':
                # Corrected: Build dict from non-key columns, then add the key
                orphan_data = {col: row[f"{col}_old"] for col in db_columns if col != merge_key}
                orphan_data[merge_key] = row[merge_key]
                orphan_data['market_component'] = row['market_component']
                orphan_data['loc_name'] = str(orphan_data.get('loc_name', '')) + " - ORPHANED"
                orphaned_records.append(orphan_data)

            elif row['_merge'] == 'right_only':
                # Corrected: Build dict from non-key columns, then add the key
                new_record = {col: row[f"{col}_new"] for col in db_columns if col != merge_key}
                new_record[merge_key] = row[merge_key]
                new_record['market_component'] = np.nan
                new_records.append(new_record)

            elif row['_merge'] == 'both':
                updated_record = {merge_key: row[merge_key]}
                for col in db_columns:
                    if col == merge_key: continue
                    old_val = row[f"{col}_old"]
                    new_val = row[f"{col}_new"]
                    if pd.isna(old_val) and pd.isna(new_val):
                        updated_record[col] = new_val
                    elif str(old_val) != str(new_val):
                        print(f"  - UPDATE DETECTED for loc '{row[merge_key]}':")
                        print(f"    - Column '{col}' changed from '{old_val}' to '{new_val}'")
                        updated_record[col] = new_val
                    else:
                        updated_record[col] = new_val
                
                updated_record['market_component'] = row['market_component']
                updates_found.append(updated_record)

        # --- 4. Assemble the Final DataFrame ---
        final_df = pd.concat([
            pd.DataFrame(updates_found),
            pd.DataFrame(new_records),
            pd.DataFrame(orphaned_records)
        ], ignore_index=True)

        final_df = final_df.reindex(columns=final_column_order)
        final_df.sort_values(by=['pipeline_name', 'loc_name'], inplace=True)
        
        # --- 5. Save the Updated Data ---
        final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nUpdate complete.")
        print(f"{len(new_records)} new locations added.")
        print(f"{len(orphaned_records)} orphaned locations marked.")
        print(f"File saved to: {output_csv_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")


if __name__ == '__main__':
    update_criterion_locs()
