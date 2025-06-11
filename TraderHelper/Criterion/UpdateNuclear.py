import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import traceback

def fetch_single_ticker_data(ticker_to_fetch):
    """
    Fetches data from the database for a single specified ticker string.
    """
    # --- Load Environment Variables ---
    # Assuming .env file is in the same directory as this script or the original ProcessNuclearData.py
    # Adjust the path if your .env file is located elsewhere relative to where you run this.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    if not os.path.exists(dotenv_path):
        dotenv_path = '.env' # Try loading from current working directory

    load_success = load_dotenv(dotenv_path=dotenv_path)
    if load_success:
        print(f"Loaded .env file from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path} or os.path.join(script_dir, '.env'). Relying on environment variables being pre-set.")

    # --- Database Connection Details ---
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = 'dda.criterionrsch.com'
    DB_PORT = 443
    DB_NAME = 'production'

    if not DB_USER or not DB_PASSWORD:
        print(f"ERROR: Database credentials (DB_USER, DB_PASSWORD) not found in environment variables.")
        print(f"Please ensure they are set in your .env file ({dotenv_path}) or your environment.")
        return pd.DataFrame()

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None

    try:
        engine = create_engine(DATABASE_URL)
        print(f"Attempting to connect to database: {DB_HOST}/{DB_NAME} as user {DB_USER}")

        if not ticker_to_fetch or not isinstance(ticker_to_fetch, str):
            print("No valid ticker provided.")
            return pd.DataFrame()

        # The query uses a database function that expects a comma-separated string of tickers.
        # For a single ticker, it's just that ticker.
        query = f"SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers('{ticker_to_fetch}')"
        print(f"Executing DB query for ticker: {ticker_to_fetch}")
        
        db_df = pd.read_sql_query(query, engine)
        
        if db_df.empty:
            print(f"No data returned from the database for ticker: {ticker_to_fetch}")
            return pd.DataFrame()

        print(f"Data successfully fetched for {ticker_to_fetch}.")
        
        if 'date' in db_df.columns:
            db_df['date'] = pd.to_datetime(db_df['date'], errors='coerce')
        if 'value' in db_df.columns:
            db_df['value'] = pd.to_numeric(db_df['value'], errors='coerce')
        
        # Drop rows where essential columns might be NaT/NaN after conversion
        db_df.dropna(subset=['date', 'value', 'ticker'], inplace=True)
        
        return db_df

    except Exception as e:
        print(f"An error occurred while fetching data for ticker {ticker_to_fetch}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        if engine:
            engine.dispose()
            print("Database connection closed.")

if __name__ == '__main__':
    # --- Specify the ticker you want to fetch ---
    target_ticker = "PWNA.NRC.GEN.0404600001.A" 
    # This ticker was provided by the user. 
    # Note: The image provided for NuclearPairs.csv did not show this specific root (040460),
    # so this test will confirm if data exists for it in the DB.

    print(f"--- Testing data fetch for ticker: {target_ticker} ---")
    
    # Create a dummy .env file if it doesn't exist for testing purposes
    # In a real scenario, the .env file should contain actual credentials
    if not os.path.exists('.env'):
        print("Creating a dummy .env file for testing structure. Please populate it with real credentials if needed.")
        with open('.env', 'w') as f:
            f.write("DB_USER=your_username\n")
            f.write("DB_PASSWORD=your_password\n")
            
    fetched_data = fetch_single_ticker_data(target_ticker)

    if not fetched_data.empty:
        print(f"\n--- Data for {target_ticker} (first 5 rows): ---")
        print(fetched_data.head())
        print(f"\n--- Data for {target_ticker} (last 5 rows): ---")
        print(fetched_data.tail())
        print(f"\nTotal rows fetched: {len(fetched_data)}")
        print(f"Date range: {fetched_data['date'].min()} to {fetched_data['date'].max()}")
    else:
        print(f"\nNo data was retrieved for {target_ticker}.")

