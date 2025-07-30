# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ICE\winter_strip_data_exporter.py

import icepython as ice
import pandas as pd
from datetime import datetime
import os

# --- Configuration ---
# Paths to your data files.
ADMIN_CSV_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\PriceAdmin.csv"
# Path for the output file.
OUTPUT_DIR = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO"
# The market suffix for the products we are analyzing.
API_SUFFIX = '-IUS'

def fetch_and_save_winter_strip_prices():
    """
    Reads PriceAdmin.csv, fetches live winter strip prices for all markets,
    and saves the results to a new CSV file in the INFO folder.
    """
    print("STEP 1: Reading market data from PriceAdmin.csv...")
    try:
        df_admin = pd.read_csv(ADMIN_CSV_PATH)
        df_admin.dropna(subset=['Fin Basis'], inplace=True)
        print(f"--> Found {len(df_admin)} potential markets with 'Fin Basis' codes.")

        # --- Construct winter strip symbols ---
        today = datetime.now()
        start_year = today.year if today.month < 11 else today.year + 1
        end_year = start_year + 1
        start_year_short = str(start_year)[-2:]
        end_year_short = str(end_year)[-2:]

        market_to_symbol = {
            row['Market Component']: f"{row['Fin Basis']} X{start_year_short}SR:{row['Fin Basis']}H{end_year_short}{API_SUFFIX}"
            for _, row in df_admin.iterrows()
        }

        # --- Fetch prices from the API ---
        print(f"\nSTEP 2: Fetching live prices for {len(market_to_symbol)} winter strips...")
        symbols_to_query = list(market_to_symbol.values())
        fields_to_request = ['bid', 'Ask', 'recset']
        
        quote_data = ice.get_quotes(symbols_to_query, fields_to_request)
        
        if not quote_data or len(quote_data) < 2:
            print("--> WARNING: No data returned from the API for any symbols.")
            return
            
        price_results = {row[0]: row[1:] for row in quote_data[1:]}

        # --- Prepare the output data ---
        output_data = []
        for market_name, symbol in market_to_symbol.items():
            prices = price_results.get(symbol)
            if prices:
                bid = pd.to_numeric(prices[0], errors='coerce')
                ask = pd.to_numeric(prices[1], errors='coerce')
                settle = pd.to_numeric(prices[2], errors='coerce')
                
                # Only include rows where we have at least one piece of price data
                if pd.notna(bid) or pd.notna(ask) or pd.notna(settle):
                    output_data.append({
                        'Market Component': market_name,
                        'Bid': bid,
                        'Ask': ask,
                        'Prev. Settle': settle,
                        'Symbol': symbol
                    })
        
        if not output_data:
            print("\n--> No valid price data was found for any of the generated symbols.")
            return

        # --- Save the results to a CSV file ---
        output_df = pd.DataFrame(output_data)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_name = f"ICE_Winter_Strip_Prices_.csv"
        full_path = os.path.join(OUTPUT_DIR, file_name)
        
        output_df.to_csv(full_path, index=False, float_format='%.4f')
        print(f"\nSTEP 3: SUCCESS! Price table with {len(output_df)} markets saved to '{full_path}'")

    except FileNotFoundError:
        print(f"--> FATAL ERROR: The file was not found at '{ADMIN_CSV_PATH}'")
    except Exception as e:
        print(f"--> An error occurred: {e}")


if __name__ == "__main__":
    fetch_and_save_winter_strip_prices()