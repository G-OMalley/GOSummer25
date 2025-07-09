# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ICE\generate_cgb_chart.py

import icepython as ice
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import os

def find_and_plot_data(search_term, target_symbol_part, days_history=90):
    """
    Finds a specific symbol, discovers its fields, fetches data,
    and intelligently decides whether to plot a candlestick or line chart.
    """
    print("="*60)
    print("STEP 1: FINDING SYMBOL DETAILS")
    print("="*60)
    print(f"Searching for symbols with term: '{search_term}'...")
    
    best_guess_symbol = None
    try:
        search_results = ice.get_search(search_term, rows=5, symbolsOnly=False)
        if not search_results:
            print(f"--> FAILURE: No symbols found for search term: '{search_term}'")
            return

        print("--> SUCCESS: Found the following potential symbols:")
        for i, result in enumerate(search_results):
            print(f"  Result {i+1}: {result}")
            # --- FIX: Find the specific symbol we want and stop ---
            if target_symbol_part in result[0] and not best_guess_symbol:
                best_guess_symbol = result[0]

        if not best_guess_symbol:
            print(f"\n--> FAILURE: Could not find a symbol containing '{target_symbol_part}' in the results.")
            best_guess_symbol = search_results[0][0] # Fallback to first result if specific one isn't found
            print(f"    Falling back to best guess: '{best_guess_symbol}'")
        else:
             print(f"\n--> SUCCESS: Selected best match: '{best_guess_symbol}'")

    except Exception as e:
        print(f"--> FAILURE: An error occurred during symbol search: {e}")
        return

    print("\n" + "="*60)
    print(f"STEP 2: FETCHING DATA FOR '{best_guess_symbol}'")
    print("="*60)

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_history)
        end_date_str, start_date_str = end_date.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d')

        # Request a standard set of fields
        fields_to_request = ['Open', 'High', 'Low', 'Settle', 'Volume']
        
        print(f"Requesting data from {start_date_str} to {end_date_str}...")
        ts_data = ice.get_timeseries(
            symbols=best_guess_symbol,
            fields=fields_to_request,
            granularity='D',
            start_date=start_date_str,
            end_date=end_date_str
        )

        if not ts_data or len(ts_data) < 2:
            print(f"--> FAILURE: No time series data was returned from the API.")
            return

        print("--> SUCCESS: Raw data returned from API.")
        header = [h.split('.')[-1] for h in ts_data[0]]
        df = pd.DataFrame(list(ts_data[1:]), columns=header)
        df.rename(columns={'Settle': 'Close'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

        for col in df.columns:
            if col != 'Time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\n--- DataFrame after initial processing ---")
        print(df.head(10))
        
        plot_intelligent_chart(df, search_term)

    except Exception as e:
        print(f"--> FAILURE: An error occurred during data fetching or processing: {e}")

def plot_intelligent_chart(df, title):
    """
    Checks for available data and plots either a candlestick or a line chart.
    """
    print("\n" + "="*60)
    print("STEP 3: PLOTTING CHART")
    print("="*60)
    
    # --- FIX: More robust check for candlestick data ---
    # Check if Open, High, and Low columns are not ALL empty.
    has_ohl_data = not df[['Open', 'High', 'Low']].isnull().all().all()

    if has_ohl_data:
        print("--> INFO: OHLC data is present. Plotting a candlestick chart.")
        # For candlestick, we need rows where all OHLC values are present
        df_cleaned = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if df_cleaned.empty:
            print("--> FAILURE: No single day has complete Open, High, Low, and Close data.")
            return
        mpf.plot(df_cleaned,
                 type='candle',
                 style='yahoo',
                 title='\n' + title,
                 ylabel='Price (USD/MMBtu)',
                 volume='Volume' in df_cleaned.columns,
                 ylabel_lower='Volume',
                 mav=(5, 10, 20),
                 figratio=(16, 9))
    elif 'Close' in df.columns and df['Close'].notna().any():
        print("--> INFO: Only Settle/Close data is available. Plotting a line chart.")
        df_cleaned = df.dropna(subset=['Close'])
        mpf.plot(df_cleaned['Close'],
                 type='line',
                 style='yahoo',
                 title='\n' + title,
                 ylabel='Settlement Price (USD/MMBtu)',
                 figratio=(16, 9))
    else:
        print("--> FAILURE: No usable OHLC or Close data found to plot.")
        return
        
    print("--> SUCCESS: Chart has been generated.")

def main():
    """
    Main function to run the program.
    """
    search_term = "NG Basis LD1 for IF Futures - CG-Mainline - Aug25"
    # We specifically want the symbol with 'CGB Q25' in it, not the spread.
    target_symbol_part = 'CGB Q25'
    
    find_and_plot_data(search_term, target_symbol_part, days_history=90)


if __name__ == "__main__":
    main()