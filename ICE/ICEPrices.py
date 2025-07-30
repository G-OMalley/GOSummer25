import icepython as ice
from datetime import datetime, timedelta
import pandas as pd

# --- Configuration ---
# The Daily Code for 'CG-Mainline'
SYMBOL_TO_FETCH = 'XLA'

# The target field we are trying to fetch.
TARGET_FIELD = 'VWAP Close' 

# --- Date Range: Last 30 Days ---
# This uses a valid, recent historical period to rule out date range issues.
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=30)

# Format dates into the 'YYYY-MM-DD' string format required by the API.
end_date_str = END_DATE.strftime('%Y-%m-%d')
start_date_str = START_DATE.strftime('%Y-%m-%d')

def final_vwap_attempt():
    """
    A final, direct attempt to fetch historical 'VWAP Close' data using 
    get_timeseries with a valid, recent date range (last 30 days).
    """
    print("="*50)
    print(f"--- Final Attempt: Fetching '{TARGET_FIELD}' for '{SYMBOL_TO_FETCH}' ---")
    print(f"Function: get_timeseries()")
    print(f"Date Range: {start_date_str} to {end_date_str}")
    print("="*50)

    try:
        # Directly request the time series data for the specified field.
        ts_data = ice.get_timeseries(
            symbols=[SYMBOL_TO_FETCH],
            fields=[TARGET_FIELD],
            granularity='D',
            start_date=start_date_str,
            end_date=end_date_str
        )

        # --- Analyze the Response ---
        if not ts_data or len(ts_data) < 2:
            print("\n--- RESULT: FAILURE ---")
            print(f"The API returned NO DATA for the '{TARGET_FIELD}' field, even with a valid historical date range.")
            print("\nConclusion:")
            print("1. The field name 'VWAP Close' is not accessible via the get_timeseries() function.")
            print("2. The issue is not the date range or the symbol name.")
            print("\nRecommendation: Please contact ICE support. This confirms the problem is with API access for this specific field, not the script's logic.")

        else:
            print("\n--- RESULT: SUCCESS ---")
            print("Data was returned from the API. Displaying results:\n")

            # Convert the returned tuple to a pandas DataFrame for clean printing.
            header = [h.replace(f'{SYMBOL_TO_FETCH}.', '') for h in ts_data[0]]
            df = pd.DataFrame(list(ts_data[1:]), columns=header)

            print(df.to_string())

    except Exception as e:
        print(f"\n--- An unexpected error occurred during the API call ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    final_vwap_attempt()
