# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ICE\ICEForwards_Final_Live.py

import icepython as ice
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# --- Configuration ---
# The base commodity code for the futures contract (e.g., 'CGB' for Canadian Gas Basis)
BASE_CODE = 'CGB'
# The suffix required by the ICE API for this specific product
API_SUFFIX = '-IUS'
# The starting month and year for the forward curve
START_CONTRACT_DATE = datetime(2025, 8, 1)
# The number of forward months to include in the analysis
NUM_FORWARD_MONTHS = 12

# --- Dynamic Date Generation ---
# This section automatically calculates the historical dates for the analysis.
today = datetime.now()
# The dictionary keys are the dates used for API calls.
# The values are the friendly names that will appear as row labels in the final table and chart legend.
HISTORICAL_DATE_NAMES = {
    (today - relativedelta(days=1)).strftime('%Y-%m-%d'): 'Yesterday',
    (today - relativedelta(days=7)).strftime('%Y-%m-%d'): 'Today -7d',
    (today - relativedelta(days=14)).strftime('%Y-%m-%d'): 'Today -14d',
    (today - relativedelta(days=28)).strftime('%Y-%m-%d'): 'Today -28d',
    (today - relativedelta(days=56)).strftime('%Y-%m-%d'): 'Today -56d',
}

# Standard futures month codes
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

def generate_forward_symbols(base_code, start_date, num_months):
    """
    Constructs a list of valid ICE futures symbols and corresponding column headers.
    
    Args:
        base_code (str): The root code for the commodity (e.g., 'CGB').
        start_date (datetime): The first contract month to generate.
        num_months (int): The number of forward months to generate.

    Returns:
        tuple: A tuple containing a list of full symbols and a list of friendly headers.
    """
    symbols, column_headers = [], []
    for i in range(num_months):
        contract_date = start_date + relativedelta(months=i)
        month_char = MONTH_CODES[contract_date.month]
        symbol_year = str(contract_date.year)[-2:]
        
        symbol = f"{base_code} {month_char}{symbol_year}{API_SUFFIX}"
        header = contract_date.strftime('%b-%y')

        symbols.append(symbol)
        column_headers.append(header)
    return symbols, column_headers

def get_settlements_on_date(symbols, date_str):
    """
    Fetches settlement prices for a list of symbols on a specific date using the icepython library.
    This function is designed to parse the "wide" data format returned by the API.
    
    Args:
        symbols (list): A list of full ICE symbols to request data for.
        date_str (str): The date for the data request in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary mapping each symbol to its settlement price. Returns an empty dict on failure.
    """
    print(f"--- Fetching live settlements for date: {date_str} ---")
    try:
        # Make the live API call to the ICE service
        ts_data = ice.get_timeseries(
            symbols=symbols,
            fields=['Settle'],
            granularity='D',
            start_date=date_str,
            end_date=date_str
        )

        # Check if the API returned any data
        if not ts_data or len(ts_data) < 2:
            print(f"--> WARNING: No data returned for {date_str} (likely a weekend or non-trading day).")
            return {}

        # The first element of the returned tuple is the header row
        header = ts_data[0]
        # The second element is the data row
        data_row = ts_data[1]
        settlements = {}

        # Parse the "wide" format where each column is 'Symbol.Field'
        for i, column_name in enumerate(header):
            if "Settle" in column_name:
                # Extract the symbol from the column header (e.g., 'CGB Q25-IUS.Settle' -> 'CGB Q25-IUS')
                symbol_key = column_name.replace('.Settle', '')
                # Get the corresponding price and convert it to a number
                price = pd.to_numeric(data_row[i], errors='coerce')
                settlements[symbol_key] = price

        print(f"--> SUCCESS: Parsed {len(settlements)} prices.")
        return settlements

    except Exception as e:
        print(f"--> FATAL ERROR: An exception occurred during API call. Reason: {e}")
        return {}

def plot_forward_curves(df, title):
    """
    Plots the forward curves from the DataFrame with custom styling.
    The "Yesterday" curve is made bold, and colors are sequential.
    
    Args:
        df (pd.DataFrame): DataFrame with historical dates as the index and contract months as columns.
        title (str): The title for the chart.
    """
    if df.isnull().all().all():
        print("Cannot plot chart because the DataFrame is empty or all values are NaN.")
        return

    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_transposed = df_numeric.T

    # --- Setup for custom plotting ---
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # --- EDIT: New sequential green color palette from dark to light ---
    colors = ['#00441b', '#1b7837', '#5aae61', '#a6dba0', '#d9f0d3']
    
    # Plot each line individually to apply custom styles
    for i, date_label in enumerate(df_transposed.columns):
        # --- EDIT: Make the 'Yesterday' line thicker ---
        linewidth = 3.5 if date_label == 'Yesterday' else 1.5
        zorder = 10 if date_label == 'Yesterday' else 5
        
        ax.plot(df_transposed.index, df_transposed[date_label], 
                color=colors[i], 
                linewidth=linewidth, 
                label=date_label, 
                marker='o', 
                markersize=4,
                zorder=zorder)

    # --- Chart Formatting ---
    ax.set_title(title, fontsize=18, pad=20, weight='bold')
    ax.set_ylabel('Settlement Price (USD/MMBtu)', fontsize=12)
    ax.set_xlabel('Contract Month', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Historical Date', fontsize=11)
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    plt.tight_layout(pad=2.0)

    # Save the chart to a file
    file_name = "forward_curves_plot_styled.png"
    plt.savefig(file_name)
    print(f"\nChart saved successfully as '{file_name}'")

def build_and_plot_curves():
    """Main function to orchestrate building the table and plotting the chart."""
    print("STEP 1: Generating Forward Contract Symbols...")
    symbols, column_headers = generate_forward_symbols(BASE_CODE, START_CONTRACT_DATE, NUM_FORWARD_MONTHS)
    
    print("\nSTEP 2: Fetching Historical Data...")
    all_data = []
    symbol_to_header = dict(zip(symbols, column_headers))

    for date_str_key, date_name in HISTORICAL_DATE_NAMES.items():
        settlements = get_settlements_on_date(symbols, date_str_key)
        
        row_data = {'Historical Date': date_name}
        for symbol, header in symbol_to_header.items():
            row_data[header] = settlements.get(symbol)
        all_data.append(row_data)

    df = pd.DataFrame(all_data).set_index('Historical Date')

    print("\n" + "="*50)
    print("STEP 3: Final Forward Curve Table")
    print("="*50)
    print(df.to_string(float_format="%.4f", na_rep="NaN"))

    print("\nSTEP 4: Generating Plot...")
    plot_forward_curves(df, f'Historical Forward Curves for {BASE_CODE}')

if __name__ == "__main__":
    build_and_plot_curves()
