# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ICE\nymex_spread_charter_multi_year.py

import icepython as ice
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- User Configuration ---

# 1. Define the number of historical years to overlay on the chart.
YEARS_OF_HISTORY = 10

# 2. Define the base symbol and suffix for the target product.
#    Based on investigation, Henry Hub LD1 futures use 'HNG' and '-IUS'.
BASE_CODE = 'HNG'
API_SUFFIX = '-IUS'

# --- End of Configuration ---

# Standard futures month codes
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}
MONTH_NAMES_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_contract_string(contract_str):
    """Parses a 'Mon-YY' string into a datetime object."""
    try:
        month_str, year_str = contract_str.split('-')
        if month_str.title() not in MONTH_NAMES_TO_NUM:
            raise ValueError(f"Invalid month: {month_str}")
        year = int(f"20{year_str}")
        month = MONTH_NAMES_TO_NUM[month_str.title()]
        return datetime(year, month, 1)
    except (ValueError, KeyError):
        print(f"ERROR: Invalid contract format '{contract_str}'. Please use 'Mon-YY' (e.g., Aug-25).")
        return None

def get_historical_settlements(symbols, start_date, end_date):
    """
    Fetches historical settlement prices for a list of symbols for a specific date range.
    """
    print(f"--- Fetching data for {symbols} from {start_date} to {end_date} ---")
    try:
        ts_data = ice.get_timeseries(
            symbols=list(symbols),
            fields=['Settle'],
            granularity='D',
            start_date=start_date,
            end_date=end_date
        )
        if not ts_data or len(ts_data) < 2:
            print("--> WARNING: No data returned from API.")
            return None

        header = [h.replace('.Settle', '') for h in ts_data[0]]
        df = pd.DataFrame(list(ts_data[1:]), columns=header)
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

        if df.index.has_duplicates:
            print(f"--> INFO: Found and removed {df.index.duplicated().sum()} duplicate date(s).")
            df = df[~df.index.duplicated(keep='last')]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    except Exception as e:
        print(f"--> FATAL ERROR during API call: {e}")
        return None

def get_live_spread_price(contract_month_1, contract_month_2):
    """
    Constructs the correct spread symbol, fetches the live bid/Ask, and returns the mark price.
    """
    print("\nSTEP 5: Fetching Live Spread Price...")
    try:
        date1 = parse_contract_string(contract_month_1)
        date2 = parse_contract_string(contract_month_2)
        if not date1 or not date2: return None

        # Construct the symbol based on the successful investigator result.
        leg1_code = f"{BASE_CODE} {MONTH_CODES[date1.month]}{str(date1.year)[-2:]}"
        leg2_code = f"{BASE_CODE}{MONTH_CODES[date2.month]}{str(date2.year)[-2:]}"
        spread_symbol = f"{leg1_code}:{leg2_code}{API_SUFFIX}"
        
        fields_to_request = ['bid', 'Ask']
        
        print(f"--- Requesting latest quote for symbol: '{spread_symbol}' with fields {fields_to_request} ---")
        quote_data = ice.get_quotes([spread_symbol], fields_to_request)
        print(f"--- Raw API response: {quote_data} ---")

        if not quote_data or len(quote_data) < 2 or '<NotEnt>' in quote_data[1][0]:
            print("--> FAILURE: Could not retrieve live quote data.")
            return None

        data_row = quote_data[1]
        bid = pd.to_numeric(data_row[1], errors='coerce')
        ask = pd.to_numeric(data_row[2], errors='coerce')
        
        print(f"--- Parsed Bid: {bid}, Parsed Ask: {ask} ---")

        if pd.notna(bid) and pd.notna(ask):
            mark_price = (bid + ask) / 2
            print("\n" + "="*40)
            print(f"  Live Spread Mark ({contract_month_1} vs {contract_month_2})")
            print(f"  Symbol: {spread_symbol}")
            print(f"  Bid:    {bid:.4f}")
            print(f"  Ask:    {ask:.4f}")
            print(f"  Mark:   {mark_price:.4f}")
            print("="*40)
            return mark_price
        else:
            print("--> WARNING: Could not calculate live mark. Bid and/or Ask was invalid.")
            return None

    except Exception as e:
        print(f"--> FATAL ERROR during live quote request: {e}")
        return None


def analyze_historical_spread(contract_month_1, contract_month_2, live_mark=None):
    """
    Main function to orchestrate the fetching, calculation, and plotting of a single spread
    across multiple historical years.
    """
    # --- STEP 1: Determine Dates and Symbols for Each Historical Year ---
    print("\nSTEP 1: Calculating historical date ranges and symbols...")
    
    base_contract_date = parse_contract_string(contract_month_1)
    if not base_contract_date: return

    month1_code = MONTH_CODES[base_contract_date.month]
    month2_code = MONTH_CODES[parse_contract_string(contract_month_2).month]

    chart_end_date = base_contract_date - relativedelta(days=5)
    
    historical_spreads = {}
    today = datetime.now()

    for i in range(YEARS_OF_HISTORY + 1):
        current_contract_expiry_date = chart_end_date - relativedelta(years=i)
        fetch_start_date = current_contract_expiry_date - relativedelta(years=1)
        
        year_label = str(base_contract_date.year - i)
        
        year1_str = str(base_contract_date.year - i)[-2:]
        year2_str = str(parse_contract_string(contract_month_2).year - i)[-2:]

        symbol1 = f"{BASE_CODE} {month1_code}{year1_str}{API_SUFFIX}".strip()
        symbol2 = f"{BASE_CODE} {month2_code}{year2_str}{API_SUFFIX}".strip()

        # --- STEP 2: Fetch Data ---
        df = get_historical_settlements(
            [symbol1, symbol2],
            fetch_start_date.strftime('%Y-%m-%d'),
            current_contract_expiry_date.strftime('%Y-%m-%d')
        )

        if df is not None and not df.empty and symbol1 in df and symbol2 in df:
            spread = df[symbol1] - df[symbol2]
            if i == 0:
                spread = spread[spread.index <= today]
            spread.index = (spread.index - current_contract_expiry_date).days
            historical_spreads[year_label] = spread

    if not historical_spreads:
        print("Could not retrieve any historical data. Halting.")
        return

    # --- STEP 4: Combine all data and plot ---
    print("\nSTEP 4: Combining data and generating plot...")
    combined_df = pd.DataFrame(historical_spreads).sort_index()
    
    latest_year_col = str(base_contract_date.year)
    last_valid_index = None
    if latest_year_col in combined_df.columns:
        last_valid_index = combined_df[latest_year_col].last_valid_index()

    combined_df.ffill(inplace=True)

    if last_valid_index is not None and latest_year_col in combined_df.columns:
        combined_df.loc[combined_df.index > last_valid_index, latest_year_col] = np.nan

    fig, ax = plt.subplots(figsize=(18, 10))
    
    latest_year = str(base_contract_date.year)
    colors = plt.cm.tab10.colors 
    
    for i, year in enumerate(sorted(combined_df.columns)):
        label = year
        
        if year == latest_year and live_mark is not None:
             label = f"{year} (Live Mark: {live_mark:.3f})"
        elif last_valid_index is not None:
            historical_value_at_today = combined_df.loc[last_valid_index].get(year, np.nan)
            if pd.notna(historical_value_at_today):
                label = f"{year} ({historical_value_at_today:.3f})"

        if year == latest_year:
            color = 'red'
            linewidth = 3.0
            alpha = 1.0
            zorder = 10
        else:
            color = colors[i % len(colors)]
            linewidth = 1.5
            alpha = 0.8
            zorder = 5
        
        ax.plot(combined_df.index, combined_df[year], label=label, color=color, linewidth=linewidth, zorder=zorder, alpha=alpha)

    # --- Chart Formatting ---
    title = f'Historical Spread Analysis: {contract_month_1} vs {contract_month_2}'
    ax.set_title(title, fontsize=20, pad=20, weight='bold')
    ax.set_ylabel('Spread Value (USD/MMBtu)', fontsize=14)
    ax.set_xlabel('Calendar Month', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend(title='Contract Year (Value)', fontsize=11, loc='upper left')
    ax.tick_params(axis='y', labelsize=10)
    
    tick_locations = []
    tick_labels = []
    for month_offset in range(13, -1, -1):
        tick_date = chart_end_date - relativedelta(months=month_offset, day=1)
        tick_location = (tick_date - chart_end_date).days
        if combined_df.index.min() <= tick_location <= combined_df.index.max():
            tick_locations.append(tick_location)
            tick_labels.append(tick_date.strftime('%b'))

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    
    ax.set_xlim(combined_df.index.min(), combined_df.index.max())
    
    fig.tight_layout(pad=2.0)
    
    output_dir = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ICE\Outputs"
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"spread_analysis_{contract_month_1}_vs_{contract_month_2}.png"
    full_path = os.path.join(output_dir, file_name)

    plt.savefig(full_path)
    print(f"\n--> SUCCESS: Chart saved as '{full_path}'")


if __name__ == "__main__":
    contract1 = input("Enter the first contract month (e.g., Aug-25): ")
    contract2 = input("Enter the second contract month (e.g., Sep-25): ")
    
    if contract1 and contract2:
        # Get the live price first
        live_mark_price = get_live_spread_price(contract1, contract2)
        # Pass the live price into the charting function
        analyze_historical_spread(contract1, contract2, live_mark_price)
    else:
        print("Both contracts must be specified. Exiting.")
