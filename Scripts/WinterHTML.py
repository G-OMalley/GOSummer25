# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\Scripts\WinterHTML.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import icepython as ice # <-- IMPORT THE ICE API LIBRARY

# --- Configuration ---
# Set to True to use the ICE API, False to use the local PRICES.csv for the strip chart
USE_ICE_API = True 

# Paths to your data files.
PRICES_CSV_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\PRICES.csv"
ADMIN_CSV_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\PriceAdmin.csv"
WEATHER_CSV_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\WEATHER.csv"
INPUT_CSV_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\ICE_Winter_Strip_Prices_.csv"
# Path for the output HTML report and generated charts.
OUTPUT_DIR = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\Scripts\MarketAnalysis_Report_Output"
# Liquidity check parameters.
LIQUIDITY_WINDOW_DAYS = 30
MINIMUM_SETTLEMENTS = 25
# Charting parameters
NUM_WINTERS_TO_CHART = 3
# The number of prior winter strips to fetch and plot from the API
NUM_PRIOR_WINTERS_FOR_STRIP_CHART = 3

def generate_winter_data_and_chart(df_winter_prices, df_winter_weather, market_name, city_symbol, winter_label, charts_subdir, current_settle):
    """
    Generates a chart and a summary data table for a single winter period.
    Returns the chart filename and the summary DataFrame.
    """
    df_winter_prices = df_winter_prices.copy()
    df_winter_weather = df_winter_weather.copy()
    
    # --- 1. Calculate Price and Basis Data ---
    df_winter_prices['Basis'] = df_winter_prices[market_name] - df_winter_prices['Henry']
    df_winter_prices['Month'] = df_winter_prices['Date'].dt.month
    
    monthly_stats = df_winter_prices.groupby('Month').agg(
        AvgB=('Basis', 'mean'),
        AvgS=(market_name, 'mean')
    )

    # --- 2. Calculate Weather Data ---
    if city_symbol and not df_winter_weather.empty:
        df_winter_weather['Month'] = df_winter_weather['Date'].dt.month
        monthly_temps = df_winter_weather.groupby('Month').agg(AvgTemp=('Avg Temp', 'mean'))
        monthly_stats = monthly_stats.join(monthly_temps)
    else:
        monthly_stats['AvgTemp'] = 'N/A'

    # --- 3. Create the Summary Table DataFrame ---
    summary_data = []
    winter_months = [11, 12, 1, 2, 3] # Nov, Dec, Jan, Feb, Mar
    month_map = {11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar'}

    for month_num in winter_months:
        if month_num in monthly_stats.index:
            row = monthly_stats.loc[month_num]
            summary_data.append({
                'Month': month_map[month_num],
                'AvgB': f"{row['AvgB']:.3f}",
                'Avg$': f"{row['AvgS']:.2f}",
                'AvgTemp': f"{row['AvgTemp']:.1f}" if isinstance(row['AvgTemp'], (int, float)) else 'N/A'
            })
    
    total_winter_stats = {
        'Month': 'Winter',
        'AvgB': f"{df_winter_prices['Basis'].mean():.3f}",
        'Avg$': f"{df_winter_prices[market_name].mean():.2f}",
        'AvgTemp': f"{df_winter_weather['Avg Temp'].mean():.1f}" if city_symbol and not df_winter_weather.empty else 'N/A'
    }
    summary_data.append(total_winter_stats)
    summary_df = pd.DataFrame(summary_data)

    # --- 4. Generate and Save the Chart ---
    fig, ax1 = plt.subplots(figsize=(12, 4.2)) 
    ax1.plot(df_winter_prices['Date'], df_winter_prices['Basis'], label='Daily Basis vs Henry', color='royalblue', zorder=10)
    
    unique_months = sorted(df_winter_prices['Month'].unique())
    avg_line_label_used = False
    for month in unique_months:
        month_data = df_winter_prices[df_winter_prices['Month'] == month]
        avg_value = monthly_stats.loc[month, 'AvgB']
        month_start_date = month_data['Date'].min()
        month_end_date = month_data['Date'].max()
        label = 'Monthly Avg Basis' if not avg_line_label_used else None
        ax1.hlines(y=avg_value, xmin=month_start_date, xmax=month_end_date, color='red', linestyle='--', linewidth=2, label=label, zorder=15)
        avg_line_label_used = True

    if pd.notna(current_settle):
        ax1.axhline(y=current_settle, color='darkgray', linestyle=':', linewidth=2, label=f'Current Settle ({current_settle:.3f})', zorder=12)

    ax1.set_ylabel('Basis to Henry (USD/MMBtu)', color='royalblue', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(df_winter_prices['Date'], df_winter_prices[market_name], color='gray', alpha=0.4, label='Market Price')
    ax2.set_ylabel('Market Price (USD/MMBtu)', color='gray', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.title(f'Winter {winter_label} Basis Analysis', fontsize=14, weight='bold')
    ax1.set_xlabel('Date', fontsize=10)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    fig.autofmt_xdate()
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    file_name = f"{market_name.replace(' ', '_').replace('/', '_')}_Winter_{winter_label}.png"
    full_path = os.path.join(charts_subdir, file_name)
    plt.savefig(full_path)
    plt.close(fig)
    print(f"     - Chart saved: {file_name}")
    
    return os.path.join('charts', file_name), summary_df

def generate_strip_settlement_chart_from_ice(market_name, basis_code, charts_subdir):
    """
    *** NEW FUNCTION ***
    Generates a historical overlay chart of the winter strip's settlement price
    by fetching data directly from the ICE API for the current and prior winters.
    """
    print(f"     - Generating pre-winter settlement chart from ICE API...")
    historical_data = {}
    min_points = {} # To store the 7-day low point for each historical year
    today = datetime.now()

    current_winter_start_year = today.year if today.month < 11 else today.year + 1
    latest_year_col = f"{current_winter_start_year}-{current_winter_start_year + 1}"

    for i in range(NUM_PRIOR_WINTERS_FOR_STRIP_CHART + 1):
        start_year = current_winter_start_year - i
        end_year = start_year + 1
        winter_label = f"{start_year}-{end_year}"

        start_year_short = str(start_year)[-2:]
        end_year_short = str(end_year)[-2:]
        
        strip_symbol = f"{basis_code} X{start_year_short}SR:{basis_code}H{end_year_short}-IUS"
        
        fetch_start_date = datetime(start_year, 6, 1).strftime('%Y-%m-%d')
        fetch_end_date = datetime(start_year, 10, 25).strftime('%Y-%m-%d')
        
        print(f"       - Fetching '{strip_symbol}' from {fetch_start_date} to {fetch_end_date}")

        try:
            ts_data = ice.get_timeseries(
                symbols=[strip_symbol], fields=['Settle'], granularity='D',
                start_date=fetch_start_date, end_date=fetch_end_date
            )

            if ts_data and len(ts_data) > 1:
                df_strip = pd.DataFrame(list(ts_data[1:]), columns=['Date', 'Settle'])
                df_strip['Date'] = pd.to_datetime(df_strip['Date'])
                df_strip.dropna(subset=['Settle'], inplace=True)
                
                if not df_strip.empty:
                    df_strip.set_index('Date', inplace=True)
                    full_date_range = pd.date_range(start=df_strip.index.min(), end=df_strip.index.max(), freq='D')
                    df_strip = df_strip.reindex(full_date_range)
                    df_strip['Settle'].ffill(inplace=True)
                    
                    # --- 7-DAY LOW CALCULATION ---
                    # Calculate only for historical years, not the current one
                    if winter_label != latest_year_col:
                        df_strip['7DayAvg'] = df_strip['Settle'].rolling(window=7).mean()
                        # Find the row with the minimum 7-day average
                        if df_strip['7DayAvg'].notna().any():
                            min_avg_row = df_strip.loc[df_strip['7DayAvg'].idxmin()]
                            min_points[winter_label] = {
                                'date': min_avg_row.name.strftime('%m-%d'),
                                'value': min_avg_row['7DayAvg']
                            }
                    # --- END 7-DAY LOW CALCULATION ---

                    df_strip.reset_index(inplace=True)
                    df_strip.rename(columns={'index': 'Date'}, inplace=True)

                    df_strip.index = df_strip['Date'].dt.strftime('%m-%d')
                    historical_data[winter_label] = df_strip['Settle']
            else:
                print(f"       - No data returned for {strip_symbol}")

        except Exception as e:
            print(f"--> ERROR: An error occurred while fetching data for {strip_symbol}: {e}")
            continue

    if not historical_data:
        print(f"     - No historical data could be fetched from ICE for {market_name}.")
        return None

    df_plot = pd.DataFrame(historical_data).sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 4.2))
    
    color_map = {} # To store colors for matching dots to lines
    
    # Plot each year's data and store the color used
    for col in sorted(df_plot.columns, reverse=True):
        line, = ax.plot(df_plot.index, df_plot[col], label=col, linewidth=2 if col == latest_year_col else 1.5, alpha=1 if col == latest_year_col else 0.7)
        color_map[col] = line.get_color()

    # --- PLOT 7-DAY LOW POINTS ---
    for year, point_data in min_points.items():
        if year in color_map:
            x_pos = point_data['date']
            y_pos = point_data['value']
            color = color_map[year]
            
            # Plot the dot
            ax.scatter(x_pos, y_pos, color=color, s=100, zorder=5, ec='black', lw=1.5)
            # Add the text label
            ax.text(x_pos, y_pos, f" {y_pos:.2f}", color='black', fontsize=9, va='center_baseline', ha='left', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2', ec='none'))
    # --- END PLOTTING POINTS ---

    ax.set_title(f'{market_name} - Pre-Winter Strip Settlement History (from ICE API)', fontsize=14, weight='bold')
    ax.set_ylabel('Settlement Price (USD/MMBtu)', fontsize=10)
    ax.set_xlabel('Date (MM-DD)', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title="Winter Strip")
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    fig.autofmt_xdate()
    
    plt.tight_layout()

    file_name = f"{market_name.replace(' ', '_').replace('/', '_')}_Strip_Settle_History_API.png"
    full_path = os.path.join(charts_subdir, file_name)
    plt.savefig(full_path)
    plt.close(fig)
    print(f"     - Chart saved: {file_name}")
    
    return os.path.join('charts', file_name)


def generate_strip_settlement_chart_from_csv(market_name, basis_code, df_prices_hist, charts_subdir):
    """
    (Original function)
    Generates a historical overlay chart of the winter strip's settlement price
    using data from a local CSV.
    """
    print(f"     - Generating pre-winter settlement chart from local CSV...")
    historical_data = {}
    today = datetime.now()
    
    current_winter_start_year = today.year if today.month < 11 else today.year + 1

    for i in range(NUM_PRIOR_WINTERS_FOR_STRIP_CHART + 1):
        start_year = current_winter_start_year - i
        end_year = start_year + 1
        start_year_short = str(start_year)[-2:]
        end_year_short = str(end_year)[-2:]
        
        fetch_start_date = datetime(start_year, 6, 1)
        fetch_end_date = datetime(start_year, 10, 25)
        
        strip_symbol = f"{basis_code} X{start_year_short}SR:{basis_code}H{end_year_short}-IUS"
        
        if strip_symbol in df_prices_hist.columns:
            df_strip = df_prices_hist[df_prices_hist['Date'].between(fetch_start_date, fetch_end_date)][['Date', strip_symbol]].copy()
            df_strip.dropna(subset=[strip_symbol], inplace=True)
            
            if not df_strip.empty:
                df_strip.index = df_strip['Date'].dt.strftime('%m-%d')
                historical_data[f"{start_year}-{end_year}"] = df_strip[strip_symbol]

    if not historical_data:
        print(f"     - No historical data found for strip symbols in PRICES.csv for {market_name}.")
        return None

    df_plot = pd.DataFrame(historical_data).sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 4.2))
    
    latest_year_col = f"{current_winter_start_year}-{current_winter_start_year + 1}"
    
    for col in sorted(df_plot.columns, reverse=True):
        ax.plot(df_plot.index, df_plot[col], label=col, linewidth=2 if col == latest_year_col else 1.5)

    ax.set_title(f'{market_name} - Pre-Winter Strip Settlement History (from CSV)', fontsize=14, weight='bold')
    ax.set_ylabel('Settlement Price (USD/MMBtu)', fontsize=10)
    ax.set_xlabel('Date (MM-DD)', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    fig.autofmt_xdate()
    
    plt.tight_layout()

    file_name = f"{market_name.replace(' ', '_').replace('/', '_')}_Strip_Settle_History_CSV.png"
    full_path = os.path.join(charts_subdir, file_name)
    plt.savefig(full_path)
    plt.close(fig)
    print(f"     - Chart saved: {file_name}")
    
    return os.path.join('charts', file_name)


def build_html_report(live_prices_df, report_data):
    """
    Generates a comprehensive HTML report with a live price table and charts/summary tables for each market.
    """
    print("\nSTEP 3: Assembling the final HTML report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Winter Market Analysis Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #212529; }}
            .container {{ max-width: 1400px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
            h1, h2, h3 {{ color: #343a40; text-align: center; }}
            h1 {{ font-size: 2.5em; }}
            h2 {{ border-top: 2px solid #dee2e6; padding-top: 40px; margin-top: 40px; font-size: 2em; }}
            h3 {{ margin-top: 30px; font-size: 1.5em; color: #495057; }}
            p.timestamp {{ text-align: center; color: #6c757d; margin-top: 0; margin-bottom: 30px; }}
            .market-section {{ margin-bottom: 40px; }}
            .winter-container {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 20px; margin-top: 20px; }}
            .chart-container {{ flex: 2; min-width: 600px; }}
            .table-container {{ flex: 1; min-width: 300px; }}
            .single-chart-container {{ text-align: center; margin-top: 20px; }}
            .single-chart-container img {{ max-width: 80%; height: auto; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
            .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
            .styled-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); }}
            .styled-table thead tr {{ background-color: #007bff; color: #ffffff; text-align: left; }}
            .summary-table thead tr {{ background-color: #6c757d; }}
            .styled-table th, .styled-table td {{ padding: 10px 12px; text-align: left; }}
            .styled-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Winter Market Analysis Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    for market, data in report_data.items():
        html_content += f'<div class="market-section"><h2>{market}</h2>'
        
        market_live_data = live_prices_df[live_prices_df['Market Component'] == market]
        if not market_live_data.empty:
            # Format numeric columns to 4 decimal places for display
            display_df = market_live_data.copy()
            for col in ['Bid', 'Ask', 'Prev. Settle']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else 'N/A')
            live_table_html = display_df.to_html(index=False, classes='styled-table', border=0)
            html_content += live_table_html

        # --- Place the pre-winter strip settlement chart first ---
        if 'strip_settle_chart' in data and data['strip_settle_chart']:
            html_content += f"""
            <div class="single-chart-container">
                <h3>Pre-Winter Strip Settlement History</h3>
                <img src="{data['strip_settle_chart']}" alt="Pre-Winter Chart for {market}">
            </div>
            """

        for winter_label in sorted(data['winters'].keys(), reverse=True):
            winter_data = data['winters'][winter_label]
            chart_file = winter_data['chart_file']
            summary_table_html = winter_data['summary_df'].to_html(index=False, classes='styled-table summary-table', border=0)
            
            html_content += f"""
            <div class="winter-container">
                <div class="chart-container">
                    <img src="{chart_file}" alt="Chart for {market} {winter_label}">
                </div>
                <div class="table-container">
                    {summary_table_html}
                </div>
            </div>
            """
        html_content += '</div>'

    html_content += """
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(OUTPUT_DIR, "Winter_Analysis_Report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"--> SUCCESS: HTML report saved to '{report_path}'")

def main():
    """
    Main function to orchestrate the entire report generation process.
    """
    try:
        df_prices = pd.read_csv(PRICES_CSV_PATH, parse_dates=['Date'])
        df_admin = pd.read_csv(ADMIN_CSV_PATH)
        df_weather = pd.read_csv(WEATHER_CSV_PATH, parse_dates=['Date'])
        df_winter_prices_live = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError as e:
        print(f"--> FATAL ERROR: Could not find a required file: {e.filename}")
        return

    print(f"STEP 1: Filtering for liquid markets...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LIQUIDITY_WINDOW_DAYS)
    df_recent_prices = df_prices[df_prices['Date'].between(start_date, end_date)]
    
    potential_markets = df_admin.dropna(subset=['Fin Basis'])['Market Component'].unique()
    liquid_markets = [
        m for m in potential_markets 
        if m in df_recent_prices.columns and df_recent_prices[m].count() >= MINIMUM_SETTLEMENTS
    ]
    print(f"--> Found {len(liquid_markets)} liquid markets.")

    if not liquid_markets:
        print("\nNo markets met the liquidity criteria. Halting report generation.")
        return

    df_liquid_live_prices = df_winter_prices_live[df_winter_prices_live['Market Component'].isin(liquid_markets)].copy()

    market_to_city = df_admin.set_index('Market Component')['City Symbol'].to_dict()
    market_to_basis_code = df_admin.set_index('Market Component')['Fin Basis'].to_dict()

    print("\nSTEP 2: Generating charts and summary tables for liquid markets...")
    charts_subdirectory = os.path.join(OUTPUT_DIR, 'charts')
    os.makedirs(charts_subdirectory, exist_ok=True)
    
    full_report_data = {}
    today = datetime.now()
    
    for market in liquid_markets:
        print(f"\n--- Processing: {market} ---")
        full_report_data[market] = {'winters': {}}
        
        current_settle_series = df_liquid_live_prices[df_liquid_live_prices['Market Component'] == market]['Prev. Settle']
        current_settle = current_settle_series.iloc[0] if not current_settle_series.empty else None

        # --- CHOOSE CHARTING FUNCTION BASED ON CONFIG ---
        basis_code = market_to_basis_code.get(market)
        if pd.notna(basis_code):
            strip_chart_file = None
            if USE_ICE_API:
                # Use the new function to get data from the API
                strip_chart_file = generate_strip_settlement_chart_from_ice(market, basis_code, charts_subdirectory)
            else:
                # Use the original function to get data from the local CSV
                strip_chart_file = generate_strip_settlement_chart_from_csv(market, basis_code, df_prices, charts_subdirectory)
            
            full_report_data[market]['strip_settle_chart'] = strip_chart_file

        # Generate the three historical winter basis charts
        for i in range(NUM_WINTERS_TO_CHART):
            winter_year = today.year - 1 - i
            start_date = datetime(winter_year, 11, 1)
            end_date = datetime(winter_year + 1, 3, 31)
            winter_label = f"{winter_year}-{winter_year+1}"
            
            df_winter_p = df_prices[df_prices['Date'].between(start_date, end_date)]
            
            city = market_to_city.get(market)
            df_winter_w = pd.DataFrame()
            if pd.notna(city):
                 df_winter_w = df_weather[
                     (df_weather['Date'].between(start_date, end_date)) & 
                     (df_weather['City Symbol'] == city)
                 ]

            if not df_winter_p.empty and market in df_winter_p.columns and 'Henry' in df_winter_p.columns:
                chart_file, summary_df = generate_winter_data_and_chart(df_winter_p, df_winter_w, market, city, winter_label, charts_subdirectory, current_settle)
                full_report_data[market]['winters'][winter_label] = {'chart_file': chart_file, 'summary_df': summary_df}
    
    build_html_report(df_liquid_live_prices, full_report_data)

if __name__ == "__main__":
    main()
