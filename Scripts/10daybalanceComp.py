import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta
import traceback

# --- Configuration ---
# Description: Define the file paths for input and output.
HISTORICAL_DATA_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\Fundy.csv"
FORECAST_DATA_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\FundyForecast.csv"
OUTPUT_FOLDER_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\Scripts\MarketAnalysis_Report_Output"
OUTPUT_FILENAME = "Market_Analysis_Report_Matplotlib.html"

# --- Charting Items ---
SINGLE_CHARTS = [
    "CONUS - Balance",
    "CONUS - Prod",
    "CONUS - LNGexp",
]

CHART_PAIRS = [
    ("CONUS - Power", "CONUS - ResCom"),
    ("Midwest - Power", "Midwest - ResCom"),
    ("Northeast - Power", "Northeast - ResCom"),
    ("Rockies - Power", "Rockies - ResCom"),
    ("Rockies[SW] - Power", "Rockies[SW] - ResCom"),
    ("Rockies[Up] - Power", "Rockies[Up] - ResCom"),
    ("SouthCentral - Power", "SouthCentral - ResCom"),
    ("SouthEast - Power", "SouthEast - ResCom"),
    ("SouthEast[Fl] - Power", "SouthEast[Fl] - ResCom"),
    ("SouthEast[Oth] - Power", "SouthEast[Oth] - ResCom"),
    ("West - Power", "West - ResCom"),
    ("West[CA] - Power", "West[CA] - ResCom"),
    ("West[PNW] - Power", "West[PNW] - ResCom"),
]


def create_yoy_matplotlib_chart(hist_df, forecast_df, column_name, today, image_path, fig_size=(12, 5)):
    """
    Creates and saves a single year-over-year Matplotlib chart for a given data column.
    """
    # 1. Check for forecast data availability
    if column_name == "CONUS - Balance":
        has_forecast = False
    else:
        has_forecast = column_name in forecast_df.columns
    
    # 2. Combine historical and (if available) forecast data
    df_hist = hist_df[[column_name]].copy()
    df_hist['source'] = 'hist'
    
    if has_forecast:
        df_forecast = forecast_df[[column_name]].copy()
        df_forecast['source'] = 'forecast'
        combined_df = pd.concat([df_hist, df_forecast])
        combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
    else:
        combined_df = df_hist

    # 3. Calculate 10-day rolling average
    avg_col_name = f'{column_name}_10D_Avg'
    combined_df[avg_col_name] = combined_df[column_name].rolling(window=10, min_periods=1).mean()
    
    # 4. Pivot data for year-over-year comparison
    combined_df['Year'] = combined_df.index.year
    combined_df['DayOfYear'] = combined_df.index.dayofyear
    yoy_df = combined_df.pivot_table(index='DayOfYear', columns='Year', values=avg_col_name)
    source_df_current_year = combined_df[combined_df['Year'] == today.year][['DayOfYear', 'source']].set_index('DayOfYear')

    # 5. Filter for current year and prior 5 years
    current_year = today.year
    start_year = current_year - 5
    years_to_plot = [year for year in range(start_year, current_year + 1) if year in yoy_df.columns]
    yoy_df = yoy_df[years_to_plot]
    
    # 6. Define plot window and filter data
    start_date = today - timedelta(days=60)
    end_date = today + timedelta(days=70)
    day_range = list(range(start_date.dayofyear, 367)) + list(range(1, end_date.dayofyear + 1)) if start_date.dayofyear > end_date.dayofyear else list(range(start_date.dayofyear, end_date.dayofyear + 1))
    yoy_df = yoy_df.reindex(day_range)
    
    # 7. Create Matplotlib figure
    fig, ax = plt.subplots(figsize=fig_size)
    plot_years = yoy_df.columns
    # Create a dummy index for plotting purposes on a consistent x-axis
    plot_index_dates = pd.to_datetime('2023-1-1') + pd.to_timedelta(yoy_df.index - 1, unit='D')

    # Plot past years
    for year in sorted([y for y in plot_years if y != current_year]):
        ax.plot(plot_index_dates, yoy_df[year], label=str(year), linewidth=2, alpha=0.7)

    # Plot current year
    if current_year in plot_years:
        current_year_data = yoy_df[current_year].copy()
        current_year_source = source_df_current_year.reindex(current_year_data.index)
        
        hist_part = current_year_data.where(current_year_source['source'] == 'hist')
        ax.plot(plot_index_dates, hist_part, label=f'{current_year} (Hist)', color='crimson', linewidth=2.5)

        if has_forecast:
            forecast_part = current_year_data.where(current_year_source['source'] == 'forecast')
            ax.plot(plot_index_dates, forecast_part, label=f'{current_year} (Fcst)', color='crimson', linewidth=2.5, linestyle='--')
            
    # 8. Style the chart
    ax.set_title(f'10-Day Avg: {column_name}', fontweight='bold')
    ax.set_ylabel('BCF')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Year')
    
    # Format x-axis to show month/day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # 9. Save the figure
    plt.tight_layout()
    fig.savefig(image_path)
    plt.close(fig) # Close the figure to free memory

def generate_html_report():
    """Main function to generate the complete HTML report with static images."""
    print("--- Starting Static HTML Report Generation (Matplotlib) ---")

    # --- 1. Load Data and Pivot ---
    try:
        today = pd.Timestamp.now()
        
        print("Loading and preparing historical data...")
        hist_df_long = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=['Date'])
        hist_df_long.drop_duplicates(subset=['Date', 'item'], keep='last', inplace=True)
        hist_df = hist_df_long.pivot_table(index='Date', columns='item', values='value')
        
        print("Loading and preparing forecast data...")
        forecast_df_long = pd.read_csv(FORECAST_DATA_PATH, parse_dates=['Date'])
        forecast_end_date = today + timedelta(days=15)
        forecast_df_long = forecast_df_long[forecast_df_long['Date'] <= forecast_end_date]
        forecast_df_long.drop_duplicates(subset=['Date', 'item'], keep='last', inplace=True)
        forecast_df = forecast_df_long.pivot_table(index='Date', columns='item', values='value')
        
        print(f"Forecast data truncated to end on {forecast_end_date.date()}.")
        
        hist_df.columns = hist_df.columns.str.strip()
        forecast_df.columns = forecast_df.columns.str.strip()
        print("Data successfully loaded and pivoted.")
        
    except Exception as e:
        print(f"FATAL ERROR: An error occurred during data loading or pivoting: {e}")
        traceback.print_exc()
        return
        
    # --- 2. Create Output Directories ---
    images_dir = os.path.join(OUTPUT_FOLDER_PATH, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # --- 3. Build HTML Report ---
    html_body = ""
    
    for item_name in SINGLE_CHARTS:
        if item_name in hist_df.columns:
            print(f"Generating image for: {item_name}")
            safe_filename = item_name.replace(' ', '_').replace('/', '_') + ".png"
            image_path = os.path.join(images_dir, safe_filename)
            relative_image_path = os.path.join("images", safe_filename)
            
            create_yoy_matplotlib_chart(hist_df, forecast_df, item_name, today, image_path, fig_size=(12, 5))
            
            html_body += f'<div class="chart-row"><div class="chart-container single-chart-container"><img src="{relative_image_path}" alt="{item_name}"></div></div>'
        else:
            print(f"--> WARNING: Skipping chart for '{item_name}' because it was not found.")

    for power_item, rescom_item in CHART_PAIRS:
        print(f"Generating images for: {power_item} and {rescom_item}")
        html_body += '<div class="chart-row">'
        
        for item_name in [power_item, rescom_item]:
            if item_name in hist_df.columns:
                safe_filename = item_name.replace(' ', '_').replace('/', '_') + ".png"
                image_path = os.path.join(images_dir, safe_filename)
                relative_image_path = os.path.join("images", safe_filename)
                
                create_yoy_matplotlib_chart(hist_df, forecast_df, item_name, today, image_path, fig_size=(6, 5))

                html_body += f'<div class="chart-container"><img src="{relative_image_path}" alt="{item_name}" style="width:100%;"></div>'
            else:
                print(f"--> WARNING: Skipping chart for '{item_name}' because it was not found.")
                html_body += f'<div class="chart-container" style="text-align:center; padding: 20px; display:flex; align-items:center; justify-content:center;"><div>Chart for<br><b>{item_name}</b><br>could not be generated.<br>Column not found.</div></div>'
        
        html_body += '</div>'

    # --- 4. Finalize and save HTML file ---
    final_html = f"""
    <html><head><title>Natural Gas Market Analysis</title><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;background-color:#f8f9fa}}h1{{text-align:center;color:#343a40;padding:20px;background-color:#e9ecef;margin-top:0}}.report-container{{padding:20px}}.chart-row{{display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:20px}}.chart-container{{flex:1;min-width:600px;box-shadow:0 4px 8px 0 rgba(0,0,0,0.1);border-radius:8px;overflow:hidden;background-color:white}}.single-chart-container{{max-width:1240px;margin:auto}}img{{max-width:100%;height:auto;display:block}}</style></head>
    <body><h1>Natural Gas Daily Fundamentals Report</h1><div class="report-container">{html_body}</div></body></html>"""
    
    output_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"\n--- Report generation complete. ---")
    print(f"Successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_html_report()
