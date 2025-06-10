import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# --- Configuration ---
# Description: Define the file paths for input and output.
HISTORICAL_DATA_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\Fundy.csv"
FORECAST_DATA_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO\FundyForecast.csv"
OUTPUT_FOLDER_PATH = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\Scripts\MarketAnalysis_Report_Output"
OUTPUT_FILENAME = "Market_Analysis_Report.html"

# --- Charting Items ---
# This list of items will be plotted if found in the data files.
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


def create_yoy_chart(hist_df, forecast_df, column_name, today):
    """
    Creates a single year-over-year Plotly chart for a given data column.
    """
    # 1. Check for forecast data availability, with a special rule for 'CONUS - Balance'
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
        combined_df = df_hist # Only use historical data

    # 3. Calculate 10-day rolling average
    avg_col_name = f'{column_name}_10D_Avg'
    combined_df[avg_col_name] = combined_df[column_name].rolling(window=10).mean()
    
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
    
    # 6. Define plot window and filter data by day of year
    start_date = today - timedelta(days=60)
    end_date = today + timedelta(days=70)
    day_range = list(range(start_date.dayofyear, 367)) + list(range(1, end_date.dayofyear + 1)) if start_date.dayofyear > end_date.dayofyear else list(range(start_date.dayofyear, end_date.dayofyear + 1))
    yoy_df = yoy_df.reindex(day_range)
    
    # 7. Create Plotly figure
    fig = go.Figure()
    plot_years = yoy_df.columns
    plot_index_dates = pd.to_datetime('2023-1-1') + pd.to_timedelta(yoy_df.index - 1, unit='D')

    # Plot past years first so they are in the background
    for year in sorted([y for y in plot_years if y != current_year]):
        fig.add_trace(go.Scatter(x=plot_index_dates, y=yoy_df[year], mode='lines', name=str(year), line=dict(width=2), opacity=0.8))

    # Plot current year on top
    if current_year in plot_years:
        current_year_data = yoy_df[current_year].copy()
        current_year_source = source_df_current_year.reindex(current_year_data.index)
        
        hist_part = current_year_data.where(current_year_source['source'] == 'hist')
        fig.add_trace(go.Scatter(x=plot_index_dates, y=hist_part, mode='lines', name=f'{current_year} (Hist)', line=dict(color='crimson', width=3.5), legendgroup=str(current_year)))

        if has_forecast:
            forecast_part = current_year_data.where(current_year_source['source'] == 'forecast')
            fig.add_trace(go.Scatter(x=plot_index_dates, y=forecast_part, mode='lines', name=f'{current_year} (Fcst)', line=dict(color='crimson', width=3.5, dash='dash'), legendgroup=str(current_year)))
            
    # 8. Style the chart
    fig.update_layout(
        title=f'<b>10-Day Avg: {column_name}</b>',
        xaxis_title='Date', yaxis_title='BCF', legend_title_text='Year',
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            tickformat='%m/%d',
            dtick="M1"  # Set ticks to the first of every month
        )
    )
    return fig

def generate_html_report():
    """Main function to generate the complete HTML report."""
    print("--- Starting HTML Report Generation ---")

    # --- 1. Load Data and Pivot ---
    try:
        today = pd.Timestamp.now()
        hist_df_long = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=['Date'])
        forecast_df_long = pd.read_csv(FORECAST_DATA_PATH, parse_dates=['Date'])
        
        # Truncate forecast data to 15 days from today before any processing
        forecast_end_date = today + timedelta(days=15)
        forecast_df_long = forecast_df_long[forecast_df_long['Date'] <= forecast_end_date]
        print(f"Forecast data truncated to end on {forecast_end_date.date()}.")

        hist_df = hist_df_long.pivot_table(index='Date', columns='item', values='value')
        forecast_df = forecast_df_long.pivot_table(index='Date', columns='item', values='value')
        
        hist_df.columns = hist_df.columns.str.strip()
        forecast_df.columns = forecast_df.columns.str.strip()
        print("Data successfully loaded and pivoted.")
        
    except Exception as e:
        print(f"FATAL ERROR: An error occurred during data loading or pivoting: {e}")
        return
        
    # --- 2. Create Output Directory ---
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    # --- 3. Build HTML Report ---
    html = """
    <html><head><title>Natural Gas Market Analysis</title><style>
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;background-color:#f8f9fa}h1{text-align:center;color:#343a40;padding:20px;background-color:#e9ecef;margin-top:0}.report-container{padding:20px}.chart-row{display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:20px}.chart-container{flex:1;min-width:600px;box-shadow:0 4px 8px 0 rgba(0,0,0,0.1);border-radius:8px;overflow:hidden;background-color:white}.single-chart-container{max-width:1240px;margin:auto}</style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
    <body><h1>Natural Gas Daily Fundamentals Report</h1><div class="report-container">"""
    
    # Generate single charts
    for item_name in SINGLE_CHARTS:
        if item_name in hist_df.columns:
            print(f"Generating chart for: {item_name}")
            fig = create_yoy_chart(hist_df, forecast_df, item_name, today)
            chart_div = fig.to_html(full_html=False, default_height=500)
            html += f'<div class="chart-row"><div class="chart-container single-chart-container">{chart_div}</div></div>'
        else:
            print(f"--> WARNING: Skipping chart for '{item_name}' because it was not found in the historical data.")

    # Generate side-by-side charts
    for power_item, rescom_item in CHART_PAIRS:
        print(f"Generating charts for: {power_item} and {rescom_item}")
        html += '<div class="chart-row">'
        
        # Power Chart
        if power_item in hist_df.columns:
            fig_power = create_yoy_chart(hist_df, forecast_df, power_item, today)
            html += f'<div class="chart-container">{fig_power.to_html(full_html=False, default_height=500)}</div>'
        else:
            print(f"--> WARNING: Skipping chart for '{power_item}' because it was not found.")
            html += f'<div class="chart-container" style="text-align:center; padding: 20px;">Chart for<br><b>{power_item}</b><br>could not be generated.<br>Column not found.</div>'

        # Rescom Chart
        if rescom_item in hist_df.columns:
            fig_rescom = create_yoy_chart(hist_df, forecast_df, rescom_item, today)
            html += f'<div class="chart-container">{fig_rescom.to_html(full_html=False, default_height=500)}</div>'
        else:
            print(f"--> WARNING: Skipping chart for '{rescom_item}' because it was not found.")
            html += f'<div class="chart-container" style="text-align:center; padding: 20px;">Chart for<br><b>{rescom_item}</b><br>could not be generated.<br>Column not found.</div>'
        html += '</div>'

    # --- 4. Finalize and save HTML file ---
    html += """</div></body></html>"""
    output_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"\n--- Report generation complete. ---")
    print(f"Successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_html_report()
