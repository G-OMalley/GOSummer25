import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import io
import numpy as np

# ==============================================================================
#  CONFIGURATION CONSTANTS
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
INFO_DIR = SCRIPT_DIR.parent / "INFO"
OUTPUT_DIR = SCRIPT_DIR / "MarketAnalysis_Report_Output"
LOCS_FILE_PATH = INFO_DIR / "locs_list.csv"
PRICES_FILE_PATH = INFO_DIR / "PRICES.csv"
CACHE_FILE_PATH = OUTPUT_DIR / "ComprehensiveFlow_DataCache.csv"
OUTPUT_FILENAME = "Comprehensive_Flow_Report.html"

DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- User-Defined Filters & Parameters ---
CATEGORIES_TO_INCLUDE = [
    'Compressor', 'Industrial', 'Interconnect', 'LDC',
    'LNG', 'Power', 'Production', 'Segment'
]

# --- Parameters for Anomaly Detection (Blueprint 1.0) ---
AGGREGATE_Z_SCORE_THRESHOLD = 2.0
SIGNIFICANCE_PERCENTAGE = 0.15
MINIMUM_ABSOLUTE_THRESHOLD = 5000
DRIVER_Z_SCORE_THRESHOLD = 1.5
TOP_N_DRIVERS = 7

# ==============================================================================
#  CORE DATA & UTILITY FUNCTIONS (No Changes Here)
# ==============================================================================
def get_db_engine():
    """Establishes a secure database connection."""
    print("INFO: Attempting to connect to database...")
    try:
        dotenv_path = SCRIPT_DIR.parent / "Criterion" / ".env"
        if not dotenv_path.exists():
            raise FileNotFoundError(f".env file not found at {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        if not db_user or not db_password:
            raise ValueError("DB_USER or DB_PASSWORD not found in environment.")
        conn_url = f"postgresql+psycopg2://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(conn_url, connect_args={"sslmode": "require", "connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("INFO: Database engine created and connection confirmed.")
        return engine
    except Exception as e:
        print(f"CRITICAL: Database connection failed. Error: {e}")
        return None

def get_active_market_components(prices_file_path, lookback_days=90, threshold=0.75):
    """Loads PRICES.csv and filters for recently active market components."""
    print(f"INFO: Loading price data from {prices_file_path.name} to find active components...")
    try:
        prices_df = pd.read_csv(prices_file_path)
        prices_df['Date'] = pd.to_datetime(prices_df['Date'], errors='coerce')
        end_date = prices_df['Date'].max()
        start_date = end_date - relativedelta(days=lookback_days)
        active_components = [
            col for col in prices_df.columns
            if col.lower() not in ['date', 'henry'] and
            (prices_df[(prices_df['Date'] >= start_date) & (prices_df['Date'] <= end_date)][col].count() / lookback_days if lookback_days > 0 else 0) >= threshold
        ]
        print(f"INFO: Found {len(active_components)} active components.")
        return active_components
    except Exception as e:
        print(f"WARNING: Could not determine active components. Error: {e}")
        return []

def get_flow_data(engine, all_tickers, locs_metadata_df):
    """Fetches, prepares, and caches all necessary flow data."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    required_cache_cols = {'date', 'ticker', 'value', 'loc_name', 'market_component', 'category_short'}
    if CACHE_FILE_PATH.exists():
        print(f"INFO: Reading from cache file: {CACHE_FILE_PATH.name}")
        df_cache = pd.read_csv(CACHE_FILE_PATH, parse_dates=['date'])
        if required_cache_cols.issubset(df_cache.columns):
            print("INFO: Cache file is valid and will be used.")
            return df_cache
        else:
            print("WARNING: Cache file is missing required columns. Refreshing.")
            CACHE_FILE_PATH.unlink()
    print("INFO: No valid cache found. Performing full data fetch...")
    try:
        with engine.connect() as connection:
            id_query = text("SELECT metadata_id, ticker FROM pipelines.metadata WHERE ticker IN :tickers")
            metadata_df = pd.read_sql_query(id_query, connection, params={'tickers': tuple(all_tickers)})
            metadata_ids = tuple(metadata_df['metadata_id'].unique())
            if not metadata_ids: return pd.DataFrame()
            flows_query = text("""
                SELECT noms.eff_gas_day AS date, meta.ticker, noms.scheduled_quantity AS value
                FROM pipelines.nomination_points AS noms
                JOIN pipelines.metadata AS meta ON noms.metadata_id = meta.metadata_id
                WHERE noms.metadata_id IN :ids
            """)
            flows_df = pd.read_sql_query(flows_query, connection, params={'ids': metadata_ids})
            if not flows_df.empty:
                print("INFO: Data fetched. Processing...")
                flows_df['date'] = pd.to_datetime(flows_df['date'])
                flows_df['value'] = pd.to_numeric(flows_df['value'], errors='coerce').fillna(0)
                processed_df = pd.merge(flows_df, locs_metadata_df, on='ticker', how='left')
                processed_df.dropna(subset=['market_component', 'category_short'], inplace=True)
                delivery_mask = processed_df['ticker'].str.endswith('.2', na=False)
                processed_df.loc[delivery_mask, 'value'] *= -1
                processed_df.to_csv(CACHE_FILE_PATH, index=False)
                print(f"INFO: Fully processed data saved to cache.")
                return processed_df
            return pd.DataFrame()
    except Exception as e:
        print(f"CRITICAL: Failed to fetch initial data. Error: {e}")
        return pd.DataFrame()

# ==============================================================================
#  ANALYSIS AND CHARTING FUNCTIONS
# ==============================================================================

def analyze_anomalies_and_drivers(all_flow_data):
    """Implements the "Trigger -> Drill-Down" logic. Returns only flagged category data."""
    print("\n--- Running Anomaly Trigger & Driver Analysis ---")
    today = datetime.now()
    flagged_categories = {}

    for (component, category), group_data in all_flow_data.groupby(['market_component', 'category_short']):
        agg_flows = group_data.groupby('date')['value'].sum().to_frame()
        if agg_flows.empty: continue

        agg_flows['flow_10d_avg'] = agg_flows['value'].rolling(window=10).mean()
        if agg_flows['flow_10d_avg'].isnull().all(): continue

        current_date = agg_flows.index.max()
        current_10d_avg = agg_flows.loc[current_date, 'flow_10d_avg']
        
        historical_vals = agg_flows[(agg_flows.index.year < current_date.year) & (agg_flows.index.dayofyear == current_date.dayofyear)]['flow_10d_avg'].dropna()

        if len(historical_vals) < 2: continue

        hist_mean = historical_vals.mean()
        hist_std = historical_vals.std()
        if hist_std == 0: continue

        z_score = (current_10d_avg - hist_mean) / hist_std
        abs_diff = abs(current_10d_avg - hist_mean)
        adaptive_magnitude_threshold = max(MINIMUM_ABSOLUTE_THRESHOLD, abs(hist_mean) * SIGNIFICANCE_PERCENTAGE)

        if abs(z_score) > AGGREGATE_Z_SCORE_THRESHOLD and abs_diff > adaptive_magnitude_threshold:
            category_key = f"{component} - {category}"
            print(f"  TRIGGER FIRED for {category_key}. Z-Score: {z_score:.2f}, Diff: {abs_diff:,.0f}.")
            
            driver_results = []
            for loc_name, loc_data in group_data.groupby('loc_name'):
                loc_flows = loc_data.groupby('date')['value'].sum().to_frame().sort_index()
                loc_flows['flow_10d_avg'] = loc_flows['value'].rolling(window=10).mean()
                if loc_flows.empty or loc_flows['flow_10d_avg'].isnull().all(): continue
                
                loc_current_10d_avg = loc_flows['flow_10d_avg'].iloc[-1]
                loc_hist_vals = loc_flows[(loc_flows.index.year < current_date.year) & (loc_flows.index.dayofyear == current_date.dayofyear)]['flow_10d_avg'].dropna()

                if len(loc_hist_vals) < 2: continue
                
                loc_hist_mean = loc_hist_vals.mean()
                loc_hist_std = loc_hist_vals.std()
                loc_z_score = ((loc_current_10d_avg - loc_hist_mean) / loc_hist_std) if loc_hist_std != 0 else 0
                
                if abs(loc_z_score) > DRIVER_Z_SCORE_THRESHOLD:
                    val_2024 = loc_hist_vals[loc_hist_vals.index.year == 2024].iloc[0] if 2024 in loc_hist_vals.index.year else np.nan
                    driver_results.append({
                        'Location': loc_name,
                        'Category': category,
                        'Current 10-Day Avg': loc_current_10d_avg,
                        'Historical Avg': loc_hist_mean,
                        'Z-Score': loc_z_score,
                        'Prior Year (2024) 10-Day Avg': val_2024,
                        'Impact': abs(loc_current_10d_avg - loc_hist_mean)
                    })

            if driver_results:
                drivers_df = pd.DataFrame(driver_results)
                flagged_categories[category_key] = drivers_df
    
    return flagged_categories

def generate_yoy_chart(data, title, date_range):
    """Generates a focused year-over-year comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    current_year = data.index.year.max()
    data_by_year = {year: df for year, df in data.groupby(data.index.year)}

    def safe_date_replace(dt, year):
        try: return dt.replace(year=year)
        except ValueError: return dt.replace(year=year, day=28)

    for year, df in sorted(data_by_year.items()):
        color, lw, z = ('darkgreen', 2.0, 5) if year == current_year else (None, 1.5, 2)
        plot_dates = df.index.map(lambda dt: safe_date_replace(dt, current_year))
        ax.plot(plot_dates, df['value'], label=str(year), color=color, linewidth=lw, zorder=z, marker='.', markersize=4)

    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_ylabel("Value"); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Year', loc='upper left'); fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(date_range)

    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def style_and_format_df(df, title):
    """Applies styling and formatting to a DataFrame for HTML output."""
    if df.empty: return ""
    
    styled = df.style.background_gradient(
        cmap='coolwarm_r', subset=['Z-Score'], vmin=-3, vmax=3
    ).format({
        'Current 10-Day Avg': '{:,.0f}',
        'Historical Avg': '{:,.0f}',
        'Z-Score': '{:.2f}',
        'Prior Year (2024) 10-Day Avg': '{:,.0f}',
    }, na_rep='-')
    
    return styled.set_caption(title).to_html(escape=False, index=False)

def generate_html_report(report_sections_html):
    """Creates a single, styled HTML report from pre-generated sections."""
    print("\n--- Generating Final HTML Report ---")
    
    final_html = f"""
    <html><head><title>Market Flow Anomaly Report</title><style>
    html {{ scroll-behavior: smooth; }}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;background-color:#f4f6f9;color:#333;}}
    h1,h2{{color:#343a40;}}
    h1{{text-align:center;color:#1a252f;padding:20px 0;background-color:#e9ecef;border-bottom:2px solid #dee2e6;margin:0;}}
    
    /* --- FIX 1: Make the main content area wider --- */
    .container{{padding:20px; max-width:1600px; margin:auto;}}

    .component-section h2{{color:#343a40;background-color:#f8f9fa;padding:10px;border-bottom:2px solid #adb5bd;margin-top:40px; border-radius: 5px;}}
    .chart-grid{{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }}
    .chart-container{{
        background-color:white;
        border-radius:8px;
        box-shadow:0 4px 8px rgba(0,0,0,0.1);
        padding:15px;
        box-sizing: border-box;
        width: calc(50% - 10px);
    }}

    /* --- FIX 2: Make chart images responsive so they don't overflow --- */
    .chart-container img {{
        max-width: 100%;
        height: auto;
    }}

    .tables-container{{margin-top:1.5em;}}
    table{{width:100%;border-collapse:collapse;margin:1.5em 0;box-shadow:0 2px 4px rgba(0,0,0,0.1);}}
    th,td{{padding:10px;text-align:left;border-bottom:1px solid #dee2e6;}} th{{background-color:#e9ecef;}}
    caption{{font-size:1.5em;font-weight:bold;margin:1em 0;text-align:left;color:#343a40;}}
    </style></head>
    <body><h1>Market Flow Anomaly Report</h1><div class="container">
    {report_sections_html}
    </div></body></html>
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    with open(output_path, 'w', encoding='utf-8') as f: f.write(final_html)
    print(f"\n--- Report generation complete. ---")
    print(f"Successfully saved to: {output_path}")

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================
def main():
    """Main function to orchestrate the entire analysis and reporting pipeline."""
    engine = get_db_engine()
    if not engine: return
    
    try:
        active_components = get_active_market_components(PRICES_FILE_PATH)
        if not active_components:
            print("CRITICAL: No active market components found. Halting."); return

        locs_df = pd.read_csv(LOCS_FILE_PATH)
        filtered_locs_df = locs_df[locs_df['market_component'].isin(active_components) & locs_df['category_short'].isin(CATEGORIES_TO_INCLUDE)]
        all_tickers = filtered_locs_df['ticker'].dropna().unique().tolist()

        all_flow_data = get_flow_data(engine, all_tickers, filtered_locs_df[['ticker', 'loc_name', 'market_component', 'category_short']])
        if all_flow_data.empty:
            print("CRITICAL: No flow data available."); return

        flagged_results = analyze_anomalies_and_drivers(all_flow_data)

        if not flagged_results:
            print("\nINFO: No significant anomalies detected.")
            # Still generate a report with all charts even if no anomalies
        
        print(f"\n--- Building Report ---")
        
        report_sections_html = ""
        
        today = datetime.now()
        start_date = (today.replace(day=1) - relativedelta(months=1))
        end_date = (today.replace(day=1) + relativedelta(months=2)) - relativedelta(days=1)
        chart_date_range = (start_date, end_date)

        for component in sorted(active_components):
            component_data = all_flow_data[all_flow_data['market_component'] == component]
            if component_data.empty: continue

            charts_html = ""
            for category, category_data in component_data.groupby('category_short'):
                agg_chart_data = category_data.groupby('date')['value'].sum().to_frame()
                if not agg_chart_data.empty:
                    base64_img = generate_yoy_chart(agg_chart_data, title=f"Aggregate Flow: {category}", date_range=chart_date_range)
                    if base64_img:
                        charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{base64_img}" alt="{component} - {category}"></div>'

            all_drivers_for_component = []
            for category_key, drivers_df in flagged_results.items():
                if category_key.startswith(component):
                    all_drivers_for_component.append(drivers_df)
            
            tables_html = ""
            if all_drivers_for_component:
                consolidated_drivers_df = pd.concat(all_drivers_for_component, ignore_index=True)
                final_drivers_raw = consolidated_drivers_df.sort_values(by='Impact', ascending=False).head(TOP_N_DRIVERS)
                
                final_drivers_ordered = final_drivers_raw[[
                    'Location', 'Category', 'Current 10-Day Avg', 
                    'Historical Avg', 'Z-Score', 'Prior Year (2024) 10-Day Avg'
                ]]
                
                tables_html = style_and_format_df(final_drivers_ordered, title=f"Top Anomaly Drivers for {component}")

            if charts_html:
                report_sections_html += (
                    f"<div class='component-section'>"
                    f"<h2>Market Component: {component}</h2>"
                    f"<div class='chart-grid'>{charts_html}</div>"
                    f"<div class='tables-container'>{tables_html}</div>"
                    f"</div>"
                )
        
        generate_html_report(report_sections_html)

    except Exception as e:
        print(f"A fatal error occurred in the main process: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("INFO: Database connection closed.")

if __name__ == "__main__":
    main()