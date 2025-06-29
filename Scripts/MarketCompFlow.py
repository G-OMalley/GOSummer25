import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import io
import numpy as np
from scipy.stats import wilcoxon

# ==============================================================================
#  CONFIGURATION CONSTANTS
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
INFO_DIR = SCRIPT_DIR.parent / "INFO"
OUTPUT_DIR = SCRIPT_DIR / "MarketAnalysis_Report_Output" # Define the output directory
LOCS_FILE_PATH = INFO_DIR / "locs_list.csv"
PRICES_FILE_PATH = INFO_DIR / "PRICES.csv"
CACHE_FILE_PATH = OUTPUT_DIR / "ComprehensiveFlow_DataCache.csv" # Dedicated cache for this script
OUTPUT_FILENAME = "Comprehensive_Flow_Report.html"

DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- User-Defined Filters & Parameters ---
CATEGORIES_TO_INCLUDE = [
    'Compressor', 'Industrial', 'Interconnect', 'LDC', 
    'LNG', 'Power', 'Production', 'Segment'
]
CHART_IMPACT_THRESHOLD = 15000
SIGNIFICANCE_LEVEL = 0.05
Z_SCORE_THRESHOLD = 2.0

# ==============================================================================
#  CORE DATA & UTILITY FUNCTIONS
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
        engine = create_engine(
            conn_url, connect_args={"sslmode": "require", "connect_timeout": 10}
        )
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
        start_date = end_date - timedelta(days=lookback_days)
        
        active_components = [
            col for col in prices_df.columns 
            if col.lower() not in ['date', 'henry'] and 
            (prices_df[(prices_df['Date'] >= start_date) & (prices_df['Date'] <= end_date)][col].count() / lookback_days if lookback_days > 0 else 0) >= threshold
        ]
        
        print(f"INFO: Found {len(active_components)} active components based on the last {lookback_days} days.")
        return active_components
    except Exception as e:
        print(f"WARNING: Could not determine active components. Error: {e}")
        return []

def get_flow_data(engine, all_tickers, locs_metadata_df):
    """
    Fetches, prepares, and caches all necessary flow data.
    This function now handles merging and value adjustments internally.
    """
    # Ensure output directory exists before trying to access cache
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Define required columns for cache validation
    required_cache_cols = {'date', 'ticker', 'value', 'loc_name', 'market_component', 'category_short'}

    if CACHE_FILE_PATH.exists():
        print(f"INFO: Reading from cache file: {CACHE_FILE_PATH.name}")
        df_cache = pd.read_csv(CACHE_FILE_PATH, parse_dates=['date'])
        
        if required_cache_cols.issubset(df_cache.columns):
            print("INFO: Cache file is valid and will be used.")
            return df_cache
        else:
            print("WARNING: Cache file is missing required columns. Deleting and performing a full refresh.")
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
                print("INFO: Data fetched. Processing and preparing for cache...")
                flows_df['date'] = pd.to_datetime(flows_df['date'])
                flows_df['value'] = pd.to_numeric(flows_df['value'], errors='coerce').fillna(0)
                
                # Merge with location metadata to add component and category info
                processed_df = pd.merge(flows_df, locs_metadata_df, on='ticker', how='left')
                processed_df.dropna(subset=['market_component', 'category_short'], inplace=True)
                
                # Adjust flow values for delivery points
                delivery_mask = processed_df['ticker'].str.endswith('.2', na=False)
                processed_df.loc[delivery_mask, 'value'] *= -1

                processed_df.to_csv(CACHE_FILE_PATH, index=False)
                print(f"INFO: Fully processed data saved to new cache at {CACHE_FILE_PATH.name}")
                return processed_df
            return flows_df
    except Exception as e:
        print(f"CRITICAL: Failed to fetch initial data. Error: {e}")
        return pd.DataFrame()

# ==============================================================================
#  CHARTING AND ANALYSIS FUNCTIONS
# ==============================================================================

def generate_yoy_chart(data_by_year, title):
    """Generates a year-over-year comparison chart and returns a base64 image string."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    if not data_by_year:
        plt.close(fig)
        return None
    current_year = max(data_by_year.keys())
    
    for year, df in sorted(data_by_year.items()):
        color, lw, z = ('darkgreen', 2.0, 5) if year == current_year else (None, 1.5, 2)
        plot_dates = df.index.map(lambda dt: dt.replace(year=current_year))
        ax.plot(plot_dates, df['value'], label=str(year), color=color, linewidth=lw, zorder=z, marker='.', markersize=4)

    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_ylabel("Value"); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Year', loc='upper left'); fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_forward_month_patterns(all_data):
    """Analyzes month-over-month flow changes for statistical patterns."""
    print("\n--- Running Forward Month Pattern Detection ---")
    today = datetime.now()
    prior_month = (today.replace(day=1) - timedelta(days=1)).month
    forward_month = (today.replace(day=1) + timedelta(days=32)).month
    
    results = []
    for loc_name, loc_data in all_data.groupby('loc_name'):
        # Correctly group by year and month, then unstack to create a DataFrame
        # with years as index and months as columns.
        monthly_totals = loc_data.groupby([loc_data['date'].dt.year, loc_data['date'].dt.month])['value'].sum().unstack()
        monthly_totals.index.name = 'year'
        monthly_totals.columns.name = 'month'

        if prior_month not in monthly_totals.columns or forward_month not in monthly_totals.columns: continue
            
        comparison_df = monthly_totals[[prior_month, forward_month]].dropna()
        if len(comparison_df) < 4: continue
        
        diff = comparison_df[forward_month] - comparison_df[prior_month]
        if diff.var() == 0: continue 
        
        stat, p_value = wilcoxon(diff, zero_method='zsplit')
        if p_value < SIGNIFICANCE_LEVEL:
            avg_change = diff.mean()
            direction = "Gain" if avg_change > 0 else "Loss"
            recent_years = sorted(comparison_df.index)[-3:]
            recent_diffs = diff.loc[recent_years]
            recency_weight = (np.sign(recent_diffs) == np.sign(avg_change)).sum() / len(recent_diffs) if len(recent_diffs) > 0 else 0
            
            results.append({"Location": loc_name, "Direction": direction, "Avg. Change": f"{avg_change:,.0f}",
                            "Stability": f"{(np.sign(diff) == np.sign(avg_change)).sum()}/{len(diff)} years",
                            "Recency Score": f"{recency_weight:.0%}", "P-Value": f"{p_value:.4f}"})
    return pd.DataFrame(results)

def analyze_current_year_anomalies(all_data):
    """Identifies locations where current year's flows are anomalous."""
    print("\n--- Running Current-Year Anomaly Detection ---")
    today = datetime.now()
    day_of_year = today.timetuple().tm_yday
    
    results = []
    for loc_name, loc_data in all_data.groupby('loc_name'):
        loc_data = loc_data.set_index('date')
        period_data = loc_data[loc_data.index.dayofyear <= day_of_year]
        current_year_data = period_data[period_data.index.year == today.year]
        historical_data = period_data[period_data.index.year < today.year]
        
        if current_year_data.empty or historical_data.empty: continue
            
        hist_ytd_totals = historical_data.groupby(historical_data.index.year)['value'].sum()
        if len(hist_ytd_totals) < 2: continue
        
        hist_mean, hist_std = hist_ytd_totals.mean(), hist_ytd_totals.std()
        if hist_std == 0: continue
            
        current_ytd_total = current_year_data['value'].sum()
        z_score = (current_ytd_total - hist_mean) / hist_std
        
        if abs(z_score) > Z_SCORE_THRESHOLD:
            results.append({"Location": loc_name, "Z-Score": f"{z_score:.2f}",
                            "Current YTD": f"{current_ytd_total:,.0f}", "Hist. Avg YTD": f"{hist_mean:,.0f}",
                            "Deviation": f"{current_ytd_total - hist_mean:,.0f}",
                            "Deviation %": f"{(current_ytd_total / hist_mean - 1):.1%}" if hist_mean !=0 else "N/A"})
    return pd.DataFrame(results)

# ==============================================================================
#  HTML REPORTING
# ==============================================================================
def generate_html_report(charts_html, pattern_df, anomaly_df):
    """Creates a single, styled HTML report with all components."""
    print("\n--- Generating Final HTML Report ---")
    
    def style_df_to_html(df, title):
        if df.empty:
            return f"<h3>{title}</h3><p>No significant signals detected.</p>"
        df_copy = df.copy()
        if 'Z-Score' in df_copy.columns:
            df_copy['Z-Score'] = pd.to_numeric(df_copy['Z-Score'])
            styled = df_copy.style.background_gradient(cmap='coolwarm_r', subset=['Z-Score'], vmin=-3, vmax=3).format({'Z-Score': "{:.2f}"})
        else:
            styled = df_copy.style
        return styled.set_caption(title).to_html(escape=False, index=False)

    pattern_html = style_df_to_html(pattern_df, "Forward Month Pattern Signals")
    anomaly_html = style_df_to_html(anomaly_df, "Current-Year Anomaly Signals")
    
    final_html = f"""
    <html><head><title>Comprehensive Flow Report</title><style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;background-color:#f4f6f9;color:#333;}}
    h1,h2,h3{{color:#343a40;}}
    h1{{text-align:center;color:#1a252f;padding:20px 0;background-color:#e9ecef;border-bottom:2px solid #dee2e6;margin:0;}}
    .container{{padding:20px;}}
    .section-title{{color:#007bff;border-bottom:2px solid #007bff;padding-bottom:10px;margin-top:2em;}}
    .component-section h2{{color:#343a40;background-color:transparent;padding:10px 0;border-bottom:2px solid #adb5bd;margin-top:20px;}}
    .chart-grid{{display:flex;flex-wrap:wrap;gap:20px;justify-content:center;}}
    .chart-container{{background-color:white;border-radius:8px;box-shadow:0 4px 8px rgba(0,0,0,0.1);padding:15px;flex:1 1 45%;min-width:500px;}}
    table{{width:100%;border-collapse:collapse;margin:2em 0;box-shadow:0 2px 4px rgba(0,0,0,0.1);}}
    th,td{{padding:10px;text-align:left;border-bottom:1px solid #dee2e6;}} th{{background-color:#e9ecef;}}
    caption{{font-size:1.5em;font-weight:bold;margin:1em 0;text-align:left;color:#343a40;}}
    </style></head>
    <body><h1>Comprehensive Flow Report</h1><div class="container">
        <h2 class="section-title">Market Flow Charts</h2>{charts_html}
        <h2 class="section-title">Statistical Flow Analysis</h2>{pattern_html}{anomaly_html}
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

        # The get_flow_data function now returns a fully processed DataFrame
        all_flow_data = get_flow_data(engine, all_tickers, filtered_locs_df[['ticker', 'loc_name', 'market_component', 'category_short']])
        if all_flow_data.empty: 
            print("CRITICAL: No flow data available.")
            return

        # --- Generate Chart HTML ---
        today = datetime.now()
        start_date_ref = (today.replace(day=1) - relativedelta(months=1))
        end_date_ref = (today.replace(day=1) + relativedelta(months=2) - timedelta(days=1))
        
        charts_html_content = ""
        for component in sorted(active_components):
            comp_flow_data = all_flow_data[all_flow_data['market_component'] == component]
            if comp_flow_data.empty: continue
            
            charts_for_component = ""
            for category in sorted(comp_flow_data['category_short'].unique()):
                category_flow_data = comp_flow_data[comp_flow_data['category_short'] == category]
                data_by_year, has_impact = {}, False
                for i in range(4):
                    target_year = today.year - (3 - i)
                    start_dt, end_dt = start_date_ref.replace(year=target_year), end_date_ref.replace(year=target_year)
                    if target_year == today.year: end_dt = today
                    
                    year_data = category_flow_data[(category_flow_data['date'] >= start_dt) & (category_flow_data['date'] <= end_dt)]
                    if year_data.empty: continue
                    
                    daily_sum = year_data.groupby('date')['value'].sum().to_frame()
                    if not has_impact and daily_sum['value'].abs().max() > CHART_IMPACT_THRESHOLD:
                        has_impact = True
                    data_by_year[target_year] = daily_sum
                
                if has_impact:
                    base64_img = generate_yoy_chart(data_by_year, category)
                    if base64_img:
                        charts_for_component += f'<div class="chart-container"><img src="data:image/png;base64,{base64_img}" alt="{category}"></div>'
            
            if charts_for_component:
                charts_html_content += f"<div class='component-section'><h2>Market: {component}</h2><div class='chart-grid'>{charts_for_component}</div></div>"
        
        # --- Run Statistical Analysis ---
        pattern_results = analyze_forward_month_patterns(all_flow_data)
        anomaly_results = analyze_current_year_anomalies(all_flow_data)
        
        # --- Generate Final Report ---
        generate_html_report(charts_html_content, pattern_results, anomaly_results)

    except Exception as e:
        print(f"A fatal error occurred in the main process: {e}")
        traceback.print_exc()
    finally:
        if engine:
            engine.dispose()
            print("INFO: Database connection closed.")

if __name__ == "__main__":
    main()
