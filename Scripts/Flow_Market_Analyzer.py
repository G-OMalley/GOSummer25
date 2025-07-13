import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
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
from jinja2 import Environment, FileSystemLoader

# ==============================================================================
# 1. CONFIGURATION CLASS
# Encapsulates all tunable parameters for the report.
# ==============================================================================
class ReportConfig:
    """Holds all configuration parameters for the analysis and report."""
    # --- Path Configuration ---
    # NOTE: These paths are relative to the script's location.
    SCRIPT_DIR: Path = Path(__file__).parent
    INFO_DIR: Path = SCRIPT_DIR.parent / "INFO"
    OUTPUT_DIR: Path = SCRIPT_DIR / "MarketAnalysis_Report_Output"
    
    # --- File Names ---
    LOCS_FILE_PATH: Path = INFO_DIR / "locs_list.csv"
    PRICES_FILE_PATH: Path = INFO_DIR / "PRICES.csv"
    CACHE_FILENAME: str = "FlowData_Only_Cache.csv"
    OUTPUT_FILENAME: str = "Comprehensive_Flow_Report.html"
    TEMPLATE_NAME: str = "report_template.html"

    # --- Database Configuration ---
    DB_HOST: str = "dda.criterionrsch.com"
    DB_PORT: int = 443
    DB_NAME: str = "production"
    
    # --- Data Filtering ---
    CATEGORIES_TO_INCLUDE: list[str] = [
        'Compressor', 'Industrial', 'Interconnect', 'LDC',
        'LNG', 'Power', 'Production', 'Segment'
    ]

    # --- Anomaly Detection Parameters ---
    HISTORICAL_YEARS: int = 3
    SEASONAL_WINDOW_DAYS: int = 30
    ROC_SHORT_WINDOW: int = 7
    ROC_LONG_WINDOW: int = 30
    
    # --- Anomaly Thresholds ---
    Z_SCORE_THRESHOLD: float = 2.0
    YOY_SHOCK_THRESHOLD_PCT: float = 0.25
    MINIMUM_IMPACT_THRESHOLD: int = 5000
    STATE_CHANGE_ZERO_THRESHOLD: int = 1000 
    
    # --- Report Output ---
    TOP_N_DRIVERS: int = 7
    COHORT_MIN_LOCATIONS: int = 5 # Min locations to trigger a category rollup

# ==============================================================================
# 2. CORE ANALYZER CLASS
# Manages the entire workflow from data fetching to report generation.
# ==============================================================================
class MarketFlowAnalyzer:
    """Orchestrates the market flow analysis and report generation."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.engine = None
        self.all_flow_data = pd.DataFrame()
        self.flagged_drivers = {}
        self.report_data = {}

    def _get_db_engine(self) -> Engine | None:
        """Establishes and returns a secure database connection."""
        print("INFO: Attempting to connect to database...")
        try:
            dotenv_path = self.config.SCRIPT_DIR.parent / "Criterion" / ".env"
            if not dotenv_path.exists():
                raise FileNotFoundError(f".env file not found at {dotenv_path}")
            
            load_dotenv(dotenv_path=dotenv_path, override=True)
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")

            if not db_user or not db_password:
                raise ValueError("DB_USER or DB_PASSWORD not found in environment.")
            
            conn_url = f"postgresql+psycopg2://{db_user}:{db_password}@{self.config.DB_HOST}:{self.config.DB_PORT}/{self.config.DB_NAME}"
            engine = create_engine(conn_url, connect_args={"sslmode": "require", "connect_timeout": 10})
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("INFO: Database engine created and connection confirmed.")
            return engine
        except Exception as e:
            print(f"CRITICAL: Database connection failed. Error: {e}")
            return None

    def _fetch_and_prepare_data(self):
        """Loads and prepares all necessary data for the analysis."""
        print("\n--- Fetching and Preparing Data ---")
        
        prices_df = pd.read_csv(self.config.PRICES_FILE_PATH)
        prices_df['Date'] = pd.to_datetime(prices_df['Date'], errors='coerce')
        end_date = prices_df['Date'].max()
        start_date = end_date - relativedelta(days=90)
        active_components = [
            col for col in prices_df.columns if col.lower() not in ['date', 'henry'] and
            (prices_df[(prices_df['Date'] >= start_date) & (prices_df['Date'] <= end_date)][col].count() / 90) >= 0.75
        ]
        print(f"INFO: Found {len(active_components)} active components.")
        self.report_data['active_components'] = sorted(active_components)

        locs_df = pd.read_csv(self.config.LOCS_FILE_PATH)
        filtered_locs_df = locs_df[
            locs_df['market_component'].isin(active_components) & 
            locs_df['category_short'].isin(self.config.CATEGORIES_TO_INCLUDE)
        ]
        all_tickers = filtered_locs_df['ticker'].dropna().unique().tolist()

        self._get_flow_data(all_tickers, filtered_locs_df)

    def _get_flow_data(self, all_tickers: list, locs_metadata_df: pd.DataFrame):
        """
        Fetches flow data using an incremental update strategy for the cache.
        If cache exists, it replaces the last 15 days. Otherwise, it performs a full fetch.
        """
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        cache_path = self.config.OUTPUT_DIR / self.config.CACHE_FILENAME

        df_to_keep = pd.DataFrame()
        original_cached_df = pd.DataFrame()
        # Default start date for a full fetch if no cache exists.
        start_date_for_fetch = datetime(2018, 1, 1)

        if cache_path.exists():
            print("INFO: Cache file exists. Preparing for incremental update.")
            try:
                original_cached_df = pd.read_csv(cache_path, parse_dates=['date'])
                if not original_cached_df.empty:
                    latest_date_in_cache = original_cached_df['date'].max()
                    # Set the fetch to start 15 days before the last known date
                    start_date_for_fetch = latest_date_in_cache - pd.Timedelta(days=15)
                    # Keep the data from before this new fetch period
                    df_to_keep = original_cached_df[original_cached_df['date'] < start_date_for_fetch].copy()
                    print(f"INFO: Keeping cached data before {start_date_for_fetch.strftime('%Y-%m-%d')}.")
            except Exception as e:
                print(f"WARNING: Could not read cache file properly: {e}. Performing a full fetch.")
        else:
            print("INFO: No cache file found. Performing a full data fetch.")

        # --- Database Fetching (now with a dynamic start date) ---
        newly_fetched_df = pd.DataFrame()
        fetch_successful = False
        try:
            print(f"INFO: Fetching data from database starting from {start_date_for_fetch.strftime('%Y-%m-%d')}...")
            with self.engine.connect() as connection:
                id_query = text("SELECT metadata_id FROM pipelines.metadata WHERE ticker IN :tickers")
                metadata_ids = tuple(pd.read_sql_query(id_query, connection, params={'tickers': tuple(all_tickers)})['metadata_id'].unique())

                if metadata_ids:
                    # The SQL query now includes a date condition
                    flows_query = text("""
                        SELECT noms.eff_gas_day AS date, meta.ticker, noms.scheduled_quantity AS value
                        FROM pipelines.nomination_points AS noms
                        JOIN pipelines.metadata AS meta ON noms.metadata_id = meta.metadata_id
                        WHERE noms.metadata_id IN :ids AND noms.eff_gas_day >= :start_date
                    """)
                    params = {'ids': metadata_ids, 'start_date': start_date_for_fetch}
                    newly_fetched_df = pd.read_sql_query(flows_query, connection, params=params)
                    print(f"INFO: Successfully fetched {len(newly_fetched_df)} new/updated records.")
                    fetch_successful = True
        except Exception as e:
            print(f"CRITICAL: Database fetch failed: {e}")
            fetch_successful = False

        # --- Combine, Save, and Process Logic ---
        final_flows_df = pd.DataFrame()
        if fetch_successful:
            # Combine the old part of the cache with the newly fetched data
            final_flows_df = pd.concat([df_to_keep, newly_fetched_df], ignore_index=True)
            # Overwrite the cache with the fresh, combined data
            print(f"INFO: Overwriting cache file with {len(final_flows_df)} total records.")
            final_flows_df.to_csv(cache_path, index=False)
        else:
            # If the fetch failed, fall back to the original, unmodified cache to avoid data loss.
            print("WARNING: Using stale cache data due to database fetch failure.")
            final_flows_df = original_cached_df

        if final_flows_df.empty:
            print("CRITICAL: No flow data could be loaded or fetched.")
            return

        print("INFO: Merging flow data with the latest location metadata...")
        processed_df = pd.merge(final_flows_df, locs_metadata_df, on='ticker', how='left')
        processed_df.dropna(subset=['market_component', 'category_short'], inplace=True)
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df['value'] = pd.to_numeric(processed_df['value'], errors='coerce').fillna(0)

        delivery_mask = processed_df['ticker'].str.endswith('.2', na=False)
        processed_df.loc[delivery_mask, 'value'] *= -1

        processed_df.set_index('date', inplace=True)
        processed_df.sort_index(inplace=True)

        self.all_flow_data = processed_df
        print("INFO: Data preparation complete.")

    def _calculate_anomaly_scores(self, df: pd.DataFrame, current_date: datetime) -> dict:
        """Calculates multiple anomaly scores for a given timeseries DataFrame."""
        scores = {}
        cfg = self.config
        
        if df.empty:
            return scores

        current_val = df['value_7d_avg'].iloc[-1]

        seasonal_start = current_date.dayofyear - cfg.SEASONAL_WINDOW_DAYS
        seasonal_end = current_date.dayofyear + cfg.SEASONAL_WINDOW_DAYS
        hist_years = range(current_date.year - cfg.HISTORICAL_YEARS, current_date.year)
        
        if seasonal_start < 1:
            day_of_year_filter = (df.index.dayofyear >= 366 + seasonal_start) | (df.index.dayofyear <= seasonal_end)
        elif seasonal_end > 366:
            day_of_year_filter = (df.index.dayofyear >= seasonal_start) | (df.index.dayofyear <= seasonal_end - 366)
        else:
            day_of_year_filter = (df.index.dayofyear >= seasonal_start) & (df.index.dayofyear <= seasonal_end)

        historical_data = df[(df.index.year.isin(hist_years)) & day_of_year_filter]['value']
        
        hist_mean = historical_data.mean() if not historical_data.empty else 0
        scores['hist_avg'] = hist_mean

        if len(historical_data) > 30:
            hist_std = historical_data.std()
            if hist_std > 0:
                scores['historical_zscore'] = (current_val - hist_mean) / hist_std

        if len(df) > cfg.ROC_LONG_WINDOW:
            roc_long_avg = df['value'].rolling(window=cfg.ROC_LONG_WINDOW).mean().iloc[-1]
            roc_long_std = df['value'].rolling(window=cfg.ROC_LONG_WINDOW).std().iloc[-1]
            if roc_long_std > 0:
                scores['roc_zscore'] = (current_val - roc_long_avg) / roc_long_std

        last_year_date = current_date - relativedelta(years=1)
        last_year_idx = df.index.get_indexer([last_year_date], method='nearest')[0]
        last_year_val = df.iloc[last_year_idx]['value_7d_avg']
        scores['yoy_avg'] = last_year_val
        if pd.notna(last_year_val) and last_year_val != 0:
            scores['yoy_shock'] = (current_val - last_year_val) / abs(last_year_val)
        
        if abs(hist_mean) < cfg.STATE_CHANGE_ZERO_THRESHOLD and abs(current_val) > cfg.MINIMUM_IMPACT_THRESHOLD:
            scores['state_change'] = True
            
        return scores

    def _analyze_anomalies(self):
        """Identifies aggregate anomalies and finds their drivers."""
        print("\n--- Running Multi-Factor Anomaly Analysis ---")
        if self.all_flow_data.empty: return

        current_date = self.all_flow_data.index.max()
        
        for (component, category), group_data in self.all_flow_data.groupby(['market_component', 'category_short']):
            agg_flows = group_data.groupby(level='date')['value'].sum().to_frame().sort_index()
            if len(agg_flows) < self.config.ROC_LONG_WINDOW: continue
            
            agg_flows['value_7d_avg'] = agg_flows['value'].rolling(window=7).mean()
            agg_flows.dropna(inplace=True)
            if agg_flows.empty: continue

            agg_scores = self._calculate_anomaly_scores(agg_flows, current_date)

            is_anomalous = (
                abs(agg_scores.get('historical_zscore', 0)) > self.config.Z_SCORE_THRESHOLD or
                abs(agg_scores.get('roc_zscore', 0)) > self.config.Z_SCORE_THRESHOLD or
                abs(agg_scores.get('yoy_shock', 0)) > self.config.YOY_SHOCK_THRESHOLD_PCT or
                agg_scores.get('state_change', False)
            )

            if not is_anomalous: continue

            print(f"  TRIGGER FIRED for {component} - {category}. Analyzing drivers...")
            self._find_drivers(group_data, current_date, component, category)

    def _find_drivers(self, group_data, current_date, component, category):
        """Analyzes individual locations to find the source of an aggregate anomaly."""
        driver_results = []
        for loc_name, loc_data in group_data.groupby('loc_name'):
            loc_flows = loc_data.groupby(level='date')['value'].sum().to_frame().sort_index()
            if len(loc_flows) < self.config.ROC_LONG_WINDOW: continue
            
            loc_flows['value_7d_avg'] = loc_flows['value'].rolling(window=7).mean()
            loc_flows.dropna(inplace=True)
            if loc_flows.empty: continue

            loc_scores = self._calculate_anomaly_scores(loc_flows, current_date)
            
            anomaly_type, max_abs_score = 'None', 0
            h_z = loc_scores.get('historical_zscore', 0)
            r_z = loc_scores.get('roc_zscore', 0)
            y_s = loc_scores.get('yoy_shock', 0)
            s_c = loc_scores.get('state_change', False)

            if s_c:
                anomaly_type = 'State Change'
            elif abs(h_z) > self.config.Z_SCORE_THRESHOLD:
                anomaly_type, max_abs_score = 'Historical', abs(h_z)
            if abs(r_z) > self.config.Z_SCORE_THRESHOLD and abs(r_z) > max_abs_score:
                anomaly_type, max_abs_score = 'Rate of Change', abs(r_z)
            if abs(y_s) > self.config.YOY_SHOCK_THRESHOLD_PCT and abs(y_s * 10) > max_abs_score:
                anomaly_type = 'YoY Shock'

            current_7d_avg = loc_flows['value_7d_avg'].iloc[-1]
            yoy_avg = loc_scores.get('yoy_avg')
            yoy_vol_change = current_7d_avg - yoy_avg if pd.notna(yoy_avg) else 0
            
            if anomaly_type != 'None' and abs(yoy_vol_change) > self.config.MINIMUM_IMPACT_THRESHOLD:
                driver_results.append({
                    'Location': loc_name, 'Anomaly Type': anomaly_type,
                    'Current 7-Day Avg': current_7d_avg, 'Historical Avg': loc_scores.get('hist_avg'),
                    'YoY Avg': yoy_avg, 'Z-Score (Hist)': h_z,
                    'Z-Score (RoC)': r_z, 'YoY % Change': y_s, 
                    'YoY Vol Change': yoy_vol_change,
                    'Component': component, 'Category': category
                })
        
        if driver_results:
            key = f"{component} - {category}"
            self.flagged_drivers[key] = pd.DataFrame(driver_results)

    def _generate_charts_and_tables(self):
        """Creates all visual components (charts, tables) for the report."""
        print("\n--- Building Report Components ---")
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        today = datetime.now()
        start_date = (today.replace(day=1) - relativedelta(months=1))
        end_date = (today.replace(day=1) + relativedelta(months=2)) - relativedelta(days=1)
        chart_date_range = (start_date, end_date)
        
        report_sections = []
        for component in self.report_data['active_components']:
            component_data = self.all_flow_data[self.all_flow_data['market_component'] == component]
            if component_data.empty: continue

            charts_html = ""
            for category, category_data in component_data.groupby('category_short'):
                agg_chart_data = category_data.groupby(level='date')['value'].sum().to_frame()
                if not agg_chart_data.empty:
                    base64_img = self._generate_yoy_chart(agg_chart_data, f"Aggregate Flow: {category}", chart_date_range)
                    if base64_img:
                        charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{base64_img}" alt="{component} - {category}"></div>'
            
            tables_html = ""
            all_drivers_list = [df for key, df in self.flagged_drivers.items() if key.startswith(component)]
            
            if all_drivers_list:
                consolidated_drivers = pd.concat(all_drivers_list, ignore_index=True)
                consolidated_drivers['Abs YoY Vol Change'] = consolidated_drivers['YoY Vol Change'].abs()
                
                # Refined Driver Selection Logic
                top_3_drivers = consolidated_drivers.nlargest(3, 'Abs YoY Vol Change')
                
                remaining_drivers = consolidated_drivers.drop(top_3_drivers.index)
                
                # Get top 1 from each remaining category
                other_drivers = remaining_drivers.loc[remaining_drivers.groupby('Category')['Abs YoY Vol Change'].idxmax()]
                
                final_drivers = pd.concat([top_3_drivers, other_drivers]).drop_duplicates(subset=['Location', 'Category']).sort_values(by='Abs YoY Vol Change', ascending=False)
                
                # Cohort Rollup (Category Summary) Logic
                cohort_summary = None
                category_counts = consolidated_drivers['Category'].value_counts()
                large_cohorts = category_counts[category_counts >= self.config.COHORT_MIN_LOCATIONS].index.tolist()
                
                if large_cohorts:
                    summary_data = []
                    for cat in large_cohorts:
                        cohort_df = consolidated_drivers[consolidated_drivers['Category'] == cat]
                        summary_data.append({
                            'Category': cat,
                            'Locations': len(cohort_df),
                            'Net YoY Volume Change': cohort_df['YoY Vol Change'].sum(),
                            'Mean % Change': cohort_df['YoY % Change'].mean()
                        })
                    cohort_summary = pd.DataFrame(summary_data)

                tables_html = self._style_and_format_df(final_drivers, f"Top Anomaly Drivers for {component}", cohort_summary)

            if charts_html:
                report_sections.append({
                    "title": f"Market Component: {component}",
                    "charts_html": charts_html,
                    "tables_html": tables_html
                })
        self.report_data['sections'] = report_sections

    def _generate_yoy_chart(self, data: pd.DataFrame, title: str, date_range: tuple) -> str:
        """Generates a base64-encoded year-over-year comparison chart."""
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        data['plot_value'] = data['value'].rolling(window=self.config.ROC_SHORT_WINDOW, min_periods=1).mean()
        
        current_year = data.index.year.max()
        years_to_plot = range(current_year - self.config.HISTORICAL_YEARS, current_year + 1)
        
        data_by_year = {year: df for year, df in data.groupby(data.index.year) if year in years_to_plot}

        for year, df in sorted(data_by_year.items()):
            color, lw, z = ('#003f5c', 2.5, 10) if year == current_year else (None, 1.5, year - current_year + 5)
            plot_dates = df.index.map(lambda dt: dt.replace(year=current_year) if not (dt.month == 2 and dt.day == 29) else dt.replace(year=current_year, day=28))
            ax.plot(plot_dates, df['plot_value'], label=str(year), color=color, linewidth=lw, zorder=z, marker='.', markersize=3, alpha=0.9)

        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_ylabel("Flow Value (7-Day Average)"); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Year', loc='upper left'); fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_xlim(date_range)
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _style_and_format_df(self, df: pd.DataFrame, title: str, cohort_summary: pd.DataFrame | None = None) -> str:
        """Applies styling and formatting to the driver and cohort summary DataFrames."""
        
        full_html = ""

        # 1. Style the Cohort Summary table if it exists
        if cohort_summary is not None and not cohort_summary.empty:
            cohort_styled = cohort_summary.style.format({
                'Net YoY Volume Change': '{:,.0f}',
                'Mean % Change': '{:+.1%}',
            }).set_caption("Category Anomaly Summary")
            full_html += cohort_styled.to_html(escape=False, index=False)

        # 2. Style the individual drivers table
        if df.empty:
             return full_html

        display_cols = ['Location', 'Category', 'Anomaly Type', 'Current 7-Day Avg', 'YoY Avg', 'YoY Vol Change', 'YoY % Change']
        df_display = df[display_cols]

        styled = df_display.style.background_gradient(
            cmap='coolwarm_r', subset=['YoY Vol Change'], axis=0
        ).format({
            'Current 7-Day Avg': '{:,.0f}',
            'YoY Avg': '{:,.0f}',
            'YoY Vol Change': '{:,.0f}',
            'YoY % Change': '{:+.1%}',
        }, na_rep='-')
        
        full_html += styled.set_caption(title).to_html(escape=False, index=False)
        
        return full_html

    def _generate_html_report(self):
        """Renders the final HTML report using a Jinja2 template."""
        print("\n--- Generating Final HTML Report ---")
        env = Environment(loader=FileSystemLoader(self.config.SCRIPT_DIR))
        template = env.get_template(self.config.TEMPLATE_NAME)
        
        final_html = template.render(
            report_title="Market Flow Anomaly Report",
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            sections=self.report_data.get('sections', [])
        )
        
        output_path = self.config.OUTPUT_DIR / self.config.OUTPUT_FILENAME
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"\n--- Report generation complete. ---")
        print(f"Successfully saved to: {output_path}")

    def run(self):
        """Executes the entire analysis and reporting pipeline."""
        self.engine = self._get_db_engine()
        if not self.engine:
            return

        try:
            self._fetch_and_prepare_data()
            self._analyze_anomalies()
            self._generate_charts_and_tables()
            self._generate_html_report()
        except Exception as e:
            print(f"A fatal error occurred in the main process: {e}")
            traceback.print_exc()
        finally:
            if self.engine:
                self.engine.dispose()
                print("INFO: Database connection closed.")

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    config = ReportConfig()
    analyzer = MarketFlowAnalyzer(config)
    analyzer.run()
