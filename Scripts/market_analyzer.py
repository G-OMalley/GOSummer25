# market_analyzer.py
# This module contains functions for data processing and generating
# various market analysis charts and tables.

import pandas as pd
import os
import numpy as np
from calendar import month_name as calendar_month_names
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import calendar as py_calendar
import io
import base64

# --- Constants ---
HENRY_HUB_NAME_INDIVIDUAL = 'Henry'
COMPLETENESS_LOOKBACK_DAYS_REGIONAL = 90
COMPLETENESS_THRESHOLD_REGIONAL = 0.75
CHART_IMAGE_FORMAT_REGIONAL = 'png'
HENRY_HUB_NAME_REGIONAL = 'Henry'


# --- Data Loading and Preparation Functions ---

def load_and_filter_prices(prices_file_path, current_processing_date, lookback_days=COMPLETENESS_LOOKBACK_DAYS_REGIONAL, threshold=COMPLETENESS_THRESHOLD_REGIONAL):
    """
    Loads PRICES.csv, converts dates and numeric columns, and filters active market components.
    """
    print(f"Attempting to load PRICES.csv from: {os.path.abspath(prices_file_path)}")
    try:
        prices_df = pd.read_csv(prices_file_path)
        print(f"Successfully loaded '{prices_file_path}'.\n")
    except FileNotFoundError:
        print(f"ERROR: PRICES.csv file not found at '{os.path.abspath(prices_file_path)}'.")
        return pd.DataFrame(), []
    except Exception as e:
        print(f"Error loading PRICES.csv: {e}")
        return pd.DataFrame(), []

    if 'Date' not in prices_df.columns:
        print("ERROR: 'Date' column not found in PRICES.csv.")
        return pd.DataFrame(), []

    prices_df['Date_dt'] = pd.to_datetime(prices_df['Date'], errors='coerce')
    prices_df['Date'] = prices_df['Date_dt'].dt.date
    prices_df = prices_df.dropna(subset=['Date_dt', 'Date'])

    potential_component_cols = [col for col in prices_df.columns if col not in ['Date', 'Date_dt', 'Unnamed: 58', 'Date.1']]
    for col in potential_component_cols:
        prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')
    print("PRICES.csv: 'Date' converted, price columns to numeric.\n")

    completeness_end_date = current_processing_date
    completeness_start_date = completeness_end_date - timedelta(days=lookback_days - 1)

    print(f"Filtering active components based on data from {completeness_start_date.strftime('%Y-%m-%d')} to {completeness_end_date.strftime('%Y-%m-%d')}...")
    active_components = []
    component_candidates = [col for col in potential_component_cols if col.lower() != HENRY_HUB_NAME_REGIONAL.lower()]

    for component in component_candidates:
        if component not in prices_df.columns: continue
        series_in_period = prices_df[
            (prices_df['Date'] >= completeness_start_date) &
            (prices_df['Date'] <= completeness_end_date)
        ][component]

        if series_in_period.empty: continue

        unique_dates_in_period_for_df = prices_df[
            (prices_df['Date'] >= completeness_start_date) &
            (prices_df['Date'] <= completeness_end_date)
        ]['Date'].nunique()

        if unique_dates_in_period_for_df == 0: continue
        non_nan_count = series_in_period.notna().sum()
        completeness_ratio = non_nan_count / unique_dates_in_period_for_df if unique_dates_in_period_for_df > 0 else 0
        if completeness_ratio >= threshold:
            active_components.append(component)

    print(f"\nFound {len(active_components)} active market components (>= {threshold*100}% data in last {lookback_days} days ending {current_processing_date.strftime('%Y-%m-%d')}).")
    prices_df_dt_indexed = prices_df.set_index('Date_dt', drop=False).sort_index()
    return prices_df_dt_indexed, active_components


def load_ice_data_and_group_by_region(ice_file_path, active_prices_components,
                                      sheet_name="Daily_pricingLIVE",
                                      component_col='Description', forward_col='Mark0', region_col='Unnamed: 35',
                                      component_name_map=None):
    """
    Loads ICE data, maps components, and groups them by region.
    """
    if component_name_map is None:
        component_name_map = {}

    print(f"Attempting to load ICE data from: {os.path.abspath(ice_file_path)}")
    try:
        ice_df = pd.read_excel(ice_file_path, sheet_name=sheet_name)
        print(f"Successfully loaded sheet '{sheet_name}' from '{ice_file_path}'.\n")
    except FileNotFoundError:
        print(f"ERROR: ICE file not found at '{os.path.abspath(ice_file_path)}'.")
        return {}, []
    except Exception as e:
        print(f"Error loading ICE Excel: {e}")
        return {}, []

    ice_data_slice = ice_df.iloc[2:].copy()
    required_cols = [component_col, forward_col, region_col]
    if not all(col in ice_data_slice.columns for col in required_cols):
        print(f"ERROR: Required column(s) not found in ICE data sheet '{sheet_name}'.")
        return {}, []

    ice_data_slice.loc[:, forward_col] = pd.to_numeric(ice_data_slice[forward_col], errors='coerce')
    ice_data_slice = ice_data_slice.dropna(subset=[region_col])
    ice_data_slice[region_col] = ice_data_slice[region_col].astype(str).str.strip()
    ice_data_slice = ice_data_slice[ice_data_slice[region_col] != '0']
    ice_data_slice = ice_data_slice[ice_data_slice[region_col] != '']

    prices_name_to_ice_desc_map = {v: k for k, v in component_name_map.items()}
    mapped_components_details = []

    print("Mapping active PRICES.csv components to ICE regions and forward values...")
    for prices_comp_name in active_prices_components:
        ice_desc_name = prices_name_to_ice_desc_map.get(prices_comp_name, prices_comp_name)
        component_data_from_ice = ice_data_slice[ice_data_slice[component_col] == ice_desc_name]

        if not component_data_from_ice.empty:
            region = component_data_from_ice.iloc[0][region_col]
            forward = component_data_from_ice.iloc[0][forward_col]
            if pd.notna(region) and str(region).strip():
                mapped_components_details.append({
                    'prices_csv_name': prices_comp_name, 'price_region': str(region).strip(), 'forward_value': forward
                })
            else:
                mapped_components_details.append({
                    'prices_csv_name': prices_comp_name, 'price_region': None, 'forward_value': forward
                })
        else:
            mapped_components_details.append({
                'prices_csv_name': prices_comp_name, 'price_region': None, 'forward_value': np.nan
            })

    regional_groups = {}
    unregioned_components_details = []
    for item in mapped_components_details:
        region = item['price_region']
        component_detail = {'prices_csv_name': item['prices_csv_name'], 'forward_value': item['forward_value']}
        if region:
            if region not in regional_groups:
                regional_groups[region] = []
            regional_groups[region].append(component_detail)
        else:
            unregioned_components_details.append(component_detail)

    print("\n--- Components Grouped by Region (for active components) ---")
    for region, comps in regional_groups.items():
        print(f"Region: {region}")
        for comp_detail in comps:
            print(f"  - Component (PRICES.csv): {comp_detail['prices_csv_name']}, Forward: {comp_detail['forward_value']}")
    if unregioned_components_details:
        print("\n--- Unregioned Active Components ---")
        for comp_detail in unregioned_components_details:
            print(f"  - Component (PRICES.csv): {comp_detail['prices_csv_name']}, Forward: {comp_detail['forward_value']}")
    print("\n")
    return regional_groups, unregioned_components_details

def load_historical_fom(fom_file_path):
    """Loads HistoricalFOM.csv."""
    try:
        df_fom = pd.read_csv(fom_file_path)
        required_cols = ['settlement_month', 'settlement_year', 'market_component', 'settlement_basis']
        if not all(col in df_fom.columns for col in required_cols):
            print(f"ERROR: {fom_file_path} is missing one or more required columns: {required_cols}")
            return pd.DataFrame()
        df_fom['settlement_year'] = df_fom['settlement_year'].astype(int)
        print(f"Historical FOM data loaded successfully from {fom_file_path}.")
        return df_fom
    except FileNotFoundError:
        print(f"ERROR: FoM file not found at {fom_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading or processing '{fom_file_path}': {e}")
        return pd.DataFrame()

def load_key_and_weather_data(price_admin_file_path, weather_file_path):
    """
    Loads PriceAdmin.csv for mapping and WEATHER.csv for temperature data.
    This version trims whitespace from symbol columns for robust matching.
    """
    component_to_city_symbol_map = {}
    city_symbol_to_title_map = {}
    df_weather = pd.DataFrame()

    try:
        df_weather = pd.read_csv(weather_file_path)
        required_weather_cols = ['Date', 'City Title', 'City Symbol', 'Avg Temp']
        if not all(col in df_weather.columns for col in required_weather_cols):
            print(f"ERROR: {weather_file_path} is missing required columns: {required_weather_cols}.")
            return {}, {}, pd.DataFrame()

        df_weather['Date'] = pd.to_datetime(df_weather['Date'])
        df_weather['City Symbol'] = df_weather['City Symbol'].str.strip() # Trim whitespace
        df_weather = df_weather.set_index('Date').sort_index()
        print(f"WEATHER data loaded successfully from {weather_file_path}.")

        symbol_title_df = df_weather[['City Symbol', 'City Title']].drop_duplicates().dropna()
        city_symbol_to_title_map = pd.Series(symbol_title_df['City Title'].values, index=symbol_title_df['City Symbol']).to_dict()
        print("Built City Symbol-to-Title map from WEATHER.csv.")

    except FileNotFoundError:
        print(f"FATAL: WEATHER file not found at {weather_file_path}. Cannot proceed.")
        return {}, {}, pd.DataFrame()
    except Exception as e:
        print(f"Error loading or processing '{weather_file_path}': {e}.")
        return {}, {}, pd.DataFrame()

    try:
        required_admin_cols = ['Market Component', 'City Symbol']
        df_price_admin = pd.read_csv(price_admin_file_path, usecols=required_admin_cols)
        df_price_admin.dropna(subset=required_admin_cols, inplace=True)
        df_price_admin['Market Component'] = df_price_admin['Market Component'].str.strip()
        df_price_admin['City Symbol'] = df_price_admin['City Symbol'].str.strip() # Trim whitespace

        component_to_city_symbol_map = pd.Series(
            df_price_admin['City Symbol'].values,
            index=df_price_admin['Market Component']
        ).to_dict()
        print(f"PriceAdmin file loaded and mapped successfully from {price_admin_file_path}.")

    except FileNotFoundError:
        print(f"Warning: PriceAdmin file not found at {price_admin_file_path}.")
    except ValueError:
        print(f"ERROR: Could not find required columns ['Market Component', 'City Symbol'] in {price_admin_file_path}.")
    except Exception as e:
        print(f"Error loading or processing '{price_admin_file_path}': {e}.")

    return component_to_city_symbol_map, city_symbol_to_title_map, df_weather

# --- Regional Report Generation Functions ---

def generate_regional_daily_overlay_chart(region_name, components_in_region, prices_df_dt_indexed,
                                            chart_start_dt, chart_end_dt, fwd_mark_dt,
                                            output_charts_dir, henry_hub_name=HENRY_HUB_NAME_REGIONAL):
    """Generates and saves the daily regional overlay chart."""
    print(f"Generating Daily Matplotlib chart for region: {region_name}")
    fig, ax = plt.subplots(figsize=(18, 7))

    ts_chart_start = pd.to_datetime(chart_start_dt)
    ts_chart_end = pd.to_datetime(chart_end_dt)

    prices_chart_period = prices_df_dt_indexed[
        (prices_df_dt_indexed.index >= ts_chart_start) &
        (prices_df_dt_indexed.index <= ts_chart_end)
    ].copy()

    if prices_chart_period.empty:
        print(f"  No price data for daily chart in region '{region_name}' for the period. Skipping.")
        plt.close(fig)
        return None
    if henry_hub_name not in prices_chart_period.columns:
        print(f"  '{henry_hub_name}' not found in price data. Skipping daily chart for region '{region_name}'.")
        plt.close(fig)
        return None

    has_data_for_chart = False
    for comp_detail in components_in_region:
        comp_name = comp_detail['prices_csv_name']
        forward_val = comp_detail['forward_value']
        if comp_name not in prices_chart_period.columns: continue

        basis_series = prices_chart_period[comp_name] - prices_chart_period[henry_hub_name]
        if basis_series.dropna().empty: continue
        
        has_data_for_chart = True
        line, = ax.plot(prices_chart_period.index, basis_series, label=comp_name, linewidth=1.5)

        if pd.notna(forward_val):
            plot_fwd_mark_dt = pd.to_datetime(fwd_mark_dt)
            
            if ts_chart_start <= plot_fwd_mark_dt <= ts_chart_end:
                ax.plot([plot_fwd_mark_dt], [forward_val], marker='*', markersize=12, color=line.get_color(), linestyle='None', label=f'_nolegend_')
                ax.text(plot_fwd_mark_dt + timedelta(days=1), forward_val, f'{forward_val:.3f}', va='center', fontsize=9, color=line.get_color())
            else:
                last_valid_date = basis_series.dropna().index.max()
                if pd.notna(last_valid_date):
                    ax.plot([last_valid_date], [forward_val], marker='*', markersize=10, color=line.get_color(), linestyle='None', label=f'_nolegend_')
                    ax.text(last_valid_date + timedelta(days=1), forward_val, f'{forward_val:.3f}', va='bottom', ha='left', fontsize=8, color=line.get_color())

    if not has_data_for_chart:
        print(f"  No valid basis data for daily chart in region '{region_name}'. Skipping.")
        plt.close(fig)
        return None

    ax.set_title(f'Regional Daily Basis Overlay: {region_name}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Basis (Component - {henry_hub_name})', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=15))
    fig.autofmt_xdate(rotation=30, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    if len(components_in_region) > 7:
        ax.legend(title='Market Components', title_fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax.legend(title='Market Components', title_fontsize=10, loc='best', fontsize=9)
        plt.tight_layout()

    safe_region_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in str(region_name)).replace(' ', '_')
    chart_file_name = f"Region_{safe_region_name}_Daily_Overlay.{CHART_IMAGE_FORMAT_REGIONAL}"
    chart_file_path = Path(output_charts_dir) / chart_file_name
    
    try:
        plt.savefig(chart_file_path, bbox_inches='tight')
        print(f"  Daily chart saved to {chart_file_path}")
    except Exception as e:
        print(f"  Error saving Daily chart for {region_name}: {e}")
        chart_file_path = None
    finally:
        plt.close(fig)
        
    return chart_file_path.name if chart_file_path else None

def generate_regional_avg_spread_grid_heatmaps(region_name, components_in_region_details, prices_df_dt_indexed,
                                              current_processing_date, output_charts_dir,
                                              num_periods=3, period_days=15, henry_hub_name=HENRY_HUB_NAME_REGIONAL):
    """Generates and saves average spread grid heatmaps for different periods."""
    component_names = [comp['prices_csv_name'] for comp in components_in_region_details]
    if len(component_names) < 2:
        print(f"  Not enough components ({len(component_names)}) in {region_name} for spread grid. Skipping.")
        return []

    generated_grid_paths = []
    baseline_matrix_for_comparison = None

    for i_display_order in range(num_periods):
        calc_idx_from_today = (num_periods - 1) - i_display_order
        period_end_dt = current_processing_date - timedelta(days=calc_idx_from_today * period_days)
        period_start_dt = period_end_dt - timedelta(days=period_days - 1)

        period_label = f"Last {period_days} Days" if calc_idx_from_today == 0 else f"Days {(calc_idx_from_today * period_days) + 1}-{(calc_idx_from_today + 1) * period_days} Ago"
        title_period_str = f"{period_start_dt.strftime('%b %d')} - {period_end_dt.strftime('%b %d, %Y')}"
        print(f"Generating Avg Spread Grid for region: {region_name}, Period: {period_label} ({title_period_str})")

        period_prices_df = prices_df_dt_indexed[(prices_df_dt_indexed.index >= pd.to_datetime(period_start_dt)) & (prices_df_dt_indexed.index <= pd.to_datetime(period_end_dt))][component_names].copy()
        if period_prices_df.empty:
            print(f"  Not enough data in period {period_label} for spread grid. Skipping.")
            continue

        current_avg_spread_matrix = pd.DataFrame(index=component_names, columns=component_names, dtype=float)
        for comp_row in component_names:
            for comp_col in component_names:
                if comp_row != comp_col and comp_row in period_prices_df.columns and comp_col in period_prices_df.columns:
                    spread_series = pd.to_numeric(period_prices_df[comp_row], errors='coerce') - pd.to_numeric(period_prices_df[comp_col], errors='coerce')
                    current_avg_spread_matrix.loc[comp_row, comp_col] = spread_series.mean()

        if current_avg_spread_matrix.isnull().all().all():
            print(f"  Could not compute valid average spreads for {region_name}, period {period_label}. Skipping.")
            continue

        data_to_annotate = current_avg_spread_matrix.astype(float)
        plt.figure(figsize=(max(8, len(component_names) * 1.2), max(6, len(component_names) * 1.0)))

        is_this_period_the_baseline = (calc_idx_from_today == num_periods - 1)
        if is_this_period_the_baseline:
            baseline_matrix_for_comparison = current_avg_spread_matrix.copy()
            sns.heatmap(data_to_annotate, annot=True, fmt=".3f", cmap=ListedColormap(['#FAFAFA']), cbar=False, linewidths=.5, linecolor='lightgray', annot_kws={"size": 9, "color": "black", "weight": "bold"})
            heatmap_title = f'Avg Spread (Row-Col)\n{period_label}: {region_name}'
        elif baseline_matrix_for_comparison is not None:
            aligned_current, aligned_base = current_avg_spread_matrix.align(baseline_matrix_for_comparison, join='outer', axis=None)
            difference_matrix = aligned_current.astype(float) - aligned_base.astype(float)
            all_color_values = difference_matrix.stack().dropna()
            v_max = all_color_values.abs().max() if not all_color_values.empty else 0.01
            sns.heatmap(difference_matrix, annot=data_to_annotate, cmap="coolwarm_r", fmt=".3f", center=0, vmin=-v_max, vmax=v_max, annot_kws={"size": 9}, linewidths=.5, linecolor='lightgray')
            heatmap_title = f'Avg Spread (Value) / Change from Base (Color)\n{period_label}: {region_name}'
        else:
            sns.heatmap(data_to_annotate, annot=True, fmt=".3f", cmap=ListedColormap(['#FAFAFA']), cbar=False, linewidths=.5, linecolor='lightgray', annot_kws={"size": 9, "color": "black", "weight": "bold"})
            heatmap_title = f'Avg Spread (Row-Col)\n{period_label}: {region_name} (Error: Base missing)'

        plt.title(heatmap_title, fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        safe_region_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in str(region_name)).replace(' ', '_')
        safe_period_label = period_label.replace(" ", "_").replace("-", "_to_")
        chart_file_name = f"Region_{safe_region_name}_AvgSpreadGrid_{safe_period_label}.{CHART_IMAGE_FORMAT_REGIONAL}"
        chart_file_path = Path(output_charts_dir) / chart_file_name
        try:
            plt.savefig(chart_file_path, bbox_inches='tight')
            print(f"  Avg Spread Grid ({period_label}) for {region_name} saved to {chart_file_path}")
            generated_grid_paths.append(chart_file_path.name)
        except Exception as e:
            print(f"  Error saving avg spread grid for {region_name}, {period_label}: {e}")
        finally:
            plt.close('all')
    return generated_grid_paths

def generate_regional_historical_monthly_basis(region_name, components_in_region_details, prices_df_dt_indexed,
                                                prior_years, month_indices, output_charts_dir,
                                                henry_hub_name=HENRY_HUB_NAME_REGIONAL):
    """Generates and saves historical monthly average basis charts for given years."""
    generated_chart_paths = []
    if henry_hub_name not in prices_df_dt_indexed.columns:
        print(f"  '{henry_hub_name}' not found. Skipping monthly charts for {region_name}.")
        return []

    for target_year in prior_years:
        sorted_month_indices = sorted(month_indices)
        month_labels = [py_calendar.month_abbr[m] + f"'{str(target_year)[-2:]}" for m in sorted_month_indices]
        print(f"Generating Monthly Avg Basis chart for region: {region_name}, Year: {target_year}, Months: {month_labels}")
        fig, ax = plt.subplots(figsize=(8, 4))
        has_data_for_this_year_chart = False

        for comp_detail in components_in_region_details:
            comp_name = comp_detail['prices_csv_name']
            if comp_name not in prices_df_dt_indexed.columns: continue
            monthly_avg_basis_values = []

            for month_idx in sorted_month_indices:
                month_data = prices_df_dt_indexed[(prices_df_dt_indexed['Date_dt'].dt.year == target_year) & (prices_df_dt_indexed['Date_dt'].dt.month == month_idx)].copy()
                if month_data.empty:
                    monthly_avg_basis_values.append(np.nan)
                    continue
                daily_basis = pd.to_numeric(month_data[comp_name], errors='coerce') - pd.to_numeric(month_data[henry_hub_name], errors='coerce')
                monthly_avg_basis_values.append(daily_basis.mean())

            if any(pd.notna(val) for val in monthly_avg_basis_values):
                has_data_for_this_year_chart = True
                ax.plot(month_labels, monthly_avg_basis_values, marker='o', linestyle='-', label=comp_name, linewidth=1.2)

        if not has_data_for_this_year_chart:
            print(f"  No valid monthly avg basis data for {region_name}, {target_year}. Skipping.")
            plt.close(fig)
            continue

        ax.set_title(f'Monthly Avg Basis {target_year}\n{region_name}', fontsize=11)
        ax.set_xlabel('Month', fontsize=9)
        ax.set_ylabel('Avg Basis', fontsize=9)
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

        if len(components_in_region_details) > 4:
            ax.legend(title='Market Components', title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
            plt.tight_layout(rect=[0, 0, 0.80, 1])
        else:
            ax.legend(title='Market Components', title_fontsize=8, loc='best', fontsize=7)
            plt.tight_layout()

        safe_region_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in str(region_name)).replace(' ', '_')
        chart_file_name = f"Region_{safe_region_name}_MonthlyAvg_{target_year}.{CHART_IMAGE_FORMAT_REGIONAL}"
        chart_file_path = Path(output_charts_dir) / chart_file_name
        try:
            plt.savefig(chart_file_path, bbox_inches='tight')
            print(f"  Monthly Avg Basis chart ({target_year}) for {region_name} saved to {chart_file_path}")
            generated_chart_paths.append(chart_file_path.name)
        except Exception as e:
            print(f"  Error saving Monthly Avg chart for {region_name}, {target_year}: {e}")
        finally:
            plt.close(fig)
    return generated_chart_paths


# --- Individual Component Report Generation Functions ---

def calculate_individual_monthly_cash_basis_table(prices_df_dt_indexed, market_component, henry_component=HENRY_HUB_NAME_INDIVIDUAL):
    """Calculates the monthly average cash basis table for an individual component."""
    if market_component not in prices_df_dt_indexed.columns:
        return None
    if henry_component not in prices_df_dt_indexed.columns:
        print(f"CRITICAL: Henry component '{henry_component}' not found in prices data. Cannot calculate basis for {market_component}.")
        return None

    component_series = pd.to_numeric(prices_df_dt_indexed[market_component], errors='coerce')
    henry_series = pd.to_numeric(prices_df_dt_indexed[henry_component], errors='coerce')
    daily_basis = component_series - henry_series
    if daily_basis.dropna().empty: return None

    monthly_avg_basis = daily_basis.resample('ME').mean()
    gas_year_data = {}
    for date_val, basis_value in monthly_avg_basis.items():
        if pd.isna(basis_value): continue
        year, month_val = date_val.year, date_val.month
        month_name_str = date_val.strftime('%B')
        gas_year_label = f"{year}-{year + 1}" if month_val >= 4 else f"{year - 1}-{year}"
        if gas_year_label not in gas_year_data: gas_year_data[gas_year_label] = {}
        gas_year_data[gas_year_label][month_name_str] = basis_value

    if not gas_year_data: return None
    df_heatmap = pd.DataFrame.from_dict(gas_year_data)
    month_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March']
    df_heatmap = df_heatmap.reindex(month_order)
    df_heatmap = df_heatmap[sorted(df_heatmap.columns)]
    df_heatmap = df_heatmap.round(3)
    df_heatmap.dropna(axis=1, how='all', inplace=True)
    return df_heatmap if not df_heatmap.empty else None

def style_individual_heatmap_table(df, title, axis_format_for_gradient):
    """Styles the heatmap table and returns HTML."""
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available to display for this heatmap.</p>"
    return (df.style.background_gradient(cmap='coolwarm_r', axis=axis_format_for_gradient)
            .format("{:.3f}", na_rep="")
            .set_caption(title)
            .set_table_styles([
                {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold'), ('text-align', 'center'), ('margin-bottom', '10px')]},
                {'selector': 'th, td', 'props': [('border', '1px solid #ccc'), ('padding', '5px')]},
                {'selector': 'th', 'props': [('background-color', '#f2f2f2')]}
            ]).to_html())

def generate_individual_fom_vs_cash_table(monthly_cash_basis_df, df_fom_historical, market_component_name, current_processing_date):
    """Generates data for FoM vs. Cash table."""
    if monthly_cash_basis_df is None:
        monthly_cash_basis_df = pd.DataFrame()

    fom_component_data = df_fom_historical[df_fom_historical['market_component'] == market_component_name]
    min_year_to_display = 2015
    current_actual_year = current_processing_date.year
    current_actual_month_idx = current_processing_date.month

    max_fom_year_with_data = 0
    if not fom_component_data.empty and 'settlement_basis' in fom_component_data.columns:
        fom_data_with_values = fom_component_data.dropna(subset=['settlement_basis'])
        if not fom_data_with_values.empty and 'settlement_year' in fom_data_with_values.columns:
            max_fom_year_with_data = fom_data_with_values['settlement_year'].max()

    max_cash_calendar_year = 0
    if not monthly_cash_basis_df.empty:
        cash_gas_years_end = [int(col.split('-')[1]) for col in monthly_cash_basis_df.columns if isinstance(col, str) and '-' in col]
        if cash_gas_years_end:
            max_cash_calendar_year = max(cash_gas_years_end)

    latest_year_to_display = max(max_fom_year_with_data, max_cash_calendar_year, min_year_to_display)

    table_data_list = []
    calendar_months_ordered = [calendar_month_names[i] for i in range(1, 13)]

    for year_iter in range(min_year_to_display, latest_year_to_display + 1):
        fom_col_name = f"'{str(year_iter)[-2:]} FoM"
        cash_col_name = f"'{str(year_iter)[-2:]} Cash"
        current_year_fom_data = []
        current_year_cash_data = []

        for month_idx, month_name_str in enumerate(calendar_months_ordered, 1):
            if year_iter == current_actual_year and month_idx > current_actual_month_idx:
                current_year_fom_data.append(np.nan)
                current_year_cash_data.append(np.nan)
                continue

            fom_val_series = fom_component_data[(fom_component_data['settlement_year'] == year_iter) & (fom_component_data['settlement_month'] == month_name_str)]['settlement_basis']
            current_year_fom_data.append(fom_val_series.iloc[0] if not fom_val_series.empty else np.nan)

            gas_year_col_for_cash = f"{year_iter}-{year_iter+1}" if month_idx >= 4 else f"{year_iter-1}-{year_iter}"
            cash_val = np.nan
            if not monthly_cash_basis_df.empty and gas_year_col_for_cash in monthly_cash_basis_df.columns and month_name_str in monthly_cash_basis_df.index:
                cash_val = monthly_cash_basis_df.loc[month_name_str, gas_year_col_for_cash]
            current_year_cash_data.append(cash_val)

        table_data_list.append(pd.Series(current_year_fom_data, index=calendar_months_ordered, name=fom_col_name))
        table_data_list.append(pd.Series(current_year_cash_data, index=calendar_months_ordered, name=cash_col_name))

    if not table_data_list:
        return pd.DataFrame(index=calendar_months_ordered)

    return pd.concat(table_data_list, axis=1).round(3)

def style_individual_fom_vs_cash_table(df, title):
    """Styles the FoM vs. Cash table and returns HTML."""
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available to display for FoM vs. Cash.</p>"

    def _color_cash_cells_styling(data_series_row):
        styles = [''] * len(data_series_row)
        col_names_in_row = data_series_row.index.tolist()

        for i in range(0, len(col_names_in_row) -1 , 2):
            fom_col_name, cash_col_name = col_names_in_row[i], col_names_in_row[i+1]
            if not (fom_col_name.endswith(" FoM") and cash_col_name.endswith(" Cash")): continue

            fom_val, cash_val = data_series_row[fom_col_name], data_series_row[cash_col_name]
            if pd.isna(fom_val) or pd.isna(cash_val): continue

            diff = cash_val - fom_val
            colors = {'dg': '#548235', 'mg': '#A9D08E', 'lg': '#C6EFCE', 'lr': '#FFCDD2', 'mr': '#F44336', 'dr': '#D32F2F'}
            text_color = 'black'
            bg_color = ''

            if diff > 0.10: bg_color, text_color = colors['dg'], 'white'
            elif diff > 0.05: bg_color = colors['mg']
            elif diff > 0.02: bg_color = colors['lg']
            elif diff < -0.10: bg_color, text_color = colors['dr'], 'white'
            elif diff < -0.05: bg_color = colors['mr']
            elif diff < -0.02: bg_color = colors['lr']

            if bg_color: styles[i+1] = f'background-color: {bg_color}; color: {text_color};'
        return styles

    return (df.style.apply(_color_cash_cells_styling, axis=1)
            .format("{:.3f}", na_rep="")
            .set_caption(title)
            .set_table_styles([
                {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold'), ('text-align', 'center'), ('margin-bottom', '10px')]},
                {'selector': 'th, td', 'props': [('border', '1px solid #ccc'), ('padding', '5px')]},
                {'selector': 'th', 'props': [('background-color', '#f2f2f2')]}
            ]).to_html())

def generate_individual_basis_history_chart(prices_df_dt_indexed, component_name, mark0_fom_value,
                                            output_charts_dir, current_processing_date,
                                            henry_name=HENRY_HUB_NAME_INDIVIDUAL):
    """
    Generates and saves the daily basis vs. history chart (as a base64 image string for HTML).
    This corrected version ensures historical data is properly aligned on a shared date axis.
    """
    try:
        if component_name not in prices_df_dt_indexed.columns or henry_name not in prices_df_dt_indexed.columns:
            return f"<p>Error: Price data missing for {component_name} or {henry_name} for basis history chart.</p>"

        today_dt = pd.to_datetime(current_processing_date)
        chart_start_date = (today_dt.replace(day=1) - relativedelta(months=1))
        chart_end_date = (today_dt.replace(day=1) + relativedelta(months=2) - relativedelta(days=1))

        basis_full_history = (pd.to_numeric(prices_df_dt_indexed[component_name], errors='coerce') -
                              pd.to_numeric(prices_df_dt_indexed[henry_name], errors='coerce')).dropna()

        if basis_full_history.empty:
            return f"<p>No basis data available for {component_name}.</p>"

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        colors = ['black', '#1f77b4', '#2ca02c', '#d62728']
        linestyles = ['-', ':', '--', '-.']
        linewidths = [2.0, 1.5, 1.5, 1.5]
        z_orders = [5, 4, 3, 2]

        for i in range(4):
            year_offset = i
            label = 'Current' if i == 0 else f'Y-{i}'
            
            hist_start = chart_start_date - relativedelta(years=year_offset)
            hist_end = chart_end_date - relativedelta(years=year_offset)
            
            data_in_window = basis_full_history[(basis_full_history.index >= hist_start) & 
                                                (basis_full_history.index <= hist_end)]
            
            if not data_in_window.empty:
                plot_index = data_in_window.index + pd.DateOffset(years=year_offset)
                
                ax.plot(plot_index, data_in_window.values, 
                        label=f'{component_name} ({label})', 
                        linewidth=linewidths[i], 
                        color=colors[i], 
                        linestyle=linestyles[i], 
                        zorder=z_orders[i])

        if pd.notna(mark0_fom_value):
            ax.axhline(y=mark0_fom_value, color='magenta', linestyle='--', 
                       label=f'FoM Mark0: {mark0_fom_value:.3f}', linewidth=1.8, zorder=6)

        ax.set_title(f'{component_name} Daily Basis vs. History & FoM Mark0', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Basis ($/MMBtu)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_xlim(chart_start_date, chart_end_date)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        fig.autofmt_xdate(rotation=30, ha='right')

        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        
        plt.tight_layout(pad=1.0)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_base64}" alt="{component_name} Basis History Chart" style="width:100%; max-width:900px; display:block; margin:auto;"/>'

    except Exception as e:
        print(f"Error generating basis history chart for {component_name}: {e}")
        return f"<p>Error generating basis history chart for {component_name}: {e}</p>"

def generate_individual_temp_scatter_plot(prices_df_dt_indexed, df_weather, component_name,
                                          city_title, city_symbol,
                                          output_charts_dir, current_processing_date,
                                          henry_name=HENRY_HUB_NAME_INDIVIDUAL,
                                          weather_forecast_file_path=None):
    """
    Generates and saves the basis vs. temp scatter plot and a formatted forecast table.
    This version manually builds the HTML table to ensure correct formatting and structure.
    """
    try:
        if component_name not in prices_df_dt_indexed.columns or henry_name not in prices_df_dt_indexed.columns:
            return f"<p>Error: Price data missing for {component_name} or {henry_name} for scatter plot.</p>"
        if df_weather.empty or city_symbol not in df_weather['City Symbol'].unique():
            return f"<p>No weather data found for City Symbol '{city_symbol}' needed for {component_name} scatter plot.</p>"

        today_dt = pd.to_datetime(current_processing_date)
        
        plot_data_all_years = {}
        basis_series = (pd.to_numeric(prices_df_dt_indexed[component_name], errors='coerce') -
                        pd.to_numeric(prices_df_dt_indexed[henry_name], errors='coerce')).dropna()

        for year_offset in range(4):
            target_year_end_date = today_dt - relativedelta(years=year_offset)
            target_year_start_date = target_year_end_date - timedelta(days=60)
            
            basis_period = basis_series[(basis_series.index >= target_year_start_date) & 
                                        (basis_series.index <= target_year_end_date)]
            if basis_period.empty: continue

            weather_period = df_weather[(df_weather['City Symbol'] == city_symbol) & 
                                        (df_weather.index >= target_year_start_date) &
                                        (df_weather.index <= target_year_end_date)]['Avg Temp'].dropna()
            if weather_period.empty: continue

            merged_data = pd.merge(basis_period.rename('basis'), weather_period.rename('avg_temp'),
                                   left_index=True, right_index=True, how='inner')
            if not merged_data.empty:
                plot_data_all_years[year_offset] = merged_data

        if not plot_data_all_years:
            return f"<p>No combined basis & temp data for {component_name}/{city_title} for required periods.</p>"

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
        polynomial_degree = 2
        current_year_poly_func = None
        
        year_colors = ['black', '#1f77b4', '#2ca02c', '#d62728']
        year_alphas = [0.7, 0.6, 0.5, 0.4]
        year_marker_sizes = [50, 30, 30, 30]
        year_line_styles = ['-', '--', ':', '-.']
        
        for year_offset, data_for_year in sorted(plot_data_all_years.items(), reverse=True):
            label_suffix = "(Current)" if year_offset == 0 else f"(Y-{year_offset})"
            color, marker_size, alpha, line_style, line_width = (
                year_colors[year_offset], year_marker_sizes[year_offset], 
                year_alphas[year_offset], year_line_styles[year_offset], 
                2 if year_offset == 0 else 1.5
            )

            ax.scatter(data_for_year['avg_temp'], data_for_year['basis'], label=f'{component_name} {label_suffix}',
                       color=color, s=marker_size, alpha=alpha, edgecolors='w', linewidth=0.5, zorder=(5 - year_offset))
            
            if len(data_for_year) >= polynomial_degree + 1:
                try:
                    coeffs = np.polyfit(data_for_year['avg_temp'], data_for_year['basis'], polynomial_degree)
                    poly_func = np.poly1d(coeffs)
                    if year_offset == 0:
                        current_year_poly_func = poly_func
                    temp_range_for_line = np.linspace(data_for_year['avg_temp'].min(), data_for_year['avg_temp'].max(), 100)
                    ax.plot(temp_range_for_line, poly_func(temp_range_for_line), color=color, linestyle=line_style,
                            linewidth=line_width, alpha=max(0.3, alpha - 0.1), zorder=(5 - year_offset))
                except Exception as e_fit:
                    print(f"Could not fit/plot trendline for {component_name} {label_suffix}: {e_fit}")

        ax.set_title(f'{component_name} Basis vs. {city_title} Avg Temp (Last 61 Days & Prior Years)', fontsize=13)
        ax.set_xlabel('Average Temperature (°F)', fontsize=11)
        ax.set_ylabel('Daily Basis ($/MMBtu)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        plt.tight_layout(pad=1.0)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        chart_html = f'<img src="data:image/png;base64,{img_base64}" alt="{component_name} Basis vs. Temp Chart" style="width:100%;"/>'

        # --- Manually Build Forecast Table HTML ---
        forecast_table_html = ""
        if current_year_poly_func is None:
            return f'<div style="width:100%; max-width:800px; margin:auto;">{chart_html}<p>Forecast table not available: no recent trendline.</p></div>'

        if not weather_forecast_file_path or not os.path.exists(weather_forecast_file_path):
             return f'<div style="width:100%; max-width:800px; margin:auto;">{chart_html}<p>Forecast table not available: forecast file missing.</p></div>'

        try:
            forecast_df = pd.read_csv(weather_forecast_file_path)
            # Ensure 'Avg Temp' is numeric, coercing errors
            forecast_df['Avg Temp'] = pd.to_numeric(forecast_df['Avg Temp'], errors='coerce')
            # Drop rows where temperature is not a valid number
            forecast_df.dropna(subset=['Avg Temp'], inplace=True)

            city_forecast_df = forecast_df[forecast_df['City Symbol'] == city_symbol].copy()

            if city_forecast_df.empty:
                return f'<div style="width:100%; max-width:800px; margin:auto;">{chart_html}<p>Forecast table not available for this city.</p></div>'
            
            city_forecast_df['Date'] = pd.to_datetime(city_forecast_df['Date'])
            city_forecast_df['Predicted Basis'] = current_year_poly_func(city_forecast_df['Avg Temp'])
            
            avg_temp = city_forecast_df['Avg Temp'].mean()
            avg_basis = city_forecast_df['Predicted Basis'].mean()

            # --- Start building the HTML string ---
            html = ['<div style="overflow-x: auto; width: 100%;">']
            html.append('<table style="font-size: 9px; border-collapse: collapse; white-space: nowrap;">')
            
            # 1. Header Row
            html.append('<tr style="background-color: #f2f2f2;">')
            html.append('<th style="padding: 4px; border: 1px solid #ccc; text-align: left; font-weight: bold;">Date</th>')
            html.append(f'<th style="padding: 4px; border: 1px solid #ccc; font-weight: bold;">Avg</th>')
            for date_val in city_forecast_df['Date']:
                # --- THIS IS THE CORRECTED LINE ---
                html.append(f'<th style="padding: 4px; border: 1px solid #ccc; font-weight: bold;">{date_val.strftime("%#m/%#d")}</th>')
            html.append('</tr>')

            # 2. Temperature Row
            html.append('<tr>')
            html.append('<td style="padding: 4px; border: 1px solid #ccc; text-align: left; font-weight: bold;">Temp</td>')
            avg_temp_str = f"{avg_temp:.0f}" if pd.notna(avg_temp) else "N/A"
            html.append(f'<td style="padding: 4px; border: 1px solid #ccc; text-align: center;">{avg_temp_str}</td>')
            for temp_val in city_forecast_df['Avg Temp']:
                cell_val = f"{temp_val:.0f}" if pd.notna(temp_val) else "N/A"
                html.append(f'<td style="padding: 4px; border: 1px solid #ccc; text-align: center;">{cell_val}</td>')
            html.append('</tr>')
            
            # 3. Basis Row
            html.append('<tr>')
            html.append('<td style="padding: 4px; border: 1px solid #ccc; text-align: left; font-weight: bold;">Basis</td>')
            avg_basis_str = f"{avg_basis:.3f}" if pd.notna(avg_basis) else "N/A"
            html.append(f'<td style="padding: 4px; border: 1px solid #ccc; text-align: center;">{avg_basis_str}</td>')
            for basis_val in city_forecast_df['Predicted Basis']:
                cell_val = f"{basis_val:.3f}" if pd.notna(basis_val) else "N/A"
                html.append(f'<td style="padding: 4px; border: 1px solid #ccc; text-align: center;">{cell_val}</td>')
            html.append('</tr>')

            html.append('</table>')
            html.append('</div>')
            
            forecast_table_html = "".join(html)

        except Exception as e:
            print(f"Error during forecast table HTML generation for {component_name}: {e}")
            forecast_table_html = f"<p>An error occurred while creating the forecast table: {e}</p>"

        return f'<div style="width:100%; max-width:800px; margin:auto;">{chart_html}{forecast_table_html}</div>'

    except Exception as e:
        print(f"FATAL Error in generate_individual_temp_scatter_plot for {component_name}: {e}")
        traceback.print_exc()
        return f"<p>Error generating basis vs. temp scatter for {component_name}: {e}</p>"