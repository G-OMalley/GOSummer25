# market_analyzer.py
# This module contains functions for data processing and generating
# various market analysis charts and tables.

import pandas as pd
import os
import numpy as np
from calendar import month_name as calendar_month_names, monthrange
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap 
import seaborn as sns # For heatmaps
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
    Returns a dictionary of regional groups and a list of unregioned components.
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
    for col_check in required_cols:
        if col_check not in ice_data_slice.columns:
            print(f"ERROR: Required column '{col_check}' not found in ICE data sheet '{sheet_name}'.")
            print(f"Available columns: {list(ice_data_slice.columns)}")
            return {}, []

    ice_data_slice.loc[:, forward_col] = pd.to_numeric(ice_data_slice[forward_col], errors='coerce')
    ice_data_slice = ice_data_slice.dropna(subset=[region_col]) 
    ice_data_slice[region_col] = ice_data_slice[region_col].astype(str)
    ice_data_slice = ice_data_slice[ice_data_slice[region_col] != '0'] 
    ice_data_slice = ice_data_slice[ice_data_slice[region_col].str.strip() != ''] 

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
                    'prices_csv_name': prices_comp_name,
                    'ice_description': ice_desc_name,
                    'price_region': str(region).strip(), 
                    'forward_value': forward
                })
            else: 
                 mapped_components_details.append({
                    'prices_csv_name': prices_comp_name,
                    'ice_description': ice_desc_name,
                    'price_region': None, 
                    'forward_value': forward
                })
        else: 
            mapped_components_details.append({
                'prices_csv_name': prices_comp_name,
                'ice_description': None, 
                'price_region': None, 
                'forward_value': np.nan 
            })

    regional_groups = {}
    unregioned_components_details = []

    for item in mapped_components_details:
        region = item['price_region']
        component_detail_for_group = {'prices_csv_name': item['prices_csv_name'], 'forward_value': item['forward_value']}
        if region: 
            if region not in regional_groups:
                regional_groups[region] = []
            regional_groups[region].append(component_detail_for_group)
        else:
            unregioned_components_details.append(component_detail_for_group) 

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

def load_key_and_weather_data(key_file_path, weather_file_path):
    """Loads KEY.xlsx for city mapping and WEATHER.csv for temperature data."""
    component_to_city_map = {}
    df_weather = pd.DataFrame()

    try:
        df_key = pd.read_excel(key_file_path, sheet_name=0, header=None, engine='openpyxl')
        if df_key.shape[1] >= 5: 
            comp_series = df_key.iloc[:, 0].astype(str).str.strip()
            city_series = df_key.iloc[:, 4].astype(str).str.strip()
            temp_df_for_map = pd.DataFrame({'component': comp_series, 'city': city_series})
            temp_df_for_map.replace('nan', np.nan, inplace=True)
            temp_df_for_map.dropna(subset=['component', 'city'], inplace=True)
            component_to_city_map = pd.Series(temp_df_for_map['city'].values, index=temp_df_for_map['component']).to_dict()
            print(f"KEY file loaded and mapped successfully from {key_file_path}.")
        else:
            print(f"Warning: KEY file at {key_file_path} does not have at least 5 columns. City mapping will be empty.")
    except FileNotFoundError:
        print(f"Warning: KEY file not found at {key_file_path}. Scatter plots requiring city mapping may be affected.")
    except Exception as e:
        print(f"Error loading or processing '{key_file_path}': {e}. City mapping may be affected.")

    try:
        df_weather = pd.read_csv(weather_file_path)
        required_weather_cols = ['Date', 'City Title', 'Avg Temp']
        if not all(col in df_weather.columns for col in required_weather_cols):
            print(f"ERROR: {weather_file_path} is missing required columns: {required_weather_cols}. Scatter plots will be affected.")
            df_weather = pd.DataFrame()
        else:
            df_weather['Date'] = pd.to_datetime(df_weather['Date'])
            df_weather = df_weather.set_index('Date').sort_index()
            print(f"WEATHER data loaded successfully from {weather_file_path}.")
    except FileNotFoundError:
        print(f"Warning: WEATHER file not found at {weather_file_path}. Scatter plots will be affected.")
    except Exception as e:
        print(f"Error loading or processing '{weather_file_path}': {e}. Scatter plots will be affected.")

    return component_to_city_map, df_weather

# --- Regional Report Generation Functions ---

def generate_regional_daily_overlay_chart(region_name, components_in_region, prices_df_dt_indexed,
                                          chart_start_dt, chart_end_dt, fwd_mark_dt,
                                          output_charts_dir, henry_hub_name=HENRY_HUB_NAME_REGIONAL):
    """Generates and saves the daily regional overlay chart."""
    print(f"Generating Daily Matplotlib chart for region: {region_name}")
    fig, ax = plt.subplots(figsize=(18, 7)) 

    prices_chart_period = prices_df_dt_indexed[
        (prices_df_dt_indexed.index >= pd.to_datetime(chart_start_dt)) &
        (prices_df_dt_indexed.index <= pd.to_datetime(chart_end_dt))
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
        if comp_name not in prices_chart_period.columns:
            continue

        basis_series = prices_chart_period[comp_name] - prices_chart_period[henry_hub_name]
        if basis_series.dropna().empty:
            continue
        has_data_for_chart = True
        line, = ax.plot(prices_chart_period.index, basis_series, label=comp_name, linewidth=1.5) 

        if pd.notna(forward_val):
            plot_fwd_mark_dt = pd.to_datetime(fwd_mark_dt)
            if prices_chart_period.index.min() <= plot_fwd_mark_dt <= prices_chart_period.index.max():
                ax.plot([plot_fwd_mark_dt], [forward_val], marker='*', markersize=12, color=line.get_color(), linestyle='None', label=f'_nolegend_')
                ax.text(plot_fwd_mark_dt + timedelta(days=0.5), forward_val, f'{forward_val:.3f}', va='center', fontsize=9, color=line.get_color())
            else:
                last_valid_date = basis_series.dropna().index.max()
                if pd.notna(last_valid_date):
                    ax.plot([last_valid_date], [forward_val], marker='*', markersize=10, color=line.get_color(), linestyle='None', label=f'_nolegend_')
                    ax.text(last_valid_date + timedelta(days=0.5), forward_val, f'{forward_val:.3f}', va='bottom', ha='left', fontsize=8, color=line.get_color())

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

        if calc_idx_from_today == 0: 
            period_label = f"Last {period_days} Days"
        else:
            days_ago_start = calc_idx_from_today * period_days + 1
            days_ago_end = (calc_idx_from_today + 1) * period_days
            period_label = f"Days {days_ago_start}-{days_ago_end} Ago"

        title_period_str = f"{period_start_dt.strftime('%b %d')} - {period_end_dt.strftime('%b %d, %Y')}"
        print(f"Generating Avg Spread Grid for region: {region_name}, Period: {period_label} ({title_period_str})")

        period_prices_df = prices_df_dt_indexed[
            (prices_df_dt_indexed.index >= pd.to_datetime(period_start_dt)) &
            (prices_df_dt_indexed.index <= pd.to_datetime(period_end_dt))
        ][component_names].copy()

        if period_prices_df.empty or len(period_prices_df) < 1:
            print(f"  Not enough data in period {period_label} for spread grid. Skipping.")
            continue

        current_avg_spread_matrix = pd.DataFrame(index=component_names, columns=component_names, dtype=float)
        for comp_row in component_names:
            for comp_col in component_names:
                if comp_row == comp_col:
                    current_avg_spread_matrix.loc[comp_row, comp_col] = np.nan
                elif comp_row in period_prices_df.columns and comp_col in period_prices_df.columns:
                    series_row = pd.to_numeric(period_prices_df[comp_row], errors='coerce')
                    series_col = pd.to_numeric(period_prices_df[comp_col], errors='coerce')
                    spread_series = series_row - series_col
                    current_avg_spread_matrix.loc[comp_row, comp_col] = spread_series.mean()
                else:
                    current_avg_spread_matrix.loc[comp_row, comp_col] = np.nan

        if current_avg_spread_matrix.isnull().all().all():
            print(f"  Could not compute valid average spreads for {region_name}, period {period_label}. Skipping.")
            continue

        data_to_annotate = current_avg_spread_matrix.astype(float)
        plt.figure(figsize=(max(8, len(component_names) * 1.2), max(6, len(component_names) * 1.0))) 

        is_this_period_the_baseline = (calc_idx_from_today == num_periods - 1)

        if is_this_period_the_baseline:
            baseline_matrix_for_comparison = current_avg_spread_matrix.copy()
            sns.heatmap(data_to_annotate, annot=data_to_annotate, fmt=".3f",
                        cmap=ListedColormap(['#FAFAFA']), 
                        cbar=False, 
                        linewidths=.5, linecolor='lightgray', 
                        annot_kws={"size": 9, "color": "black", "weight": "bold"}) 
            heatmap_title = f'Avg Spread (Row-Col)\n{period_label}: {region_name}'
            print(f"  Baseline period {period_label} for {region_name}: Displaying absolute spreads.")
        elif baseline_matrix_for_comparison is not None:
            aligned_current, aligned_base = current_avg_spread_matrix.align(baseline_matrix_for_comparison, join='outer', axis=None)
            difference_matrix = aligned_current.astype(float) - aligned_base.astype(float)
            data_for_coloring = difference_matrix
            cmap_to_use = "coolwarm_r" 
            center_val = 0
            all_color_values = data_for_coloring.stack().dropna()
            if not all_color_values.empty:
                abs_max_color_val = all_color_values.abs().max()
                vmin_color, vmax_color = -max(0.01, abs_max_color_val), max(0.01, abs_max_color_val)
            else:
                vmin_color, vmax_color = -0.01, 0.01
            sns.heatmap(data_for_coloring, annot=data_to_annotate, cmap=cmap_to_use, fmt=".3f",
                        center=center_val, vmin=vmin_color, vmax=vmax_color,
                        annot_kws={"size": 9}, linewidths=.5, linecolor='lightgray', cbar=True)
            heatmap_title = f'Avg Spread (Value) / Change from Base (Color)\n{period_label}: {region_name}'
            print(f"  Coloring {period_label} for {region_name} based on change from baseline using '{cmap_to_use}'.")
        else: 
            sns.heatmap(data_to_annotate, annot=data_to_annotate, fmt=".3f",
                        cmap=ListedColormap(['#FAFAFA']), cbar=False,
                        linewidths=.5, linecolor='lightgray',
                        annot_kws={"size": 9, "color":"black", "weight": "bold"})
            heatmap_title = f'Avg Spread (Row-Col)\n{period_label}: {region_name} (Error: Base missing for diff)'

        plt.title(heatmap_title, fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        safe_region_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in str(region_name)).replace(' ', '_')
        safe_period_label = period_label.replace(" ", "_").replace("-", "_to_").replace(":", "").replace(",", "")
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
        month_labels = [py_calendar.month_abbr[m] + f"'{str(target_year)[-2:]}" for m in month_indices]
        print(f"Generating Monthly Avg Basis chart for region: {region_name}, Year: {target_year}, Months: {month_labels}")
        fig, ax = plt.subplots(figsize=(8, 4)) 

        has_data_for_this_year_chart = False
        for comp_detail in components_in_region_details:
            comp_name = comp_detail['prices_csv_name']
            monthly_avg_basis_values = []
            if comp_name not in prices_df_dt_indexed.columns:
                continue

            for month_idx in month_indices:
                month_data = prices_df_dt_indexed[
                    (prices_df_dt_indexed['Date'].apply(lambda x: x.year) == target_year) &
                    (prices_df_dt_indexed['Date'].apply(lambda x: x.month) == month_idx)
                ].copy()

                if month_data.empty:
                    monthly_avg_basis_values.append(np.nan) 
                    continue

                daily_basis = pd.to_numeric(month_data[comp_name], errors='coerce') - pd.to_numeric(month_data[henry_hub_name], errors='coerce')
                avg_for_month = daily_basis.mean()
                monthly_avg_basis_values.append(avg_for_month if pd.notna(avg_for_month) else np.nan)

            if any(pd.notna(val) for val in monthly_avg_basis_values):
                has_data_for_this_year_chart = True
                plot_values = [val if pd.notna(val) else None for val in monthly_avg_basis_values]
                ax.plot(month_labels, plot_values, marker='o', linestyle='-', label=comp_name, linewidth=1.2)

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
        print(f"Market component '{market_component}' not found in prices data for heatmap.")
        return None
    if henry_component not in prices_df_dt_indexed.columns:
        print(f"CRITICAL: Henry component '{henry_component}' not found in prices data. Cannot calculate basis for {market_component}.")
        return None

    component_series = pd.to_numeric(prices_df_dt_indexed[market_component], errors='coerce')
    henry_series = pd.to_numeric(prices_df_dt_indexed[henry_component], errors='coerce')
    daily_basis = component_series - henry_series
    daily_basis = daily_basis.dropna()

    if daily_basis.empty: return None

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
    month_order = ['April', 'May', 'June', 'July', 'August', 'September',
                   'October', 'November', 'December', 'January', 'February', 'March']
    df_heatmap = df_heatmap.reindex(month_order)

    valid_columns = [col for col in df_heatmap.columns if isinstance(col, str) and len(col.split('-')) == 2]
    if not valid_columns and not df_heatmap.empty: return pd.DataFrame(index=month_order) 
    elif not df_heatmap.empty: df_heatmap = df_heatmap[sorted(valid_columns)]

    df_heatmap = df_heatmap.round(3)
    df_heatmap.dropna(axis=1, how='all', inplace=True)
    if df_heatmap.empty: return None
    return df_heatmap


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
    if not fom_component_data.empty:
        fom_data_with_values = fom_component_data.dropna(subset=['settlement_basis'])
        if not fom_data_with_values.empty:
            unique_fom_years_with_data = fom_data_with_values['settlement_year'].unique()
            if len(unique_fom_years_with_data) > 0:
                max_fom_year_with_data = max(unique_fom_years_with_data)

    max_cash_calendar_year = 0
    if not monthly_cash_basis_df.empty:
        cash_gas_years_end = [int(col.split('-')[1]) for col in monthly_cash_basis_df.columns if isinstance(col, str) and '-' in col and len(col.split('-')) == 2]
        if cash_gas_years_end:
            max_cash_calendar_year = max(cash_gas_years_end)

    max_data_driven_year = max(max_fom_year_with_data, max_cash_calendar_year, min_year_to_display)
    latest_year_to_display_columns = min(max_data_driven_year, current_actual_year)
    if latest_year_to_display_columns < min_year_to_display:
        latest_year_to_display_columns = min_year_to_display

    table_data_list = []
    calendar_months_ordered = [calendar_month_names[i] for i in range(1, 13)]

    for year_iter in range(min_year_to_display, latest_year_to_display_columns + 1):
        fom_col_name = f"'{str(year_iter)[-2:]} FoM"
        cash_col_name = f"'{str(year_iter)[-2:]} Cash"
        current_year_fom_data = []
        current_year_cash_data = []

        for month_idx, month_name_str in enumerate(calendar_months_ordered, 1):
            if year_iter == current_actual_year and month_idx > current_actual_month_idx:
                current_year_fom_data.append(np.nan)
                current_year_cash_data.append(np.nan)
                continue

            fom_val_series = fom_component_data[
                (fom_component_data['settlement_year'] == year_iter) &
                (fom_component_data['settlement_month'] == month_name_str)
            ]['settlement_basis']
            current_year_fom_data.append(fom_val_series.iloc[0] if not fom_val_series.empty else np.nan)

            gas_year_col_for_cash = f"{year_iter}-{year_iter+1}" if month_idx >= 4 else f"{year_iter-1}-{year_iter}"
            cash_val = np.nan
            if not monthly_cash_basis_df.empty and \
               gas_year_col_for_cash in monthly_cash_basis_df.columns and \
               month_name_str in monthly_cash_basis_df.index:
                cash_val = monthly_cash_basis_df.loc[month_name_str, gas_year_col_for_cash]
            current_year_cash_data.append(cash_val)

        table_data_list.append(pd.Series(current_year_fom_data, index=calendar_months_ordered, name=fom_col_name))
        table_data_list.append(pd.Series(current_year_cash_data, index=calendar_months_ordered, name=cash_col_name))

    if not table_data_list:
        return pd.DataFrame(index=calendar_months_ordered) 

    fom_cash_df = pd.concat(table_data_list, axis=1).round(3)
    return fom_cash_df


def style_individual_fom_vs_cash_table(df, title):
    """Styles the FoM vs. Cash table and returns HTML."""
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available to display for FoM vs. Cash.</p>"

    def _color_cash_cells_styling(data_series_row): 
        styles = [''] * len(data_series_row) 
        col_names_in_row = data_series_row.index.tolist()

        for i in range(0, len(col_names_in_row) -1 , 2): 
            fom_col_name = col_names_in_row[i]
            cash_col_name = col_names_in_row[i+1]

            if not (fom_col_name.endswith(" FoM") and cash_col_name.endswith(" Cash")):
                continue

            fom_val = data_series_row[fom_col_name]
            cash_val = data_series_row[cash_col_name]

            if pd.isna(fom_val) or pd.isna(cash_val):
                styles[i+1] = '' 
                continue

            diff = cash_val - fom_val
            colors = {'dg': '#548235', 'mg': '#A9D08E', 'lg': '#C6EFCE',
                      'lr': '#FFCDD2', 'mr': '#F44336', 'dr': '#D32F2F'}
            text_color = 'black'
            background_color = ''

            if diff > 0.10: background_color, text_color = colors['dg'], 'white'
            elif diff > 0.05: background_color = colors['mg']
            elif diff > 0.02: background_color = colors['lg']
            elif diff < -0.10: background_color, text_color = colors['dr'], 'white'
            elif diff < -0.05: background_color = colors['mr']
            elif diff < -0.02: background_color = colors['lr']

            if background_color:
                styles[i+1] = f'background-color: {background_color}; color: {text_color};'
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
    """Generates and saves the daily basis vs. history chart (as base64 image string for HTML)."""
    try:
        if component_name not in prices_df_dt_indexed.columns or henry_name not in prices_df_dt_indexed.columns:
            return f"<p>Error: Price data missing for {component_name} or {henry_name} for basis history chart.</p>"

        today_dt = pd.to_datetime(current_processing_date) 
        current_month_start = today_dt.replace(day=1)
        chart_start_date = (current_month_start - relativedelta(months=1))
        chart_end_date = (current_month_start + relativedelta(months=2) - relativedelta(days=1))

        plot_data = {}
        basis_full_history = (pd.to_numeric(prices_df_dt_indexed[component_name], errors='coerce') -
                              pd.to_numeric(prices_df_dt_indexed[henry_name], errors='coerce'))

        current_year_data_in_window = basis_full_history[
            (basis_full_history.index >= chart_start_date) &
            (basis_full_history.index <= min(pd.to_datetime(chart_end_date), today_dt)) 
        ].dropna()
        plot_data['current_year'] = current_year_data_in_window

        for i in range(1, 4): 
            year_offset = i
            historical_basis_this_year_offset = []
            temp_date_index_for_hist_year_window = pd.date_range(start=chart_start_date, end=chart_end_date, freq='D')

            for date_in_current_window_range in temp_date_index_for_hist_year_window:
                try:
                    hist_date_equivalent = date_in_current_window_range.replace(year=date_in_current_window_range.year - year_offset)
                    if hist_date_equivalent in basis_full_history.index:
                        historical_basis_this_year_offset.append(basis_full_history.loc[hist_date_equivalent])
                    else:
                        historical_basis_this_year_offset.append(np.nan)
                except ValueError: 
                    historical_basis_this_year_offset.append(np.nan)

            plot_data[f'Y-{i}'] = pd.Series(historical_basis_this_year_offset, index=temp_date_index_for_hist_year_window).dropna()

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100) 
        if not plot_data['current_year'].empty:
            ax.plot(plot_data['current_year'].index, plot_data['current_year'].values,
                    label=f'{component_name} (Current)', linewidth=2.0, color='black', zorder=5)

        colors = ['#1f77b4', '#2ca02c', '#d62728'] 
        linestyles = [':', '--', '-.']
        linewidths_hist = [1.5, 1.5, 1.5]

        for i in range(1, 4):
            year_label = f'Y-{i}'
            if not plot_data[year_label].empty:
                ax.plot(plot_data[year_label].index, plot_data[year_label].values,
                        label=f'{component_name} ({year_label})', linestyle=linestyles[i-1],
                        color=colors[i-1], alpha=0.8, linewidth=linewidths_hist[i-1])

        if pd.notna(mark0_fom_value):
            ax.axhline(y=mark0_fom_value, color='magenta', linestyle='--', label=f'FoM Mark0: {mark0_fom_value:.3f}', linewidth=1.8)

        ax.set_title(f'{component_name} Daily Basis vs. History & FoM Mark0', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Basis ($/MMBtu)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)) 
        fig.autofmt_xdate(rotation=30, ha='right')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':') 

        plt.tight_layout(pad=1.0)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" alt="{component_name} Basis History Chart" style="width:100%; max-width:900px; display:block; margin:auto;"/>'
    except Exception as e:
        print(f"Error generating basis history chart for {component_name}: {e}")
        return f"<p>Error generating basis history chart for {component_name}: {e}</p>"


def generate_individual_temp_scatter_plot(prices_df_dt_indexed, df_weather, component_name, city_name,
                                          output_charts_dir, current_processing_date,
                                          henry_name=HENRY_HUB_NAME_INDIVIDUAL):
    """Generates and saves the basis vs. temp scatter plot (as base64 image string)."""
    try:
        if component_name not in prices_df_dt_indexed.columns or henry_name not in prices_df_dt_indexed.columns:
            return f"<p>Error: Price data missing for {component_name} or {henry_name} for scatter plot.</p>"
        if df_weather.empty or city_name not in df_weather['City Title'].unique():
            return f"<p>Error: Weather data missing for city '{city_name}' for scatter plot of {component_name}.</p>"

        today_dt = pd.to_datetime(current_processing_date)
        plot_data_all_years = {}
        year_colors = ['black', '#1f77b4', '#2ca02c', '#d62728']
        year_alphas = [0.7, 0.6, 0.5, 0.4]
        year_marker_sizes = [50, 30, 30, 30]
        year_line_styles = ['-', '--', ':', '-.']

        for year_offset in range(4): 
            target_plot_year = today_dt.year - year_offset 
            end_date_of_window = today_dt.replace(year=target_plot_year)
            start_date_of_window = end_date_of_window - timedelta(days=60) 

            basis_data_year_series = (pd.to_numeric(prices_df_dt_indexed[component_name], errors='coerce') -
                                      pd.to_numeric(prices_df_dt_indexed[henry_name], errors='coerce'))
            basis_period = basis_data_year_series[
                (basis_data_year_series.index >= start_date_of_window) &
                (basis_data_year_series.index <= end_date_of_window)
            ].dropna()

            if basis_period.empty: continue

            weather_city_period = df_weather[
                (df_weather['City Title'] == city_name) &
                (df_weather.index >= start_date_of_window) &
                (df_weather.index <= end_date_of_window)
            ]['Avg Temp'].dropna()

            if weather_city_period.empty: continue

            merged_data = pd.merge(basis_period.rename('basis'), weather_city_period.rename('avg_temp'),
                                   left_index=True, right_index=True, how='inner')
            if not merged_data.empty:
                plot_data_all_years[year_offset] = merged_data.copy()

        if not plot_data_all_years:
            return f"<p>No combined basis & temp data for {component_name}/{city_name} for required periods to generate scatter.</p>"

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100) 
        polynomial_degree = 2 

        for year_offset, data_for_year in plot_data_all_years.items():
            label_suffix = "(Current)" if year_offset == 0 else f"(Y-{year_offset})"
            color = year_colors[year_offset]
            marker_size = year_marker_sizes[year_offset]
            alpha = year_alphas[year_offset]
            line_style = year_line_styles[year_offset]
            line_width = 2 if year_offset == 0 else 1.5

            ax.scatter(data_for_year['avg_temp'], data_for_year['basis'], label=f'{component_name} {label_suffix}',
                       color=color, s=marker_size, alpha=alpha, edgecolors='w', linewidth=0.5, zorder=year_offset+2)

            if len(data_for_year) >= polynomial_degree + 1: 
                try:
                    coeffs = np.polyfit(data_for_year['avg_temp'], data_for_year['basis'], polynomial_degree)
                    poly_func = np.poly1d(coeffs)
                    temp_range_for_line = np.linspace(data_for_year['avg_temp'].min(), data_for_year['avg_temp'].max(), 100)
                    ax.plot(temp_range_for_line, poly_func(temp_range_for_line), color=color, linestyle=line_style,
                            linewidth=line_width, alpha=max(0.3, alpha-0.1), zorder=year_offset+1) 
                except Exception as e_fit:
                    print(f"Could not fit curve for {component_name} {label_suffix}: {e_fit}")

        ax.set_title(f'{component_name} Basis vs. {city_name} Avg Temp (Last 61 Days & Prior Years)', fontsize=13)
        ax.set_xlabel('Average Temperature (°F)', fontsize=11)
        ax.set_ylabel('Daily Basis ($/MMBtu)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':') 

        plt.tight_layout(pad=1.0)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" alt="{component_name} Basis vs. Temp Chart" style="width:100%; max-width:800px; display:block; margin:auto;"/>'

    except Exception as e:
        print(f"Error in generate_individual_temp_scatter_plot for {component_name}: {e}")
        return f"<p>Error generating basis vs. temp scatter for {component_name}: {e}</p>"

print("market_analyzer.py parsed and loaded.") # Diagnostic print at the end of the 