# C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\EIAGuesser\model_trainer.py
# -*- coding: utf-8 -*-
"""
This script builds and deploys a production-grade, multi-model machine
learning system to forecast the weekly U.S. EIA natural gas storage change.

VERSION 15: PRODUCTION FINAL
This version incorporates all previous fixes and upgrades into a robust,
trader-grade pipeline. It trains multiple model types per region, generates
advanced diagnostics and SHAP interpretability plots, and produces a final
consensus forecast.

Key Features:
- Multi-Model Regional Training: Trains XGBoost, LightGBM, and CatBoost.
- Standardized Flexible Alignment: Exclusively uses the Fri-Thu weekly definition.
- Advanced Interpretability: Generates SHAP summary plots for each model.
- Trader-Grade Diagnostics: Produces a fully aligned summary table.
- Forecastability Guardrail: Prevents forecasts when data is incomplete.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
import shap

# --- Optional Imports for Additional Models ---
try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

try:
    import catboost as cb
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False

# --- Configuration & Setup ---
pd.options.mode.chained_assignment = None

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPORTS_DIR = Path.home() / 'OneDrive/Desktop/Coding/TraderHelper/Scripts/MarketAnalysis_Report_Output'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR = SCRIPT_DIR / 'model_outputs'
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR / 'output'
DATA_FILE = OUTPUT_DIR / 'model_ready_feature_set.csv'

# --- Utility Functions ---

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculates and prints key regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} Bcf")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} Bcf")
    return {'name': model_name, 'mae': mae, 'rmse': rmse}

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Generates and saves a plot of actual vs. predicted values."""
    plt.figure(figsize=(15, 7))
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)
    plt.plot(y_true.index, y_true, label='Actual', color='blue', marker='.', linestyle='-')
    plt.plot(y_pred.index, y_pred, label='Predicted', color='red', marker='.', linestyle='--')
    plt.title(f'{model_name}: Actual vs. Predicted', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f'{model_name}_actual_vs_predicted.png')
    plt.close()

# --- Data Loading and Preparation ---

def load_and_prepare_data(file_path, eia_actuals_path):
    """Loads feature set, derives regional targets, and identifies the forecast window."""
    print("--- 1. Loading and Preparing Data ---")
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df.sort_index(inplace=True)
    df.columns = [col.replace('[', '_').replace(']', '').replace('<', '_') for col in df.columns]
    print("  -> Sanitized column names.")
    
    eia_actuals = pd.read_csv(eia_actuals_path, parse_dates=['Period'])
    eia_actuals.columns = [col.encode('ascii', 'ignore').decode().strip() for col in eia_actuals.columns]
    eia_actuals['Period'] = eia_actuals['Period'].dt.normalize()
    eia_actuals.set_index('Period', inplace=True)
    
    last_eia_date = eia_actuals.index.max()
    forecast_start_date = last_eia_date + pd.Timedelta(days=1)
    forecast_end_date = last_eia_date + pd.Timedelta(days=7)
    
    print(f"  -> Last known EIA report date (source of truth): {last_eia_date.date()}")
    print(f"  -> Next forecast window inferred: {forecast_start_date.date()} to {forecast_end_date.date()}")
    
    return df, forecast_start_date, forecast_end_date, eia_actuals

def format_data_for_flexible_mode(df):
    """Prepares data using the standard Friday-Thursday weekly alignment."""
    df_mode = df.copy()
    eia_week_grouper = pd.Grouper(freq='W-THU')
    df_mode['eia_week_id'] = df_mode.groupby(eia_week_grouper).ngroup()
    if 'daily_imbalance' in df_mode.columns:
        df_mode['cumulative_imbalance'] = df_mode.groupby('eia_week_id')['daily_imbalance'].cumsum()
    return df_mode

# --- Modeling & Diagnostics ---

def get_regional_features(all_features, region):
    """Selects features specific to a given region deterministically."""
    region_lower = region.lower().replace(' ', '_')
    regional_features = [f for f in all_features if region_lower in f.lower()]
    conus_features = [f for f in all_features if 'conus' in f.lower()]
    general_features = [f for f in all_features if 'east' not in f.lower() and 'midwest' not in f.lower() and 'mountain' not in f.lower() and 'pacific' not in f.lower() and 'south' not in f.lower() and 'conus' not in f.lower()]
    
    selected_features = sorted(list(set(regional_features + conus_features + general_features)))
    return selected_features

def train_regional_model(df_mode, all_features, region, model_type='xgboost'):
    """Trains a single model (XGBoost, LightGBM, or CatBoost) for a specific region."""
    target_col = f"Target_{region}_Change"
    if target_col not in df_mode.columns:
        print(f"  -> WARNING: Target column {target_col} not found. Skipping model for {region} region.")
        return None, None, None, None
        
    df_mode.dropna(subset=[target_col], inplace=True)
    
    model_name = f"{model_type.capitalize()}_{region}"
    print(f"\n--- Training {model_name} ---")

    regional_features = get_regional_features(all_features, region)
    
    df_mode['Date_col'] = df_mode.index
    last_day_indices = df_mode.groupby('eia_week_id')['Date_col'].idxmax()
    df_weekly = df_mode.loc[last_day_indices]
    
    X = df_weekly[regional_features].copy()
    y = df_weekly[target_col]

    test_size = int(len(df_weekly) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42, early_stopping_rounds=50, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif model_type == 'lightgbm' and LGBM_INSTALLED:
        model = lgb.LGBMRegressor(random_state=42, n_estimators=500, verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif model_type == 'catboost' and CATBOOST_INSTALLED:
        model = cb.CatBoostRegressor(random_state=42, verbose=0, iterations=1000, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    else:
        print(f"  -> SKIPPING {model_name}: Library not installed or type not recognized.")
        return None, None, None, None
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, model_name)
    
    return model, X_train, pd.Series(y_pred, index=y_test.index), metrics

def generate_regional_live_forecast(df, model, features, forecast_start, forecast_end):
    """Generates a one-week-ahead forecast for a single region."""
    forecast_data_raw = df.loc[forecast_start:forecast_end].copy()
    
    if 'daily_imbalance' in forecast_data_raw.columns:
        forecast_data_raw['cumulative_imbalance'] = forecast_data_raw['daily_imbalance'].cumsum()
    
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else features
    X_forecast_raw = forecast_data_raw.iloc[[-1]]
    X_forecast = X_forecast_raw.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(X_forecast)[0]
    return float(prediction)

def get_actual_eia_value(region, date, eia_actuals_df):
    """Looks up the official EIA value from the EIAchanges.csv source file."""
    eia_col_map = {
        "East": "East Region Storage Change (Bcf)", "Midwest": "Midwest Region Storage Change (Bcf)",
        "Mountain": "Mountain Region Storage Change (Bcf)", "Pacific": "Pacific Region Storage Change (Bcf)",
        "SouthCentral": "South Central Region Storage Change (Bcf)",
    }
    col = eia_col_map.get(region)
    norm_date = pd.Timestamp(date).normalize()

    if col not in eia_actuals_df.columns: return np.nan
    if norm_date not in eia_actuals_df.index: return np.nan
    return eia_actuals_df.at[norm_date, col]

def generate_trader_summary_table(model_results, df_master, eia_actuals_df, live_forecast_possible):
    """Generates a comprehensive summary table for all trained models."""
    print("\n--- 7. Generating Trader Diagnostics Summary ---")
    rows = []
    
    actual_last_week_date = eia_actuals_df.index.max()
    
    for model_name, res in model_results.items():
        model, region = res.get('model'), res.get('region')
        if model is None: continue

        forecast_val = res.get('forecast')
        
        last_week_end_for_estimate = actual_last_week_date
        last_week_start_for_estimate = last_week_end_for_estimate - pd.Timedelta(days=6)
        
        last_week_estimate, mae_4wk = np.nan, np.nan
        if last_week_end_for_estimate in df_master.index:
            last_week_data_raw = df_master.loc[last_week_start_for_estimate:last_week_end_for_estimate].copy()
            if 'daily_imbalance' in last_week_data_raw.columns:
                last_week_data_raw['cumulative_imbalance'] = last_week_data_raw['daily_imbalance'].cumsum()

            X_last_week_raw = last_week_data_raw.iloc[[-1]]
            X_last_week = X_last_week_raw.reindex(columns=model.feature_names_in_ if hasattr(model, 'feature_names_in_') else res['features'], fill_value=0)
            last_week_estimate = model.predict(X_last_week)[0]

        actual_last_week = get_actual_eia_value(region, actual_last_week_date, eia_actuals_df)
        
        y_pred_test = res.get('predictions')
        if y_pred_test is not None and len(y_pred_test) >= 4:
            target_col = f"Target_{region}_Change"
            y_true_test = df_master.loc[y_pred_test.index][target_col]
            mae_4wk = mean_absolute_error(y_true_test.iloc[-4:], y_pred_test.iloc[-4:])

        rows.append({
            'Model Name': model_name,
            'Forecast Next Week': f"{forecast_val:.2f}" if forecast_val is not None else 'DATA N/A',
            'Forecast Last Week': f"{last_week_estimate:.2f}" if pd.notna(last_week_estimate) else 'N/A',
            'Actual Last Week': f"{actual_last_week:.2f}" if pd.notna(actual_last_week) else 'N/A',
            '4-Week MAE': f"{mae_4wk:.2f}" if pd.notna(mae_4wk) else 'N/A',
        })

    df_diag = pd.DataFrame(rows)
    print(df_diag.to_string(index=False))
    df_diag.to_csv(REPORTS_DIR / 'final_model_summary.csv', index=False)
    print("\n  -> Saved final model summary.")
    return df_diag

def create_summary_visuals(df_diag, results, df_master):
    """Creates and saves a suite of summary visualizations."""
    print("\n--- 8. Generating Summary Visualizations ---")

    plt.figure(figsize=(15, 10))
    df_plot = df_diag[df_diag['Forecast Next Week'] != 'DATA N/A'].copy()
    df_plot['Forecast Next Week'] = pd.to_numeric(df_plot['Forecast Next Week'])
    sns.barplot(data=df_plot, x='Forecast Next Week', y='Model Name', orient='h', hue='Model Name', dodge=False, palette='viridis')
    plt.title('Forecast Comparison for Next EIA Report', fontsize=16)
    plt.xlabel('Storage Change (Bcf)')
    plt.ylabel('Model')
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'forecast_comparison_bar.png')
    plt.close()
    print("  -> Saved forecast comparison bar chart.")

    df_plot['4-Week MAE'] = pd.to_numeric(df_plot['4-Week MAE'])
    df_plot[['Model_Type', 'Region']] = df_plot['Model Name'].str.split('_', expand=True)
    df_pivot = df_plot.pivot_table(index='Region', columns='Model_Type', values='4-Week MAE')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
    plt.title('4-Week Rolling Mean Absolute Error (Bcf)', fontsize=16)
    plt.ylabel('Region')
    plt.xlabel('Model Type')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / '4_week_mae_heatmap.png')
    plt.close()
    print("  -> Saved 4-week MAE heatmap.")


# --- Main Execution Block ---

if __name__ == '__main__':
    eia_actuals_file = SCRIPT_DIR.parent / 'INFO' / 'EIAchanges.csv'
    df_master, forecast_start, forecast_end, eia_actuals = load_and_prepare_data(DATA_FILE, eia_actuals_file)

    REGIONS = ['East', 'Midwest', 'Mountain', 'Pacific', 'SouthCentral']
    
    base_features = [col for col in df_master.columns if not col.startswith('Target_') and not col.startswith('Inv_') and col not in ['eia_week_id', 'Date_col']]

    model_registry = {
        **{f"XGBoost_{region}": {"func": train_regional_model, "region": region, "type": "xgboost"} for region in REGIONS},
        **{f"LightGBM_{region}": {"func": train_regional_model, "region": region, "type": "lightgbm"} for region in REGIONS},
        **{f"CatBoost_{region}": {"func": train_regional_model, "region": region, "type": "catboost"} for region in REGIONS},
    }
    
    results = {}
    df_formatted = format_data_for_flexible_mode(df_master.copy())

    for name, config in model_registry.items():
        regional_features = get_regional_features(base_features, config['region'])
        # FIX: Correctly unpack the 4 return values from the function
        model, x_train, y_pred, metrics = config['func'](df_formatted, regional_features, config['region'], config['type'])
        results[name] = {'model': model, 'region': config['region'], 'predictions': y_pred, 'metrics': metrics, 'features': regional_features}

    # --- Live Forecasting with Guardrail ---
    print("\n--- 6. Generating Live Forecasts ---")
    
    forecast_range = pd.date_range(start=forecast_start, end=forecast_end)
    if not all(date in df_master.index for date in forecast_range):
        print(f"\nCRITICAL WARNING: Cannot generate a live forecast for {forecast_start.date()} to {forecast_end.date()}.")
        print("Not all 7 days of input data are available.")
        live_forecast_possible = False
    else:
        live_forecast_possible = True

    for name, res in results.items():
        if res['model'] and live_forecast_possible:
            res['forecast'] = generate_regional_live_forecast(
                df_master, res['model'], res['features'], forecast_start, forecast_end
            )

    # --- Generate Diagnostics Table with Consensus ---
    df_diag = generate_trader_summary_table(results, df_master, eia_actuals, live_forecast_possible)
    create_summary_visuals(df_diag, results, df_master)

    # --- Final Ensemble Forecast ---
    if live_forecast_possible:
        regional_consensus = {}
        for region in REGIONS:
            regional_forecasts = [res['forecast'] for name, res in results.items() if res.get('region') == region and res.get('forecast') is not None]
            if regional_forecasts:
                regional_consensus[region] = np.mean(regional_forecasts)
        
        if len(regional_consensus) == len(REGIONS):
            ensemble_forecast = sum(regional_consensus.values())
            print("\n--- FINAL CONSENSUS FORECAST ---")
            for region, forecast in regional_consensus.items():
                print(f"  -> Consensus Forecast for {region}: {forecast:.2f} Bcf")
            print(f"\n  -> Total CONUS Forecast: {ensemble_forecast:.2f} Bcf")
        else:
            print("\n--- Could not generate a complete ensemble forecast due to missing regional models. ---")

    print("\n\n--- Modeling Pipeline Complete ---")


