import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import warnings
import re
import json

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", message=".*XGBoost.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Path Configuration ---
# Hardcoded paths as requested by the user
BASE_PROJECT_DIR = Path(r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper')
OUTPUT_DIR = BASE_PROJECT_DIR / 'EIAGuesser' / 'output'
INFO_DIR = BASE_PROJECT_DIR / 'INFO'
MODEL_OUTPUT_DIR = OUTPUT_DIR / 'models'
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET_FILE = OUTPUT_DIR / 'final_feature_set.csv'
EIA_CHANGES_FILE = INFO_DIR / 'EIAchanges.csv'
PREDICTIONS_FILE = MODEL_OUTPUT_DIR / 'predictions.txt'

# --- Constants & Model Definitions ---
REGIONS_TO_PREDICT = ['East', 'Midwest', 'Mountain', 'Pacific', 'South Central']
EIA_COLUMN_MAP = {
    'East': 'East Region Storage Change (Bcf)', 'Midwest': 'Midwest Region Storage Change (Bcf)',
    'Mountain': 'Mountain Region Storage Change (Bcf)', 'Pacific': 'Pacific Region Storage Change (Bcf)',
    'South Central': 'South Central Region Storage Change (Bcf)', 'Lower 48': 'Lower 48 States Storage Change (Bcf)'
}
DAILY_TARGET_MAP = {
    'East': 'East_Criterion_Storage_Change', 'Midwest': 'Midwest_Criterion_Storage_Change',
    'Mountain': 'Mountain_Criterion_Storage_Change', 'Pacific': 'Pacific_Criterion_Storage_Change',
    'South Central': 'South_Central_Criterion_Storage_Change', 'Lower 48': 'Daily_Storage_Change'
}
MODEL_CONFIG = {
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, early_stopping_rounds=50),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10, n_jobs=-1),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

# --- Core Functions ---

def sanitize_feature_names(df):
    """Sanitizes column names to be compatible with all models."""
    cols = df.columns
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in cols]
    df.columns = new_cols
    return df

def load_data():
    """Loads, sanitizes, and aligns daily features and weekly EIA actuals."""
    print("Loading and aligning data...")
    df_features = pd.read_csv(FEATURE_SET_FILE, index_col='Date', parse_dates=True)
    df_features = sanitize_feature_names(df_features)
    df_eia = pd.read_csv(EIA_CHANGES_FILE, index_col='Period', parse_dates=True)
    df_eia.columns = df_eia.columns.str.strip()
    print(f"Loaded daily feature set with shape: {df_features.shape}")
    print(f"Loaded weekly EIA changes with shape: {df_eia.shape}")
    return df_features, df_eia

def find_optimal_features(X, y, region_name):
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to find the
    most predictive subset of features for a given region. Caches results for speed.
    """
    region_safe_name = region_name.replace(' ', '_')
    cache_file = MODEL_OUTPUT_DIR / f'optimal_features_{region_safe_name}.json'

    if cache_file.exists():
        print(f"Loading cached optimal features for {region_name}...")
        with open(cache_file, 'r') as f:
            feature_list = json.load(f)
        return feature_list

    print(f"\n--- Finding Optimal Features for {region_name} using RFECV ---")
    print("This may take a few minutes...")

    estimator = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
        selector = RFECV(estimator, step=1, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        selector = selector.fit(X.values, y.values)

    optimal_features = X.columns[selector.support_].tolist()
    print(f"RFECV complete. Found {len(optimal_features)} optimal features for {region_name}.")

    with open(cache_file, 'w') as f:
        json.dump(optimal_features, f)
    print(f"Saved optimal features to {cache_file}")
    
    return optimal_features

def train_and_evaluate_models(X, y, df_eia_weekly, region_name):
    """Trains, evaluates, and returns a dictionary of trained models and their performance."""
    print("\n" + "="*60)
    print(f"STARTING MODEL TRAINING PIPELINE FOR: {region_name}")
    print(f"(Using {X.shape[1]} optimized features)")
    print("="*60)

    last_eia_date = df_eia_weekly.index.max()
    split_date = last_eia_date - pd.DateOffset(months=6)
    
    X_train, y_train = X.loc[:split_date], y.loc[:split_date]
    X_test, y_test = X.loc[split_date:last_eia_date], y.loc[split_date:last_eia_date]

    trained_models = {}
    for model_name, model in MODEL_CONFIG.items():
        print(f"\n--- Training {model_name} ---")
        if model_name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        elif model_name == 'LightGBM':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model.fit(X_train, y_train)

        daily_preds = model.predict(X_test)
        weekly_preds = pd.Series(daily_preds, index=X_test.index).resample('W-FRI').sum()
        
        eia_target_col = EIA_COLUMN_MAP[region_name]
        df_eval = pd.DataFrame({'weekly_pred': weekly_preds}).join(df_eia_weekly[eia_target_col]).dropna()

        if df_eval.empty:
            print(f"ERROR: Could not evaluate {model_name}.")
            continue

        mae_full = mean_absolute_error(df_eval[eia_target_col], df_eval['weekly_pred'])
        r2_full = r2_score(df_eval[eia_target_col], df_eval['weekly_pred'])
        
        recent_start_date = df_eval.index.max() - pd.DateOffset(weeks=4)
        df_recent_eval = df_eval.loc[df_eval.index > recent_start_date]
        mae_recent = mean_absolute_error(df_recent_eval[eia_target_col], df_recent_eval['weekly_pred'])
        
        print(f"Evaluation (Full 6-mo): MAE={mae_full:.2f}, R2={r2_full:.3f}")
        print(f"Evaluation (Last 4-wk): MAE={mae_recent:.2f} <-- Used for ranking")

        region_safe_name = region_name.replace(' ', '_')
        model_path = MODEL_OUTPUT_DIR / f'model_{region_safe_name}_{model_name}.joblib'
        joblib.dump(model, model_path)

        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            imp_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(20)
            plt.barh(imp_df['feature'], imp_df['importance'])
            plt.title(f'Top 20 Feature Importance: {region_name} - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plot_path = MODEL_OUTPUT_DIR / f'feature_importance_{region_safe_name}_{model_name}.png'
            plt.savefig(plot_path)
            plt.close()

        trained_models[model_name] = {'model': model, 'mae_recent': mae_recent, 'path': model_path}
        
    return trained_models

def generate_ensemble_forecast(X, trained_models, df_eia_weekly):
    """
    Selects top models and generates an ensemble forecast for the next week
    using pre-existing feature data.
    """
    print("\n--- Generating Ensemble Forecast for Next Week ---")
    if not trained_models: return None, []

    sorted_models = sorted(trained_models.items(), key=lambda item: item[1]['mae_recent'])
    top_models = sorted_models[:3]
    
    if not top_models: return None, []

    print("Top 3 models selected for ensemble (based on recent MAE):")
    for name, data in top_models: print(f"  - {name} (MAE: {data['mae_recent']:.2f})")

    last_eia_date = df_eia_weekly.index.max()
    prediction_start = last_eia_date + pd.Timedelta(days=1)
    prediction_end = last_eia_date + pd.Timedelta(days=7)
    
    X_future = X.loc[prediction_start:prediction_end]

    if X_future.shape[0] != 7:
        print(f"ERROR: Expected 7 days of feature data for forecast but found {X_future.shape[0]}.")
        print("Please ensure 'feature_engineering.py' has run correctly.")
        return None, []

    if X_future.isnull().values.any():
        print("WARNING: NaN values detected in forecast features. Applying forward/backward fill.")
        X_future = X_future.ffill().bfill()
    
    if X_future.isnull().values.any():
        print("FATAL ERROR: Could not impute all NaN values in the forecast data. Aborting forecast.")
        return None, []

    all_preds = [data['model'].predict(X_future) for _, data in top_models]
    ensemble_weekly_total = np.mean(all_preds, axis=0).sum()
    
    print(f"\nEnsemble Forecast for week ending ~{prediction_end.date()}: {ensemble_weekly_total:.2f} Bcf")
    return ensemble_weekly_total, [name for name, _ in top_models]

# --- Main Execution Block ---

if __name__ == '__main__':
    df_features, df_eia = load_data()
    
    if PREDICTIONS_FILE.exists(): PREDICTIONS_FILE.unlink()

    all_regions = REGIONS_TO_PREDICT + ['Lower 48']
    for region in all_regions:
        daily_target_col = DAILY_TARGET_MAP.get(region)
        y_full = df_features[daily_target_col]
        cols_to_drop = [col for col in df_features.columns if 'Storage_Change' in col or 'Inv_' in col]
        X_full = df_features.drop(columns=cols_to_drop)
        
        optimal_feature_names = find_optimal_features(X_full, y_full, region)
        X_optimal = X_full[optimal_feature_names]

        trained_models = train_and_evaluate_models(X_optimal, y_full, df_eia, region)
        
        forecast, top_model_names = generate_ensemble_forecast(X_optimal, trained_models, df_eia)
        
        with open(PREDICTIONS_FILE, 'a') as f:
            f.write("="*60 + "\n")
            f.write(f"FORECAST SUMMARY FOR: {region}\n")
            f.write("="*60 + "\n")
            if forecast is not None:
                f.write(f"Optimal features found: {len(optimal_feature_names)}\n")
                f.write(f"Top Models Used for Ensemble: {', '.join(top_model_names)}\n")
                f.write(f"Ensemble Forecast for next week: {forecast:.2f} Bcf\n\n")
            else:
                f.write("Forecast could not be generated for this region.\n\n")

    print("\n--- Forecasting Engine Run Complete ---")
    print(f"Check '{MODEL_OUTPUT_DIR}' for models, plots, feature lists, and the 'predictions.txt' summary.")
