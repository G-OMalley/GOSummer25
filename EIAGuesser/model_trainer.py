import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import re

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'output'
INFO_DIR = SCRIPT_DIR.parent / 'INFO'
MODEL_OUTPUT_DIR = OUTPUT_DIR / 'models'

# Create directory for saving models and plots
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
FEATURE_SET_FILE = OUTPUT_DIR / 'final_feature_set.csv'
EIA_CHANGES_FILE = INFO_DIR / 'EIAchanges.csv'
ALL_REGIONS = ['East', 'Midwest', 'Mountain', 'Pacific', 'South Central']


# --- Main Functions ---

def load_data():
    """Loads the pre-processed feature set and the EIA target data."""
    print("Loading data...")
    if not FEATURE_SET_FILE.exists() or not EIA_CHANGES_FILE.exists():
        raise FileNotFoundError("Required data files not found. Ensure 'feature_engineering.py' has been run.")
    
    df_features = pd.read_csv(FEATURE_SET_FILE, index_col='Date', parse_dates=True)
    df_eia = pd.read_csv(EIA_CHANGES_FILE, index_col='Period', parse_dates=True)
    
    print(f"Loaded feature set with shape: {df_features.shape}")
    print(f"Loaded EIA changes with shape: {df_eia.shape}")
    return df_features, df_eia

def select_features_for_region(df_full, region_name):
    """
    Selects the relevant columns for a specific regional model.
    A regional model uses CONUS (non-regional) data + data specific to that region.
    """
    # The CONUS model uses all available features.
    if region_name == 'Lower 48 States':
        print("Selecting all features for Lower 48 (CONUS) model.")
        return df_full

    # For regional models, we select only CONUS features and features for that specific region.
    region_prefix = region_name.split(' ')[0] # 'East', 'Midwest', etc.
    
    # Identify columns that are NOT for other regions.
    # A column is kept if it's general (contains no region names) OR it contains the current region's name.
    cols_to_keep = []
    for col in df_full.columns:
        # Check if the column name contains any of the other region names.
        is_other_region = any(other_region in col for other_region in ALL_REGIONS if other_region != region_prefix)
        
        # If it's not another region's column, we keep it.
        if not is_other_region:
            cols_to_keep.append(col)
            
    print(f"Selected {len(cols_to_keep)} features for {region_name} model.")
    return df_full[cols_to_keep]


def train_and_evaluate(df_features, df_target_weekly, target_column_name):
    """
    Trains a model to predict daily changes and evaluates by summing to weekly changes.
    """
    print("\n" + "="*50)
    print(f"STARTING MODEL TRAINING FOR: {target_column_name}")
    print("="*50)

    # --- 1. Prepare Data ---
    # The model learns to predict the daily implied storage change.
    y = df_features['Daily_Storage_Change']
    X = df_features.drop(columns=['Daily_Storage_Change'])
    
    # FIX: Aggressively sanitize column names for XGBoost compatibility.
    # This replaces any character that is not a letter, number, or underscore with '_'.
    X.columns = X.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
    
    # Time-based split: Train on data up to the start of the last 6 months.
    split_date = X.index.max() - pd.DateOffset(months=6)
    X_train = X[X.index <= split_date]
    y_train = y[y.index <= split_date]
    X_test = X[X.index > split_date]
    y_test = y[y.index > split_date]

    print(f"Training data from {X_train.index.min().date()} to {X_train.index.max().date()} ({len(X_train)} days)")
    print(f"Testing data from {X_test.index.min().date()} to {X_test.index.max().date()} ({len(X_test)} days)")

    # --- 2. Train XGBoost Model ---
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 # Add early stopping
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    print("Model training complete.")

    # --- 3. Make Daily Predictions and Aggregate to Weekly ---
    print("Making daily predictions and aggregating to weekly totals...")
    daily_predictions = model.predict(X_test)
    
    df_results = pd.DataFrame({'daily_pred': daily_predictions}, index=X_test.index)
    
    # Resample daily predictions to weekly, summing them up. 'W-FRI' aligns with EIA's weekly reporting.
    weekly_preds = df_results['daily_pred'].resample('W-FRI').sum()

    # --- 4. Align and Evaluate ---
    df_eval = pd.DataFrame(weekly_preds).join(df_target_weekly[target_column_name])
    df_eval.dropna(inplace=True) 
    
    if df_eval.empty:
        print("ERROR: No matching weekly actuals found for the test period. Cannot evaluate.")
        return

    mae = mean_absolute_error(df_eval[target_column_name], df_eval['daily_pred'])
    r2 = r2_score(df_eval[target_column_name], df_eval['daily_pred'])

    print("\n--- Model Evaluation (Weekly) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} Bcf")
    print(f"R-squared (R2): {r2:.2f}")

    # --- 5. Save Model and Feature Importance ---
    region_name_safe = target_column_name.split(' ')[0]
    model_path = MODEL_OUTPUT_DIR / f'model_{region_name_safe}.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax, title=f'Feature Importance ({region_name_safe})')
    plt.tight_layout()
    plot_path = MODEL_OUTPUT_DIR / f'feature_importance_{region_name_safe}.png'
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()


# --- Main execution block ---
if __name__ == '__main__':
    df_features_full, df_eia_weekly = load_data()
    
    # Clean up column names in EIA data for easier matching.
    df_eia_weekly.columns = [col.strip() for col in df_eia_weekly.columns]
    
    # Loop through each target region defined in the EIA changes file
    for target_col in df_eia_weekly.columns:
        region_name = target_col.replace('Storage Change (Bcf)', '').strip()
        
        df_regional_features = select_features_for_region(df_features_full, region_name)
        
        train_and_evaluate(df_regional_features, df_eia_weekly, target_col)

    print("\n--- All models trained successfully! ---")