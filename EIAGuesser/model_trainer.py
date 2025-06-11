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
    
    # Sanitize feature columns right after loading
    df_features.columns = df_features.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
    
    print(f"Loaded feature set with shape: {df_features.shape}")
    print(f"Loaded EIA changes with shape: {df_eia.shape}")
    return df_features, df_eia

def select_features_and_target(df_full, region_name):
    """
    Selects the relevant features AND the correct daily target column for a model.
    """
    # The CONUS model uses all available features.
    if region_name == 'Lower 48 States':
        print("Selecting all features for Lower 48 (CONUS) model.")
        y = df_full['Daily_Storage_Change']
        X = df_full.drop(columns=['Daily_Storage_Change'])
        return X, y

    region_prefix = region_name.split(' ')[0]
    
    # Define the specific daily target column for this region.
    # This column was created in feature_engineering.py
    regional_target_col = f'{region_prefix}_Criterion_Storage_Change'
    
    if regional_target_col not in df_full.columns:
        raise KeyError(f"The regional daily target '{regional_target_col}' was not found in the feature set.")
        
    y = df_full[regional_target_col]

    # For features, we select non-regional (CONUS) features and features for the current region.
    cols_to_keep = []
    for col in df_full.columns:
        is_conus = not any(col.startswith(r) for r in ALL_REGIONS)
        is_current_region = col.startswith(region_prefix)
        
        if is_conus or is_current_region:
            cols_to_keep.append(col)
    
    X = df_full[cols_to_keep]
    # Remove all potential target columns from the features to prevent data leakage
    y_cols_to_drop = [col for col in X.columns if 'Storage_Change' in col]
    X = X.drop(columns=y_cols_to_drop, errors='ignore')
    
    print(f"Selected {len(X.columns)} features and target '{regional_target_col}' for {region_name} model.")
    return X, y


def train_and_evaluate(X, y, df_target_weekly, target_column_name):
    """
    Trains a model to predict daily changes and evaluates by summing to weekly changes.
    """
    print("\n" + "="*50)
    print(f"STARTING MODEL TRAINING FOR: {target_column_name}")
    print("="*50)

    split_date = X.index.max() - pd.DateOffset(months=6)
    X_train = X[X.index <= split_date]
    y_train = y[y.index <= split_date]
    X_test = X[X.index > split_date]
    y_test = y[y.index > split_date]

    print(f"Training data from {X_train.index.min().date()} to {X_train.index.max().date()} ({len(X_train)} days)")
    print(f"Testing data from {X_test.index.min().date()} to {X_test.index.max().date()} ({len(X_test)} days)")

    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    print("Model training complete.")

    print("Making daily predictions and aggregating to weekly totals...")
    daily_predictions = model.predict(X_test)
    
    df_results = pd.DataFrame({'daily_pred': daily_predictions}, index=X_test.index)
    
    weekly_preds = df_results['daily_pred'].resample('W-FRI').sum()

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

    region_name_safe = target_column_name.replace(' ', '_').replace('(Bcf)', '').strip('_')
    model_path = MODEL_OUTPUT_DIR / f'model_{region_name_safe}.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax, title=f'Feature Importance ({target_column_name})')
    plt.tight_layout()
    plot_path = MODEL_OUTPUT_DIR / f'feature_importance_{region_name_safe}.png'
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()


# --- Main execution block ---
if __name__ == '__main__':
    df_features_full, df_eia_weekly = load_data()
    
    df_eia_weekly.columns = [col.strip() for col in df_eia_weekly.columns]
    
    for target_col in df_eia_weekly.columns:
        region_name = target_col.replace('Storage Change (Bcf)', '').strip()
        
        # This now returns BOTH the features and the correct target series
        X_regional, y_regional = select_features_and_target(df_features_full, region_name)
        
        train_and_evaluate(X_regional, y_regional, df_eia_weekly, target_col)

    print("\n--- All models trained successfully! ---")
