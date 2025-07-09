import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
print("--- Initializing Iterative Modeling Pipeline ---")

# --- Paths ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    # The feature engineering script saves its output to 'output_features', so we read from there.
    INPUT_DIR = SCRIPT_DIR / 'output_features' 
    # UPDATED: Pointing final model results to the requested 'output' directory.
    OUTPUT_DIR = SCRIPT_DIR / 'output'
except NameError:
    BASE_DIR = Path('C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/EIAGuesser')
    INPUT_DIR = BASE_DIR / 'output_features'
    OUTPUT_DIR = BASE_DIR / 'output' 
    print(f"Warning: Using fallback paths. Output will be saved to {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(exist_ok=True)

# --- Modeling Parameters ---
INPUT_FILE = INPUT_DIR / 'master_weekly_features.csv'
REGION_TO_MODEL = 'East'
TARGET_COLUMN = 'Target_Storage_Change'
DATE_COLUMN = 'Week_Ending_Friday'
N_SPLITS_BACKTEST = 8
RECENT_WEEKS_MAE = 4

# --- Feature Group Definitions for Toggling ---
# This part reads the file multiple times, which is inefficient but simple for configuration.
# In a production system, this could be optimized.
try:
    FEATURE_GROUPS = {
        "weather": [col for col in pd.read_csv(INPUT_FILE).columns if 'Weather_' in col or 'HDD' in col or 'CDD' in col],
        "power": [col for col in pd.read_csv(INPUT_FILE).columns if 'Power_' in col],
        "fundy": [col for col in pd.read_csv(INPUT_FILE).columns if 'Fundy_' in col],
        "storage": [col for col in pd.read_csv(INPUT_FILE).columns if 'Storage_' in col],
        "inventory": [col for col in pd.read_csv(INPUT_FILE).columns if 'Inventory_' in col],
        "intra_week": [col for col in pd.read_csv(INPUT_FILE).columns if '_7d' in col or '_3d' in col or '_5d' in col or 'acceleration' in col or 'surge' in col or 'last_day' in col],
        "lags_deltas": [col for col in pd.read_csv(INPUT_FILE).columns if '_lag_' in col or '_delta_' in col]
    }
except FileNotFoundError:
    print(f"WARNING: Input file not found at {INPUT_FILE}. Feature groups cannot be defined. The script might fail.")
    FEATURE_GROUPS = {}


# --- Model Definitions ---
MODELS = {
    "LightGBM": lgb.LGBMRegressor(objective='regression_l1', random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Ridge": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(random_state=42))])
}

# =============================================================================
# --- 2. DATA PREPARATION & FEATURE SELECTION ---
# =============================================================================
def load_and_select_features(file_path, region, target_col, date_col, excluded_groups=None):
    """Loads data and selects features based on excluded groups."""
    print(f"\n--- Loading data for region '{region}' and selecting features ---")
    if excluded_groups is None:
        excluded_groups = []
    
    if not file_path.exists():
        print(f"CRITICAL ERROR: Input file not found at {file_path}")
        return None, None

    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df_region = df[df['Region'] == region].sort_values(by=date_col).set_index(date_col)
    
    # Drop columns from excluded feature groups
    cols_to_drop = []
    for group in excluded_groups:
        if group in FEATURE_GROUPS:
            cols_to_drop.extend(FEATURE_GROUPS[group])
    
    X = df_region.drop(columns=[target_col, 'Region'] + cols_to_drop, errors='ignore')
    y = df_region[target_col]

    # Data Hygiene
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    low_variance_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if low_variance_cols:
        X = X.drop(columns=low_variance_cols)
    
    # FIX: Use modern pandas methods for filling missing values
    X.ffill(inplace=True)
    X.bfill(inplace=True)
    
    print(f"  - Excluded feature groups: {excluded_groups if excluded_groups else 'None'}")
    print(f"  - Final feature shape: {X.shape}")
    return X, y

# =============================================================================
# --- 3. MODELING & EVALUATION ---
# =============================================================================
def run_backtest(X, y, model):
    """Performs a time-series cross-validation backtest for a single model."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_BACKTEST)
    oof_preds, oof_actuals, oof_dates = [], [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        oof_preds.extend(preds)
        oof_actuals.extend(y_test.values)
        oof_dates.extend(y_test.index)
        
    df_backtest = pd.DataFrame({'Date': oof_dates, 'Actual': oof_actuals, 'Predicted': oof_preds}).set_index('Date')
    
    full_mae = mean_absolute_error(df_backtest['Actual'], df_backtest['Predicted'])
    recent_mae = mean_absolute_error(df_backtest['Actual'].tail(RECENT_WEEKS_MAE), df_backtest['Predicted'].tail(RECENT_WEEKS_MAE))
    
    return full_mae, recent_mae

def train_final_model_and_predict(X, y, model):
    """Trains a final model on all data and predicts the next week."""
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_predict = X.iloc[[-1]]
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_predict)[0]
    prediction_date = X_predict.index[0] + pd.Timedelta(days=7) # Predict next week
    
    return model, prediction, prediction_date

# =============================================================================
# --- 4. DIAGNOSTICS & LOGGING ---
# =============================================================================
def log_results(diagnostics_log_path, scenario_name, model_name, full_mae, recent_mae):
    """Logs the results of a single model run to a CSV."""
    log_entry = {
        'Timestamp': [datetime.now()],
        'Region': [REGION_TO_MODEL],
        'Scenario': [scenario_name],
        'Model': [model_name],
        'Full_MAE': [full_mae],
        'Recent_4w_MAE': [recent_mae]
    }
    df_log = pd.DataFrame(log_entry)
    
    if diagnostics_log_path.exists():
        df_log.to_csv(diagnostics_log_path, mode='a', header=False, index=False)
    else:
        df_log.to_csv(diagnostics_log_path, mode='w', header=True, index=False)

def generate_feedback_report(diagnostics_log_path):
    """Analyzes the diagnostics log to provide feedback on feature groups."""
    print("\n--- Generating Model Intelligence Feedback Report ---")
    if not diagnostics_log_path.exists():
        print("  - Diagnostics log not found. Run models to generate it.")
        return

    df_log = pd.read_csv(diagnostics_log_path)
    df_log = df_log.sort_values('Timestamp').drop_duplicates(subset=['Scenario', 'Model'], keep='last')
    
    baseline = df_log[df_log['Scenario'] == 'all_features']
    if baseline.empty:
        print("  - Baseline 'all_features' run not found in log. Cannot generate feedback.")
        return

    print("--- Feedback Loop: Impact of Dropping Feature Groups (vs. baseline) ---")
    report = []
    for model_name in baseline['Model'].unique():
        baseline_mae = baseline[baseline['Model'] == model_name]['Recent_4w_MAE'].iloc[0]
        report.append(f"\nModel: {model_name} (Baseline Recent MAE: {baseline_mae:.2f})")
        
        comparison_runs = df_log[(df_log['Scenario'] != 'all_features') & (df_log['Model'] == model_name)]
        
        impacts = []
        for _, row in comparison_runs.iterrows():
            scenario = row['Scenario']
            mae = row['Recent_4w_MAE']
            impact = mae - baseline_mae # Positive value means dropping the feature hurt performance
            group_name = scenario.replace('no_', '')
            impacts.append({'group': group_name, 'impact_on_mae': impact})
        
        if not impacts:
            report.append("  - No comparison scenarios found to generate feedback.")
            continue

        df_impact = pd.DataFrame(impacts).sort_values('impact_on_mae', ascending=False)
        
        helpful_features = df_impact[df_impact['impact_on_mae'] > 0.1]
        hurting_features = df_impact[df_impact['impact_on_mae'] < -0.1]
        
        report.append("  - Top Helpful Feature Groups (dropping them hurts MAE most):")
        if not helpful_features.empty:
            for _, row in helpful_features.iterrows():
                report.append(f"    - {row['group']} (MAE increased by {row['impact_on_mae']:.2f})")
        else:
            report.append("    - None with significant positive impact.")
            
        report.append("  - Potentially Hurting Feature Groups (dropping them helps MAE most):")
        if not hurting_features.empty:
            for _, row in hurting_features.iterrows():
                report.append(f"    - {row['group']} (MAE improved by {-row['impact_on_mae']:.2f})")
        else:
            report.append("    - None with significant negative impact.")
            
        if not hurting_features.empty:
            report.append(f"  - SUGGESTED DROP for {model_name}: {list(hurting_features['group'])}")
        else:
             report.append(f"  - SUGGESTED DROP for {model_name}: None")

    final_report = "\n".join(report)
    print(final_report)
    
    # Save report to a text file
    report_path = OUTPUT_DIR / 'model_feedback_report.txt'
    with open(report_path, 'w') as f:
        f.write(final_report)
    print(f"\n  - ✅ Feedback report saved to {report_path}")

# =============================================================================
# --- 5. MAIN EXECUTION ---
# =============================================================================
def main():
    """Main function to orchestrate the iterative modeling pipeline."""
    
    diagnostics_log_path = OUTPUT_DIR / 'model_diagnostics.csv'

    # Define the scenarios to run
    scenarios = {
        "all_features": [],
        "no_weather": ["weather"],
        "no_power": ["power"],
        "no_fundy": ["fundy"],
        "no_storage": ["storage"],
        "no_inventory": ["inventory"]
    }

    for scenario_name, excluded_groups in scenarios.items():
        print(f"\n{'='*20} RUNNING SCENARIO: {scenario_name.upper()} {'='*20}")
        
        X, y = load_and_select_features(INPUT_FILE, REGION_TO_MODEL, TARGET_COLUMN, DATE_COLUMN, excluded_groups)
        if X is None or y.empty:
            print(f"Skipping scenario {scenario_name} due to data loading issues.")
            continue
            
        for model_name, model_instance in MODELS.items():
            full_mae, recent_mae = run_backtest(X, y, model_instance)
            log_results(diagnostics_log_path, scenario_name, model_name, full_mae, recent_mae)

            # For the main scenario, train final model and save outputs
            if scenario_name == 'all_features':
                final_model, prediction, pred_date = train_final_model_and_predict(X, y, model_instance)
                
                # Save prediction
                pred_df = pd.DataFrame({'Prediction_Date': [pred_date], f'Predicted_Storage_Change_{model_name}': [prediction]})
                pred_path = OUTPUT_DIR / f'prediction_{model_name}_{REGION_TO_MODEL}.csv'
                pred_df.to_csv(pred_path, index=False)
                print(f"  - ✅ Prediction for {model_name} saved to {pred_path}")

                # FIX: Save feature importance for different model types
                if hasattr(final_model, 'feature_importances_'): # For LGBM and RandomForest
                    if hasattr(final_model, 'feature_name_'): # LGBM
                        feature_names = final_model.feature_name_
                    else: # Scikit-learn tree-based models
                        feature_names = final_model.feature_names_in_
                    
                    importance = pd.Series(final_model.feature_importances_, index=feature_names).nlargest(20)
                    title = f'Feature Importance - {model_name} ({REGION_TO_MODEL})'

                elif hasattr(final_model, 'named_steps') and 'ridge' in final_model.named_steps: # For Ridge pipeline
                    ridge_coefs = final_model.named_steps['ridge'].coef_
                    feature_names = X.columns # The scaler doesn't change the order/names
                    
                    importance = pd.Series(np.abs(ridge_coefs), index=feature_names).nlargest(20)
                    title = f'Feature Importance (Absolute Coefs) - {model_name} ({REGION_TO_MODEL})'
                
                else:
                    importance = None

                if importance is not None:
                    plt.figure(figsize=(10, 8))
                    importance.sort_values().plot(kind='barh')
                    plt.title(title)
                    plt.tight_layout()
                    plot_path = OUTPUT_DIR / f'feature_importance_{model_name}_{REGION_TO_MODEL}.png'
                    plt.savefig(plot_path)
                    plt.close()


    # After all scenarios run, generate the final feedback report
    generate_feedback_report(diagnostics_log_path)

    print("\n--- Iterative Modeling Pipeline Complete ---")


if __name__ == '__main__':
    main()

