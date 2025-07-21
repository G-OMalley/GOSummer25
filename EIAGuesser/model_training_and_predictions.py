# model_training_and_predictions.py
# ---------------------------------
# EIA Natural Gas Storage Weekly Forecast Model
#
# This script provides an end-to-end machine learning pipeline for forecasting
# the weekly U.S. EIA natural gas storage report. It ingests raw daily
# fundamental data, performs feature engineering, evaluates a diverse zoo of
# machine learning models, and generates a forecast for the upcoming week.
# See the docstring below for more details.

"""
EIA Natural Gas Storage Weekly Forecast Model

This script provides an end-to-end machine learning pipeline for forecasting
the weekly U.S. Energy Information Administration (EIA) natural gas storage
report. It ingests raw daily fundamental data, performs sophisticated feature
engineering, evaluates a diverse zoo of machine learning models, and generates
a precise, actionable forecast for the upcoming week.

Key Decisions & Features:
- Data Pipeline: Automates the entire workflow from raw data ingestion to
  final prediction, requiring no manual intervention.
- Feature Engineering: Creates a robust feature set, including 7-day
  rolling sums, Gas-Weighted Degree Days (GWDD), and calendar features.
- Model-Specific Aggregation: Aggregates daily data to a weekly frequency
  using a dynamic dictionary that applies .sum() to flow/demand metrics and
  .mean() to others.
- Model Zoo: Competes a comprehensive suite of models for each EIA region,
  including classical regressors (Ridge, RF, GBDT, etc.) and deep learning
  architectures (MLP, LSTM, GRU, Transformer).
- Champion Selection: Selects the "champion" model based on the lowest Mean
  Absolute Error (MAE) over the most recent 12 weeks, ensuring adaptation to
  the current market regime.
- Interpretability: Generates and saves feature importance plots for tree-based
  models and SHAP summary plots for neural network champions.
- Operational Modes: A MODE flag allows switching between "weekly" (default)
  and "daily" (sequence-based) training strategies.

How to Run:
1. Ensure the required directory structure and data files are in place.
2. Verify all required packages are installed.
3. Execute from the command line: `python model_training_and_predictions.py`
4. All outputs will be saved to the `EIAGuesser/model_outputs/` and
   `EIAGuesser/output/` directories.
"""

import os
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
import holidays
from datetime import datetime

# --- ML/DL Frameworks ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# --- Optional Tree Boosters ---
try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

try:
    import xgboost as xgb
    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False

try:
    import catboost as cb
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False

# --- Deep Learning & Interpretability ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Flatten
    import shap
    TENSORFLOW_INSTALLED = True
except ImportError:
    TENSORFLOW_INSTALLED = False

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages

# ─── Configuration ────────────────────────────────────────────────────────
# General
TOP_N_FEATURES  = 30      # Number of top features to save for tree models
RECENT_WEEKS    = 12      # Evaluation window to pick “best” model
WEIGHT_RECENT   = True    # Use linear recency weights when fitting classical models
RANDOM_STATE    = 42      # Seed for reproducibility
MISSING_THRESHOLD = 0.05  # Drop columns with more than 5% missing values

# Modeling Mode: "weekly" or "daily"
# "weekly": Aggregates daily features to a weekly frequency before training.
# "daily": Trains sequence models on the last 7 days of daily features.
MODE = "weekly"

# ─── Paths ────────────────────────────────────────────────────────────────
ROOT_PATH       = r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper'
EIAGUESSER_PATH = os.path.join(ROOT_PATH, 'EIAGuesser')
OUTPUT_PATH     = os.path.join(EIAGUESSER_PATH, 'output')
MODEL_PATH      = os.path.join(EIAGUESSER_PATH, 'model_outputs')
INFO_PATH       = os.path.join(ROOT_PATH, 'INFO')
FEAT_FILE       = os.path.join(OUTPUT_PATH, 'Combined_Wide_Data.csv')
TARGET_FILE     = os.path.join(INFO_PATH, 'EIAchanges.csv')

# Create output directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'feature_importances'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'shap_plots'), exist_ok=True)

# ─── Helper Functions ─────────────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be code-friendly."""
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_')
    )
    return df

def date_label(dt: pd.Timestamp) -> str:
    """Formats a date for display, accounting for Windows peculiarities."""
    return dt.strftime('%#m/%#d') if platform.system() == 'Windows' else dt.strftime('%-m/%-d')

def make_sample_weight(idx: pd.DatetimeIndex) -> np.ndarray:
    """Creates linearly increasing weights to prioritize recent data."""
    return np.linspace(0.1, 1.0, len(idx))

def get_agg_dict(columns: pd.Index) -> dict:
    """Builds an aggregation dictionary based on column name patterns."""
    agg_dict = {}
    sum_keywords = ('_flow', '_demand', '_stor', '_prod', '_exp', '_imp', '_gen')
    for col in columns:
        if any(keyword in col for keyword in sum_keywords):
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'mean'
    return agg_dict

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from the raw daily data."""
    df_eng = df.select_dtypes(include=np.number).copy()

    # 7-day rolling sums for additive metrics
    sum_cols = [col for col in df_eng.columns if any(k in col for k in ['flow', 'demand', 'stor', 'prod'])]
    roll7 = df_eng[sum_cols].rolling(7, min_periods=1).sum()
    roll7.columns = [f'{c}_sum7' for c in roll7.columns]

    # GWDD feature
    df_eng['gwdd'] = (
        df_eng.filter(like='_weather_').filter(like='_cdd').sum(axis=1) -
        df_eng.filter(like='_weather_').filter(like='_hdd').sum(axis=1)
    )

    # Calendar features
    us_holidays = holidays.US()
    df_eng['is_holiday'] = df.index.to_series().apply(lambda x: x in us_holidays).astype(int)
    dow = pd.get_dummies(df.index.dayofweek, prefix='dow', drop_first=True).astype(int)
    dow.index = df.index

    return pd.concat([df_eng, roll7, dow], axis=1)

# ─── Deep Learning Model Builders ─────────────────────────────────────────
def build_mlp(input_shape, name='MLP'):
    """Builds a simple Multi-Layer Perceptron model."""
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs) if len(input_shape) > 1 else inputs
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def build_lstm(input_shape, name='LSTM'):
    """Builds an LSTM model."""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def build_gru(input_shape, name='GRU'):
    """Builds a GRU model."""
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=False)(inputs) # return_sequences=False is a good baseline
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def build_transformer(input_shape, head_size=128, num_heads=4, ff_dim=4, num_blocks=4, name='Transformer'):
    """Builds a Transformer Encoder model."""
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x_norm = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x_norm, x_norm)
        attention_output = Dropout(0.1)(attention_output)
        x = x + attention_output

        x_norm = LayerNormalization(epsilon=1e-6)(x)
        ffn = Dense(ff_dim, activation="relu")(x_norm)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(input_shape[-1])(ffn)
        x = x + ffn

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model

def generate_shap_summary(model, X_data, X_cols, region):
    """Generates and saves a SHAP summary plot for a given model."""
    print(f"   Generating SHAP summary for {model.name}...")
    try:
        background_data = X_data[np.random.choice(X_data.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(X_data)
        if len(X_data.shape) == 3:
            shap_values_avg = np.mean(shap_values[0], axis=1)
            shap_df = pd.DataFrame(np.mean(X_data, axis=1), columns=X_cols)
            shap.summary_plot(shap_values_avg, shap_df, show=False, max_display=TOP_N_FEATURES)
        else:
            shap.summary_plot(shap_values[0], pd.DataFrame(X_data, columns=X_cols), show=False, max_display=TOP_N_FEATURES)
        import matplotlib.pyplot as plt
        plot_path = os.path.join(OUTPUT_PATH, 'shap_plots', f'{region}_{model.name}_shap.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"   SHAP plot saved to {plot_path}")
    except Exception as e:
        print(f"   Could not generate SHAP plot for {model.name}: {e}")

# ─── Load Data ────────────────────────────────────────────────────────────
print('-' * 60 + '\nLoading data…')
features_daily = pd.read_csv(FEAT_FILE, index_col='date', parse_dates=True)
targets_raw    = pd.read_csv(TARGET_FILE)

features_daily = clean_columns(features_daily)
targets_raw = clean_columns(targets_raw)
targets_raw.rename(columns={'period': 'date'}, inplace=True)
targets_raw['date'] = pd.to_datetime(targets_raw['date'])
targets_raw.set_index('date', inplace=True)

target_map = {
    'conus'          : 'lower_48_states_storage_change_bcf',
    'east'           : 'east_region_storage_change_bcf',
    'midwest'        : 'midwest_region_storage_change_bcf',
    'south_central'  : 'south_central_region_storage_change_bcf',
    'mountain'       : 'mountain_region_storage_change_bcf',
    'pacific'        : 'pacific_region_storage_change_bcf'
}
targets_raw = targets_raw[list(target_map.values())]

# ─── Imputation and Feature Engineering ──────────────────────────────────
# Drop columns with a high percentage of missing values
missing_pct = features_daily.isnull().sum() / len(features_daily)
cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index
if len(cols_to_drop) > 0:
    features_daily.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} columns with >{MISSING_THRESHOLD:.0%} missing values.")

# Impute remaining NaNs using interpolation and forward/backward fill
print("Imputing missing values...")
features_daily.interpolate(method='linear', limit_direction='both', inplace=True)
print(f"Data imputed. Remaining NaNs: {features_daily.isnull().sum().sum()}")

features_daily_eng = engineer_features(features_daily)
print("Feature engineering complete.")

# ─── Aggregation ──────────────────────────────────────────────────────────
if MODE == "weekly":
    agg_dict = get_agg_dict(features_daily_eng.columns)
    weekly_feat = features_daily_eng.resample('W-THU').agg(agg_dict)
    weekly_feat.index += pd.Timedelta(days=1)
    print(f"Data aggregated to weekly frequency (Fridays).")
elif MODE == "daily":
    weekly_feat = features_daily_eng
    print(f"Using daily data for sequence modeling.")
else:
    raise ValueError(f"Invalid MODE: '{MODE}'. Choose 'weekly' or 'daily'.")

# --- Join with Targets ---
weekly_feat.index = weekly_feat.index.tz_localize(None)
targets_raw.index = targets_raw.index.tz_localize(None)

if MODE == "weekly":
    weekly_all = weekly_feat.join(targets_raw, how='inner')
elif MODE == "daily":
    targets_on_thursday = targets_raw.copy()
    targets_on_thursday.index -= pd.Timedelta(days=1)
    weekly_all = weekly_feat.join(targets_on_thursday, how='left')

weekly_all.dropna(axis=0, how='any', subset=list(target_map.values()), inplace=True)

if weekly_all.empty:
    raise ValueError("The merged dataframe is empty after inner join. Check for date overlap and format consistency.")

# ─── Prepare for Prediction ──────────────────────────────────────────────
last_report_date = weekly_all.index.max()
pred_week_label = last_report_date + pd.Timedelta(days=7)

if MODE == "weekly":
    if pred_week_label not in weekly_feat.index:
        raise ValueError(f'Missing weekly aggregated data for week ending {pred_week_label.date()}')
elif MODE == "daily":
    pred_week_thursday = pred_week_label - pd.Timedelta(days=1)
    required_days = pd.date_range(start=pred_week_thursday - pd.Timedelta(days=6), end=pred_week_thursday)
    if not all(day in features_daily_eng.index for day in required_days):
        raise ValueError(f'Missing one or more daily rows for week ending {pred_week_thursday.date()}')

print(f'Last EIA week (Friday label): {last_report_date.date()} | Forecasting for week: {pred_week_label.date()}')

# ─── Model Zoo ────────────────────────────────────────────────────────────
base_models = {
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=400, random_state=RANDOM_STATE)
}
if LGBM_INSTALLED:
    base_models['LightGBM'] = lgb.LGBMRegressor(n_estimators=800, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
if XGBOOST_INSTALLED:
    base_models['XGBoost'] = xgb.XGBRegressor(n_estimators=800, random_state=RANDOM_STATE, n_jobs=-1)
if CATBOOST_INSTALLED:
    base_models['CatBoost'] = cb.CatBoostRegressor(iterations=800, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)

if TENSORFLOW_INSTALLED:
    base_models.update({ 'MLP': 'build_mlp', 'LSTM': 'build_lstm', 'GRU': 'build_gru', 'Transformer': 'build_transformer' })

def extract_scalar(pred):
   return pred[0, 0] if pred.ndim == 2 else pred[0]

# model_training_and_predictions.py - Cleaned & Corrected Version
# ------------------------------------------------------------
# EIA Natural Gas Storage Weekly Forecast Model
# Full script with corrected logic and indentation

# [Truncated top portion for brevity – assume all previous code remains unchanged until the Main Loop]

# ─── Main Loop ────────────────────────────────────────────────────────────
results = []
SEQUENCE_LENGTH = 7

for region, tgt_col in target_map.items():
    print(f"\n{'=' * 22} {region.upper().replace('_', ' ')} {'=' * 22}")

    # Select target series and raw feature set
    y_all = weekly_all[tgt_col]
    X_all_raw = weekly_all.drop(columns=target_map.values())

    # Narrow to top features if available or region-specific defaults
    imp_file = os.path.join(OUTPUT_PATH, 'feature_importances', f'{region}_feature_importances.csv')
    if os.path.exists(imp_file):
        top_feats = pd.read_csv(imp_file)['feature'].head(TOP_N_FEATURES).tolist()
    else:
        if region != 'conus':
            top_feats = [c for c in X_all_raw.columns if c.startswith((region, 'conus'))]
        else:
            top_feats = X_all_raw.columns.tolist()

    X_all_raw = X_all_raw[[c for c in top_feats if c in X_all_raw.columns]]
    X_all = X_all_raw.copy()
    date_index = y_all.index

    # Split into training and evaluation sets based on RECENT_WEEKS window
    eval_cut_date = date_index.max() - pd.Timedelta(weeks=RECENT_WEEKS)
    train_mask = date_index <= eval_cut_date
    eval_mask  = date_index > eval_cut_date

    train_X, train_y = X_all[train_mask], y_all[train_mask]
    eval_X,  eval_y  = X_all[eval_mask],  y_all[eval_mask]

    if len(train_X) < 40 or len(eval_X) < 4:
        print('   Skipping region (not enough data)')
        continue

    # Standardize if needed
    scaler = StandardScaler()
    if MODE == 'daily':
        # Sequence models expect 3D input
        train_X_scaled = scaler.fit_transform(train_X.values.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        eval_X_scaled  = scaler.transform(eval_X.values.reshape(-1, eval_X.shape[-1])).reshape(eval_X.shape)
    else:
        train_X_scaled = scaler.fit_transform(train_X)
        eval_X_scaled  = scaler.transform(eval_X)

    # Model selection
    best_model_obj, best_mae, champion_is_nn, best_model_name = None, np.inf, False, ''
    for name, model_obj in base_models.items():
        is_nn = isinstance(model_obj, str)
        if is_nn and not TENSORFLOW_INSTALLED:
            continue
        if MODE == 'weekly' and name in ['LSTM', 'GRU', 'Transformer']:
            continue
        if MODE == 'daily' and name == 'Ridge':
            continue

        # Prepare data for training
        X_tr, X_ev = (train_X_scaled, eval_X_scaled) if is_nn else (train_X, eval_X)
        y_tr, y_ev = train_y, eval_y

        # Train
        if is_nn:
            model = globals()[model_obj](X_tr.shape[1:], name=name)
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
            model.fit(X_tr, y_tr, validation_data=(X_ev, y_ev), epochs=150, batch_size=32,
                      callbacks=callbacks, verbose=0)
        else:
            model = clone(model_obj)
            fit_params = {'sample_weight': make_sample_weight(train_y.index)} if WEIGHT_RECENT else {}
            model.fit(X_tr, y_tr, **fit_params)

        # Evaluate
        preds = model.predict(X_ev).flatten()
        r_mae = mean_absolute_error(y_ev, preds)
        print(f'   {name:<16} recent {RECENT_WEEKS}-wk MAE: {r_mae:5.2f}')

        if r_mae < best_mae:
            best_mae, champion_is_nn, best_model_name = r_mae, is_nn, name
            if is_nn:
                model.save_weights(f'{name}_temp_weights.weights.h5')
                best_model_arch = model.to_json()
            else:
                best_model_obj = clone(model)

    print(f' → Champion: {best_model_name} (MAE {best_mae:4.2f})')

    # Retrain champion on full data
    if champion_is_nn:
        full_X = X_all_scaled if MODE == 'daily' else scaler.fit_transform(X_all)
        champion_model = tf.keras.models.model_from_json(best_model_arch)
        champion_model.load_weights(f'{best_model_name}_temp_weights.weights.h5')
        champion_model.compile(optimizer='adam', loss='mean_absolute_error')
        champion_model.fit(full_X, y_all, epochs=150, batch_size=32,
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)], verbose=0)
        os.remove(f'{best_model_name}_temp_weights.weights.h5')
    else:
        champion_model = best_model_obj
        fit_params = {'sample_weight': make_sample_weight(y_all.index)} if WEIGHT_RECENT else {}
        champion_model.fit(X_all, y_all, **fit_params)

    # Save model
    model_fname = f"{region}_{best_model_name}_{last_report_date:%Y%m%d}"
    if champion_is_nn:
        model_fpath = os.path.join(MODEL_PATH, model_fname + '.keras')
        champion_model.save(model_fpath)
        generate_shap_summary(champion_model, full_X if MODE=='daily' else scaler.transform(X_all), X_all.columns, region)
    else:
        model_fpath = os.path.join(MODEL_PATH, model_fname + '.joblib')
        joblib.dump(champion_model, model_fpath)
        if hasattr(champion_model, 'feature_importances_'):
            fi_df = pd.Series(champion_model.feature_importances_, index=X_all.columns)
            fi_df.sort_values(ascending=False).head(TOP_N_FEATURES)
            fi_df.reset_index().rename(columns={'index':'feature',0:'importance'}).to_csv(
                os.path.join(OUTPUT_PATH,'feature_importances',f'{region}_feature_importances.csv'), index=False)
    print(f'   Saved champion model to {model_fpath}')

    # Prepare prediction inputs
    if MODE == 'daily':
        next_week_data = features_daily_eng.loc[required_days, X_all.columns].values.reshape(1, SEQUENCE_LENGTH, -1)
        prior_week_data = X_all.iloc[[-1]].values.reshape(1, *X_all.iloc[[-1]].values.shape)
        if champion_is_nn:
            next_week_data = scaler.transform(next_week_data.reshape(-1, next_week_data.shape[-1])).reshape(next_week_data.shape)
            prior_week_data = scaler.transform(prior_week_data.reshape(-1, prior_week_data.shape[-1])).reshape(prior_week_data.shape)
    else:
        next_week_data = weekly_feat.loc[[pred_week_label], X_all.columns].values
        prior_week_data = X_all.iloc[[-1]].values
        if champion_is_nn:
            next_week_data = scaler.transform(next_week_data)
            prior_week_data = scaler.transform(prior_week_data)

    # Generate predictions
    next_pred = extract_scalar(champion_model.predict(next_week_data))
    prior_pred = extract_scalar(champion_model.predict(prior_week_data))
    prior_act  = y_all.iloc[-1]

    # Append to results
    results.append({
        'Region': region.replace('conus','Lower 48').replace('_',' ').title(),
        'Model': best_model_name,
        f'{date_label(pred_week_label)} Est': next_pred,
        f'{RECENT_WEEKS} wk MAE': best_mae,
        f'{date_label(last_report_date)} Est': prior_pred,
        f'{date_label(last_report_date)} actual': prior_act
    })



# ─── Save Summary ─────────────────────────────────────────────────────────
if not results: print("\nNo models were trained. Exiting."); exit()
res_df = pd.DataFrame(results).set_index(['Region', 'Model']).round(2).sort_index()
print('\n' + '-' * 60 + '\nFinal Prediction Summary\n' + '-' * 60); print(res_df)
csv_out = os.path.join(MODEL_PATH, 'predictions.csv')
res_df.to_csv(csv_out)
print(f'\nForecast table → {csv_out}')

try:
    l48_est = res_df.loc[('Lower 48', slice(None)), f'{date_label(pred_week_label)} Est'].iloc[0]
    over_under = int(round(l48_est))
    ou_df = pd.DataFrame(
        {f'{date_label(pred_week_label)} Est': [f'{over_under} (Bcf)'], f'{RECENT_WEEKS} wk MAE': ['—'],
         f'{date_label(last_report_date)} Est': ['—'], f'{date_label(last_report_date)} actual': ['—']},
        index=pd.MultiIndex.from_tuples([('OVER / UNDER', '')], names=['Region', 'Model'])
    )
    res_df_with_ou = pd.concat([res_df, ou_df]); res_df_with_ou.to_csv(csv_out)
    print(f'Over/Under line (Lower-48 rounded): {over_under:+} Bcf')
except (KeyError, IndexError): print("Could not generate Over/Under line.")

print('Model objects  →', MODEL_PATH); print('Done', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))