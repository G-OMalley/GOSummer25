# model_training_and_predictions.py
# ---------------------------------
# Select the model with the lowest MAE over the last N weeks, retrain on
# full data, and forecast next week's EIA storage changes.

import os
import platform
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.base import clone

# -- optional tree boosters ------------------------------------------------
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

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# ─── Paths ────────────────────────────────────────────────────────────────
root           = r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper'
eiaguesser     = os.path.join(root, 'EIAGuesser')
output_path    = os.path.join(eiaguesser, 'output')           # processed_daily_data.csv
model_path     = os.path.join(eiaguesser, 'model_outputs')
info_path      = os.path.join(root, 'INFO')                   # EIAchanges.csv
feat_file      = os.path.join(output_path, 'processed_daily_data.csv')
target_file    = os.path.join(info_path, 'EIAchanges.csv')

os.makedirs(model_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────
TOP_N_FEATURES  = 40
CV_SPLITS       = 4
RECENT_WEEKS    = 12        # evaluation window to pick “best” model
WEIGHT_RECENT   = True      # use linear recency weights when fitting

# ─── Helpers ──────────────────────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
                 .str.lower()
                 .str.replace(r'[^a-z0-9]+', '_', regex=True)
                 .str.strip('_')
    )
    return df

def date_label(dt: pd.Timestamp) -> str:
    return dt.strftime('%#m/%#d') if platform.system() == 'Windows' else dt.strftime('%-m/%-d')

def make_sample_weight(idx: pd.DatetimeIndex) -> np.ndarray:
    # Linear recency weights (oldest=1, newest=len)
    return np.arange(1, len(idx) + 1, dtype=float)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    roll7 = df.rolling(7, min_periods=1).sum()
    roll7.columns = roll7.columns + '_sum7'

    df['gwdd'] = (
        df.filter(like='_weather_').filter(like='_cdd').sum(axis=1)
        - df.filter(like='_weather_').filter(like='_hdd').sum(axis=1)
    )

    dow = pd.get_dummies(df.index.dayofweek, prefix='dow', drop_first=True)
    dow.index = df.index

    return pd.concat([df, roll7, dow], axis=1)

# ─── Load Data ────────────────────────────────────────────────────────────
print('-' * 60 + '\nLoading data…')
features_daily = pd.read_csv(feat_file, index_col='date', parse_dates=True)
targets_raw    = clean_columns(pd.read_csv(target_file))

targets_raw.rename(columns={'period': 'date'}, inplace=True)
targets_raw['date'] = pd.to_datetime(targets_raw['date'])
targets_raw.set_index('date', inplace=True)

target_map = {
    'conus48'      : 'lower_48_states_storage_change_bcf',
    'east'         : 'east_region_storage_change_bcf',
    'midwest'      : 'midwest_region_storage_change_bcf',
    'southcentral' : 'south_central_region_storage_change_bcf',
    'mountain'     : 'mountain_region_storage_change_bcf',
    'pacific'      : 'pacific_region_storage_change_bcf'
}
targets_raw = targets_raw[list(target_map.values())]

# ─── Feature Engineering ─────────────────────────────────────────────────
features_daily = engineer_features(features_daily)

weekly_feat = features_daily.resample('W-THU').mean()
weekly_feat.index = weekly_feat.index + pd.Timedelta(days=1)  # align to Friday

weekly_all = weekly_feat.join(targets_raw, how='inner').dropna(how='any')
last_report, pred_week = weekly_all.index.max(), weekly_all.index.max() + pd.Timedelta(days=7)

if pred_week not in weekly_feat.index:
    raise ValueError(f'Missing daily data for week ending {pred_week.date()}')

print(f'Last EIA week: {last_report.date()} | Forecasting: {pred_week.date()}')

# ─── Model zoo ────────────────────────────────────────────────────────────
base_models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=400, random_state=42)
}

if LGBM_INSTALLED:
    base_models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=800, random_state=42, n_jobs=-1, verbose=-1
    )
if CATBOOST_INSTALLED:
    base_models['CatBoost'] = cb.CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.05,
        random_state=42, verbose=False, allow_writing_files=False
    )

# ─── Main Loop ────────────────────────────────────────────────────────────
results = []

for region, tgt in target_map.items():
    print(f'\n{"=" * 22} {region.upper()} {"=" * 22}')
    y_all = weekly_all[tgt]
    X_all = weekly_all.drop(columns=target_map.values())

    # --- feature list ----------------------------------------------------
    imp_file = os.path.join(output_path, f'{region}_feature_importances.csv')
    if os.path.exists(imp_file):
        top_feats = pd.read_csv(imp_file)['feature'].head(TOP_N_FEATURES).tolist()
    else:
        if region == 'conus48':
            top_feats = X_all.columns.tolist()
        else:
            top_feats = [
                c for c in X_all.columns
                if c.startswith(region + '_') or c.startswith('conus48_')
            ]
    X_all = X_all[top_feats]

    # --- split into train / recent eval ----------------------------------
    eval_cut = X_all.index.max() - pd.Timedelta(weeks=RECENT_WEEKS)
    train_X, train_y = X_all[X_all.index <= eval_cut], y_all[y_all.index <= eval_cut]
    eval_X,  eval_y  = X_all[X_all.index > eval_cut],  y_all[y_all.index > eval_cut]

    if len(train_X) < 40 or len(eval_X) < 4:
        print('Skipping (not enough data).')
        continue

    best_model, best_mae, fi_accum = None, np.inf, None

    # --- evaluate each candidate on RECENT window ------------------------
    for name, model in base_models.items():
        sw = make_sample_weight(train_X.index) if WEIGHT_RECENT and name != 'Ridge' else None
        if sw is not None:
            model.fit(train_X, train_y, sample_weight=sw)
        else:
            model.fit(train_X, train_y)

        preds = model.predict(eval_X)
        r_mae = mean_absolute_error(eval_y, preds)
        print(f'   {name:<15} recent {RECENT_WEEKS}-wk MAE: {r_mae:5.2f}')

        if r_mae < best_mae:
            best_model = clone(model)
            best_mae = r_mae
            fi_accum = getattr(model, 'feature_importances_', None)

    # --- retrain champion on *all* data ----------------------------------
    sw_all = make_sample_weight(X_all.index) if WEIGHT_RECENT and isinstance(best_model, (RandomForestRegressor, GradientBoostingRegressor)) else None
    if sw_all is not None:
        best_model.fit(X_all, y_all, sample_weight=sw_all)
    else:
        best_model.fit(X_all, y_all)

    # save champion model
    champ_name = best_model.__class__.__name__.replace('Regressor', '')
    model_fname = f"{region}_{champ_name}_{last_report:%Y%m%d}.joblib"
    joblib.dump(best_model, os.path.join(model_path, model_fname))
    print(f' → Champion: {champ_name} (MAE {best_mae:4.2f}) | saved {model_fname}')

    # update importance CSV
    if fi_accum is not None:
        fi_df = (
            pd.Series(fi_accum, index=X_all.columns)
              .sort_values(ascending=False)
              .head(TOP_N_FEATURES)
              .reset_index()
              .rename(columns={'index': 'feature', 0: 'importance'})
        )
        fi_df.to_csv(imp_file, index=False)

    # predictions
    next_pred  = best_model.predict(weekly_feat.loc[[pred_week], X_all.columns])[0]
    prior_pred = best_model.predict(X_all.iloc[[-1]])[0]
    prior_act  = y_all.iloc[-1]

    results.append({
        'Region': region.replace('conus48', 'Lower 48').capitalize(),
        'Model': champ_name,
        f'{date_label(pred_week)} Est': next_pred,
        f'{RECENT_WEEKS} wk MAE': round(best_mae, 2),
        f'{date_label(last_report)} Est': prior_pred,
        f'{date_label(last_report)} actual': prior_act
    })

# ─── Save summary ─────────────────────────────────────────────────────────
res_df = (
    pd.DataFrame(results)
      .set_index(['Region', 'Model'])
      .round(2)
      .sort_index()
)

print('\n' + '-' * 60 + '\nFinal Prediction Summary\n' + '-' * 60)
print(res_df)

csv_out = os.path.join(model_path, 'predictions.csv')
res_df.to_csv(csv_out)
print(f'\nForecast table → {csv_out}')
print('Model objects  →', model_path)
print('Done', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# ─── Add Over / Under line -----------------------------------------------
next_label = date_label(pred_week)  # e.g. '7/11'
l48_est = res_df.loc[('Lower 48', slice(None)), f'{next_label} Est'].iloc[0]
over_under = int(round(l48_est))

ou_df = pd.DataFrame(
    {
        f'{next_label} Est': [f'{over_under} (Bcf)'],
        f'{RECENT_WEEKS} wk MAE': ['—'],
        f'{date_label(last_report)} Est': ['—'],
        f'{date_label(last_report)} actual': ['—']
    },
    index=pd.MultiIndex.from_tuples([('OVER / UNDER', '')], names=['Region', 'Model'])
)

res_df_with_ou = pd.concat([res_df, ou_df])

print('\n' + '-' * 60 + '\nFinal Prediction Summary\n' + '-' * 60)
print(res_df_with_ou)

csv_out = os.path.join(model_path, 'predictions.csv')
res_df_with_ou.to_csv(csv_out)
print(f'\nForecast table → {csv_out}')
print(f'Over/Under line (Lower-48 rounded): +{over_under} Bcf')
print('Model objects  →', model_path)
print('Done', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
