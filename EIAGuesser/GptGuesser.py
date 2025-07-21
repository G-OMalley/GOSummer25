# Fully fixed version: suppress LightGBM warnings and skip week if not enough data

import os, warnings, joblib, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Suppress LightGBM warnings globally
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import holidays; US_HOLIDAYS = holidays.US()
except ImportError:
    US_HOLIDAYS = None

RANDOM_STATE = 42
MISSING_THRESHOLD = 0.05
N_SPLITS = 5
TEST_SIZE = 12
ROOT = r'C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper'
DATA_DIR = os.path.join(ROOT, 'EIAGuesser', 'output')
INFO_DIR = os.path.join(ROOT, 'INFO')
MODEL_DIR = os.path.join(ROOT, 'EIAGuesser', 'model_outputs')
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_columns(df):
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.strip('_')
    )
    return df

def engineer_daily(df):
    num = df.select_dtypes(include=np.number).copy()
    num['gwdd'] = num.filter(like='_cdd').sum(axis=1) - num.filter(like='_hdd').sum(axis=1)
    num['dow'] = df.index.dayofweek
    num['month'] = df.index.month
    num['is_holiday'] = df.index.isin(US_HOLIDAYS).astype(int) if US_HOLIDAYS else 0
    return num

def aggregate_weekly(df_daily):
    agg = {
        col: ('sum' if any(k in col for k in ['_flow', '_demand', '_prod', '_imp', '_exp', '_hdd', '_cdd']) else 'mean')
        for col in df_daily.columns
    }
    weekly = df_daily.resample('W-FRI').agg(agg)
    return weekly

def add_weekly_features(df_w, targets):
    df = df_w.copy()
    for t in targets:
        for lag in range(1, 5):
            df[f'{t}_lag{lag}'] = df[t].shift(lag)
        df[f'{t}_vol4'] = df[t].shift(1).rolling(4).std()
    df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)
    df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
    return df.dropna()

def oof_stack_preds(models, X, y):
    oof = pd.DataFrame(index=X.index)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    for name, mdl in models.items():
        preds = np.zeros(len(X))
        for tr, te in splitter.split(X):
            m = mdl.__class__(**mdl.get_params())
            m.fit(X.iloc[tr], y.iloc[tr])
            preds[te] = m.predict(X.iloc[te])
        oof[name] = preds
    return oof

def train_region(region, target, weekly_feat, daily_eng, tgt_df):
    merged = weekly_feat.join(tgt_df[[target]], how='inner')
    adv = add_weekly_features(merged, [target])
    X, y = adv.drop(columns=[target]), adv[target]
    if len(X) < N_SPLITS * TEST_SIZE + 1:
        print(f"[Skip] {region}: insufficient history ({len(X)} rows)")
        return None

    base = {'rf': RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE)}
    if HAS_LGBM:
        base['lgbm'] = lgb.LGBMRegressor(n_estimators=1200, random_state=RANDOM_STATE)
    if HAS_XGB:
        base['xgb'] = xgb.XGBRegressor(n_estimators=1200, random_state=RANDOM_STATE, verbosity=0)

    oof = oof_stack_preds(base, X, y)
    meta = Ridge(random_state=RANDOM_STATE)
    meta.fit(oof, y)
    cv_mae = mean_absolute_error(y, meta.predict(oof))

    for m in base.values(): m.fit(X, y)

    last_fri = weekly_feat.index.max()
    next_fri = last_fri + timedelta(days=7)
    win_start = next_fri - timedelta(days=6)
    win = daily_eng.loc[win_start:next_fri]
    if len(win) < 7:
        print(f"[Skip] {region}: insufficient daily for {next_fri.date()}")
        return None

    Xn = aggregate_weekly(win).loc[[next_fri]]
    for lag in range(1, 5): Xn[f'{target}_lag{lag}'] = adv[target].iloc[-lag]
    Xn[f'{target}_vol4'] = adv[target].iloc[-4:].std()
    Xn['is_winter'] = int(next_fri.month in [12, 1, 2])
    Xn['is_summer'] = int(next_fri.month in [6, 7, 8])
    Xn = Xn[X.columns]

    base_preds = {n: m.predict(Xn)[0] for n, m in base.items()}
    next_est = meta.predict(pd.DataFrame([base_preds]))[0]

    tag = region.replace(' ', '_').lower()
    stamp = datetime.now().strftime('%Y%m%d')
    joblib.dump({'base': base, 'meta': meta}, os.path.join(MODEL_DIR, f'{tag}_stack_{stamp}.joblib'))

    return {'Region': region.title(), 'CV_MAE': round(cv_mae, 2), 'Next_Est': round(next_est, 2), 'Last_Act': y.iloc[-1]}

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    daily = clean_columns(pd.read_csv(os.path.join(DATA_DIR, 'Combined_Wide_Data.csv'), index_col='date', parse_dates=True))
    tgt = clean_columns(pd.read_csv(os.path.join(INFO_DIR, 'EIAchanges.csv')))
    tgt.rename(columns={'period': 'date'}, inplace=True)
    tgt['date'] = pd.to_datetime(tgt['date'])
    tgt.set_index('date', inplace=True)

    thr = int((1 - MISSING_THRESHOLD) * len(daily))
    daily.dropna(axis=1, thresh=thr, inplace=True)
    daily.interpolate(limit_direction='both', inplace=True)

    daily_eng = engineer_daily(daily)
    weekly_feat = aggregate_weekly(daily_eng)

    regions = {
        'conus': 'lower_48_states_storage_change_bcf',
        'east': 'east_region_storage_change_bcf',
        'midwest': 'midwest_region_storage_change_bcf',
        'south_central': 'south_central_region_storage_change_bcf',
        'mountain': 'mountain_region_storage_change_bcf',
        'pacific': 'pacific_region_storage_change_bcf'
    }

    res = []
    for r, t in regions.items():
        out = train_region(r, t, weekly_feat, daily_eng, tgt)
        if out: res.append(out)

    df_res = pd.DataFrame(res).set_index('Region')
    print('\nFinal Results:\n', df_res)
    df_res.to_csv(os.path.join(MODEL_DIR, 'predictions.csv'))
    print(f"\nSaved predictions to {os.path.join(MODEL_DIR, 'predictions.csv')}\n")