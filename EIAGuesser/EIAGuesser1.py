# EIAGuesser1.py - Finalized Overhauled Version
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import us

warnings.filterwarnings('ignore')

# ------------------- PATH SETUP -------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    INFO_DIR = SCRIPT_DIR.parent / 'INFO'
    OUTPUT_DIR = SCRIPT_DIR / 'output'
except NameError:
    BASE_DIR = Path('C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/EIAGuesser')
    INFO_DIR = BASE_DIR.parent / 'INFO'
    OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------- CONFIG -------------------
REGIONS = ['East', 'Midwest', 'South Central', 'Mountain', 'Pacific']
DAILY_TARGET = 'Storage_Criterion_Regional_Sum'
WEEKLY_TARGET_PREFIX = 'Region Storage Change (Bcf)'
BACKTEST_WEEKS = 5

MODELS = {
    'LightGBM': lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=500),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=3),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(random_state=42))])
}

REGION_MAP = {
    'CT': 'East', 'DE': 'East', 'FL': 'East', 'GA': 'East', 'MA': 'East', 'MD': 'East', 'ME': 'East', 'NC': 'East',
    'NH': 'East', 'NJ': 'East', 'NY': 'East', 'OH': 'East', 'PA': 'East', 'RI': 'East', 'SC': 'East', 'VA': 'East',
    'VT': 'East', 'WV': 'East',
    'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KY': 'Midwest', 'MI': 'Midwest', 'MN': 'Midwest',
    'MO': 'Midwest', 'TN': 'Midwest', 'WI': 'Midwest',
    'AZ': 'Mountain', 'CO': 'Mountain', 'ID': 'Mountain', 'MT': 'Mountain', 'NE': 'Mountain', 'NV': 'Mountain',
    'NM': 'Mountain', 'ND': 'Mountain', 'SD': 'Mountain', 'UT': 'Mountain', 'WY': 'Mountain',
    'CA': 'Pacific', 'OR': 'Pacific', 'WA': 'Pacific',
    'AL': 'South Central', 'AR': 'South Central', 'KS': 'South Central', 'LA': 'South Central', 'MS': 'South Central',
    'OK': 'South Central', 'TX': 'South Central'
}

# ------------------- LOADER -------------------
def load_data():
    print("\n--- Loading Data ---")
    def read_file(name):
        path = INFO_DIR / name
        if path.exists():
            print(f"  - Reading {name}...")
            return pd.read_csv(path)
        else:
            print(f"  - Missing {name}.")
            return pd.DataFrame()

    files = {
        'eia_changes': read_file('EIAchanges.csv'),
        'fundy': read_file('Fundy.csv'),
        'weather': read_file('WEATHER.csv'),
        'power': read_file('PlattsPowerFundy.csv'),
        'totals': read_file('EIAtotals.csv'),
        'storage': read_file('CriterionStorageChange.csv'),
        'locs': read_file('locs_list.csv')
    }
    return files

# ------------------- ENGINEER -------------------
def engineer_daily_features(data):
    print("\n--- Engineering Features ---")
    df = data['storage']
    locs = data['locs']
    df['storage_name'] = df['storage_name'].str.strip().str.lower()
    locs['storage_name'] = locs['storage_name'].str.strip().str.lower()

    locs = locs[['storage_name', 'state_name']].dropna().drop_duplicates()
    locs['state_name'] = locs['state_name'].str.strip().str.upper()
    locs['state_abbr'] = locs['state_name'].map({s.name.upper(): s.abbr for s in us.states.STATES})
    locs['Region'] = locs['state_abbr'].map(REGION_MAP)

    merged = pd.merge(df, locs, on='storage_name', how='left')
    merged.dropna(subset=['Region'], inplace=True)
    merged.rename(columns={'eff_gas_day': 'Date'}, inplace=True)
    merged['Date'] = pd.to_datetime(merged['Date'])

    regional = merged.groupby(['Date', 'Region'])['daily_storage_change'].sum().reset_index()
    regional.rename(columns={'daily_storage_change': DAILY_TARGET}, inplace=True)
    return regional

# ------------------- FEATURES -------------------
def create_modeling_features(df):
    print("\n--- Generating Lags & Rolls ---")
    df = df.sort_values(['Region', 'Date']).copy()
    df['target'] = df.groupby('Region')[DAILY_TARGET].shift(-1)
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df.groupby('Region')[DAILY_TARGET].shift(lag)
    for win in [3, 7]:
        df[f'roll_mean_{win}'] = df.groupby('Region')[DAILY_TARGET].transform(lambda x: x.rolling(win).mean())
        df[f'roll_std_{win}'] = df.groupby('Region')[DAILY_TARGET].transform(lambda x: x.rolling(win).std())
    df['dayofweek'] = df['Date'].dt.dayofweek
    df.dropna(inplace=True)
    return df

# ------------------- MODELING -------------------
def run_regional_modeling(df_modeling, region, eia_changes):
    print(f"\n=== Modeling {region} ===")
    df = df_modeling[df_modeling['Region'] == region].copy()
    if df.empty:
        return None
    features = [c for c in df.columns if c not in ['Date', 'Region', 'target', DAILY_TARGET]]
    X, y = df[features], df['target']
    last_date = df['Date'].max()
    cutoff = last_date - pd.Timedelta(days=(BACKTEST_WEEKS * 7) - 1)
    train = df[df['Date'] < cutoff]
    test = df[df['Date'] >= cutoff]
    result = {}
    for name, model in MODELS.items():
        model.fit(train[features], train['target'])
        preds = model.predict(test[features])
        test[f'pred_{name}'] = preds
        result[name] = {'model': model, 'mae': mean_absolute_error(test['target'], preds)}
    weekly_map = pd.concat([
        pd.DataFrame({'Date': pd.date_range(end=friday, periods=7), 'Week_Ending': friday})
        for friday in pd.to_datetime(eia_changes['Period'].unique())
    ])
    test = pd.merge(test, weekly_map, on='Date', how='left').dropna(subset=['Week_Ending'])
    weekly_preds = test.groupby('Week_Ending').mean(numeric_only=True).reset_index()
    actuals = eia_changes[['Period', f'{region} {WEEKLY_TARGET_PREFIX}']].rename(columns={'Period': 'Week_Ending', f'{region} {WEEKLY_TARGET_PREFIX}': 'Actual'})
    out = pd.merge(weekly_preds, actuals, on='Week_Ending', how='left')
    return out, result

# ------------------- MAIN -------------------
def main():
    data = load_data()
    daily = engineer_daily_features(data)
    enriched = create_modeling_features(daily)
    all_outputs = []
    for region in REGIONS:
        result, models = run_regional_modeling(enriched, region, data['eia_changes'])
        if result is not None:
            result.to_csv(OUTPUT_DIR / f'weekly_forecast_{region}.csv', index=False)
            all_outputs.append(result)
    print("\n✅ All regions modeled and forecasts saved.")

if __name__ == '__main__':
    main()
