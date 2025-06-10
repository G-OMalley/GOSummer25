# GEMAgent_Enhanced.py
# An improved Q-Learning agent that uses advanced features for trading decisions.

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import warnings

# Suppress common warnings from pandas for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Global Setup and Enhanced Data Loading ---

def load_all_data():
    """
    Loads all necessary data files, including those for advanced features.
    This version includes robust column name cleaning and correct date handling.
    """
    try:
        base_path = './'
        info_path = os.path.join(base_path, 'INFO')

        paths = {
            'fom': os.path.join(info_path, 'HistoricalFOM.csv'),
            'prices': os.path.join(info_path, 'PRICES.csv'),
            'weather': os.path.join(info_path, 'WEATHER.csv'),
            'fundy': os.path.join(info_path, 'Fundy.csv'),
            'eia_totals': os.path.join(info_path, 'EIAtotals.csv'),
            'eia_changes': os.path.join(info_path, 'EIAchanges.csv'),
            'power_prices': os.path.join(info_path, 'PowerPrices.csv')
        }

        dataframes = {name: pd.read_csv(path) for name, path in paths.items()}
        
        # --- ROBUST COLUMN CLEANING AND DATE HANDLING (CORRECTED LOGIC) ---
        for name, df in dataframes.items():
            # Clean column names: strip whitespace, lowercase, replace spaces/dashes with underscores
            df.columns = (df.columns.str.strip().str.lower()
                          .str.replace(' ', '_')
                          .str.replace('-', '_'))
            dataframes[name] = df

        # Handle date conversions AFTER cleaning, using the NEW, clean column names
        for name in ['prices', 'weather', 'fundy', 'power_prices']:
            if 'date' in dataframes[name].columns:
                dataframes[name]['date'] = pd.to_datetime(dataframes[name]['date'])

        # Correctly identify cleaned column names and rename them to the standard 'date'
        if 'end_of_week' in dataframes['eia_totals'].columns:
            dataframes['eia_totals'] = dataframes['eia_totals'].rename(columns={'end_of_week': 'date'})
        if 'date' in dataframes['eia_totals'].columns:
             dataframes['eia_totals']['date'] = pd.to_datetime(dataframes['eia_totals']['date'])

        if 'week_ending' in dataframes['eia_changes'].columns:
            dataframes['eia_changes'] = dataframes['eia_changes'].rename(columns={'week_ending': 'date'})
        if 'date' in dataframes['eia_changes'].columns:
            dataframes['eia_changes']['date'] = pd.to_datetime(dataframes['eia_changes']['date'])
        
        print("All data files loaded successfully.")
        return dataframes
    
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a data file: {e.filename}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 2. Advanced Feature Engineering and Reward Functions ---

def get_actual_monthly_average_basis(prices_df, market_component, month, year):
    """Calculates the actual average basis for a given month from PRICES.csv."""
    if 'date' not in prices_df.columns: return np.nan
    target_prices = prices_df[(prices_df['date'].dt.month == month) & (prices_df['date'].dt.year == year)]
    
    # Clean the component name to match the cleaned column headers in the prices_df
    cleaned_component_name = (market_component.strip().lower()
                              .replace(' ', '_')
                              .replace('-', '_')
                              .replace('(', '')
                              .replace(')', ''))
                              
    if cleaned_component_name in target_prices.columns and not target_prices[cleaned_component_name].isnull().all():
        return target_prices[cleaned_component_name].mean()
    return np.nan

def get_advanced_features(data, trade_date):
    """Calculates a rich set of features for the agent's state for a given trade date."""
    
    target_dt = trade_date + pd.DateOffset(months=1)
    target_month, target_year = target_dt.month, target_dt.year
    
    # Weather Feature
    weather_df = data['weather']
    weather_period = weather_df[(weather_df['date'].dt.month == target_month) & (weather_df['date'].dt.year == target_year) & (weather_df['date'].dt.day <= 15)]
    weather_anomaly = (weather_period['hdd'] - weather_period['10yr_hdd']).sum() if 'hdd' in weather_period.columns and '10yr_hdd' in weather_period.columns and not weather_period.empty else 0

    # Storage Level Feature
    eia_totals_df = data['eia_totals']
    relevant_storage_series = eia_totals_df[eia_totals_df['date'] <= trade_date]
    storage_vs_5yr_avg = (relevant_storage_series.iloc[-1]['total'] - relevant_storage_series.iloc[-1]['5_yr_avg']) if not relevant_storage_series.empty else 0

    # Storage Momentum Feature
    eia_changes_df = data['eia_changes']
    recent_changes = eia_changes_df[eia_changes_df['date'] <= trade_date].tail(4)
    storage_momentum = recent_changes['total'].sum() if not recent_changes.empty else 0

    # Power Price Momentum Feature
    power_prices_df = data['power_prices']
    power_period = power_prices_df[(power_prices_df['date'] <= trade_date) & (power_prices_df['date'] > trade_date - pd.Timedelta(days=15))]
    power_momentum = power_period['ercot_north'].mean() if 'ercot_north' in power_period.columns and not power_period.empty else 0

    # Discretize features
    features = {
        'weather': int(np.round(weather_anomaly / 20)),
        'storage_level': int(np.round(storage_vs_5yr_avg / 50)),
        'storage_momentum': int(np.round(storage_momentum / 10)),
        'power_momentum': int(np.round(power_momentum / 5))
    }
    return tuple(features.values())

# --- 3. Training Function ---

def train_agent(data, train_dates):
    """Trains the Q-learning agent with advanced features."""
    print("\n--- Starting Agent Training ---")
    
    q_table = defaultdict(lambda: np.zeros(3)) # 0=buy, 1=sell, 2=hold
    params = {'lr': 0.1, 'gamma': 0.95, 'epsilon': 1.0, 'decay': 0.999, 'min_epsilon': 0.01, 'episodes': 1000}
    market_components = sorted(data['fom']['market_component'].unique())

    for episode in range(params['episodes']):
        if episode % 100 == 0:
            print(f"  Training Episode: {episode}/{params['episodes']}")

        for trade_date in train_dates:
            features = get_advanced_features(data, trade_date)
            target_dt = trade_date + pd.DateOffset(months=1)
            target_month, target_year = target_dt.month, target_dt.year

            for component in market_components:
                fom_row = data['fom'][(data['fom']['market_component'] == component) & (data['fom']['settlement_month_num'] == target_month) & (data['fom']['settlement_year'] == target_year)]
                if fom_row.empty: continue
                fom_basis = fom_row['settlement_basis'].iloc[0]
                
                state = (component, int(np.round(fom_basis * 10)),) + features

                action = np.random.choice([0, 1, 2]) if np.random.uniform(0, 1) < params['epsilon'] else np.argmax(q_table[state])
                
                actual_basis = get_actual_monthly_average_basis(data['prices'], component, target_month, target_year)
                if pd.isna(actual_basis): continue
                
                reward = (actual_basis - fom_basis) if action == 0 else (fom_basis - actual_basis) if action == 1 else 0
                
                old_q = q_table[state][action]
                new_q = old_q + params['lr'] * (reward - old_q)
                q_table[state][action] = new_q

        if params['epsilon'] > params['min_epsilon']:
            params['epsilon'] *= params['decay']
            
    print("--- Training Finished ---")
    return q_table

# --- 4. Evaluation and Inference ---

def evaluate_agent(q_table, data, test_dates):
    """Evaluates the agent on unseen test data."""
    print("\n--- Evaluating Agent Performance ---")
    
    monthly_pnl = []
    for trade_date in test_dates:
        features = get_advanced_features(data, trade_date)
        target_dt = trade_date + pd.DateOffset(months=1)
        target_month, target_year = target_dt.month, target_dt.year
        
        monthly_trades_pnl = 0
        for component in sorted(data['fom']['market_component'].unique()):
            fom_row = data['fom'][(data['fom']['market_component'] == component) & (data['fom']['settlement_month_num'] == target_month) & (data['fom']['settlement_year'] == target_year)]
            if fom_row.empty: continue
            fom_basis = fom_row['settlement_basis'].iloc[0]
            
            state = (component, int(np.round(fom_basis * 10)),) + features
            action = np.argmax(q_table[state]) if state in q_table else 2
            
            actual_basis = get_actual_monthly_average_basis(data['prices'], component, target_month, target_year)
            if pd.isna(actual_basis): continue
            
            reward = (actual_basis - fom_basis) if action == 0 else (fom_basis - actual_basis) if action == 1 else 0
            monthly_trades_pnl += reward
        
        monthly_pnl.append(monthly_trades_pnl)

    total_pnl = sum(monthly_pnl)
    sharpe = np.mean(monthly_pnl) / np.std(monthly_pnl) if np.std(monthly_pnl) > 0 else 0
    print(f"Total PnL on Test Data: ${total_pnl:,.2f}")
    print(f"Monthly Sharpe Ratio: {sharpe:.2f}")
    print("--- Evaluation Finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    data = load_all_data()

    if data:
        # Pre-process month names to numbers
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
        
        # --- CORRECTED LOGIC ---
        # Convert the actual data in the settlement_month column to lowercase before mapping
        data['fom']['settlement_month_num'] = data['fom']['settlement_month'].str.lower().map(month_map)

        # Define training and testing periods
        date_df = data['fom'][['settlement_year', 'settlement_month_num']].drop_duplicates().rename(columns={'settlement_year': 'year', 'settlement_month_num': 'month'})
        date_df.dropna(inplace=True) 
        date_df['day'] = 1
        
        all_contract_dates = sorted(pd.to_datetime(date_df).unique())
        trade_dates = [d - pd.DateOffset(months=1) + pd.offsets.MonthEnd(0) for d in all_contract_dates]
        
        if not trade_dates:
            print("\nERROR: Could not generate any tradeable dates from 'HistoricalFOM.csv'.")
            print("Please check the 'settlement_month' and 'settlement_year' columns in that file.")
        else:
            print(f"\nFound data for {len(trade_dates)} total tradeable months, from {trade_dates[0].date()} to {trade_dates[-1].date()}.")
            
            train_period_end = '2022-12-31'
            train_dates = [d for d in trade_dates if d <= pd.to_datetime(train_period_end)]
            
            max_price_date = data['prices']['date'].max()
            print(f"Latest price data available until: {max_price_date.date()}")
            
            potential_test_dates = [d for d in trade_dates if d > pd.to_datetime(train_period_end)]
            test_dates = []
            for d in potential_test_dates:
                contract_month_end = d + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
                if contract_month_end <= max_price_date:
                    test_dates.append(d)

            print(f"Created {len(train_dates)} training dates and {len(test_dates)} testing dates.")

            if not train_dates or not test_dates:
                 print("\nError: Not enough data to create a valid train/test split. Check data files and date ranges.")
            else:
                trained_q_table = train_agent(data, train_dates)
                evaluate_agent(trained_q_table, data, test_dates)