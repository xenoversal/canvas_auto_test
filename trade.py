import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Technical indicators
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# ML + Model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

###############################################################################
# 1) Data Loading & Feature Engineering

def load_all_stocks(data_folder):
    """
    Loads all CSVs in data_folder (each is a different stock).
    Assumes CSV columns:
      TimeStamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
    Then renames them to match the script's expectations:
      Date, Open, High, Low, Close, Volume
    """
    data_dict = {}
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "")
            path = os.path.join(data_folder, filename)
            
            # Attempt to specify dtypes to reduce memory usage (optional).
            dtypes = {
                'OpenPrice':  'float32',
                'HighPrice':  'float32',
                'LowPrice':   'float32',
                'ClosePrice': 'float32',
                'Volume':     'float32'
            }
            
            # parse_dates -> interpret the "TimeStamp" column as a datetime
            df = pd.read_csv(path, parse_dates=['Date'], dtype=dtypes)
            
            # Rename columns to what the script expects:
            df.rename(columns={
                'Date':  'Date',
                'OpenPrice':  'Open',
                'HighPrice':  'High',
                'LowPrice':   'Low',
                'ClosePrice': 'Close'
            }, inplace=True)
            
            # Sort by date ascending and reset index
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            data_dict[ticker] = df
    return data_dict

def add_technical_indicators(df, short_window=10, long_window=50):
    """
    Adds common technical indicators (SMA, EMA, RSI, Bollinger Bands)
    to the DataFrame.
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_Short'] = SMAIndicator(close=df['Close'], window=short_window).sma_indicator()
    df['SMA_Long']  = SMAIndicator(close=df['Close'], window=long_window).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_Short'] = EMAIndicator(close=df['Close'], window=short_window).ema_indicator()
    df['EMA_Long']  = EMAIndicator(close=df['Close'], window=long_window).ema_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low']  = bb_indicator.bollinger_lband()
    
    return df

def build_features_and_labels(df, forecast_horizon=1):
    """
    Builds features (X) and labels (y).
    Let's define 'label' as whether the Close price will go UP (1) or DOWN (0)
    after `forecast_horizon` days.
    """
    df = df.copy()
    
    # 1) Create target based on shift
    df['Future_Close'] = df['Close'].shift(-forecast_horizon)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)  # 1 if future close is higher, else 0
    
    # 2) Drop rows near the end that don't have future data
    df.dropna(subset=['Future_Close'], inplace=True)
    
    # 3) Potential feature columns
    feature_cols = [
        'SMA_Short', 'SMA_Long',
        'EMA_Short', 'EMA_Long', 
        'RSI', 'BB_High', 'BB_Low', 
        'Volume'
    ]
    # Add daily returns
    # Note: fill_method=None => no auto-fill for missing data
    df['Return_1D'] = df['Close'].pct_change(fill_method=None)
    feature_cols.append('Return_1D')
    
    # Build X, y
    X = df[feature_cols].copy()
    y = df['Target'].copy()
    
    # Forward/backward fill or handle missing values
    X = X.ffill().bfill()
    
    # Convert ±∞ to NaN, then drop them
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    
    # Align target with X
    y = y[X.index]
    
    return X, y, df

###############################################################################
# 2) Training + Time-Based Splitting

def train_and_evaluate_ticker(
    df,
    train_end_date='2020-01-01',
    forecast_horizon=1,
    param_grid=None,
    verbose=False
):
    """
    1) Adds technical indicators + builds features/labels.
    2) Splits data into train & test based on a date cutoff.
    3) Optionally does a GridSearchCV over param_grid.
    4) Trains + evaluates final model on test set.
    5) Runs a basic backtest on the test period.
    
    Returns: Dict with metrics & final DataFrame for the backtest.
    """
    # Add indicators
    df_ind = add_technical_indicators(df)
    # Build features
    X, y, df_feat = build_features_and_labels(df_ind, forecast_horizon=forecast_horizon)
    
    # We must ensure we have a Date column in df_feat
    df_feat['Date'] = df_ind.loc[df_feat.index, 'Date']
    
    # Time-based split
    cutoff = pd.to_datetime(train_end_date)
    df_feat.sort_values('Date', inplace=True)
    # We sort X + y to keep them aligned with df_feat's sorted index
    X = X.loc[df_feat.index]
    y = y.loc[df_feat.index]
    
    train_mask = df_feat['Date'] < cutoff
    test_mask  = df_feat['Date'] >= cutoff
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test   = X[test_mask], y[test_mask]
    df_test_feat     = df_feat[test_mask].copy()
    
    if len(X_train) < 30 or len(X_test) < 30:
        # Not enough data to train or test
        return {
            'status': 'Not enough data after date split',
            'ticker_df': df_test_feat
        }
    
    # Choose a default param_grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
        }
    
    # Use TimeSeriesSplit inside GridSearchCV for time-safe cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"Best params: {grid.best_params_}")
        print(f"Test Accuracy: {accuracy:.2f}")
    
    # Now do a simple backtest on the test set
    pred_probs = best_model.predict_proba(X_test)[:, 1]
    # If prob > 0.5 => go long
    df_test_feat['Signal'] = (pred_probs > 0.5).astype(int)
    
    # Shift the signal by 1 day to avoid lookahead bias
    df_test_feat['Position'] = df_test_feat['Signal'].shift(1, fill_value=0)
    
    # Calculate daily returns
    df_test_feat['Daily_Return'] = df_test_feat['Return_1D'] * df_test_feat['Position']
    
    # Strategy performance
    df_test_feat['Strategy_Cumulative'] = (1 + df_test_feat['Daily_Return']).cumprod()
    
    # Compare to buy and hold (starting from the first test day)
    df_test_feat['BuyHold_Cumulative'] = (1 + df_test_feat['Return_1D']).cumprod()
    
    final_strategy = df_test_feat['Strategy_Cumulative'].iloc[-1]
    final_buyhold  = df_test_feat['BuyHold_Cumulative'].iloc[-1]
    
    return {
        'status': 'OK',
        'best_params': grid.best_params_,
        'test_accuracy': accuracy,
        'final_strategy_multiplier': final_strategy,
        'final_buyhold_multiplier': final_buyhold,
        'ticker_df': df_test_feat
    }

###############################################################################
# 3) Main Workflow: Run for Each Ticker

if __name__ == "__main__":
    # 1) Load data from a folder
    data_folder = "stocks"  # <-- Update this to your folder containing AA.csv (and any others)
    data_dict = load_all_stocks(data_folder)
    
    # Date to end training. Everything after is "test."
    train_end_date = "2020-01-01"
    
    results_summary = []
    
    for ticker, df in data_dict.items():
        print(f"\n=== Processing {ticker} ===")
        # If your CSV has very little data, skip
        if len(df) < 300:
            print(f"[{ticker}] Not enough total data. Skipping.")
            continue
        
        # Train & evaluate
        res = train_and_evaluate_ticker(
            df,
            train_end_date=train_end_date,
            forecast_horizon=1,  # Next-day forecast
            param_grid={
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
            },
            verbose=True
        )
        
        if res['status'] == 'OK':
            print(f"[{ticker}] Accuracy={res['test_accuracy']:.2f}, "
                  f"Strategy={res['final_strategy_multiplier']:.2f}x, "
                  f"Buy&Hold={res['final_buyhold_multiplier']:.2f}x")
        else:
            print(f"[{ticker}] {res['status']}")
        
        results_summary.append({
            'Ticker': ticker,
            'Status': res['status'],
            'Best_Params': res.get('best_params'),
            'Test_Accuracy': res.get('test_accuracy'),
            'Strategy_Multiplier': res.get('final_strategy_multiplier'),
            'BuyHold_Multiplier': res.get('final_buyhold_multiplier')
        })
    
    # Show summary
    results_df = pd.DataFrame(results_summary)
    print("\n===== Summary Across Tickers =====")
    print(results_df)
