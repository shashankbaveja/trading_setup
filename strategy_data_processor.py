import warnings
import pandas as pd
from myKiteLib import kiteAPIs
from datetime import date, datetime # Added datetime for string conversion
import numpy as np # Added for np.nan
from ta.trend import SMAIndicator, ADXIndicator, CCIIndicator, MACD, EMAIndicator, AroonIndicator, aroon_up, aroon_down # Added EMAIndicator, AroonIndicator, aroon_up, aroon_down
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel # Added DonchianChannel
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator, ROCIndicator, KAMAIndicator # MACD removed from here
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, VolumeWeightedAveragePrice # Added VolumeWeightedAveragePrice
import os # Ensure os is imported
import argparse # <--- ADDED argparse

import sys
import os
# Add the project root (parent of model_trader) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from model_trader import config # Import the new config file
# fillna=False is generally preferred for ta library to inspect NaNs, will handle them later if needed.

# Suppress specific warnings for cleaner pipeline output during VIF debugging
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define parameters for data fetching
# token_to_fetch = 15792898
# Using date objects as provided by the user
# start_dt_obj = date(2025, 3, 1)
# end_dt_obj = date(2025, 5, 13) # Updated end_dt_val
# interval_val = 'minute'

# Convert date objects to string format 'YYYY-MM-DD' for the DB query function
# start_dt_str = start_dt_obj.strftime('%Y-%m-%d')
# end_dt_str = end_dt_obj.strftime('%Y-%m-%d')

# Constants from strategy.md
# SMA_PERIOD = 15 # No longer used for primary trigger
# TREND_THRESHOLD = 0.05  # No longer used for primary trigger
# REVERSAL_WINDOW = config.REVERSAL_WINDOW_LABEL    # Using from config
# REVERSAL_MAGNITUDE = config.REVERSAL_MAGNITUDE_LABEL # Using from config

# New fixed trigger definition
# MAX_TRIGGER_LOOKBACK_CANDLES = config.MAX_TRIGGER_LOOKBACK_CANDLES # Using from config
# TRIGGER_DROP_PCT = config.TRIGGER_DROP_PCT    # Using from config

# Define periods for new TA feature variations
EXTENDED_TA_PERIODS = [5, 7, 9, 10, 11, 15, 20, 30]

def fetch_data(token_to_process, start_date_obj, end_date_obj, interval_val='minute'):
    """
    Fetches historical data directly from the database using extract_data_from_db.
    Now accepts token, start_date, and end_date as parameters.
    """
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')

    print(f"Initializing Kite APIs for DB access...")
    k_apis = kiteAPIs()

    print(f"Fetching historical data from DB for token {token_to_process} from {start_date_str} to {end_date_str} for interval {interval_val}...")
    # Use the direct database extraction function
    historical_df = k_apis.extract_data_from_db(
        from_date=start_date_str,
        to_date=end_date_str,
        interval=interval_val,
        instrument_token=token_to_process
    )

    if historical_df is not None and not historical_df.empty:
        print("Successfully fetched data from DB. DataFrame head:")
        print(historical_df.head())
        
        if 'timestamp' in historical_df.columns:
            historical_df.rename(columns={'timestamp': 'date'}, inplace=True) 
            print("Renamed 'timestamp' column to 'date'.")
        
        if 'date' not in historical_df.columns:
            print("Critical Error: Neither 'date' nor 'timestamp' column found in DataFrame from DB.")
            return pd.DataFrame() # Return empty if no date column
            
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_numeric:
            if col in historical_df.columns:
                historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
            else:
                print(f"Warning: Expected column '{col}' not found for numeric conversion.")

        historical_df.sort_values(by=['instrument_token', 'date'], inplace=True)
        historical_df.drop_duplicates(subset=['instrument_token', 'date'], keep='first', inplace=True)
        historical_df.reset_index(drop=True, inplace=True)

        print("\nData fetching from DB and initial preparation complete.")
        return historical_df
    else:
        print("Failed to fetch data from DB or the DataFrame is empty.")
        return pd.DataFrame()

def calculate_sma(df, window):
    """Calculates SMA for the 'close' price using ta library."""
    print(f"Calculating {window}-period SMA using ta library...")
    
    # Helper function to apply SMAIndicator within a pandas groupby().transform()
    def sma_transformer(series_close):
        indicator = SMAIndicator(close=series_close, window=window, fillna=False)
        return indicator.sma_indicator()

    df[f'sma_{window}'] = df.groupby('instrument_token')['close'].transform(sma_transformer)
    return df

# def label_trends(df): # No longer using SMA-based trends for the primary trigger
#     """Labels trends based on strategy.md 3.1."""
#     print("Labeling trends...")
#     df['trend'] = 'NO_TREND'
#     sma_col = f'sma_{SMA_PERIOD}'
    
#     # Uptrend: C_t > SMA15_t * (1 + TREND_THRESHOLD)
#     df.loc[df['close'] > df[sma_col] * (1 + TREND_THRESHOLD), 'trend'] = 'UPTREND'
#     # Downtrend: C_t < SMA15_t * (1 - TREND_THRESHOLD)
#     df.loc[df['close'] < df[sma_col] * (1 - TREND_THRESHOLD), 'trend'] = 'DOWNTREND'
#     return df

def label_reversals_with_fixed_trigger(df):
    """
    Labels DOWNTREND_REVERSAL_SIGNAL based on a fixed trigger:
    A drop of > TRIGGER_DROP_PCT within any window from 1 to MAX_TRIGGER_LOOKBACK_CANDLES minutes,
    ending at t_eval-1. The reversal is then checked against REVERSAL_MAGNITUDE_LABEL within REVERSAL_WINDOW_LABEL from t_eval.
    """
    print(f"Labeling reversal signals based on flexible trigger: {config.TRIGGER_DROP_PCT*100}% drop in up to {config.MAX_TRIGGER_LOOKBACK_CANDLES} mins...")
    df['reversal_signal'] = 'NO_REVERSAL'

    # Iterate through the DataFrame.
    # `i` represents the candle for which we are trying to label a signal (t_eval).
    # The trigger condition is checked for a window ending at `i-1`.
    # The reversal check window is [i, i+config.REVERSAL_WINDOW_LABEL-1].
    # Need at least 1 for lookback (shortest trigger window) and config.REVERSAL_WINDOW_LABEL for lookahead.
    # Loop from `1` (min trigger window) to `len(df) - config.REVERSAL_WINDOW_LABEL`.
    
    for i in range(1, len(df) - config.REVERSAL_WINDOW_LABEL):
        trigger_confirmed = False
        trigger_eval_point_idx = i - 1 # This is t_eval - 1

        # Check for trigger condition over various lookback durations up to config.MAX_TRIGGER_LOOKBACK_CANDLES
        for lookback_duration_candles in range(1, config.MAX_TRIGGER_LOOKBACK_CANDLES + 1):
            if trigger_eval_point_idx - lookback_duration_candles < 0:
                continue # Not enough data for this lookback_duration
            
            start_of_lookback_idx = trigger_eval_point_idx - lookback_duration_candles
            price_at_trigger_eval_point = df['close'].iloc[trigger_eval_point_idx]
            price_at_start_of_lookback = df['close'].iloc[start_of_lookback_idx]

            if price_at_trigger_eval_point <= price_at_start_of_lookback * (1 - config.TRIGGER_DROP_PCT):
                trigger_confirmed = True
                break # Trigger found for this t_eval-1, no need to check shorter/longer windows for this same point
        
        if trigger_confirmed:
            # Trigger condition met. Now check for upward reversal from candle `i` (t_eval).
            # C_ref for reversal check is the price at the end of the trigger period (t_eval-1).
            c_ref_for_reversal = df['close'].iloc[trigger_eval_point_idx]
            
            # Look at high prices from t_eval (candle `i`) to t_eval + config.REVERSAL_WINDOW_LABEL - 1
            max_high_in_reversal_window = df['high'].iloc[i : i + config.REVERSAL_WINDOW_LABEL].max()
            
            if max_high_in_reversal_window >= c_ref_for_reversal * (1 + config.REVERSAL_MAGNITUDE_LABEL):
                df.loc[i, 'reversal_signal'] = 'DOWNTREND_REVERSAL_SIGNAL'
                
    # UPTREND_REVERSAL_SIGNAL logic is removed as per focus on downtrend reversals.
    print(f"Reversal signal distribution:\n{df['reversal_signal'].value_counts(dropna=False)}")
    return df

# def label_reversals(df): # This function is replaced by label_reversals_with_fixed_trigger
#     """Labels reversal signals based on strategy.md 3.2."""
#     print("Labeling reversal signals...")
#     df['reversal_signal'] = 'NO_REVERSAL'

#     # Shift trend and close data to get values for t_eval-1
#     df['prev_close'] = df.groupby('instrument_token')['close'].shift(1)
#     df['prev_trend'] = df.groupby('instrument_token')['trend'].shift(1)

#     # Iterate through the DataFrame to apply reversal logic
#     for i in range(len(df) - REVERSAL_WINDOW): 
#         if pd.isna(df['prev_trend'].iloc[i]) or pd.isna(df['prev_close'].iloc[i]):
#             continue

#         # Check for UPTREND at t_eval-1
#         if df['prev_trend'].iloc[i] == 'UPTREND':
#             min_low_in_window = df['low'].iloc[i : i + REVERSAL_WINDOW].min()
#             c_ref = df['prev_close'].iloc[i]
#             if min_low_in_window <= c_ref * (1 - REVERSAL_MAGNITUDE):
#                 df.loc[i, 'reversal_signal'] = 'UPTREND_REVERSAL_SIGNAL'
        
#         # Check for DOWNTREND at t_eval-1
#         elif df['prev_trend'].iloc[i] == 'DOWNTREND':
#             max_high_in_window = df['high'].iloc[i : i + REVERSAL_WINDOW].max()
#             c_ref = df['prev_close'].iloc[i]
#             if max_high_in_window >= c_ref * (1 + REVERSAL_MAGNITUDE):
#                 df.loc[i, 'reversal_signal'] = 'DOWNTREND_REVERSAL_SIGNAL'
                
#     df.drop(columns=['prev_close', 'prev_trend'], inplace=True)
#     return df

# --- New Feature Engineering Functions ---

def add_candlestick_features(df):
    print("Adding candlestick features...")
    df['candle_body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['price_range'] = df['high'] - df['low']
    return df

def add_cumulative_features(df):
    print("Adding cumulative features...")
    # Price Percentage Change
    for n in [5, 10]:
        df[f'price_pct_change_{n}min'] = df.groupby('instrument_token')['close'].transform(
            lambda x: x.pct_change(periods=n) * 100
        )
    # Volume Sum
    for n in [5, 10]:
        df[f'volume_sum_{n}min'] = df.groupby('instrument_token')['volume'].transform(
            lambda x: x.rolling(window=n, min_periods=1).sum()
        )
    return df

def add_lagged_features(df, feature_columns, lag_periods):
    print(f"Adding lagged features for columns: {feature_columns} with lags: {lag_periods}...")
    for col in feature_columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df.groupby('instrument_token')[col].shift(lag)
    return df

def add_all_ta_features(df):
    print("Adding original comprehensive TA features from 'ta' library...")
    
    # Ensure high, low, close, open, volume are present and numeric (already done in fetch_data)
    # Helper to apply indicators that need H, L, C, O, V
    # Most 'ta' library indicators operate on Series, so direct assignment is fine.
    # For groupby().transform(), a helper function is needed if applying per group,
    # but for single token processing, direct application is simpler.
    # Assuming single token context for now as per current usage. If multiple tokens were in df,
    # these would need to be wrapped in groupby().transform().

    # Volatility
    bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=False)
    df['bb_mavg'] = bb_indicator.bollinger_mavg()
    df['bb_hband'] = bb_indicator.bollinger_hband()
    df['bb_lband'] = bb_indicator.bollinger_lband()
    df['bb_pband'] = bb_indicator.bollinger_pband() # %B indicator
    df['bb_wband'] = bb_indicator.bollinger_wband() # Bandwidth

    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False)
    df['atr'] = atr_indicator.average_true_range()

    # Momentum
    df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=False).rsi()
    
    stoch_indicator = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3, fillna=False)
    df['stoch_k'] = stoch_indicator.stoch()
    df['stoch_d'] = stoch_indicator.stoch_signal()

    macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_diff'] = macd_indicator.macd_diff()

    df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14, fillna=False).williams_r()
    df['awesome_oscillator'] = AwesomeOscillatorIndicator(high=df['high'], low=df['low'], window1=5, window2=34, fillna=False).awesome_oscillator()
    df['roc'] = ROCIndicator(close=df['close'], window=8, fillna=False).roc()
    # df['kama'] = KAMAIndicator(close=df['close'], window=10, pow1=2, pow2=30, fillna=False).kama() # KAMA can be slow

    # Volume
    df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'], fillna=False).on_balance_volume()
    df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20, fillna=False).chaikin_money_flow()
    # df['eom'] = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'], window=14, fillna=False).ease_of_movement() # Needs high, low, volume
    df['vpt'] = VolumePriceTrendIndicator(close=df['close'], volume=df['volume'], fillna=False).volume_price_trend()


    # Trend
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False)
    df['adx'] = adx_indicator.adx()
    df['adx_pos'] = adx_indicator.adx_pos()
    df['adx_neg'] = adx_indicator.adx_neg()
    
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20, constant=0.015, fillna=False).cci()

    return df

def add_bollinger_bands_10_1(df):
    """Adds Bollinger Bands with 10-period window and 1 standard deviation."""
    print("Adding Bollinger Bands (10,1)...")
    bb_indicator_10_1 = BollingerBands(close=df['close'], window=10, window_dev=1, fillna=False)
    df['bb_mavg_10_1'] = bb_indicator_10_1.bollinger_mavg()
    df['bb_hband_10_1'] = bb_indicator_10_1.bollinger_hband()
    df['bb_lband_10_1'] = bb_indicator_10_1.bollinger_lband()
    df['bb_pband_10_1'] = bb_indicator_10_1.bollinger_pband()
    df['bb_wband_10_1'] = bb_indicator_10_1.bollinger_wband()
    return df

def add_bollinger_band_flags(df):
    """Adds flags based on bb_pband (from 20,2 BBs) conditions over the last 10 periods."""
    print("Adding Bollinger Band flags...")
    if 'bb_pband' not in df.columns:
        print("Warning: 'bb_pband' (from 20,2 BBs) not found. Cannot create BB flags.")
        return df

    # Flag if bb_pband > 0.9 anytime in the last 10 periods
    df['bb_pband_gt_0_9_flag_10p'] = df['bb_pband'].rolling(window=10, min_periods=1).apply(lambda x: (x > 0.9).any(), raw=True).fillna(0).astype(int)
    
    # Flag if bb_pband < 0.1 anytime in the last 10 periods
    df['bb_pband_lt_0_1_flag_10p'] = df['bb_pband'].rolling(window=10, min_periods=1).apply(lambda x: (x < 0.1).any(), raw=True).fillna(0).astype(int)
    return df

def add_roc_flags(df):
    """Adds flags based on 8-period ROC crossovers over the last 10 periods."""
    print("Adding ROC flags...")
    if 'roc' not in df.columns: # Should be 8-period ROC
        print("Warning: 'roc' (8-period) not found. Cannot create ROC flags.")
        return df

    roc_prev = df['roc'].shift(1)
    
    # Bullish crossover: ROC was < 0 and now is > 0
    bullish_cross_roc = (roc_prev < 0) & (df['roc'] > 0)
    df['roc_bullish_cross_zero_flag_10p'] = bullish_cross_roc.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    
    # Bearish crossover: ROC was > 0 and now is < 0
    bearish_cross_roc = (roc_prev > 0) & (df['roc'] < 0)
    df['roc_bearish_cross_zero_flag_10p'] = bearish_cross_roc.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_macd_flags(df, macd_col, signal_col, suffix=""):
    """Adds MACD crossover flags for a given MACD and signal line over the last 10 periods."""
    print(f"Adding MACD flags for {suffix}...")
    if macd_col not in df.columns or signal_col not in df.columns:
        print(f"Warning: MACD columns {macd_col} or {signal_col} not found for suffix {suffix}. Cannot create MACD flags.")
        return df

    macd_line_prev = df[macd_col].shift(1)
    signal_line_prev = df[signal_col].shift(1)

    positive_cross = (macd_line_prev < signal_line_prev) & (df[macd_col] > df[signal_col])
    df[f'macd_positive_cross_flag_10p{suffix}'] = positive_cross.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    negative_cross = (macd_line_prev > signal_line_prev) & (df[macd_col] < df[signal_col])
    df[f'macd_negative_cross_flag_10p{suffix}'] = negative_cross.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_stochastic_flags(df, k_col, d_col, suffix=""):
    """Adds Stochastic Oscillator buy/sell signal flags over the last 10 periods."""
    print(f"Adding Stochastic flags for {suffix}...")
    if k_col not in df.columns or d_col not in df.columns:
        print(f"Warning: Stochastic columns {k_col} or {d_col} not found for suffix {suffix}. Cannot create Stochastic flags.")
        return df

    k_curr = df[k_col]
    d_curr = df[d_col]
    k_prev = df[k_col].shift(1)
    d_prev = df[d_col].shift(1)

    buy_crossover = (k_prev < d_prev) & (k_curr > d_curr)
    buy_condition = buy_crossover & (k_prev < 20) & (d_prev < 20) # Check condition at t-1 for the state before crossover
    df[f'stoch_buy_signal_flag_10p{suffix}'] = buy_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    sell_crossover = (k_prev > d_prev) & (k_curr < d_curr)
    sell_condition = sell_crossover & (k_prev > 80) & (d_prev > 80) # Check condition at t-1
    df[f'stoch_sell_signal_flag_10p{suffix}'] = sell_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_willr_flags(df, wr_col, suffix=""):
    """Adds Williams %R buy/sell signal flags over the last 10 periods."""
    print(f"Adding Williams %R flags for {suffix}...")
    if wr_col not in df.columns:
        print(f"Warning: Williams %R column {wr_col} not found for suffix {suffix}. Cannot create Williams %R flags.")
        return df
        
    wr_curr = df[wr_col]
    wr_prev = df[wr_col].shift(1)

    buy_condition = (wr_prev < -80) & (wr_curr > -80) # Crossed back above -80
    df[f'wr_buy_signal_flag_10p{suffix}'] = buy_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    sell_condition = (wr_prev > -20) & (wr_curr < -20) # Crossed back below -20
    df[f'wr_sell_signal_flag_10p{suffix}'] = sell_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_ao_flags(df):
    """Adds Awesome Oscillator zero-line crossover flags over the last 10 periods."""
    print("Adding Awesome Oscillator flags...")
    if 'awesome_oscillator' not in df.columns:
        print("Warning: 'awesome_oscillator' not found. Cannot create AO flags.")
        return df

    ao_curr = df['awesome_oscillator']
    ao_prev = df['awesome_oscillator'].shift(1)

    bullish_cross = (ao_prev < 0) & (ao_curr > 0)
    df['ao_bullish_cross_zero_flag_10p'] = bullish_cross.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    bearish_cross = (ao_prev > 0) & (ao_curr < 0)
    df['ao_bearish_cross_zero_flag_10p'] = bearish_cross.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_cci_flags(df):
    """Adds Commodity Channel Index buy/sell signal flags over the last 10 periods."""
    print("Adding CCI flags...")
    if 'cci' not in df.columns:
        print("Warning: 'cci' not found. Cannot create CCI flags.")
        return df

    cci_curr = df['cci']
    cci_prev = df['cci'].shift(1)

    buy_condition = (cci_prev < -100) & (cci_curr > -100) # Crossed back above -100
    df['cci_buy_signal_flag_10p'] = buy_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    sell_condition = (cci_prev > 100) & (cci_curr < 100) # Crossed back below 100
    df['cci_sell_signal_flag_10p'] = sell_condition.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_adx_flags(df, adx_col, adx_pos_col, adx_neg_col, suffix=""):
    """Adds ADX trend and signal flags over the last 10 periods."""
    print(f"Adding ADX flags for {suffix}...")
    if not all(col in df.columns for col in [adx_col, adx_pos_col, adx_neg_col]):
        print(f"Warning: ADX columns ({adx_col}, {adx_pos_col}, {adx_neg_col}) not found for suffix {suffix}. Cannot create ADX flags.")
        return df

    adx = df[adx_col]
    plus_di = df[adx_pos_col]
    minus_di = df[adx_neg_col]
    plus_di_prev = df[adx_pos_col].shift(1)
    minus_di_prev = df[adx_neg_col].shift(1)

    # Trend flags
    bullish_trend_cond = plus_di > minus_di
    df[f'adx_bullish_trend_flag_10p{suffix}'] = bullish_trend_cond.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    
    bearish_trend_cond = minus_di > plus_di
    df[f'adx_bearish_trend_flag_10p{suffix}'] = bearish_trend_cond.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    # Crossover signal flags
    long_cross = (plus_di_prev < minus_di_prev) & (plus_di > minus_di)
    long_signal_cond = long_cross & (adx > 25)
    df[f'adx_long_signal_flag_10p{suffix}'] = long_signal_cond.rolling(window=10, min_periods=1).max().fillna(0).astype(int)

    short_cross = (minus_di_prev < plus_di_prev) & (minus_di > plus_di)
    short_signal_cond = short_cross & (adx > 25)
    df[f'adx_short_signal_flag_10p{suffix}'] = short_signal_cond.rolling(window=10, min_periods=1).max().fillna(0).astype(int)
    return df

def add_extended_ta_features(df, periods):
    """Adds new technical indicators with multiple period variations and Fibonacci features."""
    print(f"Adding extended TA features for periods: {periods}...")

    # RSI
    for p in periods:
        df[f'rsi_{p}'] = RSIIndicator(close=df['close'], window=p, fillna=False).rsi()

    # Stochastic Oscillator
    for p in periods:
        stoch_ind = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=p, smooth_window=3, fillna=False)
        df[f'stoch_k_{p}'] = stoch_ind.stoch()
        df[f'stoch_d_{p}'] = stoch_ind.stoch_signal()

    # ADX
    for p in periods:
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=p, fillna=False)
        df[f'adx_{p}'] = adx_ind.adx()
        df[f'adx_pos_{p}'] = adx_ind.adx_pos()
        df[f'adx_neg_{p}'] = adx_ind.adx_neg()

    # Aroon Indicator
    for p in periods:
        # User suggested modification
        aroon_ind = AroonIndicator(df['high'], df['low'], window=p, fillna=True)
        df[f'aroon_up_{p}'] = aroon_ind.aroon_up()
        df[f'aroon_down_{p}'] = aroon_ind.aroon_down()
        df[f'aroon_osc_{p}'] = aroon_ind.aroon_indicator() # This is Aroon Oscillator

    # EMA
    for p in periods:
        df[f'ema_{p}'] = EMAIndicator(close=df['close'], window=p, fillna=False).ema_indicator()

    # On-Balance Volume (OBV) - Standard OBV is already calculated in add_all_ta_features if 'obv' is desired there.
    # Let's ensure it's calculated once, and then add EMAs of OBV.
    # If not already present from add_all_ta_features, calculate standard OBV.
    if 'obv' not in df.columns: # Assuming 'obv' is the name from the original function
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'], fillna=False).on_balance_volume()
    
    # EMAs of OBV
    if 'obv' in df.columns: # Check if OBV exists to calculate its EMAs
        for p in periods:
            # EMAIndicator can take any series, so we pass df['obv']
            # Temporarily fill NaNs in OBV for EMA calculation if any, then revert or handle
            obv_series = df['obv'].fillna(method='bfill').fillna(method='ffill').fillna(0) # Robust fill for EMA
            df[f'obv_ema_{p}'] = EMAIndicator(close=obv_series, window=p, fillna=False).ema_indicator()
    else:
        print("Warning: OBV column not found, cannot calculate OBV EMAs.")


    # Donchian Channels
    for p in periods:
        dc_ind = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=p, offset=0, fillna=False)
        df[f'dc_hband_{p}'] = dc_ind.donchian_channel_hband()
        df[f'dc_lband_{p}'] = dc_ind.donchian_channel_lband()
        df[f'dc_mband_{p}'] = dc_ind.donchian_channel_mband()
        # df[f'dc_pband_{p}'] = dc_ind.donchian_channel_pband() # If needed
        # df[f'dc_wband_{p}'] = dc_ind.donchian_channel_wband() # If needed

    # Volume-Weighted Average Price (VWAP)
    for p in periods:
        df[f'vwap_{p}'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=p, fillna=False).volume_weighted_average_price()

    # Williams %R
    for p in periods:
        df[f'wr_{p}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=p, fillna=False).williams_r()
        # Add flags for this variant
        df = add_willr_flags(df, f'wr_{p}', suffix=f"_{p}")

    # MACD
    for p in periods:
        fast_period = p
        slow_period = max(p + 1, int(p * 2.0)) # Ensure slow > fast
        signal_period = max(1, int(p * 0.75))
        
        macd_ind = MACD(close=df['close'], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period, fillna=False)
        df[f'macd_{fast_period}_{slow_period}_{signal_period}'] = macd_ind.macd()
        df[f'macd_signal_{fast_period}_{slow_period}_{signal_period}'] = macd_ind.macd_signal()
        df[f'macd_diff_{fast_period}_{slow_period}_{signal_period}'] = macd_ind.macd_diff()
        # Add flags for this variant
        df = add_macd_flags(df, f'macd_{fast_period}_{slow_period}_{signal_period}', 
                            f'macd_signal_{fast_period}_{slow_period}_{signal_period}', 
                            suffix=f"_{fast_period}_{slow_period}_{signal_period}")

    # Stochastic Oscillator (Extended) - needs to be added if not already present in this function
    # For consistency, let's assume stoch_k_{p} and stoch_d_{p} are already calculated in this function
    for p in periods:
        # This loop is already present for stoch_k_p, stoch_d_p. We just add the flag call.
        if f'stoch_k_{p}' in df.columns and f'stoch_d_{p}' in df.columns:
             df = add_stochastic_flags(df, f'stoch_k_{p}', f'stoch_d_{p}', suffix=f"_{p}")
        else:
            # This part of stochastic calculation was already in add_extended_ta_features
            # stoch_ind = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=p, smooth_window=3, fillna=False)
            # df[f'stoch_k_{p}'] = stoch_ind.stoch()
            # df[f'stoch_d_{p}'] = stoch_ind.stoch_signal()
            # df = add_stochastic_flags(df, f'stoch_k_{p}', f'stoch_d_{p}', suffix=f"_{p}")
            pass # Assuming stoch_k_p and stoch_d_p are calculated earlier in this function as per existing code

    # ADX (Extended)
    for p in periods:
        # This loop is already present for adx_p, adx_pos_p, adx_neg_p. We just add the flag call.
        if f'adx_{p}' in df.columns and f'adx_pos_{p}' in df.columns and f'adx_neg_{p}' in df.columns:
            df = add_adx_flags(df, f'adx_{p}', f'adx_pos_{p}', f'adx_neg_{p}', suffix=f"_{p}")
        else:
            # This part of ADX calculation was already in add_extended_ta_features
            # adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=p, fillna=False)
            # df[f'adx_{p}'] = adx_ind.adx()
            # df[f'adx_pos_{p}'] = adx_ind.adx_pos()
            # df[f'adx_neg_{p}'] = adx_ind.adx_neg()
            # df = add_adx_flags(df, f'adx_{p}', f'adx_pos_{p}', f'adx_neg_{p}', suffix=f"_{p}")
            pass # Assuming these are calculated earlier in this function

    # Fibonacci Retracement Features (Simplified Rolling Window Approach)
    print("Adding Fibonacci-based features...")
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
    for lookback_p in periods:
        swing_high_p = df['high'].rolling(window=lookback_p, min_periods=max(1,int(lookback_p*0.5))).max() # Ensure some min_periods
        swing_low_p = df['low'].rolling(window=lookback_p, min_periods=max(1,int(lookback_p*0.5))).min()
        swing_range_p = swing_high_p - swing_low_p
        
        # Avoid division by zero if range is 0 (e.g., flat price for the lookback period)
        # Use a small epsilon or replace 0 with NaN then fill
        swing_range_p_safe = swing_range_p.replace(0, np.nan)

        for f_level in fib_levels:
            f_level_str = str(f_level).replace('.','') # e.g. 0236

            # Up-swing assumption (price retracing down from swing_high_p)
            fib_level_val_up = swing_high_p - f_level * swing_range_p
            df[f'close_dist_norm_fib_up_{f_level_str}_{lookback_p}'] = (df['close'] - fib_level_val_up) / swing_range_p_safe
            df[f'close_above_fib_up_{f_level_str}_{lookback_p}'] = (df['close'] > fib_level_val_up).astype(int)
            # Replace NaNs that arose from swing_range_p_safe being NaN (where swing_range_p was 0)
            # For binary features, if range is 0, it might mean close is at the level or indeterminate, default to 0 (false)
            df[f'close_dist_norm_fib_up_{f_level_str}_{lookback_p}'].fillna(0, inplace=True)
            df[f'close_above_fib_up_{f_level_str}_{lookback_p}'].where(swing_range_p > 0, 0, inplace=True)


            # Down-swing assumption (price retracing up from swing_low_p)
            fib_level_val_down = swing_low_p + f_level * swing_range_p
            df[f'close_dist_norm_fib_down_{f_level_str}_{lookback_p}'] = (df['close'] - fib_level_val_down) / swing_range_p_safe
            df[f'close_below_fib_down_{f_level_str}_{lookback_p}'] = (df['close'] < fib_level_val_down).astype(int)
            # Fill NaNs for these too
            df[f'close_dist_norm_fib_down_{f_level_str}_{lookback_p}'].fillna(0, inplace=True)
            df[f'close_below_fib_down_{f_level_str}_{lookback_p}'].where(swing_range_p > 0, 0, inplace=True)
            
    return df

def add_all_features(df):
    """Main function to add all types of features."""
    df = add_candlestick_features(df)
    df = add_cumulative_features(df)
    df = add_all_ta_features(df) # Adds the original broad range of TA features
    df = add_bollinger_bands_10_1(df) # Add new BB (10,1)
    df = add_extended_ta_features(df, EXTENDED_TA_PERIODS) # Adds new extended TA features AND THEIR FLAGS

    # Add flags for base indicators from add_all_ta_features
    df = add_bollinger_band_flags(df) # Uses bb_pband from (20,2)
    df = add_roc_flags(df)           # Uses 8-period roc
    df = add_macd_flags(df, 'macd', 'macd_signal', suffix="_default") # For default MACD
    df = add_stochastic_flags(df, 'stoch_k', 'stoch_d', suffix="_default") # For default Stochastic
    df = add_willr_flags(df, 'williams_r', suffix="_default") # For default Williams %R
    df = add_ao_flags(df)
    df = add_cci_flags(df)
    df = add_adx_flags(df, 'adx', 'adx_pos', 'adx_neg', suffix="_default") # For default ADX
    
    # Define features for lagging (Example - can be expanded or moved)
    # This section might need review based on the final feature set.
    # For now, commenting out to avoid issues with new feature names.
    # features_to_lag = ['close', 'rsi', 'macd', 'volume_sum_5min'] 
    # lag_periods_list = [1, 2, 3, 5]
    # df = add_lagged_features(df, features_to_lag, lag_periods_list)

    # Handle NaNs. Current strategy is fill with 0. This might be aggressive.
    # Consider how NaNs from various window sizes should be handled.
    # For example, very short periods might produce NaNs at the start for longer period indicators.
    # Filling with 0 might distort initial data points.
    # df.fillna(0, inplace=True) # Moved to the end of main processing for now.
    # print(f"Shape after adding all features (before NaN handling specific to this func): {df.shape}")
    
    return df

# --- End of New Feature Engineering Functions ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Data Processor") # <--- ADDED
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files.") # <--- ADDED
    args = parser.parse_args() # <--- ADDED

    # Define global parameters at the top of the script or pass them around.
    # For simplicity here, they are global within the script context.
    # token_to_fetch = 15792898
    # start_dt_obj = date(2025, 3, 1)
    # end_dt_obj = date(2025, 5, 13)
    # interval_val = 'minute'
    # start_dt_str = start_dt_obj.strftime('%Y-%m-%d')
    # end_dt_str = end_dt_obj.strftime('%Y-%m-%d')
    # SMA_PERIOD = 15 # No longer used

    ALL_RUN_MODES = ["INDEX_TRAINING", "INDEX_SIMULATION_PERIOD", "OPTION_SIMULATION_OHLCV"]

    for RUN_MODE in ALL_RUN_MODES:
        print(f"\n================== Starting Run Mode: {RUN_MODE} ==================")
        # --- Configuration for current run --- 
        # This section will be manually changed or ideally set by command-line args in future
        # Example: RUN_MODE = "INDEX_TRAINING"
        # RUN_MODE = "INDEX_SIMULATION_PERIOD"
        # RUN_MODE = "OPTION_SIMULATION_OHLCV" # Change this to generate different files

        if RUN_MODE == "INDEX_TRAINING":
            print("--- Running in INDEX_TRAINING mode ---")
            current_token_to_fetch = config.TRAINING_TOKEN
            current_start_dt_obj = config.TRAINING_START_DATE
            current_end_dt_obj = config.TRAINING_END_DATE
            output_file_label = "INDEX_TRAINING"
            process_features_and_labels_flag = True
            output_filename_base = f"featured_data_TOKEN_{current_token_to_fetch}_{output_file_label}_{current_start_dt_obj}_to_{current_end_dt_obj}.csv"
        elif RUN_MODE == "INDEX_SIMULATION_PERIOD":
            print("--- Running in INDEX_SIMULATION_PERIOD mode ---")
            current_token_to_fetch = config.TRAINING_TOKEN # Still use Index token
            current_start_dt_obj = config.TRADING_SIMULATION_START_DATE # But for simulation dates
            current_end_dt_obj = config.TRADING_SIMULATION_END_DATE
            output_file_label = "INDEX_SIMULATION_PERIOD"
            process_features_and_labels_flag = True # Yes, we need features to predict on
            output_filename_base = f"featured_data_TOKEN_{current_token_to_fetch}_{output_file_label}_{current_start_dt_obj}_to_{current_end_dt_obj}.csv"
        elif RUN_MODE == "OPTION_SIMULATION_OHLCV":
            print("--- Running in OPTION_SIMULATION_OHLCV mode ---")
            current_token_to_fetch = config.TRADING_SIMULATION_TOKEN # Option token
            current_start_dt_obj = config.TRADING_SIMULATION_START_DATE # Simulation dates
            current_end_dt_obj = config.TRADING_SIMULATION_END_DATE
            output_file_label = "OPTION_SIMULATION_OHLCV"
            process_features_and_labels_flag = False # No, just raw OHLCV (or minimal processing if needed for simulator)
            output_filename_base = f"ohlcv_data_TOKEN_{current_token_to_fetch}_{output_file_label}_{current_start_dt_obj}_to_{current_end_dt_obj}.csv"
        else:
            print(f"Error: Unknown RUN_MODE: {RUN_MODE}. Exiting.")
            sys.exit(1)
        
        # Determine the output directory to use # <--- MODIFIED BLOCK
        if args.output_dir:
            output_dir_to_use = args.output_dir
            print(f"DEBUG STRATEGY_PROCESSOR: Using provided --output_dir: {os.path.abspath(output_dir_to_use)}")
        else:
            output_dir_to_use = "cursor_logs" # Default if not running from sensitivity script
            print(f"DEBUG STRATEGY_PROCESSOR: Using default output_dir (relative to script CWD): {output_dir_to_use}")

        os.makedirs(output_dir_to_use, exist_ok=True)

        interval_val = 'minute' # Assuming always minute data for now

        processed_data = fetch_data(current_token_to_fetch, current_start_dt_obj, current_end_dt_obj, interval_val)
        
        if not processed_data.empty:
            if process_features_and_labels_flag:
                print("Processing features and labels...")
                # 1. Calculate SMA15 (used for initial trend definition) - NO LONGER PRIMARY
                # processed_data = calculate_sma(processed_data, SMA_PERIOD) 
                # Drop rows where initial SMA is NaN before labeling trends
                # processed_data.dropna(subset=[f'sma_{SMA_PERIOD}'], inplace=True) 
                # processed_data.reset_index(drop=True, inplace=True)
                
                # 2. Implement Trend Labeling - NO LONGER PRIMARY
                # processed_data = label_trends(processed_data)

                # 3. Implement Reversal Signal Labeling with new fixed trigger
                processed_data = label_reversals_with_fixed_trigger(processed_data)
                
                # --- Add other features ---
                print("\nStarting feature engineering...")
                processed_data = add_all_features(processed_data) # New main call to add features
                print("Feature engineering complete.")

                # --- Post-feature engineering steps ---
                nan_counts_before_fill = processed_data.isnull().sum()
                print("\nNaN counts per column before filling:")
                print(nan_counts_before_fill[nan_counts_before_fill > 0])

                processed_data.fillna(0, inplace=True) # Simple fill with 0 strategy

                print("\nNaN counts per column after filling with 0:")
                nan_counts_after_fill = processed_data.isnull().sum()
                print(nan_counts_after_fill[nan_counts_after_fill > 0])
                if nan_counts_after_fill.sum() == 0:
                    print("All NaNs successfully filled with 0.")

                # 4. Output descriptive statistics of labels
                print("\n--- Label Statistics (after feature engineering) ---")
                print("\nReversal Signal Labels Distribution:")
                print(processed_data['reversal_signal'].value_counts(dropna=False))
                
                target_map = {
                    'NO_REVERSAL': 0,
                    'DOWNTREND_REVERSAL_SIGNAL': 1 
                }
                processed_data['target_numerical'] = processed_data['reversal_signal'].map(target_map)
                processed_data['target_numerical'].fillna(0, inplace=True) 

                # --- Feature-Target Correlation Analysis (only if features were processed) ---
                if 'target_numerical' in processed_data.columns:
                    print("\nCalculating feature correlation with target ('target_numerical')...")
                    base_exclude_cols = ['id', 'instrument_token', 'date', 'trend', 'reversal_signal', 'target_numerical']
                    potential_feature_cols = [col for col in processed_data.columns if col not in base_exclude_cols]
                    correlations = {}
                    for col in potential_feature_cols:
                        if pd.api.types.is_numeric_dtype(processed_data[col]):
                            if processed_data[col].nunique() > 1:
                                try:
                                    corr_val = processed_data[col].corr(processed_data['target_numerical'])
                                    if pd.notna(corr_val):
                                        correlations[col] = corr_val
                                    else:
                                        print(f"Warning: Correlation for feature '{col}' resulted in NaN.")
                                except Exception as e:
                                    print(f"Warning: Could not calculate correlation for feature '{col}': {e}")
                            else:
                                print(f"Warning: Feature '{col}' is constant. Skipping correlation.")
                        else:
                            print(f"Warning: Feature '{col}' is not numeric. Skipping correlation.")

                    if correlations:
                        sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
                        print("\nTop features correlated with 'target_numerical' (absolute value, descending):")
                        for feature, corr_value in sorted_correlations:
                            print(f"  {feature}: {corr_value:.4f}")
                    else:
                        print("No valid feature-target correlations could be calculated.")
                else:
                    print("Warning: 'target_numerical' column not found. Cannot calculate feature-target correlations.")
            else: # if process_features_and_labels_flag is False (e.g., for OPTION_SIMULATION_OHLCV)
                print("Skipping feature engineering and label processing for this run mode.")

            # output_dir = "cursor_logs" # HARDCODED! # <--- REMOVED THIS LINE
            # os.makedirs(output_dir, exist_ok=True) # <--- REMOVED THIS LINE (covered by output_dir_to_use)
            
            # Save label count summary only if labels were processed
            if process_features_and_labels_flag and 'reversal_signal' in processed_data.columns:
                label_count_file = os.path.join(output_dir_to_use, f"summary_label_count_{output_file_label}.txt") # <--- MODIFIED
                try:
                    with open(label_count_file, 'w') as f:
                        signal_counts = processed_data['reversal_signal'].value_counts()
                        downtrend_reversals = signal_counts.get('DOWNTREND_REVERSAL_SIGNAL', 0)
                        f.write(f"DOWNTREND_REVERSAL_SIGNAL:{downtrend_reversals}")
                    print(f"Label count saved to {label_count_file}")
                except Exception as e:
                    print(f"Error saving label count: {e}")

            # output_filename_base = f"featured_data_token_{token_to_fetch}_{start_dt_obj}_to_{end_dt_obj}_RW{REVERSAL_WINDOW_LABEL}_RM{int(REVERSAL_MAGNITUDE_LABEL*100)}.csv"
            # Use fixed filename based on RUN_MODE defined earlier
            output_filename = os.path.join(output_dir_to_use, output_filename_base) # <--- MODIFIED
            
            print(f"\nSaving processed data to {os.path.abspath(output_filename)}...") # <--- MODIFIED (added abspath)
            processed_data.to_csv(output_filename, index=False)
            print("Processing complete.")
            
            print("\nFinal DataFrame with features head:")
            print(processed_data.head())
            print("\nFinal DataFrame with features tail:")
            print(processed_data.tail())
            print(f"\nShape of final DataFrame with features: {processed_data.shape}")
            print("\nColumns in final DataFrame:")
            print(processed_data.columns.tolist())

            # --- End of Phase 2.A ---

        else:
            print("No data to process.") 
        print(f"================== Finished Run Mode: {RUN_MODE} ==================") 