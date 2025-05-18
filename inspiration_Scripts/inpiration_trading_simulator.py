import pandas as pd
import numpy as np
import argparse
import os
from datetime import timedelta

import sys
# Add the project root (parent of model_trader) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model_trader import config # Import the new config file

# --- Default Trade Parameters (sourced from config.py) ---
DEFAULT_TRADE_UNITS = config.SIM_DEFAULT_TRADE_UNITS
DEFAULT_PROFIT_TARGET_PCT = config.SIM_DEFAULT_PROFIT_TARGET_PCT
DEFAULT_STOP_LOSS_PCT = config.SIM_DEFAULT_STOP_LOSS_PCT
DEFAULT_MAX_HOLDING_PERIOD_MINUTES = config.SIM_DEFAULT_MAX_HOLDING_PERIOD_MINUTES

# New fixed output filenames for the simulator
SIMULATION_TRADES_LOG_FILENAME = "simulation_trades_output.csv"
SIMULATION_SUMMARY_FILENAME = "simulation_summary.txt"

def load_ohlcv_data(filepath):
    """Loads OHLCV data from a CSV file."""
    print(f"Loading OHLCV data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: OHLCV data file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if 'date' not in df.columns:
            print("Error: 'date' column not found in OHLCV data.")
            return None
        df['date'] = pd.to_datetime(df['date'])
        # Ensure required columns are present
        required_cols = ['date', 'open', 'high', 'low', 'close', 'instrument_token']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: OHLCV data must contain columns: {required_cols}")
            return None
        print(f"OHLCV data loaded successfully. Shape: {df.shape}")
        return df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"Error loading OHLCV data: {e}")
        return None

def load_signals_data(filepath):
    """
    Loads signals data from a CSV file.
    Expected columns: 'date', 'predicted_signal' (0: No Signal, 1: BUY), 'instrument_token' (optional for now)
    """
    print(f"Loading signals data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Signals data file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        # Ensure 'date' and the correct signal column ('predicted_signal') are present
        if 'date' not in df.columns or 'predicted_signal' not in df.columns: # Changed 'signal' to 'predicted_signal'
            print("Error: Signals data must contain 'date' and 'predicted_signal' columns.") # Updated error message
            return None
        df['date'] = pd.to_datetime(df['date'])
        # Rename 'predicted_signal' to 'signal' internally for the rest of the simulator's logic
        df.rename(columns={'predicted_signal': 'signal'}, inplace=True)
        print(f"Signals data loaded successfully. Shape: {df.shape}. Renamed 'predicted_signal' to 'signal' internally.")
        return df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"Error loading signals data: {e}")
        return None

def run_simulation(ohlcv_df, signals_df, trade_params):
    """
    Simulates trades based on signals and OHLCV data.
    
    Args:
        ohlcv_df (pd.DataFrame): DataFrame with OHLCV data, indexed by datetime.
        signals_df (pd.DataFrame): DataFrame with trade signals (date, signal type).
        trade_params (dict): Dictionary containing trading parameters.
                             Keys: 'units', 'profit_target_pct', 'stop_loss_pct', 
                                   'max_holding_period_minutes'.
    
    Returns:
        pd.DataFrame: DataFrame containing details of all executed trades.
    """
    print("\nStarting trade simulation...")
    
    if ohlcv_df is None or signals_df is None:
        print("Error: OHLCV data or signals data is missing. Cannot run simulation.")
        return pd.DataFrame()

    # Ensure data is sorted by date
    ohlcv_df = ohlcv_df.sort_values(by='date').reset_index(drop=True)
    signals_df = signals_df.sort_values(by='date').reset_index(drop=True)

    executed_trades = []
    
    # For faster lookups, set date as index for ohlcv_df if not already
    # ohlcv_df_indexed = ohlcv_df.set_index('date')

    for idx, signal_row in signals_df.iterrows():
        signal_time = signal_row['date']
        signal_type = signal_row['signal'] # This will now use the renamed column

        if signal_type not in [1]: # Process only BUY (1) signals from the model (as it predicts DOWNTREND_REVERSAL for BUY)
            continue

        # Find the candle in OHLCV data that corresponds to the signal time
        # This assumes 1-minute data. The signal is for candle t. We enter at open of candle t.
        entry_candle_index = ohlcv_df[ohlcv_df['date'] == signal_time].index
        
        if entry_candle_index.empty:
            # print(f"Warning: No OHLCV data found for signal at {signal_time}. Skipping trade.")
            continue
        
        entry_candle_index = entry_candle_index[0] # Get the first match
        entry_price = ohlcv_df.loc[entry_candle_index, 'open']
        
        # Define exit targets based on entry price and signal type
        if signal_type == 1: # BUY
            profit_target_price = entry_price * (1 + trade_params['profit_target_pct'])
            stop_loss_price = entry_price * (1 - trade_params['stop_loss_pct'])
        elif signal_type == 2: # SELL
            profit_target_price = entry_price * (1 - trade_params['profit_target_pct'])
            stop_loss_price = entry_price * (1 + trade_params['stop_loss_pct'])
        else:
            continue # Should not happen if filtered above

        exit_price = None
        exit_time = None
        exit_reason = None
        
        # Simulate holding period (e.g., next 8 candles)
        # The entry candle is candle 0 for the holding period.
        # We look from entry_candle_index up to entry_candle_index + max_holding_period_minutes -1
        max_holding_candles = trade_params['max_holding_period_minutes']
        
        for i in range(max_holding_candles):
            current_candle_idx = entry_candle_index + i
            if current_candle_idx >= len(ohlcv_df):
                # Ran out of data before max holding period
                exit_reason = "End of Data"
                # If no exit yet, use last available close
                if i > 0: # Must have held for at least one candle past entry
                    exit_price = ohlcv_df.loc[entry_candle_index + i -1, 'close']
                    exit_time = ohlcv_df.loc[entry_candle_index + i -1, 'date']
                break 

            current_high = ohlcv_df.loc[current_candle_idx, 'high']
            current_low = ohlcv_df.loc[current_candle_idx, 'low']
            current_close = ohlcv_df.loc[current_candle_idx, 'close']
            current_candle_time = ohlcv_df.loc[current_candle_idx, 'date']

            if signal_type == 1: # BUY trade active
                # Check for profit target (high of current candle)
                if current_high >= profit_target_price:
                    exit_price = profit_target_price # Assume execution at target
                    exit_time = current_candle_time
                    exit_reason = "Profit Target"
                    break
                # Check for stop loss (low of current candle)
                if current_low <= stop_loss_price:
                    exit_price = stop_loss_price # Assume execution at stop
                    exit_time = current_candle_time
                    exit_reason = "Stop Loss"
                    break
            
            elif signal_type == 2: # SELL trade active
                # Check for profit target (low of current candle)
                if current_low <= profit_target_price:
                    exit_price = profit_target_price # Assume execution at target
                    exit_time = current_candle_time
                    exit_reason = "Profit Target"
                    break
                # Check for stop loss (high of current candle)
                if current_high >= stop_loss_price:
                    exit_price = stop_loss_price # Assume execution at stop
                    exit_time = current_candle_time
                    exit_reason = "Stop Loss"
                    break
            
            # If it's the last candle of the holding period
            if i == max_holding_candles - 1:
                exit_price = current_close # Exit at close of the last holding candle
                exit_time = current_candle_time
                exit_reason = "Max Hold Time"
                break
        
        if exit_price is not None:
            pnl_per_unit = 0
            if signal_type == 1: # BUY
                pnl_per_unit = exit_price - entry_price
            elif signal_type == 2: # SELL
                pnl_per_unit = entry_price - exit_price
            
            total_pnl = pnl_per_unit * trade_params['units']
            
            executed_trades.append({
                'entry_time': ohlcv_df.loc[entry_candle_index, 'date'],
                'entry_price': entry_price,
                'signal_type': 'BUY' if signal_type == 1 else 'SELL',
                'exit_time': exit_time,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_per_unit': pnl_per_unit,
                'total_pnl': total_pnl,
                'units': trade_params['units']
            })

    print(f"Simulation complete. Total trades considered based on signals: {len(signals_df[signals_df['signal'].isin([1])])}") # Adjusted for only BUY signal type 1
    return pd.DataFrame(executed_trades)

def calculate_performance_metrics(trades_df, initial_capital=100000): # Assuming initial capital for drawdown %
    """Calculates trading performance metrics and returns them as a dictionary and a string."""
    print("\n--- Trading Performance Metrics ---")
    metrics_summary_dict = {}
    
    if trades_df.empty:
        print("No trades were executed. Cannot calculate metrics.")
        metrics_summary_dict["message"] = "No trades were executed."
        return metrics_summary_dict, "No trades were executed."

    num_total_trades = len(trades_df)
    trades_df['pnl_per_unit'] = pd.to_numeric(trades_df['pnl_per_unit'], errors='coerce')
    trades_df['total_pnl'] = pd.to_numeric(trades_df['total_pnl'], errors='coerce')
    
    winning_trades = trades_df[trades_df['total_pnl'] > 0]
    losing_trades = trades_df[trades_df['total_pnl'] < 0]
    
    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)
    
    win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0
    
    total_profit_loss = trades_df['total_pnl'].sum()
    average_pnl_per_trade = trades_df['total_pnl'].mean() if num_total_trades > 0 else 0
    
    gross_profit = winning_trades['total_pnl'].sum()
    gross_loss = abs(losing_trades['total_pnl'].sum()) # abs because losses are negative
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max Drawdown Calculation
    trades_df['cumulative_pnl'] = trades_df['total_pnl'].cumsum()
    trades_df['peak_pnl'] = trades_df['cumulative_pnl'].cummax()
    trades_df['drawdown'] = trades_df['peak_pnl'] - trades_df['cumulative_pnl']
    max_drawdown_value = trades_df['drawdown'].max()
    
    # Store metrics in dictionary
    metrics_summary_dict = {
        "Total Trades Executed": num_total_trades,
        "Winning Trades": num_winning_trades,
        "Losing Trades": num_losing_trades,
        "Win Rate (%)": f"{win_rate:.2f}",
        "Total Profit/Loss": f"{total_profit_loss:.2f}",
        "Average Profit/Loss per Trade": f"{average_pnl_per_trade:.2f}",
        "Gross Profit": f"{gross_profit:.2f}",
        "Gross Loss": f"{gross_loss:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Maximum Drawdown (Value)": f"{max_drawdown_value:.2f}"
    }

    # Print metrics (as before)
    print(f"Total Trades Executed: {num_total_trades}")
    print(f"Winning Trades: {num_winning_trades}")
    print(f"Losing Trades: {num_losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit_loss:.2f}")
    print(f"Average Profit/Loss per Trade: {average_pnl_per_trade:.2f}")
    print(f"Gross Profit (from winning trades): {gross_profit:.2f}")
    print(f"Gross Loss (from losing trades): {gross_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown (Value): {max_drawdown_value:.2f}")

    # Format summary string for saving to text file
    summary_str_lines = ["--- Trading Performance Metrics ---"]
    for key, value in metrics_summary_dict.items():
        summary_str_lines.append(f"{key}: {value}")
    
    return metrics_summary_dict, "\n".join(summary_str_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade simulator for evaluating trading strategies.")
    # parser.add_argument("--ohlcv_file", type=str, required=True, help="Path to the OHLCV data CSV file.")
    # parser.add_argument("--signals_file", type=str, required=True, help="Path to the signals CSV file.")
    
    # Arguments will now be less direct, filenames constructed based on config
    # We can add specific overrides if needed later

    parser.add_argument("--units", type=int, default=DEFAULT_TRADE_UNITS, help="Number of units per trade.")
    parser.add_argument("--target_pct", type=float, default=DEFAULT_PROFIT_TARGET_PCT, help="Profit target percentage (e.g., 0.05 for 5%).")
    parser.add_argument("--sl_pct", type=float, default=DEFAULT_STOP_LOSS_PCT, help="Stop loss percentage (e.g., 0.025 for 2.5%).")
    parser.add_argument("--hold_min", type=int, default=DEFAULT_MAX_HOLDING_PERIOD_MINUTES, help="Max holding period in minutes.")
    parser.add_argument("--output_dir", type=str, default="cursor_logs", help="Directory to save simulation output files.")
    parser.add_argument("--model-name", type=str, default="LOGISTIC", help="Name of the model (e.g., LOGISTIC, XGBOOST) to determine signals file.")
    parser.add_argument("--summary_filename_suffix", type=str, default="", help="Suffix to append to the summary filename (e.g., '_run1').")

    args = parser.parse_args()

    # --- Construct input filenames based on config.py --- 
    option_ohlcv_filename = f"ohlcv_data_TOKEN_{config.TRADING_SIMULATION_TOKEN}_OPTION_SIMULATION_OHLCV_{config.TRADING_SIMULATION_START_DATE}_to_{config.TRADING_SIMULATION_END_DATE}.csv"
    option_ohlcv_filepath = os.path.join(args.output_dir, option_ohlcv_filename) # Assuming signals are also in output_dir which is cursor_logs

    # Construct signals filename based on model_name
    model_prefix = ""
    if args.model_name.upper() == "XGBOOST":
        model_prefix = "XGBOOST_" # Ensure there's an underscore if the prefix exists
    elif args.model_name.upper() == "LOGISTIC":
        model_prefix = "" # Original filename structure for Logistic Regression
    else:
        # Potentially handle other model names or raise an error for unsupported names
        print(f"Warning: Model name '{args.model_name}' not explicitly handled. Assuming no prefix for signals file.")
        model_prefix = args.model_name.upper() + "_" if args.model_name else ""


    index_signals_filename = f"model_signals_{model_prefix}INDEX_FOR_SIMULATION_TOKEN_{config.TRAINING_TOKEN}_DATES_{config.TRADING_SIMULATION_START_DATE}_to_{config.TRADING_SIMULATION_END_DATE}.csv"
    index_signals_filepath = os.path.join(args.output_dir, index_signals_filename)

    print(f"Expecting Option OHLCV data at: {option_ohlcv_filepath}")
    print(f"Expecting Index signals at: {index_signals_filepath}")

    trade_parameters = {
        'units': args.units,
        'profit_target_pct': args.target_pct,
        'stop_loss_pct': args.sl_pct,
        'max_holding_period_minutes': args.hold_min
    }

    ohlcv_data = load_ohlcv_data(option_ohlcv_filepath) # Load Option OHLCV
    signals_data = load_signals_data(index_signals_filepath) # Load Index signals

    if ohlcv_data is not None and signals_data is not None:
        executed_trades_df = run_simulation(ohlcv_data, signals_data, trade_parameters)
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        if not executed_trades_df.empty:
            # Save detailed trades log
            trades_log_path = os.path.join(args.output_dir, SIMULATION_TRADES_LOG_FILENAME)
            executed_trades_df.to_csv(trades_log_path, index=False)
            print(f"\nDetailed trades log saved to: {trades_log_path}")

            # Calculate and save performance metrics summary
            metrics_dict, metrics_summary_str = calculate_performance_metrics(executed_trades_df)
            
            # Construct summary filename with optional suffix
            base_summary_name, summary_ext = os.path.splitext(SIMULATION_SUMMARY_FILENAME)
            actual_summary_filename = f"{base_summary_name}{args.summary_filename_suffix}{summary_ext}"
            summary_file_path = os.path.join(args.output_dir, actual_summary_filename)

            print(f"DEBUG TRADING_SIMULATOR: Attempting to save summary to absolute path: {os.path.abspath(summary_file_path)}") # DEBUG PRINT
            try:
                with open(summary_file_path, 'w') as f:
                    f.write(metrics_summary_str)
                print(f"Performance summary saved to: {summary_file_path}")
            except IOError as e:
                print(f"Error saving performance summary: {e}")
        else:
            print("Simulation resulted in no trades to analyze.")
            # Create an empty summary file if no trades
            base_summary_name, summary_ext = os.path.splitext(SIMULATION_SUMMARY_FILENAME)
            actual_summary_filename = f"{base_summary_name}{args.summary_filename_suffix}{summary_ext}"
            summary_file_path = os.path.join(args.output_dir, actual_summary_filename)
            
            print(f"DEBUG TRADING_SIMULATOR: Attempting to save EMPTY summary to absolute path: {os.path.abspath(summary_file_path)}") # DEBUG PRINT
            try:
                with open(summary_file_path, 'w') as f:
                    f.write("Simulation resulted in no trades to analyze.")
                print(f"Empty performance summary saved to: {summary_file_path}")
            except IOError as e:
                print(f"Error saving empty performance summary: {e}")

    else:
        print("Exiting due to errors in loading data.") 