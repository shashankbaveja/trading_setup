# Automated Trading System Implementation

## Project Objective

The primary goal is to develop an automated trading system that:
1.  Fetches real-time data from the Kite API.
2.  Analyzes data based on predefined technical strategies.
3.  Executes and manages trades.

## System Components

The system will be divided into the following main parts:

1.  **Data Preparation**: Involves acquiring, cleaning, and storing market data. This will be supported by a `DataPrep` class within a dedicated `trading_strategies.py` file, responsible for fetching and preparing data for strategy analysis.
2.  **Strategy Definition & Signal Generation (`trading_strategies.py`)**:
    *   A new file, `trading_strategies.py`, will house all strategy-related logic.
    *   It will include:
        *   The `DataPrep` class mentioned above.
        *   A `BaseStrategy` abstract class to define a common interface for all trading strategies (e.g., a `generate_signals(self, data)` method).
        *   Individual strategy classes (e.g., `StrategyReversalV1`) inheriting from `BaseStrategy`, each implementing its unique logic for generating trading signals.
3.  **Trading Loop & Orchestration**:
    *   A central orchestrator script will manage the overall workflow.
    *   **Trading Simulation (`TradingSimulator`)**:
        *   A dedicated `TradingSimulator` module will be developed.
        *   This simulator will instantiate `DataPrep` and strategy classes from `trading_strategies.py`.
        *   It will be responsible for running strategies against historical or real-time data, simulating trades, and evaluating performance.
        *   A `StrategyManager` (or similar component) within the `TradingSimulator` will manage the lifecycle of strategies, pass data to them, and collect generated signals.
        *   The output of the simulation (often a "signal file" or performance report) will be generated here.
    *   **Trade Execution & Monitoring**: Handles the actual placement of trades via the Kite API and monitors their performance.

The existing `myKiteLib.py` will serve as a foundational library for basic Kite API interactions and other common functionalities, utilized by `DataPrep` and other components.

## Immediate Next Steps

1.  **Daily Options Data Ingestion (Minute Level)**:
    *   Develop a daily loop to run every evening to fetch minute-level options data.
    *   **New Instruments**: Ensure new options instruments are identified and included.
    *   **Data Looping**: Implement a robust loop to fetch data for all relevant options.
    *   **One-Time Backfill**: Perform a one-time data backfill for the last two months.
    *   **Daily Delta**: Modify the loop to fetch data for the last two days on a daily basis.

2.  **Candle Data Resampling Function**:
    *   Create a function within `myKiteLib.py`.
    *   **Input**: Accept an instrument token or a list of tokens.
    *   **Process**: Fetch minute-level candle data from the database for the given token(s).
    *   **Output**: Convert the 1-minute data into 2-minute, 3-minute, 4-minute, 5-minute, and 10-minute intervals and return as a Pandas DataFrame.

# Implementation Details and Progress

This document tracks the implementation progress, technical details, and architectural decisions for the intraday options trading model.

## Recent Progress (as of last update)

The primary focus has been on developing and refining the `TradingSimulator` and integrating it with a configuration-driven approach for strategies.

1.  **Configuration-Driven Simulator**:
    *   `trading_simulator.py` has been significantly refactored to load all crucial parameters from `trading_config.ini`. This includes:
        *   Global simulator settings (index token, initial capital).
        *   Strategy selection (e.g., `STRATEGY_CONFIG_DonchianStandard`).
        *   Strategy-specific indicator parameters (e.g., `length`, `exit_option` for Donchian).
        *   Strategy-specific trading parameters (e.g., `option_type`, `trade_interval`, `trade_units`, `profit_target_pct`, `stop_loss_pct`, `max_holding_period_minutes`).
    *   The `if __name__ == '__main__':` block in `trading_simulator.py` now dynamically instantiates the selected strategy and the simulator based on the configuration.

2.  **Data Preparation Enhancements**:
    *   The `DataPrep.fetch_and_prepare_data` method in `trading_strategies.py` was updated to always fetch 1-minute interval data from the database.
    *   If a different `trade_interval` (e.g., '5minute') is specified in the strategy configuration, the 1-minute data is then resampled to the target interval using `DataPrep.convert_minute_data_interval`. This ensures consistency and allows strategies to operate on various timeframes using a unified base data source.

3.  **Debugging and Refinements**:
    *   **Configuration Parsing**: Fixed a `ValueError` caused by inline comments in `trading_config.ini` for integer parameters (e.g., `trade_units`). Comments were moved to separate lines.
    *   **Attribute Errors**: Resolved an `AttributeError` in `trading_simulator.py` where `self.dp` was used instead of the correctly initialized `self.data_prep`.
    *   **Option Data Fetching Logic**:
        *   Corrected the logic for fetching option OHLCV data. The `option_data_end_date` in `TradingSimulator.run_simulation` is now correctly set to the overall simulation end date (`self.trade_end_date`) to ensure sufficient data is available for trades initiated near the end of a day or period.
        *   Removed an outdated and unused method `_fetch_ohlcv_for_option_token` from `trading_simulator.py` to prevent confusion and ensure the new `DataPrep.fetch_and_prepare_data` is used consistently.
    *   **Performance Metrics Formatting**: Fixed a `ValueError` in `TradingSimulator.calculate_performance_metrics` related to an invalid f-string format specifier for `profit_factor` when it was `np.inf`. The value is now pre-formatted into a string.
    *   **SQLAlchemy Warnings**: Observed `UserWarning: pandas only supports SQLAlchemy connectable...` during database operations. While not critical errors, these suggest reviewing the database connection handling in `myKiteLib.py` and `trading_simulator.py` for optimal SQLAlchemy integration.
    *   **MySQL Shutdown Errors**: Noticed `TypeError: 'NoneType' object cannot be interpreted as an integer` during MySQL socket shutdown at the end of the script execution. This seems to be an issue within the `mysql.connector` or `ssl` library when closing the connection.

4.  **Successful Simulation Run**:
    *   The `trading_simulator.py` script, configured for `STRATEGY_CONFIG_DonchianStandard`, completed a simulation run from May 1, 2025, to May 16, 2025.
    *   Trades were generated, and performance metrics were calculated and saved to `cursor_logs/`.

## Technical Details

### Core Components:

*   **`trading_config.ini`**: Central configuration file.
    *   `[SIMULATOR_SETTINGS]`: Global settings for the simulator.
    *   `[DATA_PREP_DEFAULTS]`: Default parameters for `DataPrep` if not overridden by strategy needs.
    *   `[STRATEGY_CONFIG_*]`: Sections for each strategy, defining:
        *   `strategy_class_name`: The Python class for the strategy.
        *   Indicator-specific parameters (e.g., `length` for Donchian).
        *   Trading parameters (option type, interval, units, PNL targets, holding period).

*   **`trading_simulator.py`**:
    *   **`TradingSimulator` Class**:
        *   `__init__`: Initializes with index token, instantiated strategy object, trade dates, option type, interval, specific trade parameters (profit target, stop loss, etc.), and initial capital. All these are derived from `trading_config.ini`.
        *   `_find_closest_CE_option` / `_find_closest_PE_option`: Selects the nearest strike option (CE or PE) based on the NIFTY price at the signal time using SQL queries against the `instruments_zerodha` table. Queries filter for options expiring in the current month of the signal.
        *   `_simulate_single_trade_on_option`:
            *   Takes option OHLCV data, NIFTY signal time, NIFTY price, and a series of NIFTY exit signals for the trade window.
            *   Option Entry: At the `open` price of the option candle at or immediately following the NIFTY BUY signal.
            *   Exit Conditions (checked per minute, in order of priority):
                1.  **NIFTY Strategy Exit Signal**: If the underlying NIFTY strategy (e.g., Donchian) generates an explicit exit signal (`-1`) during the holding period.
                2.  **Profit Target**: If `option_high >= entry_price * (1 + profit_target_pct)`. Exit at target price.
                3.  **Stop Loss**: If `option_low <= entry_price * (1 - stop_loss_pct)`. Exit at stop-loss price.
                4.  **Max Holding Period**: If `max_holding_period_minutes` is reached. Exit at `close` of the last candle.
                5.  **End of Option Data**: If option data for the selected token runs out. Exit at `close` of the last available candle.
            *   Logs detailed information for each simulated trade.
        *   `run_simulation`:
            1.  Fetches and prepares NIFTY index data using `DataPrep.fetch_and_prepare_data` for the specified `trade_interval`.
            2.  Calculates technical indicators on the NIFTY data using `DataPrep.calculate_statistics` (leveraging strategy-specific parameters like `length` if provided by the strategy object).
            3.  Generates BUY/SELL/HOLD signals on NIFTY data using the `generate_signals` method of the instantiated strategy object.
            4.  Iterates through BUY signals on NIFTY:
                *   Finds the appropriate option token (CE/PE based on config).
                *   Fetches option OHLCV data using `DataPrep.fetch_and_prepare_data` for the period from the signal date to the simulation end date.
                *   Calls `_simulate_single_trade_on_option` to simulate the trade using the fetched option data and NIFTY exit signals.
            5.  Collects all executed trades into a DataFrame.
        *   `calculate_performance_metrics`: Computes metrics like win rate, PNL, profit factor, max drawdown.
        *   `save_results`: Saves the detailed trades log (CSV) and performance summary (TXT) to the `cursor_logs/` directory.
    *   **`if __name__ == '__main__':` Block**:
        *   Loads global configuration from `trading_config.ini`.
        *   Selects a strategy configuration section (e.g., `STRATEGY_CONFIG_DonchianStandard`).
        *   Extracts simulator settings and strategy-specific parameters.
        *   Dynamically instantiates the chosen strategy class (e.g., `DonchianBreakoutStrategy`) with its indicator parameters.
        *   Instantiates `TradingSimulator` with all necessary configured parameters.
        *   Calls `run_simulation`, then `calculate_performance_metrics` and `save_results`.

*   **`trading_strategies.py`**:
    *   **`DataPrep` Class**:
        *   `fetch_and_prepare_data`:
            *   Fetches raw 1-minute OHLCV data from the MySQL database (`myKiteLib.get_historical_data_from_db_for_token_and_interval_new`).
            *   Handles column renaming, type conversion, sorting, and dropping duplicates.
            *   If `interval` is > 1 minute, calls `convert_minute_data_interval` to resample.
        *   `convert_minute_data_interval`: Resamples 1-minute data to '3minute', '5minute', etc.
        *   `calculate_statistics`: Calculates technical indicators (e.g., Donchian Channels via `_add_donchian_channels`). Uses strategy-specific parameters if available (e.g., `donchian_length`), otherwise falls back to defaults from `[DATA_PREP_DEFAULTS]` in `trading_config.ini`.
    *   **`DonchianBreakoutStrategy` Class**:
        *   `__init__`: Takes `length` and `exit_option` as parameters (from config).
        *   `generate_signals`:
            *   Requires DataFrame with 'high', 'low', 'close', 'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid'.
            *   BUY signal (1): If `close` crosses above `Donchian_Upper`.
            *   SELL/EXIT signal (-1):
                *   If `exit_option == 1`: If `close` crosses below `Donchian_Lower`.
                *   If `exit_option == 2`: If `close` crosses below `Donchian_Mid`.
            *   HOLD signal (0): Otherwise.

*   **`myKiteLib.py`**:
    *   Provides database connection (`mysqlDB`) and data fetching utilities (`kiteAPIs.get_historical_data_from_db_for_token_and_interval_new`).
    *   The `UserWarning`s and shutdown errors observed might originate from interactions within this library or how it's used by `DataPrep` and `TradingSimulator`.

## Next Steps / Areas for Improvement

*   Investigate and resolve SQLAlchemy `UserWarning`s.
*   Investigate and resolve MySQL connection shutdown `TypeError`s.
*   Expand strategy library with more complex indicators and logic.
*   Implement more sophisticated risk management and position sizing.
*   Enhance reporting and visualization of simulation results.
*   Consider parameter optimization techniques. 