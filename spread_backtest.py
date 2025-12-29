#!/usr/bin/env python3
"""
Spread Trading Backtester
=========================

Tests spread trading strategies across multiple currency pairs and timeframes.
Uses VectorBT for fast backtesting and comparison.

Usage:
    export OANDA_API_KEY="your_key"
    export OANDA_ACCOUNT_ID="your_account"
    python spread_backtest.py

Or run with sample data (no OANDA credentials needed):
    python spread_backtest.py --sample
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import vectorbt as vbt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# =============================================================================
# Configuration
# =============================================================================

# Pairs to test (pair1, pair2)
SPREAD_PAIRS = [
    ('EUR_USD', 'USD_CHF'),   # Inverse relationship - your original
    ('EUR_JPY', 'GBP_JPY'),   # European vs JPY crosses
    ('AUD_USD', 'NZD_USD'),   # Oceania twins - usually strong cointegration
    ('EUR_USD', 'GBP_USD'),   # European majors
    ('USD_CAD', 'USD_NOK'),   # Oil/commodity currencies
]

# Timeframes to test
TIMEFRAMES = ['D', 'W']  # Daily and Weekly

# Strategy parameters to test
ENTRY_Z_SCORES = [1.5, 2.0, 2.3, 2.5]  # Added 1.5 for more trades
EXIT_Z_SCORES = [0.0, 0.2, 0.5]
LOOKBACK_PERIODS = [20, 30, 60]

# Backtest settings
YEARS_OF_DATA = 5
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.1  # 10% of capital per trade


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_oanda_data(instrument: str, granularity: str, count: int) -> Optional[pd.DataFrame]:
    """Fetch historical data from OANDA"""
    try:
        import requests
        
        api_key = os.environ.get('OANDA_API_KEY')
        account_id = os.environ.get('OANDA_ACCOUNT_ID')
        
        if not api_key or not account_id:
            return None
        
        base_url = "https://api-fxpractice.oanda.com"
        url = f"{base_url}/v3/instruments/{instrument}/candles"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        candles = []
        for c in data.get('candles', []):
            if c['complete']:
                candles.append({
                    'time': pd.to_datetime(c['time']),
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c']),
                    'volume': int(c['volume'])
                })
        
        df = pd.DataFrame(candles)
        df.set_index('time', inplace=True)
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch {instrument}: {e}")
        return None


def generate_sample_data(pair: str, granularity: str, count: int) -> pd.DataFrame:
    """Generate realistic sample forex data for testing without OANDA"""
    # Use a fixed seed based on pair name for reproducibility
    seed = sum(ord(c) for c in pair) % 10000
    np.random.seed(seed)
    
    # Base prices for different pairs
    base_prices = {
        'EUR_USD': 1.10,
        'GBP_USD': 1.27,
        'USD_CHF': 0.88,
        'USD_JPY': 145.0,
        'EUR_JPY': 160.0,
        'GBP_JPY': 184.0,
        'AUD_USD': 0.65,
        'NZD_USD': 0.60,
        'USD_CAD': 1.36,
        'USD_NOK': 10.5,
    }
    
    base = base_prices.get(pair, 1.0)
    
    # Generate dates - use a fixed end date for consistency
    end_date = pd.Timestamp('2024-12-27')
    
    if granularity == 'W':
        dates = pd.date_range(end=end_date, periods=count, freq='W-FRI')
    else:  # Daily
        dates = pd.date_range(end=end_date, periods=count, freq='D')
    
    # Generate correlated random walk with mean reversion
    volatility = 0.003 if granularity == 'D' else 0.008
    returns = np.random.normal(0, volatility, count)
    
    # Add some mean reversion
    prices = [base]
    for r in returns[1:]:
        # Mean reversion factor
        mean_rev = -0.01 * (prices[-1] - base) / base
        new_price = prices[-1] * (1 + r + mean_rev)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC
    noise = np.random.uniform(0.0005, 0.002, count)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, count)),
        'high': prices * (1 + noise),
        'low': prices * (1 - noise),
        'close': prices,
        'volume': np.random.randint(1000, 10000, count)
    }, index=dates)
    
    return df


# Cache for correlated sample data
_sample_data_cache = {}

def generate_correlated_pair_data(pair1: str, pair2: str, granularity: str, count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate cointegrated sample data for a pair of currencies.
    Uses an error correction model to ensure actual cointegration.
    """
    cache_key = f"{pair1}_{pair2}_{granularity}"
    if cache_key in _sample_data_cache:
        return _sample_data_cache[cache_key]
    
    np.random.seed(42 + len(cache_key))  # Consistent per pair
    
    # Base prices
    base_prices = {
        'EUR_USD': 1.10, 'GBP_USD': 1.27, 'USD_CHF': 0.88, 'USD_JPY': 145.0,
        'EUR_JPY': 160.0, 'GBP_JPY': 184.0, 'AUD_USD': 0.65, 'NZD_USD': 0.60,
        'USD_CAD': 1.36, 'USD_NOK': 10.5,
    }
    
    base1 = base_prices.get(pair1, 1.0)
    base2 = base_prices.get(pair2, 1.0)
    
    # Hedge ratio (cointegrating vector)
    is_inverse = pair1 == 'EUR_USD' and pair2 == 'USD_CHF'
    if is_inverse:
        hedge_ratio = -1.2  # Inverse relationship
    else:
        hedge_ratio = base1 / base2  # Approximate
    
    # Generate dates
    end_date = pd.Timestamp('2024-12-27')
    if granularity == 'W':
        dates = pd.date_range(end=end_date, periods=count, freq='W-FRI')
    else:
        dates = pd.date_range(end=end_date, periods=count, freq='D')
    
    # Volatility settings
    vol = 0.004 if granularity == 'D' else 0.012
    
    # Generate cointegrated series using error correction model
    # Price2 follows Price1 with a mean-reverting spread
    
    # First, generate Price1 as a random walk
    returns1 = np.random.normal(0, vol, count)
    prices1 = [base1]
    for r in returns1[1:]:
        prices1.append(prices1[-1] * (1 + r))
    prices1 = np.array(prices1)
    
    # Generate Price2 that maintains cointegration with Price1
    # spread = price1 - hedge_ratio * price2 should be stationary
    # So price2 = (price1 - spread) / hedge_ratio
    # where spread follows an OU process
    
    # Mean reversion speed (determines half-life)
    # Half-life = ln(2) / kappa, so kappa = ln(2) / half_life
    target_half_life = 15 if granularity == 'D' else 4  # days or weeks
    kappa = np.log(2) / target_half_life
    
    # Generate mean-reverting spread
    spread_vol = vol * abs(hedge_ratio) * base2 * 0.5
    spread = [0.0]
    for i in range(1, count):
        # OU process: dS = -kappa * S * dt + sigma * dW
        drift = -kappa * spread[-1]
        noise = np.random.normal(0, spread_vol)
        spread.append(spread[-1] + drift + noise)
    spread = np.array(spread)
    
    # Compute price2 from cointegration relationship
    # spread = price1 - hedge_ratio * price2
    # price2 = (price1 - spread) / hedge_ratio
    if is_inverse:
        # For inverse pairs: spread = price1 + |hedge_ratio| * price2
        prices2 = (spread - prices1) / hedge_ratio
    else:
        prices2 = (prices1 - spread) / hedge_ratio
    
    # Ensure prices are positive
    prices2 = np.maximum(prices2, base2 * 0.5)
    
    # Build DataFrames
    def make_df(prices):
        noise = np.random.uniform(0.0003, 0.001, count)
        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.0005, 0.0005, count)),
            'high': prices * (1 + noise),
            'low': prices * (1 - noise),
            'close': prices,
            'volume': np.random.randint(1000, 10000, count)
        }, index=dates)
    
    df1 = make_df(prices1)
    df2 = make_df(prices2)
    
    _sample_data_cache[cache_key] = (df1, df2)
    return df1, df2


def get_data(pair: str, granularity: str, use_sample: bool = False) -> Optional[pd.DataFrame]:
    """Get data from OANDA or generate sample data"""
    
    # Calculate count based on timeframe and years
    if granularity == 'W':
        count = YEARS_OF_DATA * 52 + 10  # Weekly
    else:
        count = YEARS_OF_DATA * 252 + 10  # Daily (trading days)
    
    if use_sample:
        print(f"  Generating sample data for {pair} ({granularity})...")
        return generate_sample_data(pair, granularity, count)
    else:
        print(f"  Fetching {pair} ({granularity}) from OANDA...")
        return fetch_oanda_data(pair, granularity, count)


# =============================================================================
# Cointegration Analysis
# =============================================================================

@dataclass
class CointegrationResult:
    pair1: str
    pair2: str
    is_cointegrated: bool
    p_value: float
    hedge_ratio: float
    half_life: float
    correlation: float


def calculate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Calculate hedge ratio via OLS regression"""
    x_with_const = add_constant(x)
    model = OLS(y, x_with_const).fit()
    return model.params.iloc[1]


def calculate_half_life(spread: pd.Series) -> float:
    """Calculate half-life of mean reversion using AR(1) model"""
    spread_clean = spread.dropna()
    
    if len(spread_clean) < 30:
        return float('inf')
    
    # Use levels, not differences
    spread_lag = spread_clean.shift(1).dropna()
    spread_curr = spread_clean.iloc[1:]
    
    # Align
    min_len = min(len(spread_lag), len(spread_curr))
    spread_lag = spread_lag.iloc[:min_len]
    spread_curr = spread_curr.iloc[:min_len]
    
    try:
        x_with_const = add_constant(spread_lag)
        model = OLS(spread_curr.values, x_with_const).fit()
        
        # AR(1) coefficient: spread_t = a + rho * spread_{t-1}
        rho = model.params[1]
        
        if rho >= 1 or rho <= 0:
            return float('inf')
        
        # Half-life = -ln(2) / ln(rho)
        half_life = -np.log(2) / np.log(rho)
        return max(0.1, half_life)  # Ensure positive
    except:
        return float('inf')


def test_cointegration(price1: pd.Series, price2: pd.Series, 
                       pair1: str, pair2: str) -> CointegrationResult:
    """Test cointegration between two price series"""
    
    # Align series
    combined = pd.concat([price1, price2], axis=1).dropna()
    p1 = combined.iloc[:, 0]
    p2 = combined.iloc[:, 1]
    
    # Cointegration test
    coint_stat, p_value, _ = coint(p1, p2)
    is_cointegrated = p_value < 0.05
    
    # Hedge ratio
    hedge_ratio = calculate_hedge_ratio(p1, p2)
    
    # Spread and half-life
    spread = p1 - hedge_ratio * p2
    half_life = calculate_half_life(spread)
    
    # Correlation
    correlation = p1.corr(p2)
    
    return CointegrationResult(
        pair1=pair1,
        pair2=pair2,
        is_cointegrated=is_cointegrated,
        p_value=p_value,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        correlation=correlation
    )


# =============================================================================
# Spread Trading Strategy
# =============================================================================

def calculate_spread_zscore(price1: pd.Series, price2: pd.Series, 
                            hedge_ratio: float, lookback: int) -> pd.Series:
    """Calculate rolling z-score of the spread"""
    spread = price1 - hedge_ratio * price2
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    z_score = (spread - mean) / std
    return z_score


def generate_signals(z_score: pd.Series, entry_z: float, exit_z: float) -> Tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals based on z-score thresholds.
    
    LONG_SPREAD when z_score < -entry_z (spread is cheap)
    SHORT_SPREAD when z_score > entry_z (spread is expensive)
    EXIT when |z_score| < exit_z (spread reverted to mean)
    """
    # Long entries: z-score below -entry_z
    long_entries = z_score < -entry_z
    
    # Short entries: z-score above +entry_z
    short_entries = z_score > entry_z
    
    # Exits: z-score crosses back toward zero
    exits = abs(z_score) < exit_z
    
    return long_entries, short_entries, exits


# =============================================================================
# Backtesting
# =============================================================================

@dataclass
class BacktestResult:
    pair1: str
    pair2: str
    timeframe: str
    entry_z: float
    exit_z: float
    lookback: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_trade_duration: float
    is_cointegrated: bool
    half_life: float
    
    def score(self) -> float:
        """Combined score for ranking strategies"""
        if self.num_trades < 5:
            return -999  # Not enough trades
        if not self.is_cointegrated:
            return -998  # Not cointegrated
        
        # Weighted score: Sharpe is king, but penalize huge drawdowns
        score = self.sharpe_ratio * 100
        score -= abs(self.max_drawdown) * 50  # Penalize drawdown
        score += self.win_rate * 20  # Bonus for high win rate
        return score


def run_single_backtest(price1: pd.Series, price2: pd.Series,
                        pair1: str, pair2: str, timeframe: str,
                        entry_z: float, exit_z: float, lookback: int,
                        coint_result: CointegrationResult) -> BacktestResult:
    """Run a single backtest configuration - simplified manual approach"""
    
    # Calculate z-score
    z_score = calculate_spread_zscore(price1, price2, coint_result.hedge_ratio, lookback)
    z_score = z_score.dropna()
    
    if len(z_score) < lookback + 10:
        return BacktestResult(
            pair1=pair1, pair2=pair2, timeframe=timeframe,
            entry_z=entry_z, exit_z=exit_z, lookback=lookback,
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            num_trades=0, win_rate=0.0, avg_trade_duration=0.0,
            is_cointegrated=coint_result.is_cointegrated,
            half_life=coint_result.half_life
        )
    
    # Manual simulation
    spread = (price1 - coint_result.hedge_ratio * price2).loc[z_score.index]
    
    position = 0  # 1 = long spread, -1 = short spread, 0 = flat
    entry_price = 0.0
    trades = []
    pnl_history = [0.0]
    
    for i in range(len(z_score)):
        z = z_score.iloc[i]
        s = spread.iloc[i]
        
        # Entry logic
        if position == 0:
            if z < -entry_z:  # Spread is cheap, go long
                position = 1
                entry_price = s
            elif z > entry_z:  # Spread is expensive, go short
                position = -1
                entry_price = s
        
        # Exit logic
        elif position != 0:
            if abs(z) < exit_z:  # Mean reversion complete
                exit_price = s
                pnl = (exit_price - entry_price) * position
                trades.append({
                    'entry_z': entry_z if position == -1 else -entry_z,
                    'exit_z': z,
                    'pnl': pnl,
                    'position': position
                })
                position = 0
                entry_price = 0.0
        
        # Track cumulative P/L
        if position != 0:
            unrealized = (s - entry_price) * position
            pnl_history.append(pnl_history[-1] + unrealized - (pnl_history[-1] if i > 0 else 0))
        else:
            pnl_history.append(pnl_history[-1])
    
    # Close any open position at end
    if position != 0:
        exit_price = spread.iloc[-1]
        pnl = (exit_price - entry_price) * position
        trades.append({
            'entry_z': entry_z if position == -1 else -entry_z,
            'exit_z': z_score.iloc[-1],
            'pnl': pnl,
            'position': position
        })
    
    num_trades = len(trades)
    
    if num_trades == 0:
        return BacktestResult(
            pair1=pair1, pair2=pair2, timeframe=timeframe,
            entry_z=entry_z, exit_z=exit_z, lookback=lookback,
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            num_trades=0, win_rate=0.0, avg_trade_duration=0.0,
            is_cointegrated=coint_result.is_cointegrated,
            half_life=coint_result.half_life
        )
    
    # Calculate metrics
    total_pnl = sum(t['pnl'] for t in trades)
    spread_std = spread.std()
    total_return = total_pnl / spread_std / 10  # Normalize
    
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / num_trades
    
    # Simple Sharpe approximation
    trade_returns = [t['pnl'] / spread_std for t in trades]
    if len(trade_returns) > 1:
        sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 0.001) * np.sqrt(len(trade_returns))
    else:
        sharpe = 0.0
    
    # Max drawdown from P/L history
    cumulative = np.cumsum([t['pnl'] for t in trades])
    if len(cumulative) > 0:
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / (running_max + 0.001)
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    else:
        max_dd = 0.0
    
    return BacktestResult(
        pair1=pair1,
        pair2=pair2,
        timeframe=timeframe,
        entry_z=entry_z,
        exit_z=exit_z,
        lookback=lookback,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_trade_duration=0.0,
        is_cointegrated=coint_result.is_cointegrated,
        half_life=coint_result.half_life
    )


# =============================================================================
# Main Runner
# =============================================================================

def run_full_backtest(use_sample: bool = False) -> List[BacktestResult]:
    """Run backtests across all pairs, timeframes, and parameters"""
    
    all_results = []
    
    print("\n" + "=" * 70)
    print("  SPREAD TRADING BACKTEST")
    print("=" * 70)
    print(f"\n  Testing {len(SPREAD_PAIRS)} pairs × {len(TIMEFRAMES)} timeframes")
    print(f"  Parameter combinations: {len(ENTRY_Z_SCORES) * len(EXIT_Z_SCORES) * len(LOOKBACK_PERIODS)}")
    print(f"  Data: {YEARS_OF_DATA} years")
    
    for pair1, pair2 in SPREAD_PAIRS:
        spread_name = f"{pair1}/{pair2}"
        print(f"\n{'='*70}")
        print(f"  {spread_name}")
        print(f"{'='*70}")
        
        for timeframe in TIMEFRAMES:
            tf_name = "Daily" if timeframe == 'D' else "Weekly"
            print(f"\n  [{tf_name}]")
            
            # Calculate count
            if timeframe == 'W':
                count = YEARS_OF_DATA * 52 + 10
            else:
                count = YEARS_OF_DATA * 252 + 10
            
            # Fetch data - use correlated generation for sample data
            if use_sample:
                print(f"  Generating correlated sample data...")
                data1, data2 = generate_correlated_pair_data(pair1, pair2, timeframe, count)
            else:
                data1 = get_data(pair1, timeframe, use_sample=False)
                data2 = get_data(pair2, timeframe, use_sample=False)
            
            if data1 is None or data2 is None:
                print(f"    ✗ Failed to get data, skipping")
                continue
            
            # Align data
            price1 = data1['close']
            price2 = data2['close']
            combined = pd.concat([price1, price2], axis=1).dropna()
            price1 = combined.iloc[:, 0]
            price2 = combined.iloc[:, 1]
            
            print(f"    Data points: {len(price1)}")
            
            if len(price1) < 100:
                print(f"    ✗ Not enough data points, skipping")
                continue
            
            # Test cointegration
            coint_result = test_cointegration(price1, price2, pair1, pair2)
            
            coint_status = "✓ YES" if coint_result.is_cointegrated else "✗ NO"
            hl_str = f"{coint_result.half_life:.1f}" if coint_result.half_life < 1000 else "∞"
            print(f"    Cointegrated: {coint_status} (p={coint_result.p_value:.4f})")
            print(f"    Hedge ratio: {coint_result.hedge_ratio:.4f}")
            print(f"    Half-life: {hl_str} periods")
            print(f"    Correlation: {coint_result.correlation:.4f}")
            
            # Debug: show z-score range
            z_score = calculate_spread_zscore(price1, price2, coint_result.hedge_ratio, 20)
            z_clean = z_score.dropna()
            if len(z_clean) > 0:
                print(f"    Z-score range: [{z_clean.min():.2f}, {z_clean.max():.2f}]")
            
            # Run backtests for all parameter combinations
            best_result = None
            best_score = -999
            
            for entry_z in ENTRY_Z_SCORES:
                for exit_z in EXIT_Z_SCORES:
                    for lookback in LOOKBACK_PERIODS:
                        result = run_single_backtest(
                            price1, price2, pair1, pair2, timeframe,
                            entry_z, exit_z, lookback, coint_result
                        )
                        all_results.append(result)
                        
                        if result.score() > best_score:
                            best_score = result.score()
                            best_result = result
            
            # Print best result for this pair/timeframe
            if best_result and best_result.num_trades > 0:
                print(f"\n    Best params: entry_z={best_result.entry_z}, exit_z={best_result.exit_z}, lookback={best_result.lookback}")
                print(f"    Return: {best_result.total_return*100:.2f}%")
                print(f"    Sharpe: {best_result.sharpe_ratio:.2f}")
                print(f"    Max DD: {best_result.max_drawdown*100:.2f}%")
                print(f"    Trades: {best_result.num_trades}")
                print(f"    Win rate: {best_result.win_rate*100:.1f}%")
    
    return all_results


def print_summary(results: List[BacktestResult]):
    """Print summary table of all results"""
    
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY - TOP 10 CONFIGURATIONS")
    print("=" * 70)
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x.score(), reverse=True)
    
    # Filter to only show results with trades
    valid_results = [r for r in sorted_results if r.num_trades >= 5]
    
    if not valid_results:
        print("\n  No valid results found (need at least 5 trades)")
        return
    
    print(f"\n  {'Pair':<20} {'TF':<6} {'Entry':<6} {'Exit':<6} {'LB':<4} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10} {'Trades':<8} {'WinRate':<8} {'Coint':<6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    
    for r in valid_results[:10]:
        spread_name = f"{r.pair1[:3]}/{r.pair2[:3]}"
        tf = "D" if r.timeframe == 'D' else "W"
        ret = f"{r.total_return*100:+.1f}%"
        sharpe = f"{r.sharpe_ratio:.2f}"
        dd = f"{r.max_drawdown*100:.1f}%"
        coint = "✓" if r.is_cointegrated else "✗"
        win = f"{r.win_rate*100:.0f}%"
        
        print(f"  {spread_name:<20} {tf:<6} {r.entry_z:<6.1f} {r.exit_z:<6.1f} {r.lookback:<4} {ret:<10} {sharpe:<8} {dd:<10} {r.num_trades:<8} {win:<8} {coint:<6}")
    
    # Print recommendation
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    
    top = valid_results[0]
    print(f"\n  Best configuration:")
    print(f"    Pair: {top.pair1}/{top.pair2}")
    print(f"    Timeframe: {'Daily' if top.timeframe == 'D' else 'Weekly'}")
    print(f"    Entry Z-score: {top.entry_z}")
    print(f"    Exit Z-score: {top.exit_z}")
    print(f"    Lookback: {top.lookback} periods")
    print(f"    Expected annual return: {top.total_return*100/YEARS_OF_DATA:.1f}%")
    print(f"    Sharpe ratio: {top.sharpe_ratio:.2f}")
    print(f"    Max drawdown: {top.max_drawdown*100:.1f}%")
    print(f"    Trades over {YEARS_OF_DATA} years: {top.num_trades}")
    
    # Compare daily vs weekly for the best pair
    best_pair = (top.pair1, top.pair2)
    daily_results = [r for r in valid_results if (r.pair1, r.pair2) == best_pair and r.timeframe == 'D']
    weekly_results = [r for r in valid_results if (r.pair1, r.pair2) == best_pair and r.timeframe == 'W']
    
    if daily_results and weekly_results:
        best_daily = max(daily_results, key=lambda x: x.score())
        best_weekly = max(weekly_results, key=lambda x: x.score())
        
        print(f"\n  Daily vs Weekly comparison for {top.pair1}/{top.pair2}:")
        print(f"    Daily:  Return={best_daily.total_return*100:+.1f}%, Sharpe={best_daily.sharpe_ratio:.2f}, Trades={best_daily.num_trades}")
        print(f"    Weekly: Return={best_weekly.total_return*100:+.1f}%, Sharpe={best_weekly.sharpe_ratio:.2f}, Trades={best_weekly.num_trades}")


def main():
    parser = argparse.ArgumentParser(description='Spread Trading Backtester')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of OANDA')
    args = parser.parse_args()
    
    use_sample = args.sample
    
    # Check for OANDA credentials if not using sample
    if not use_sample:
        if not os.environ.get('OANDA_API_KEY') or not os.environ.get('OANDA_ACCOUNT_ID'):
            print("[WARN] OANDA credentials not found. Using sample data.")
            print("       Set OANDA_API_KEY and OANDA_ACCOUNT_ID to use real data.")
            use_sample = True
    
    # Run backtests
    results = run_full_backtest(use_sample)
    
    # Print summary
    print_summary(results)
    
    print("\n")


if __name__ == "__main__":
    main()