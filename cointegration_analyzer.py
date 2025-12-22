#!/usr/bin/env python3
"""
Cointegration Analyzer for Spread Trading
==========================================

Tests currency pairs for statistical cointegration and calculates:
- Cointegration test (Engle-Granger)
- Half-life of mean reversion
- Optimal hedge ratio
- Historical spread behavior

This helps identify which pairs are actually statistically sound for 
mean reversion trading vs just "kinda correlated."

Usage:
    export OANDA_API_KEY="your_key"
    export OANDA_ACCOUNT_ID="your_account"
    python cointegration_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Statistical tests
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from oanda_client import OandaClient


# =============================================================================
# Spread definitions to test
# =============================================================================
SPREADS_TO_TEST = [
    ('EUR_USD', 'GBP_USD'),   # European majors
    ('AUD_USD', 'NZD_USD'),   # Oceania twins
    ('USD_CAD', 'USD_NOK'),   # Oil exporters
    ('EUR_JPY', 'GBP_JPY'),   # European vs safe haven
    ('AUD_JPY', 'NZD_JPY'),   # Oceania vs safe haven (alternative)
    ('USD_CHF', 'USD_JPY'),   # Safe havens
    ('EUR_USD', 'USD_CHF'),   # Inverse relationship
]

# How much history to analyze
LOOKBACK_DAYS = 365  # 1 year of daily data
GRANULARITY = 'D'    # Daily candles


class CointegrationAnalyzer:
    """Analyzes currency pairs for cointegration and mean reversion properties"""
    
    def __init__(self):
        self.client = OandaClient()
        self.price_data = {}
    
    def fetch_historical_data(self, instrument: str, days: int = 365) -> pd.Series:
        """Fetch daily close prices for an instrument"""
        
        if instrument in self.price_data:
            return self.price_data[instrument]
        
        print(f"  Fetching {instrument}...", end=" ", flush=True)
        
        candles = self.client.get_candles(
            instrument,
            granularity=GRANULARITY,
            count=days
        )
        
        if not candles:
            print("FAILED")
            return None
        
        # Convert to pandas Series with datetime index
        dates = [c['time'][:10] for c in candles]  # Just the date part
        closes = [c['close'] for c in candles]
        
        series = pd.Series(closes, index=pd.to_datetime(dates), name=instrument)
        self.price_data[instrument] = series
        
        print(f"OK ({len(series)} days)")
        return series
    
    def calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """
        Calculate optimal hedge ratio using OLS regression.
        
        y = beta * x + residual
        
        Returns beta (hedge ratio)
        """
        x_with_const = add_constant(x)
        model = OLS(y, x_with_const).fit()
        return model.params.iloc[1]  # The slope (beta)
    
    def calculate_spread(self, pair1: pd.Series, pair2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate the spread: pair1 - hedge_ratio * pair2"""
        return pair1 - hedge_ratio * pair2
    
    def test_stationarity(self, series: pd.Series) -> dict:
        """
        Run Augmented Dickey-Fuller test for stationarity.
        
        If p-value < 0.05, the series is stationary (mean-reverting).
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def test_cointegration(self, pair1: pd.Series, pair2: pd.Series) -> dict:
        """
        Run Engle-Granger cointegration test.
        
        If p-value < 0.05, the pairs are cointegrated.
        """
        # Align the series
        combined = pd.concat([pair1, pair2], axis=1).dropna()
        
        result = coint(combined.iloc[:, 0], combined.iloc[:, 1])
        
        return {
            'coint_statistic': result[0],
            'p_value': result[1],
            'critical_values': {
                '1%': result[2][0],
                '5%': result[2][1],
                '10%': result[2][2]
            },
            'is_cointegrated': result[1] < 0.05
        }
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion using AR(1) model.
        
        spread_t = rho * spread_{t-1} + noise
        half_life = -log(2) / log(rho)
        
        Returns half-life in periods (days if using daily data).
        """
        spread_clean = spread.dropna()
        spread_lag = spread_clean.shift(1).dropna()
        spread_diff = spread_clean.iloc[1:]
        
        # Align
        spread_lag = spread_lag.iloc[:len(spread_diff)]
        
        # Regress spread on lagged spread
        x_with_const = add_constant(spread_lag)
        model = OLS(spread_diff.values, x_with_const).fit()
        
        # rho is approximately 1 + coefficient on lagged spread
        # For mean reversion: spread_t - spread_{t-1} = (rho - 1) * spread_{t-1}
        rho = model.params.iloc[1] + 1
        
        if rho >= 1 or rho <= 0:
            return float('inf')  # Not mean-reverting
        
        half_life = -np.log(2) / np.log(rho)
        return half_life
    
    def calculate_spread_stats(self, spread: pd.Series) -> dict:
        """Calculate statistics about the spread"""
        return {
            'mean': spread.mean(),
            'std': spread.std(),
            'min': spread.min(),
            'max': spread.max(),
            'current': spread.iloc[-1],
            'current_z_score': (spread.iloc[-1] - spread.mean()) / spread.std(),
            'pct_time_outside_1std': (abs(spread - spread.mean()) > spread.std()).mean() * 100,
            'pct_time_outside_2std': (abs(spread - spread.mean()) > 2 * spread.std()).mean() * 100,
        }
    
    def analyze_pair(self, pair1_symbol: str, pair2_symbol: str) -> dict:
        """Full analysis of a spread pair"""
        
        # Fetch data
        pair1 = self.fetch_historical_data(pair1_symbol, LOOKBACK_DAYS)
        pair2 = self.fetch_historical_data(pair2_symbol, LOOKBACK_DAYS)
        
        if pair1 is None or pair2 is None:
            return {'error': 'Failed to fetch data'}
        
        # Align data
        combined = pd.concat([pair1, pair2], axis=1).dropna()
        pair1_aligned = combined.iloc[:, 0]
        pair2_aligned = combined.iloc[:, 1]
        
        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(pair1_aligned, pair2_aligned)
        
        # Calculate spread
        spread = self.calculate_spread(pair1_aligned, pair2_aligned, hedge_ratio)
        
        # Run tests
        cointegration = self.test_cointegration(pair1_aligned, pair2_aligned)
        stationarity = self.test_stationarity(spread)
        half_life = self.calculate_half_life(spread)
        spread_stats = self.calculate_spread_stats(spread)
        
        # Calculate correlation for comparison
        correlation = pair1_aligned.corr(pair2_aligned)
        
        return {
            'pair1': pair1_symbol,
            'pair2': pair2_symbol,
            'data_points': len(combined),
            'hedge_ratio': hedge_ratio,
            'correlation': correlation,
            'cointegration': cointegration,
            'stationarity': stationarity,
            'half_life_days': half_life,
            'spread_stats': spread_stats,
            'spread_series': spread  # For plotting if needed
        }
    
    def score_pair(self, analysis: dict) -> float:
        """
        Score a pair based on how good it is for spread trading.
        Higher = better.
        
        Factors:
        - Cointegration (must pass)
        - Short half-life (faster mean reversion)
        - High correlation (more predictable)
        - Reasonable z-score frequency (enough trading opportunities)
        """
        if 'error' in analysis:
            return 0
        
        score = 0
        
        # Cointegration is the big one
        if analysis['cointegration']['is_cointegrated']:
            score += 40
            # Bonus for stronger cointegration
            if analysis['cointegration']['p_value'] < 0.01:
                score += 10
        
        # Stationarity of spread
        if analysis['stationarity']['is_stationary']:
            score += 20
        
        # Half-life: prefer 5-30 days for your timeframe
        hl = analysis['half_life_days']
        if 5 <= hl <= 30:
            score += 20
        elif 30 < hl <= 60:
            score += 10
        elif hl > 60 or hl < 3:
            score += 0
        
        # Correlation
        corr = abs(analysis['correlation'])
        if corr > 0.9:
            score += 10
        elif corr > 0.8:
            score += 5
        
        return score


def print_analysis(analysis: dict) -> None:
    """Pretty print the analysis results"""
    
    if 'error' in analysis:
        print(f"  ERROR: {analysis['error']}")
        return
    
    pair_name = f"{analysis['pair1']}/{analysis['pair2']}"
    
    print(f"\n{'='*70}")
    print(f"  {pair_name}")
    print(f"{'='*70}")
    
    print(f"\n  Data: {analysis['data_points']} daily observations")
    print(f"  Correlation: {analysis['correlation']:.4f}")
    print(f"  Optimal hedge ratio: {analysis['hedge_ratio']:.4f}")
    
    # Cointegration result
    coint = analysis['cointegration']
    coint_status = "✓ YES" if coint['is_cointegrated'] else "✗ NO"
    print(f"\n  Cointegration Test:")
    print(f"    Cointegrated: {coint_status} (p-value: {coint['p_value']:.4f})")
    
    # Stationarity
    stat = analysis['stationarity']
    stat_status = "✓ YES" if stat['is_stationary'] else "✗ NO"
    print(f"\n  Spread Stationarity (ADF Test):")
    print(f"    Stationary: {stat_status} (p-value: {stat['p_value']:.4f})")
    
    # Half-life
    hl = analysis['half_life_days']
    if hl == float('inf'):
        hl_str = "∞ (not mean-reverting)"
    else:
        hl_str = f"{hl:.1f} days"
    print(f"\n  Mean Reversion Half-Life: {hl_str}")
    
    # Spread stats
    stats = analysis['spread_stats']
    print(f"\n  Spread Statistics:")
    print(f"    Current Z-Score: {stats['current_z_score']:+.2f}")
    print(f"    Time outside 1σ: {stats['pct_time_outside_1std']:.1f}%")
    print(f"    Time outside 2σ: {stats['pct_time_outside_2std']:.1f}%")
    
    # Trading implications
    print(f"\n  Trading Implications:")
    if coint['is_cointegrated'] and hl < 60:
        print(f"    → Good candidate for spread trading")
        print(f"    → Expect reversion within ~{hl*2:.0f} days (2x half-life)")
        if stats['pct_time_outside_2std'] > 5:
            print(f"    → Gets to 2σ about {stats['pct_time_outside_2std']:.0f}% of the time")
        if abs(stats['current_z_score']) > 1.5:
            direction = "LONG" if stats['current_z_score'] < 0 else "SHORT"
            print(f"    → Currently extended: consider {direction} spread")
    else:
        if not coint['is_cointegrated']:
            print(f"    → NOT cointegrated - correlation may be spurious")
        if hl >= 60:
            print(f"    → Half-life too long for practical trading")


def main():
    print("\n" + "="*70)
    print("  COINTEGRATION ANALYZER FOR SPREAD TRADING")
    print("="*70)
    print(f"\n  Analyzing {len(SPREADS_TO_TEST)} spread pairs...")
    print(f"  Lookback: {LOOKBACK_DAYS} days of daily data")
    print(f"\n  Fetching price data from OANDA...")
    
    analyzer = CointegrationAnalyzer()
    results = []
    
    for pair1, pair2 in SPREADS_TO_TEST:
        print(f"\n  Analyzing {pair1}/{pair2}...")
        analysis = analyzer.analyze_pair(pair1, pair2)
        analysis['score'] = analyzer.score_pair(analysis)
        results.append(analysis)
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Print detailed results
    print("\n" + "="*70)
    print("  DETAILED RESULTS (sorted by score)")
    print("="*70)
    
    for analysis in results:
        print_analysis(analysis)
    
    # Summary table
    print("\n" + "="*70)
    print("  SUMMARY RANKING")
    print("="*70)
    print(f"\n  {'Spread':<25} {'Coint?':<8} {'Half-Life':<12} {'Corr':<8} {'Score':<6}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*8} {'-'*6}")
    
    for r in results:
        if 'error' in r:
            continue
        
        name = f"{r['pair1'][:3]}/{r['pair2'][:3]}"
        coint = "✓" if r['cointegration']['is_cointegrated'] else "✗"
        hl = f"{r['half_life_days']:.1f}d" if r['half_life_days'] < 1000 else "∞"
        corr = f"{r['correlation']:.3f}"
        score = f"{r['score']:.0f}"
        
        print(f"  {name:<25} {coint:<8} {hl:<12} {corr:<8} {score:<6}")
    
    # Recommendations
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    good_pairs = [r for r in results if r.get('score', 0) >= 50]
    
    if good_pairs:
        print("\n  Strong candidates for spread trading:")
        for r in good_pairs:
            hl = r['half_life_days']
            print(f"    • {r['pair1']}/{r['pair2']} - half-life {hl:.0f} days")
        
        print("\n  Suggested parameters for conservative approach:")
        print("    • Entry: Z-score ≥ 2.5 (fewer but higher-conviction trades)")
        print("    • Exit: Z-score ≤ 0.0 (ride it back to the mean)")
        print("    • Timeframe: Daily candles, 60-period lookback")
        print("    • Expect: 2-4 trades per pair per year")
    else:
        print("\n  No pairs scored high enough for recommendation.")
        print("  Consider testing different pairs or longer timeframes.")
    
    print("\n")


if __name__ == "__main__":
    main()
