"""
Forex Pairs Trading Bot
Main entry point - runs the trading loop
"""

import os
import sys
import time
import signal
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from oanda_client import OandaClient
from pairs_analyzer import MultiPairAnalyzer, SpreadSignal


# Configuration from environment (will come from ConfigMap/Secrets in k8s)
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '60'))  # seconds
LOOKBACK_PERIODS = int(os.environ.get('LOOKBACK_PERIODS', '20'))
ENTRY_Z_SCORE = float(os.environ.get('ENTRY_Z_SCORE', '2.0'))
EXIT_Z_SCORE = float(os.environ.get('EXIT_Z_SCORE', '0.5'))
TRADE_UNITS = int(os.environ.get('TRADE_UNITS', '1000'))  # Units per leg
# Logic was backwards - should check if == 'true', not == 'False'
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'
GRANULARITY = os.environ.get('GRANULARITY', 'H1')  # Candle size for historical data
MAX_TRADES_PER_DAY = int(os.environ.get('MAX_TRADES_PER_DAY', '1'))
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', '1'))
ALLOW_LIVE_TRADES = os.environ.get('ALLOW_LIVE_TRADES', 'false').lower() == 'true'
STOP_LOSS_PIPS = float(os.environ.get('STOP_LOSS_PIPS', '50'))  # Stop-loss distance in pips
CLOSE_ON_SHUTDOWN = os.environ.get('CLOSE_ON_SHUTDOWN', 'true').lower() == 'true'
COINT_CHECK_INTERVAL_HOURS = int(os.environ.get('COINT_CHECK_INTERVAL_HOURS', '24'))
COINT_P_THRESHOLD = float(os.environ.get('COINT_P_THRESHOLD', '0.05'))
COINT_LOOKBACK_DAYS = int(os.environ.get('COINT_LOOKBACK_DAYS', '180'))
# Z-score beyond this on an open position = spread is blowing out, exit immediately
ZSCORE_BLOWOUT = float(os.environ.get('ZSCORE_BLOWOUT', '3.5'))


# SPREADS_CONFIG -- candidate universe. Bot auto-promotes/demotes pairs
# based on daily cointegration screening. No redeploy needed to switch pairs.
# Set via: flyctl secrets set SPREADS_CONFIG='[{...}]' --app forex-spread-trading
# JSON: [{"pair1":"EUR_USD","pair2":"GBP_USD","hedge_ratio":1.0}, ...]
import json as _json

_DEFAULT_SPREADS_CONFIG = [
    {"pair1": "EUR_USD", "pair2": "GBP_USD", "hedge_ratio": 1.0},
    {"pair1": "AUD_USD", "pair2": "NZD_USD", "hedge_ratio": 1.0},
    {"pair1": "EUR_USD", "pair2": "USD_CHF", "hedge_ratio": -1.1895},
    {"pair1": "EUR_JPY", "pair2": "GBP_JPY", "hedge_ratio": 1.0},
    {"pair1": "AUD_JPY", "pair2": "NZD_JPY", "hedge_ratio": 1.0}
]

def _load_spreads_config() -> list[dict]:
    raw = os.environ.get('SPREADS_CONFIG', '').strip()
    if not raw:
        print("[CONFIG] SPREADS_CONFIG not set -- using hardcoded defaults")
        return _DEFAULT_SPREADS_CONFIG
    try:
        cfg = _json.loads(raw)
        if not isinstance(cfg, list) or not cfg:
            raise ValueError("must be a non-empty JSON array")
        for item in cfg:
            for k in ('pair1', 'pair2', 'hedge_ratio'):
                if k not in item:
                    raise ValueError(f"missing key '{k}' in spread entry: {item}")
        print(f"[CONFIG] Loaded {len(cfg)} candidate spread(s) from SPREADS_CONFIG")
        return cfg
    except Exception as e:
        print(f"[CONFIG] ERROR parsing SPREADS_CONFIG: {e} -- falling back to defaults")
        return _DEFAULT_SPREADS_CONFIG

_SPREADS_CONFIG  = _load_spreads_config()
SPREADS          = [(s['pair1'], s['pair2']) for s in _SPREADS_CONFIG]
INSTRUMENTS      = list(dict.fromkeys(p for pair in SPREADS for p in pair))
HEDGE_RATIOS     = {f"{s['pair1']}/{s['pair2']}": s['hedge_ratio'] for s in _SPREADS_CONFIG}

# Half-life thresholds for the runtime cointegration screen
COINT_MIN_HALF_LIFE = float(os.environ.get('COINT_MIN_HALF_LIFE', '3'))
COINT_MAX_HALF_LIFE = float(os.environ.get('COINT_MAX_HALF_LIFE', '40'))


class TradingBot:
    """Main bot class that orchestrates everything"""
    
    def __init__(self):
        self.running = False
        self.client = OandaClient()
        self.open_positions = {}

        self.trade_day = datetime.utcnow().date()
        
        self.trades_today = 0
        
        self.analyzer = MultiPairAnalyzer()
        
        # Track open spread positions
        # Key: spread name, Value: {'pair1_units': X, 'pair2_units': Y}
        self.open_positions = {}

        # Cointegration health tracking.
        # Starts False -- pairs must PASS the daily screen before entries are allowed.
        # This means on first startup the bot will idle until the first 24h check runs
        # and positively confirms the pair. Prevents trading stale config on boot.
        self.spread_cointegrated = {f"{p1}/{p2}": False for p1, p2 in SPREADS}
        self.last_coint_check: datetime | None = None

        # Per-spread entry/exit z-scores -- updated by the daily screen as
        # the Kalman hedge ratio and OU half-life change over time.
        self.spread_entry_z: dict[str, float] = {}
        self.spread_exit_z:  dict[str, float] = {}
        for s in _SPREADS_CONFIG:
            sname = f"{s['pair1']}/{s['pair2']}"
            self.spread_entry_z[sname] = s.get('entry_z', ENTRY_Z_SCORE)
            self.spread_exit_z[sname]  = s.get('exit_z',  EXIT_Z_SCORE)

        # Set up spread analyzers with initial z-score values.
        # The daily screen will update entry/exit thresholds in-place.
        for pair1, pair2 in SPREADS:
            sname = f"{pair1}/{pair2}"
            self.analyzer.add_spread(
                pair1, pair2,
                lookback=LOOKBACK_PERIODS,
                entry_z=self.spread_entry_z.get(sname, ENTRY_Z_SCORE),
                exit_z=self.spread_exit_z.get(sname, EXIT_Z_SCORE),
            )
        
        print("[INIT] Bot initialized")
        print(f"[INIT] Tracking spreads: {[f'{p1}/{p2}' for p1, p2 in SPREADS]}")
        print(f"[INIT] Settings: lookback={LOOKBACK_PERIODS}, entry_z={ENTRY_Z_SCORE}, exit_z={EXIT_Z_SCORE}, blowout_z={ZSCORE_BLOWOUT}")
        print(f"[INIT] Trade units: {TRADE_UNITS}, Dry run: {DRY_RUN}")
        print(f"[INIT] Stop-loss: {STOP_LOSS_PIPS} pips, Close on shutdown: {CLOSE_ON_SHUTDOWN}")
    
    # =========================================================================
    # Reconcile positions from OANDA on startup
    # =========================================================================
    def reconcile_positions(self) -> int:
        """
        Query OANDA for open positions and reconstruct internal state.
        This ensures we don't orphan trades after a pod restart.
        
        Returns:
            Number of spread positions reconciled
        """
        print("\n[RECONCILE] Checking OANDA for existing positions...")
        
        oanda_positions = self.client.get_open_positions()
        if oanda_positions is None:
            print("[RECONCILE] Failed to fetch positions from OANDA")
            return 0
        
        # DEBUG: Show raw position data from OANDA
        if oanda_positions:
            print("[RECONCILE] Raw positions from OANDA:")
            for pos in oanda_positions:
                print(f"[RECONCILE]   {pos['instrument']}: long={pos['long_units']}, "
                      f"short={pos['short_units']}, net={pos['net_units']}")
        
        # Build a map of instrument -> net_units
        position_map = {}
        for pos in oanda_positions:
            if pos['net_units'] != 0:
                position_map[pos['instrument']] = pos['net_units']
                print(f"[RECONCILE] Active: {pos['instrument']} = {pos['net_units']:+.0f} units")
        
        if not position_map:
            print("[RECONCILE] No open positions at OANDA")
            return 0
        
        # Try to match positions to our spread definitions
        reconciled = 0
        for pair1, pair2 in SPREADS:
            spread_name = f"{pair1}/{pair2}"
            
            # Check if both legs of this spread are open
            if pair1 in position_map and pair2 in position_map:
                pair1_units = position_map[pair1]
                pair2_units = position_map[pair2]
                
                # Check if this is an inverse correlation pair
                hedge_ratio = HEDGE_RATIOS.get(spread_name, 1.0)
                is_inverse = hedge_ratio < 0
                
                # Determine the spread direction based on position signs
                if is_inverse:
                    # INVERSE pairs: both legs same direction
                    # LONG_SPREAD = both positive
                    # SHORT_SPREAD = both negative
                    if pair1_units > 0 and pair2_units > 0:
                        side = "LONG_SPREAD"
                    elif pair1_units < 0 and pair2_units < 0:
                        side = "SHORT_SPREAD"
                    else:
                        print(f"[RECONCILE] {spread_name}: inverse pair but opposite directions - invalid")
                        continue
                else:
                    # Normal pairs: opposite directions
                    # LONG_SPREAD = long pair1, short pair2
                    # SHORT_SPREAD = short pair1, long pair2
                    if pair1_units > 0 and pair2_units < 0:
                        side = "LONG_SPREAD"
                    elif pair1_units < 0 and pair2_units > 0:
                        side = "SHORT_SPREAD"
                    else:
                        print(f"[RECONCILE] {spread_name}: positions exist but not a valid spread (same direction)")
                        continue
                
                # Reconstruct the position tracking
                self.open_positions[spread_name] = {
                    'pair1_units': pair1_units,
                    'pair2_units': pair2_units,
                    'entry_z': None,  # Unknown - we lost this on restart
                    'entry_time': None,  # Unknown
                    'reconciled': True  # Flag that this was recovered
                }
                
                # Update the analyzer's internal state
                for analyzer in self.analyzer.analyzers.values():
                    if analyzer.name == spread_name:
                        analyzer.update_position_state(entered=True, side=side)
                
                inverse_note = " (inverse pair)" if is_inverse else ""
                print(f"[RECONCILE] âœ“ Recovered {spread_name}: {side}{inverse_note} "
                      f"({pair1}={pair1_units:+.0f}, {pair2}={pair2_units:+.0f})")
                reconciled += 1
                
                # Remove from map so we can detect orphans
                del position_map[pair1]
                del position_map[pair2]
        
        # Auto-close orphan positions -- no matching spread partner.
        # These are naked/unhedged leftovers from a config change or partial close.
        if position_map:
            print(f"[RECONCILE] WARNING: {len(position_map)} orphan position(s) -- auto-closing:")
            for inst, units in list(position_map.items()):
                print(f"[RECONCILE]   Closing orphan {inst}: {units:+.0f} units")
                if not DRY_RUN:
                    result = self.client.close_position(inst)
                    if result is not None:
                        print(f"[RECONCILE]   OK {inst} closed")
                    else:
                        print(f"[RECONCILE]   FAILED {inst} -- check OANDA manually")
                else:
                    print(f"[RECONCILE]   DRY RUN -- would close {inst}")

        print(f"[RECONCILE] Complete. Recovered {reconciled} spread position(s)\n")
        return reconciled
    # =========================================================================
    
    def _roll_trade_day_if_needed(self) -> None:
        today = datetime.utcnow().date()
        if today != self.trade_day:
            self.trade_day = today
            self.trades_today = 0

    def warm_up(self) -> bool:
        """Load historical data to warm up the z-score calculations"""
        print(f"\n[WARMUP] Loading historical data (granularity={GRANULARITY})...")
        
        # Fetch candles for all instruments
        candle_data = {}
        for instrument in INSTRUMENTS:
            candles = self.client.get_candles(
                instrument, 
                granularity=GRANULARITY, 
                count=LOOKBACK_PERIODS + 10  # Extra buffer
            )
            if candles:
                candle_data[instrument] = candles
                print(f"[WARMUP] {instrument}: loaded {len(candles)} candles")
            else:
                print(f"[WARMUP] Failed to load candles for {instrument}")
                return False
        
        # Load into analyzers
        for key, analyzer in self.analyzer.analyzers.items():
            if analyzer.pair1 in candle_data and analyzer.pair2 in candle_data:
                analyzer.load_historical_ratios(
                    candle_data[analyzer.pair1],
                    candle_data[analyzer.pair2]
                )
        
        print("[WARMUP] Complete\n")
        return True
    
    def get_current_prices(self) -> dict:
        """Fetch current prices for all instruments"""
        prices = {}
        for instrument in INSTRUMENTS:
            price_data = self.client.get_current_price(instrument)
            if price_data:
                prices[instrument] = price_data['mid']
        return prices
    
    # =========================================================================
    # Calculate USD value per unit for beta-weighted sizing
    # =========================================================================
    def get_usd_value_per_unit(self, instrument: str, prices: dict) -> float:
        """
        Calculate USD value of 1 unit of an instrument's base currency.
        
        Examples:
            EUR_USD at 1.17 â†’ 1 EUR = $1.17
            USD_CAD at 1.35 â†’ 1 USD = $1.00
            EUR_JPY at 184, EUR_USD at 1.17 â†’ 1 EUR = $1.17
        """
        base = instrument[:3]   # EUR, GBP, AUD, etc.
        quote = instrument[4:]  # USD, JPY, CAD, etc.
        
        if quote == 'USD':
            # XXX/USD - price IS the USD value per unit of base
            return prices.get(instrument, 1.0)
        elif base == 'USD':
            # USD/XXX - base is already USD, so 1 unit = $1
            return 1.0
        else:
            # XXX/YYY (e.g., EUR/JPY) - need XXX/USD rate
            usd_pair = f"{base}_USD"
            if usd_pair in prices:
                return prices[usd_pair]
            else:
                print(f"[WARN] Cannot determine USD value for {instrument}, using 1.0")
                return 1.0
    # =========================================================================
    
    def execute_spread_trade(self, signal: SpreadSignal, prices: dict) -> bool:
        """
        Execute a spread trade based on signal
        
        For POSITIVE correlation (normal):
            LONG_SPREAD: Buy pair1, Sell pair2
            SHORT_SPREAD: Sell pair1, Buy pair2
            
        For NEGATIVE correlation (inverse relationship like EUR_USD/USD_CHF):
            LONG_SPREAD: Buy pair1, Buy pair2 (both same direction)
            SHORT_SPREAD: Sell pair1, Sell pair2 (both same direction)
        
        Uses beta-weighted sizing for dollar-neutral exposure.
        """
        spread_name = f"{signal.pair1}/{signal.pair2}"
        now = datetime.utcnow().strftime('%H:%M')
        
        # =====================================================================
        # SAFETY CHECK -- only block if the conflicting instrument belongs to a
        # *tracked* spread already in self.open_positions. Orphan positions from
        # a prior deploy are cleaned up by reconcile; they must not block new entries.
        # =====================================================================
        if signal.signal in ['LONG_SPREAD', 'SHORT_SPREAD'] and not DRY_RUN:
            oanda_positions = self.client.get_open_positions()
            if oanda_positions:
                tracked_instruments = set()
                for sname, pos in self.open_positions.items():
                    p1, p2 = sname.split('/')
                    tracked_instruments.add(p1)
                    tracked_instruments.add(p2)
                oanda_open  = {p['instrument'] for p in oanda_positions if p['net_units'] != 0}
                conflicting = oanda_open & tracked_instruments
                if signal.pair1 in conflicting:
                    print(f"\n[TRADE {now}] BLOCKED: {signal.pair1} held by a tracked spread")
                    print(f"[TRADE {now}] Skipping {signal.signal} on {spread_name}")
                    return False
                if signal.pair2 in conflicting:
                    print(f"\n[TRADE {now}] BLOCKED: {signal.pair2} held by a tracked spread")
                    print(f"[TRADE {now}] Skipping {signal.signal} on {spread_name}")
                    return False
        # =====================================================================
        
        # Calculate beta-weighted units for dollar-neutral exposure
        pair1_usd_value = self.get_usd_value_per_unit(signal.pair1, prices)
        pair2_usd_value = self.get_usd_value_per_unit(signal.pair2, prices)
        
        # pair1 gets base TRADE_UNITS, pair2 adjusted to match USD exposure
        pair2_adjusted = int(TRADE_UNITS * pair1_usd_value / pair2_usd_value)
        
        # Check if this is an inverse correlation pair
        hedge_ratio = HEDGE_RATIOS.get(spread_name, 1.0)
        is_inverse = hedge_ratio < 0
        
        if signal.signal == 'LONG_SPREAD':
            pair1_units = TRADE_UNITS      # Buy
            if is_inverse:
                # Inverse: both same direction
                pair2_units = pair2_adjusted   # Also Buy
            else:
                pair2_units = -pair2_adjusted  # Sell (normal)
        elif signal.signal == 'SHORT_SPREAD':
            pair1_units = -TRADE_UNITS     # Sell
            if is_inverse:
                # Inverse: both same direction
                pair2_units = -pair2_adjusted  # Also Sell
            else:
                pair2_units = pair2_adjusted   # Buy (normal)
        elif signal.signal == 'CLOSE':
            # Reverse existing position - use ACTUAL units held, not recalculated
            if spread_name in self.open_positions:
                pos = self.open_positions[spread_name]
                pair1_units = -pos['pair1_units']
                pair2_units = -pos['pair2_units']
            else:
                print(f"[TRADE] No position to close for {spread_name}")
                return False
        else:
            return False
        
        print(f"\n[TRADE {now}] {'=' * 50}")
        print(f"[TRADE {now}] Signal: {signal.signal} on {spread_name}")
        print(f"[TRADE {now}] Z-Score: {signal.z_score:.4f}")
        # Show the beta weighting calculation
        if signal.signal != 'CLOSE':
            inverse_note = " (INVERSE pair)" if is_inverse else ""
            print(f"[TRADE {now}] Beta weighting: {signal.pair1}=${pair1_usd_value:.4f}/unit, {signal.pair2}=${pair2_usd_value:.4f}/unit{inverse_note}")
        print(f"[TRADE {now}] {signal.pair1}: {pair1_units:+.0f} units")
        print(f"[TRADE {now}] {signal.pair2}: {pair2_units:+.0f} units")
        
        if DRY_RUN:
            print(f"[TRADE {now}] DRY RUN - No actual orders placed")
            success = True
        elif signal.signal == 'CLOSE':
            # Use close_position endpoint (targets open position directly, no stop-loss needed)
            result1 = self.client.close_position(signal.pair1)
            result2 = self.client.close_position(signal.pair2)

            if result1 is not None:
                print(f"[TRADE {now}] {signal.pair1} close order accepted")
            else:
                print(f"[TRADE {now}] {signal.pair1} close order rejected/failed - verifying OANDA state...")
            if result2 is not None:
                print(f"[TRADE {now}] {signal.pair2} close order accepted")
            else:
                print(f"[TRADE {now}] {signal.pair2} close order rejected/failed - verifying OANDA state...")

            # Ground truth: check what OANDA actually has open right now.
            # A reject can mean "already flat" (stop-loss fired, manual close, etc.)
            # Don't rely on the API return value alone to decide state.
            actual_positions = self.client.get_open_positions() or []
            still_open = {p['instrument'] for p in actual_positions if p['net_units'] != 0}

            leg1_flat = signal.pair1 not in still_open
            leg2_flat = signal.pair2 not in still_open
            success = leg1_flat and leg2_flat

            if not leg1_flat:
                print(f"[TRADE {now}] {signal.pair1} still open at OANDA - will retry next poll")
            if not leg2_flat:
                print(f"[TRADE {now}] {signal.pair2} still open at OANDA - will retry next poll")
            if success:
                print(f"[TRADE {now}] Both legs confirmed flat at OANDA")
        else:
            # Entry orders - attach stop-loss
            result1 = self.client.place_market_order(signal.pair1, pair1_units, stop_loss_pips=STOP_LOSS_PIPS)
            result2 = self.client.place_market_order(signal.pair2, pair2_units, stop_loss_pips=STOP_LOSS_PIPS)
            success = result1 is not None and result2 is not None

            if result1 is not None:
                print(f"[TRADE {now}] {signal.pair1} filled at {result1['price']}")
            else:
                print(f"[TRADE {now}] FAILED to fill {signal.pair1}")
            if result2 is not None:
                print(f"[TRADE {now}] {signal.pair2} filled at {result2['price']}")
            else:
                print(f"[TRADE {now}] FAILED to fill {signal.pair2}")

            if not success:
                print(f"[TRADE {now}] Order execution failed!")

        # Update position tracking
        if success:
            if signal.signal == 'CLOSE':
                if spread_name in self.open_positions:
                    del self.open_positions[spread_name]
                # Update analyzer state
                for a in self.analyzer.analyzers.values():
                    if a.name == spread_name:
                        a.update_position_state(exited=True)
            else:
                self.open_positions[spread_name] = {
                    'pair1_units': pair1_units,
                    'pair2_units': pair2_units,
                    'entry_z': signal.z_score,
                    'entry_time': signal.timestamp
                }
                # Update analyzer state
                for a in self.analyzer.analyzers.values():
                    if a.name == spread_name:
                        a.update_position_state(entered=True, side=signal.signal)
        
        print(f"[TRADE {now}] {'=' * 50}\n")
        return success
    
    def print_status(self, prices: dict, signals: list) -> None:
        """Print current status to console"""
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\n[{now}] Status Update")
        print("-" * 60)
        
        # Current prices
        print("Prices:")
        for inst, price in prices.items():
            print(f"  {inst}: {price:.5f}")
        
        # Spread status
        print("\nSpreads:")
        for sig in signals:
            spread_name = f"{sig.pair1}/{sig.pair2}"
            pos_indicator = ""
            if spread_name in self.open_positions:
                pos = self.open_positions[spread_name]
                side = "LONG" if pos['pair1_units'] > 0 else "SHORT"
                # Show if position was reconciled from restart
                reconciled_flag = " (reconciled)" if pos.get('reconciled') else ""
                pos_indicator = f" [POSITION: {side}{reconciled_flag}]"
            
            print(f"  {spread_name}:")
            print(f"    Ratio: {sig.ratio:.6f} | Z-Score: {sig.z_score:+.4f} | Signal: {sig.signal}{pos_indicator}")
        
        # Account summary
        if not DRY_RUN:
            acct = self.client.get_account_summary()
            if acct:
                print(f"\nAccount: Balance=${acct['balance']:.2f}, Unrealized P/L=${acct['unrealized_pl']:.2f}")
        
        print("-" * 60)
    
    def check_zscore_blowout(self, signals: list) -> None:
        """
        Fast real-time check run every poll.
        If the z-score on an open position exceeds ZSCORE_BLOWOUT in the wrong
        direction, the spread is behaving abnormally — exit immediately rather
        than waiting for the 24h Engle-Granger check.

        "Wrong direction" means the spread kept moving against the position
        instead of reverting, i.e. a LONG_SPREAD position with z-score going
        further negative, or a SHORT_SPREAD with z-score going further positive.
        """
        prices = None  # Lazy-fetch only if we need to close

        for sig in signals:
            spread_name = f"{sig.pair1}/{sig.pair2}"

            if spread_name not in self.open_positions:
                continue  # Not in a position, nothing to protect

            pos = self.open_positions[spread_name]
            z = sig.z_score
            side = 'LONG_SPREAD' if pos['pair1_units'] > 0 else 'SHORT_SPREAD'

            # Check if z-score is blowing out in the wrong direction
            blowout = (
                (side == 'LONG_SPREAD'  and z <= -ZSCORE_BLOWOUT) or
                (side == 'SHORT_SPREAD' and z >=  ZSCORE_BLOWOUT)
            )

            if blowout:
                print(f"\n[BLOWOUT] {spread_name}: z={z:+.4f} exceeded ±{ZSCORE_BLOWOUT} threshold on {side}")
                print(f"[BLOWOUT] Spread is diverging — closing position to limit loss")

                if prices is None:
                    prices = self.get_current_prices()

                fake_signal = SpreadSignal(
                    pair1=sig.pair1,
                    pair2=sig.pair2,
                    ratio=sig.ratio,
                    z_score=z,
                    mean=sig.mean,
                    std=sig.std,
                    signal='CLOSE',
                    timestamp=datetime.utcnow(),
                )
                self.execute_spread_trade(fake_signal, prices)

    def check_cointegration(self) -> None:
        """
        Full cointegration screen -- runs every COINT_CHECK_INTERVAL_HOURS.
        Uses the same methodology as cointegration_analyzer.py:
          1. Kalman filter hedge ratio on the 180-day rolling window
          2. Dynamic spread = pair1 - kalman_beta * pair2
          3. Engle-Granger cointegration test on the dynamic spread
          4. OU fit for kappa and half-life

        A pair becomes entry-eligible when ALL pass:
          - p_value < COINT_P_THRESHOLD
          - kappa > 0
          - COINT_MIN_HALF_LIFE <= half_life <= COINT_MAX_HALF_LIFE

        When a pair transitions fail->pass, entries are enabled and
        the analyzer's z-score thresholds are updated from the OU fit.
        When it transitions pass->fail, any open position is force-closed.
        """
        now = datetime.utcnow()

        if self.last_coint_check is not None:
            hours_since = (now - self.last_coint_check).total_seconds() / 3600
            if hours_since < COINT_CHECK_INTERVAL_HOURS:
                return

        print(f"\n[COINT] Running full cointegration screen ({COINT_LOOKBACK_DAYS}-day window)...")
        self.last_coint_check = now

        for pair1, pair2 in SPREADS:
            spread_name = f"{pair1}/{pair2}"
            was_eligible = self.spread_cointegrated.get(spread_name, False)

            # --- fetch daily candles ---
            candles1 = self.client.get_candles(pair1, granularity='D', count=COINT_LOOKBACK_DAYS)
            candles2 = self.client.get_candles(pair2, granularity='D', count=COINT_LOOKBACK_DAYS)
            if not candles1 or not candles2:
                print(f"[COINT] {spread_name}: candle fetch failed, skipping")
                continue

            # --- align on date ---
            closes1 = {c['time'][:10]: c['close'] for c in candles1}
            closes2 = {c['time'][:10]: c['close'] for c in candles2}
            common  = sorted(set(closes1) & set(closes2))
            if len(common) < 60:
                print(f"[COINT] {spread_name}: only {len(common)} overlapping days, need 60+")
                continue

            s1 = pd.Series([closes1[d] for d in common], dtype=float)
            s2 = pd.Series([closes2[d] for d in common], dtype=float)

            # --- Kalman hedge ratio (same as analyzer) ---
            beta, P, R, Q = 0.0, 1.0, 0.001, 0.01
            betas = []
            for yt, xt in zip(s1, s2):
                P_pred = P + Q
                if xt == 0:
                    betas.append(beta); continue
                e = yt - beta * xt
                S = xt * xt * P_pred + R
                K = P_pred * xt / S
                beta = beta + K * e
                P    = (1 - K * xt) * P_pred
                betas.append(beta)
            kalman_betas = pd.Series(betas, dtype=float)
            kalman_latest = kalman_betas.iloc[-1]

            # --- dynamic spread ---
            spread = s1.values - kalman_betas.values * s2.values
            spread = pd.Series(spread, dtype=float).dropna()

            # --- cointegration test on aligned raw series ---
            try:
                _, p_value, _ = coint(s1, s2)
            except Exception as exc:
                print(f"[COINT] {spread_name}: coint() error: {exc}")
                continue

            # --- OU fit for kappa + half-life ---
            kappa, half_life = float('nan'), float('inf')
            if len(spread) >= 20:
                x_ou = spread.shift(1).dropna()
                y_ou = spread.iloc[1:]
                try:
                    model = OLS(y_ou.values, add_constant(x_ou.values)).fit()
                    a_ou, b_ou = model.params[0], model.params[1]
                    if 0 < b_ou < 1:
                        kappa     = -np.log(b_ou)
                        half_life =  np.log(2) / kappa
                except Exception as exc:
                    print(f"[COINT] {spread_name}: OU fit error: {exc}")

            # --- gate: all three criteria must pass ---
            p_ok  = p_value   < COINT_P_THRESHOLD
            k_ok  = np.isfinite(kappa) and kappa > 0
            hl_ok = np.isfinite(half_life) and COINT_MIN_HALF_LIFE <= half_life <= COINT_MAX_HALF_LIFE
            now_eligible = p_ok and k_ok and hl_ok

            hl_str = f"{half_life:.1f}d" if np.isfinite(half_life) else "inf"
            k_str  = f"{kappa:.4f}"      if np.isfinite(kappa)     else "nan"
            status = "PASS" if now_eligible else "FAIL"
            reasons = []
            if not p_ok:  reasons.append(f"p={p_value:.4f}>={COINT_P_THRESHOLD}")
            if not k_ok:  reasons.append(f"kappa={k_str}<=0")
            if not hl_ok: reasons.append(f"HL={hl_str} outside [{COINT_MIN_HALF_LIFE},{COINT_MAX_HALF_LIFE}]d")
            reason_str = ", ".join(reasons) if reasons else "all criteria met"
            print(f"[COINT] {spread_name}: {status} | p={p_value:.4f} kappa={k_str} HL={hl_str} hedge={kalman_latest:.4f} | {reason_str}")

            self.spread_cointegrated[spread_name] = now_eligible

            # --- transition: fail -> pass ---
            if now_eligible and not was_eligible:
                print(f"[COINT] {spread_name}: NOW ELIGIBLE -- updating z-score thresholds")
                # Derive entry_z from half-life using the same heuristic as the analyzer
                if   half_life < 5:  entry_z = ENTRY_Z_SCORE + 0.3
                elif half_life <= 15: entry_z = ENTRY_Z_SCORE
                elif half_life <= 40: entry_z = ENTRY_Z_SCORE + 0.2
                else:                 entry_z = ENTRY_Z_SCORE + 0.4
                exit_z = EXIT_Z_SCORE
                self.spread_entry_z[spread_name] = entry_z
                self.spread_exit_z[spread_name]  = exit_z
                # Update the live analyzer so it uses the new thresholds immediately
                for a in self.analyzer.analyzers.values():
                    if a.name == spread_name:
                        a.entry_z = entry_z
                        a.exit_z  = exit_z
                        print(f"[COINT] {spread_name}: entry_z={entry_z:.2f} exit_z={exit_z:.2f} (HL={hl_str})")
                # Also refresh the hedge ratio in HEDGE_RATIOS so sizing is current
                HEDGE_RATIOS[spread_name] = kalman_latest

            # --- transition: pass -> fail ---
            elif not now_eligible and was_eligible:
                print(f"[COINT] {spread_name}: NO LONGER ELIGIBLE -- disabling entries")
                if spread_name in self.open_positions:
                    print(f"[COINT] {spread_name}: open position detected -- force closing")
                    prices = self.get_current_prices()
                    fake_signal = SpreadSignal(
                        pair1=pair1, pair2=pair2,
                        ratio=0, z_score=0, mean=0, std=0,
                        signal='CLOSE', timestamp=now,
                    )
                    self.execute_spread_trade(fake_signal, prices)

        eligible = [k for k, v in self.spread_cointegrated.items() if v]
        ineligible = [k for k, v in self.spread_cointegrated.items() if not v]
        print(f"[COINT] Screen complete | eligible: {eligible or 'none'} | idle: {ineligible or 'none'}\n")

    def run_once(self) -> None:
        """Run a single iteration of the bot logic"""
        # Cointegration health check (throttled internally to COINT_CHECK_INTERVAL_HOURS)
        self.check_cointegration()

        # Get current prices
        prices = self.get_current_prices()
        if len(prices) != len(INSTRUMENTS):
            print(f"[WARN] Only got prices for {list(prices.keys())}, expected {INSTRUMENTS}")
            return

        # Get signals for all spreads
        signals = self.analyzer.get_all_signals(prices)

        # Fast blowout check — runs every poll, exits immediately if z-score diverges too far
        self.check_zscore_blowout(signals)

        # Print status
        self.print_status(prices, signals)

        # Act on signals
        for sig in signals:
            spread_name = f"{sig.pair1}/{sig.pair2}"
            is_entry = sig.signal in ['LONG_SPREAD', 'SHORT_SPREAD']
            is_close = sig.signal == 'CLOSE'

            # Block new entries on spreads that failed cointegration check
            if is_entry and not self.spread_cointegrated.get(spread_name, True):
                print(f"[SKIP] {spread_name}: cointegration failed, not entering new position")
                continue

            if is_entry or is_close:
                self.execute_spread_trade(sig, prices)
    
    def run(self) -> None:
        """Main bot loop"""
        print("\n[BOT] Starting trading bot...")
        
        # Verify API connection
        acct = self.client.get_account_summary()
        if acct is None:
            print("[ERROR] Failed to connect to OANDA API. Check your credentials.")
            sys.exit(1)
        
        print(f"[BOT] Connected to OANDA. Account balance: ${acct['balance']:.2f}")
        
        # Warm up with historical data
        if not self.warm_up():
            print("[ERROR] Failed to warm up with historical data")
            sys.exit(1)
        
        # =====================================================================
        # Reconcile any existing positions from OANDA
        # =====================================================================
        self.reconcile_positions()
        # =====================================================================
        
        # Main loop
        self.running = True
        print(f"[BOT] Entering main loop (poll interval: {POLL_INTERVAL}s)")
        print("[BOT] Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                self.run_once()
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                print("\n[BOT] Shutting down...")
                self.running = False
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                time.sleep(POLL_INTERVAL)
    
    def stop(self) -> None:
        """Stop the bot gracefully"""
        self.running = False


def main():
    # Handle SIGTERM for graceful k8s shutdown
    bot = TradingBot()
    
    def handle_sigterm(signum, frame):
        print("\n[BOT] Received SIGTERM, shutting down...")
        if CLOSE_ON_SHUTDOWN and not DRY_RUN:
            print("[BOT] Closing all open positions...")
            closed = bot.client.close_all_positions()
            print(f"[BOT] Closed {closed} positions")
        bot.stop()
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Run the bot
    bot.run()
    
    print("[BOT] Goodbye!")


if __name__ == "__main__":
    main()