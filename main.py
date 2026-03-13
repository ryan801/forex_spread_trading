"""
Forex Pairs Trading Bot - Multi-Spread Edition
Main entry point - runs the trading loop

Supports multiple spreads with different timeframes:
- EUR_JPY/GBP_JPY on Weekly
- EUR_USD/GBP_USD on Daily
"""

import os
import sys
import time
import signal
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

from oanda_client import OandaClient
from pairs_analyzer import PairsAnalyzer, SpreadSignal


# =============================================================================
# Configuration
# =============================================================================

POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '3600'))  # 1 hour default
TRADE_UNITS = int(os.environ.get('TRADE_UNITS', '10000'))
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'
SPREAD_MAX_LOSS_USD = float(os.environ.get('SPREAD_MAX_LOSS_USD', '200'))
CLOSE_ON_SHUTDOWN = os.environ.get('CLOSE_ON_SHUTDOWN', 'true').lower() == 'true'


# =============================================================================
# Spread Configurations - Each spread has its own settings
# =============================================================================

@dataclass
class SpreadConfig:
    """Configuration for a single spread strategy"""
    pair1: str
    pair2: str
    granularity: str      # 'D' for daily, 'W' for weekly
    entry_z: float
    exit_z: float
    lookback: int
    hedge_ratio: float    # From backtest - positive = normal, negative = inverse
    description: str


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CYAN: Updated spread configurations based on backtest results              │
# └─────────────────────────────────────────────────────────────────────────────┘
SPREAD_CONFIGS = [
    SpreadConfig(
        pair1='EUR_JPY',
        pair2='GBP_JPY',
        granularity='W',      # Weekly
        entry_z=1.5,
        exit_z=0.2,
        lookback=20,
        hedge_ratio=0.8188,   # From backtest
        description='EUR vs GBP (JPY crosses) - Weekly'
    ),
    SpreadConfig(
        pair1='AUD_USD',
        pair2='NZD_USD',
        granularity='D',      # Daily
        entry_z=1.5,
        exit_z=0.2,
        lookback=20,
        hedge_ratio=0.8715,   # From backtest
        description='AUD vs NZD (USD crosses) - Daily'
    ),
]

# Build instrument list from configs
INSTRUMENTS = list(set(
    inst for cfg in SPREAD_CONFIGS for inst in [cfg.pair1, cfg.pair2]
))


# =============================================================================
# Trading Bot
# =============================================================================

class TradingBot:
    """Main bot class that orchestrates multiple spread strategies"""
    
    def __init__(self):
        self.running = False
        self.client = OandaClient()
        
        # Track open positions per spread
        # Key: spread name, Value: {'pair1_units': X, 'pair2_units': Y, ...}
        self.open_positions: Dict[str, dict] = {}
        
        # Track last check time per granularity to avoid over-checking
        self.last_check: Dict[str, datetime] = {}
        
        # Create analyzer for each spread with its specific config
        self.analyzers: Dict[str, PairsAnalyzer] = {}
        self.configs: Dict[str, SpreadConfig] = {}
        
        for cfg in SPREAD_CONFIGS:
            spread_name = f"{cfg.pair1}/{cfg.pair2}"
            self.analyzers[spread_name] = PairsAnalyzer(
                cfg.pair1,
                cfg.pair2,
                lookback=cfg.lookback,
                entry_z=cfg.entry_z,
                exit_z=cfg.exit_z,
                mode='hedged_spread',
                hedge_ratio=cfg.hedge_ratio,
            )
            self.configs[spread_name] = cfg
        
        print("[INIT] Multi-Spread Trading Bot initialized")
        print(f"[INIT] Tracking {len(SPREAD_CONFIGS)} spreads:")
        for cfg in SPREAD_CONFIGS:
            print(f"[INIT]   • {cfg.pair1}/{cfg.pair2} ({cfg.granularity}) - "
                  f"entry_z={cfg.entry_z}, exit_z={cfg.exit_z}, lookback={cfg.lookback}")
        print(f"[INIT] Trade units: {TRADE_UNITS}, Dry run: {DRY_RUN}")
        print(f"[INIT] Spread stop-loss: ${SPREAD_MAX_LOSS_USD} combined P&L")
    
    # =========================================================================
    # Warm Up - Load historical data for each spread
    # =========================================================================
    def warm_up(self) -> bool:
        """Load historical data for each spread based on its granularity"""
        print("\n[WARMUP] Loading historical data for all spreads...")
        
        success = True
        
        for spread_name, cfg in self.configs.items():
            print(f"\n[WARMUP] {spread_name} ({cfg.granularity})...")
            
            # Fetch candles for both pairs
            candles1 = self.client.get_candles(
                cfg.pair1,
                granularity=cfg.granularity,
                count=cfg.lookback + 10
            )
            candles2 = self.client.get_candles(
                cfg.pair2,
                granularity=cfg.granularity,
                count=cfg.lookback + 10
            )
            
            if not candles1 or not candles2:
                print(f"[WARMUP] ✗ Failed to load data for {spread_name}")
                success = False
                continue
            
            # Load into analyzer
            analyzer = self.analyzers[spread_name]
            loaded = analyzer.load_historical_ratios(candles1, candles2)
            print(f"[WARMUP] ✓ Loaded {loaded} historical ratios")
            
            # Initialize last check time
            self.last_check[cfg.granularity] = datetime.utcnow()
        
        print("\n[WARMUP] Complete")
        return success
    
    # =========================================================================
    # Reconcile positions from OANDA on startup
    # =========================================================================
    def reconcile_positions(self) -> int:
        """Query OANDA for open positions and reconstruct internal state"""
        print("\n[RECONCILE] Checking OANDA for existing positions...")
        
        oanda_positions = self.client.get_open_positions()
        if oanda_positions is None:
            print("[RECONCILE] Failed to fetch positions from OANDA")
            return 0
        
        # Build map of instrument -> net_units
        position_map = {}
        for pos in oanda_positions:
            if pos['net_units'] != 0:
                position_map[pos['instrument']] = pos['net_units']
                print(f"[RECONCILE] Found: {pos['instrument']} = {pos['net_units']:+.0f} units")
        
        if not position_map:
            print("[RECONCILE] No open positions at OANDA")
            return 0
        
        reconciled = 0
        
        for spread_name, cfg in self.configs.items():
            pair1, pair2 = cfg.pair1, cfg.pair2
            
            if pair1 in position_map and pair2 in position_map:
                pair1_units = position_map[pair1]
                pair2_units = position_map[pair2]
                
                # Determine spread direction
                # Both positive correlation pairs: LONG = long pair1, short pair2
                if pair1_units > 0 and pair2_units < 0:
                    side = "LONG_SPREAD"
                elif pair1_units < 0 and pair2_units > 0:
                    side = "SHORT_SPREAD"
                else:
                    # Same direction - might be valid for some strategies
                    if pair1_units > 0:
                        side = "LONG_SPREAD"
                    else:
                        side = "SHORT_SPREAD"
                
                self.open_positions[spread_name] = {
                    'pair1_units': pair1_units,
                    'pair2_units': pair2_units,
                    'entry_z': None,
                    'entry_time': None,
                    'reconciled': True
                }
                
                self.analyzers[spread_name].update_position_state(entered=True, side=side)
                
                print(f"[RECONCILE] ✓ Recovered {spread_name}: {side}")
                reconciled += 1
                
                del position_map[pair1]
                del position_map[pair2]
        
        if position_map:
            print(f"[RECONCILE] ⚠ Unmatched positions:")
            for inst, units in position_map.items():
                print(f"[RECONCILE]   {inst}: {units:+.0f} units")
        
        print(f"[RECONCILE] Complete. Recovered {reconciled} spread position(s)\n")
        return reconciled
    
    # =========================================================================
    # Check if we should evaluate a spread based on its timeframe
    # =========================================================================
    def should_check_spread(self, cfg: SpreadConfig) -> bool:
        """
        Determine if we should check this spread based on its granularity.
        
        Daily spreads: Check once per day (after daily bar closes)
        Weekly spreads: Check once per week (after weekly bar closes)
        """
        now = datetime.utcnow()
        last = self.last_check.get(cfg.granularity, datetime.min)
        
        if cfg.granularity == 'D':
            # Check if we've crossed into a new day (5pm ET = 10pm UTC for forex)
            # Simplified: just check if 24+ hours since last check
            hours_since = (now - last).total_seconds() / 3600
            return hours_since >= 23  # Check roughly once per day
        
        elif cfg.granularity == 'W':
            # Check if we've crossed into a new week
            # Weekly bars close Friday 5pm ET
            days_since = (now - last).days
            return days_since >= 6  # Check roughly once per week
        
        return True  # Default: always check
    
    # =========================================================================
    # Get USD value per unit for position sizing
    # =========================================================================
    def get_usd_value_per_unit(self, instrument: str, prices: dict) -> float:
        """Calculate USD value of 1 unit of an instrument's base currency"""
        base = instrument[:3]
        quote = instrument[4:]
        
        if quote == 'USD':
            return prices.get(instrument, 1.0)
        elif base == 'USD':
            return 1.0
        else:
            # Cross pair - need to find USD conversion
            usd_pair = f"{base}_USD"
            if usd_pair in prices:
                return prices[usd_pair]
            # Try inverse
            usd_pair_inv = f"USD_{base}"
            if usd_pair_inv in prices:
                return 1.0 / prices[usd_pair_inv]
            print(f"[WARN] Cannot find USD rate for {instrument}")
            return 1.0
    
    # =========================================================================
    # Execute a spread trade
    # =========================================================================
    def execute_spread_trade(self, spread_name: str, signal: SpreadSignal, 
                            prices: dict) -> bool:
        """Execute a spread trade based on signal"""
        cfg = self.configs[spread_name]
        now = datetime.utcnow().strftime('%H:%M')
        
        # Safety check - verify no existing position on these instruments
        if signal.signal in ['LONG_SPREAD', 'SHORT_SPREAD'] and not DRY_RUN:
            oanda_positions = self.client.get_open_positions()
            if oanda_positions:
                for pos in oanda_positions:
                    if pos['net_units'] != 0:
                        if pos['instrument'] in [cfg.pair1, cfg.pair2]:
                            print(f"[TRADE {now}] BLOCKED: {pos['instrument']} already has position")
                            return False
        
        # Use cointegration hedge ratio for position sizing:
        # pair2_units = hedge_ratio * pair1_units (keeps the spread stationary)
        pair2_hedge_units = int(TRADE_UNITS * cfg.hedge_ratio)

        # Determine trade direction
        # For positive correlation pairs: LONG = buy pair1, sell pair2
        if signal.signal == 'LONG_SPREAD':
            pair1_units = TRADE_UNITS
            pair2_units = -pair2_hedge_units
        elif signal.signal == 'SHORT_SPREAD':
            pair1_units = -TRADE_UNITS
            pair2_units = pair2_hedge_units
        elif signal.signal == 'CLOSE':
            if spread_name in self.open_positions:
                pos = self.open_positions[spread_name]
                pair1_units = -pos['pair1_units']
                pair2_units = -pos['pair2_units']
            else:
                print(f"[TRADE] No position to close for {spread_name}")
                return False
        else:
            return False
        
        print(f"\n[TRADE {now}] {'='*50}")
        print(f"[TRADE {now}] {spread_name} ({cfg.granularity})")
        print(f"[TRADE {now}] Signal: {signal.signal}")
        print(f"[TRADE {now}] Z-Score: {signal.z_score:.4f}")
        print(f"[TRADE {now}] {cfg.pair1}: {pair1_units:+.0f} units")
        print(f"[TRADE {now}] {cfg.pair2}: {pair2_units:+.0f} units")
        
        if DRY_RUN:
            print(f"[TRADE {now}] DRY RUN - No actual orders placed")
            success = True
        else:
            # No per-leg stop-losses: individual stops break the hedge by closing one
            # leg while leaving the other open. Spread-level stops are used instead.
            result1 = self.client.place_market_order(cfg.pair1, pair1_units)
            result2 = self.client.place_market_order(cfg.pair2, pair2_units)
            success = result1 is not None and result2 is not None
            
            if success:
                print(f"[TRADE {now}] ✓ {cfg.pair1} filled at {result1['price']}")
                print(f"[TRADE {now}] ✓ {cfg.pair2} filled at {result2['price']}")
            else:
                print(f"[TRADE {now}] ✗ Order execution failed!")
        
        # Update position tracking
        if success:
            if signal.signal == 'CLOSE':
                if spread_name in self.open_positions:
                    del self.open_positions[spread_name]
                self.analyzers[spread_name].update_position_state(exited=True)
            else:
                self.open_positions[spread_name] = {
                    'pair1_units': pair1_units,
                    'pair2_units': pair2_units,
                    'entry_z': signal.z_score,
                    'entry_time': signal.timestamp
                }
                self.analyzers[spread_name].update_position_state(
                    entered=True, side=signal.signal
                )
        
        print(f"[TRADE {now}] {'='*50}\n")
        return success
    
    # =========================================================================
    # Monitor open spreads against OANDA's actual positions
    # =========================================================================
    def check_leg_closures(self, oanda_map: dict) -> None:
        """
        Detect if OANDA closed one leg of a spread externally (e.g., margin call,
        manual close) and immediately close the surviving leg to avoid unhedged exposure.

        oanda_map: {instrument: net_units} from get_open_positions()
        """
        spreads_to_remove = []

        for spread_name, pos in self.open_positions.items():
            cfg = self.configs[spread_name]
            pair1_open = cfg.pair1 in oanda_map
            pair2_open = cfg.pair2 in oanda_map

            if not pair1_open and not pair2_open:
                print(f"[MONITOR] {spread_name}: both legs closed externally, removing from tracking")
                spreads_to_remove.append(spread_name)
            elif not pair1_open and pair2_open:
                print(f"[MONITOR] {spread_name}: {cfg.pair1} was closed externally — closing {cfg.pair2} to eliminate unhedged exposure")
                if not DRY_RUN:
                    self.client.close_position(cfg.pair2)
                spreads_to_remove.append(spread_name)
            elif pair1_open and not pair2_open:
                print(f"[MONITOR] {spread_name}: {cfg.pair2} was closed externally — closing {cfg.pair1} to eliminate unhedged exposure")
                if not DRY_RUN:
                    self.client.close_position(cfg.pair1)
                spreads_to_remove.append(spread_name)

        for spread_name in spreads_to_remove:
            del self.open_positions[spread_name]
            self.analyzers[spread_name].update_position_state(exited=True)

    def check_spread_stops(self, oanda_positions: list) -> None:
        """
        Close both legs of a spread if the combined unrealized P&L exceeds
        SPREAD_MAX_LOSS_USD. This replaces per-leg stop-losses.
        """
        if not self.open_positions or DRY_RUN:
            return

        pl_map = {pos['instrument']: pos['unrealized_pl'] for pos in oanda_positions}

        spreads_to_close = []
        for spread_name, pos in self.open_positions.items():
            cfg = self.configs[spread_name]
            pl1 = pl_map.get(cfg.pair1, 0.0)
            pl2 = pl_map.get(cfg.pair2, 0.0)
            total_pl = pl1 + pl2

            if total_pl <= -SPREAD_MAX_LOSS_USD:
                print(f"[STOP] {spread_name}: combined P&L ${total_pl:.2f} hit spread stop "
                      f"(limit -${SPREAD_MAX_LOSS_USD:.2f}). Closing both legs.")
                spreads_to_close.append(spread_name)

        for spread_name in spreads_to_close:
            cfg = self.configs[spread_name]
            self.client.close_position(cfg.pair1)
            self.client.close_position(cfg.pair2)
            del self.open_positions[spread_name]
            self.analyzers[spread_name].update_position_state(exited=True)

    # =========================================================================
    # Get current prices for all instruments
    # =========================================================================
    def get_current_prices(self) -> dict:
        """Fetch current prices for all instruments we need"""
        prices = {}
        
        # Get prices for spread instruments
        for inst in INSTRUMENTS:
            price_data = self.client.get_current_price(inst)
            if price_data:
                prices[inst] = price_data['mid']
        
        # Also fetch USD crosses for position sizing
        for base in ['EUR', 'GBP', 'JPY']:
            usd_pair = f"{base}_USD"
            if usd_pair not in prices:
                price_data = self.client.get_current_price(usd_pair)
                if price_data:
                    prices[usd_pair] = price_data['mid']
        
        return prices
    
    # =========================================================================
    # Print status
    # =========================================================================
    def print_status(self, prices: dict) -> None:
        """Print current status"""
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\n[{now}] Status Update")
        print("-" * 60)
        
        for spread_name, analyzer in self.analyzers.items():
            cfg = self.configs[spread_name]
            status = analyzer.get_status()
            
            pos_str = ""
            if spread_name in self.open_positions:
                pos = self.open_positions[spread_name]
                side = "LONG" if pos['pair1_units'] > 0 else "SHORT"
                pos_str = f" [POSITION: {side}]"
            
            print(f"\n  {spread_name} ({cfg.granularity}):")
            print(f"    Z-Score: {status['z_score']:+.4f}")
            print(f"    Entry threshold: ±{cfg.entry_z}")
            print(f"    Ready: {status['ready']}{pos_str}")
        
        if not DRY_RUN:
            acct = self.client.get_account_summary()
            if acct:
                print(f"\n  Account: ${acct['balance']:.2f}, "
                      f"Unrealized P/L: ${acct['unrealized_pl']:.2f}")
        
        print("-" * 60)
    
    # =========================================================================
    # Main loop iteration
    # =========================================================================
    def run_once(self) -> None:
        """Run a single iteration of the bot logic"""
        prices = self.get_current_prices()

        if len(prices) < len(INSTRUMENTS):
            print(f"[WARN] Missing prices for some instruments")
            return

        # Check OANDA's actual positions to detect externally-closed legs and
        # enforce spread-level stops. Do this before evaluating new signals.
        if self.open_positions and not DRY_RUN:
            oanda_positions = self.client.get_open_positions()
            if oanda_positions is not None:
                oanda_map = {
                    pos['instrument']: pos['net_units']
                    for pos in oanda_positions
                    if pos['net_units'] != 0
                }
                self.check_leg_closures(oanda_map)
                self.check_spread_stops(oanda_positions)

        # Check each spread
        for spread_name, analyzer in self.analyzers.items():
            cfg = self.configs[spread_name]
            
            # Only check if appropriate for this timeframe
            if not self.should_check_spread(cfg):
                continue
            
            # Get current prices for this spread
            if cfg.pair1 not in prices or cfg.pair2 not in prices:
                continue
            
            # Get signal
            signal = analyzer.get_signal(prices[cfg.pair1], prices[cfg.pair2])
            
            # Act on actionable signals
            if signal.signal in ['LONG_SPREAD', 'SHORT_SPREAD', 'CLOSE']:
                self.execute_spread_trade(spread_name, signal, prices)
        
        # Update last check times
        now = datetime.utcnow()
        for cfg in SPREAD_CONFIGS:
            if self.should_check_spread(cfg):
                self.last_check[cfg.granularity] = now
        
        # Print status
        self.print_status(prices)
    
    # =========================================================================
    # Main run loop
    # =========================================================================
    def run(self) -> None:
        """Main bot loop"""
        print("\n[BOT] Starting Multi-Spread Trading Bot...")
        
        # Verify API connection
        acct = self.client.get_account_summary()
        if acct is None:
            print("[ERROR] Failed to connect to OANDA API")
            sys.exit(1)
        
        print(f"[BOT] Connected to OANDA. Balance: ${acct['balance']:.2f}")
        
        # Warm up
        if not self.warm_up():
            print("[ERROR] Failed to warm up")
            sys.exit(1)
        
        # Reconcile positions
        self.reconcile_positions()
        
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
                import traceback
                traceback.print_exc()
                time.sleep(POLL_INTERVAL)
    
    def stop(self) -> None:
        """Stop the bot gracefully"""
        self.running = False


# =============================================================================
# Main entry point
# =============================================================================

def main():
    bot = TradingBot()
    
    def handle_sigterm(signum, frame):
        print("\n[BOT] Received SIGTERM...")
        if CLOSE_ON_SHUTDOWN and not DRY_RUN:
            print("[BOT] Closing all positions...")
            closed = bot.client.close_all_positions()
            print(f"[BOT] Closed {closed} positions")
        bot.stop()
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    bot.run()
    print("[BOT] Goodbye!")


if __name__ == "__main__":
    main()