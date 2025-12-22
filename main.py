"""
Forex Pairs Trading Bot
Main entry point - runs the trading loop
"""

import os
import sys
import time
import signal
from datetime import datetime

from oanda_client import OandaClient
from pairs_analyzer import MultiPairAnalyzer, SpreadSignal


# Configuration from environment (will come from ConfigMap/Secrets in k8s)
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '60'))  # seconds
LOOKBACK_PERIODS = int(os.environ.get('LOOKBACK_PERIODS', '20'))
ENTRY_Z_SCORE = float(os.environ.get('ENTRY_Z_SCORE', '2.0'))
EXIT_Z_SCORE = float(os.environ.get('EXIT_Z_SCORE', '0.5'))
TRADE_UNITS = int(os.environ.get('TRADE_UNITS', '1000'))  # Units per leg
# FIXED: Logic was backwards - should check if == 'true', not == 'False'
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'
GRANULARITY = os.environ.get('GRANULARITY', 'H1')  # Candle size for historical data
MAX_TRADES_PER_DAY = int(os.environ.get('MAX_TRADES_PER_DAY', '1'))
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', '1'))
ALLOW_LIVE_TRADES = os.environ.get('ALLOW_LIVE_TRADES', 'false').lower() == 'true'
STOP_LOSS_PIPS = float(os.environ.get('STOP_LOSS_PIPS', '50'))  # Stop-loss distance in pips
CLOSE_ON_SHUTDOWN = os.environ.get('CLOSE_ON_SHUTDOWN', 'true').lower() == 'true'


# Currency pairs we're trading
INSTRUMENTS = ['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_NOK', 'EUR_JPY', 'GBP_JPY']

# Spread definitions: (pair1, pair2)
SPREADS = [
    ('EUR_USD', 'GBP_USD'),  # Tight correlation - essentially EUR/GBP
    ('EUR_USD', 'AUD_USD'),  # Softer correlation
    ('AUD_USD', 'NZD_USD'),  # Very tight - Oceania twins
    ('USD_CAD', 'USD_NOK'),  # Oil exporters - both track crude
    ('EUR_JPY', 'GBP_JPY'),  # European majors vs safe haven
]


class TradingBot:
    """Main bot class that orchestrates everything"""
    
    def __init__(self):
        self.running = False
        # FIXED: client was undefined, should be OandaClient()
        self.client = OandaClient()
        self.open_positions = {}

        # FIXED: typo 'datetiem' -> 'datetime', added missing ()
        self.trade_day = datetime.utcnow().date()
        # FIXED: Changed from {} to 0 to match usage later
        self.trades_today = 0
        
        self.analyzer = MultiPairAnalyzer()
        
        # Track open spread positions
        # Key: spread name, Value: {'pair1_units': X, 'pair2_units': Y}
        self.open_positions = {}
        
        # Set up spread analyzers
        for pair1, pair2 in SPREADS:
            self.analyzer.add_spread(
                pair1, pair2,
                lookback=LOOKBACK_PERIODS,
                entry_z=ENTRY_Z_SCORE,
                exit_z=EXIT_Z_SCORE
            )
        
        print("[INIT] Bot initialized")
        print(f"[INIT] Tracking spreads: {[f'{p1}/{p2}' for p1, p2 in SPREADS]}")
        print(f"[INIT] Settings: lookback={LOOKBACK_PERIODS}, entry_z={ENTRY_Z_SCORE}, exit_z={EXIT_Z_SCORE}")
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
        
        # Show raw position data from OANDA
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
                
                # Determine the spread direction based on position signs
                # LONG_SPREAD = long pair1, short pair2 (pair1 positive, pair2 negative)
                # SHORT_SPREAD = short pair1, long pair2 (pair1 negative, pair2 positive)
                if pair1_units > 0 and pair2_units < 0:
                    side = "LONG_SPREAD"
                elif pair1_units < 0 and pair2_units > 0:
                    side = "SHORT_SPREAD"
                else:
                    # Both same direction - not a proper spread trade, skip
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
                
                print(f"[RECONCILE] ✓ Recovered {spread_name}: {side} "
                      f"({pair1}={pair1_units:+.0f}, {pair2}={pair2_units:+.0f})")
                reconciled += 1
                
                # Remove from map so we can detect orphans
                del position_map[pair1]
                del position_map[pair2]
        
        # Warn about any positions that don't match our spreads
        if position_map:
            print(f"[RECONCILE] ⚠ Unmatched positions (not part of tracked spreads):")
            for inst, units in position_map.items():
                print(f"[RECONCILE]   {inst}: {units:+.0f} units - consider closing manually")
        
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
    
    def execute_spread_trade(self, signal: SpreadSignal) -> bool:
        """
        Execute a spread trade based on signal
        
        LONG_SPREAD: Buy pair1, Sell pair2
        SHORT_SPREAD: Sell pair1, Buy pair2
        """
        spread_name = f"{signal.pair1}/{signal.pair2}"
        
        if signal.signal == 'LONG_SPREAD':
            pair1_units = TRADE_UNITS   # Buy
            pair2_units = -TRADE_UNITS  # Sell
        elif signal.signal == 'SHORT_SPREAD':
            pair1_units = -TRADE_UNITS  # Sell
            pair2_units = TRADE_UNITS   # Buy
        elif signal.signal == 'CLOSE':
            # Reverse existing position
            if spread_name in self.open_positions:
                pos = self.open_positions[spread_name]
                pair1_units = -pos['pair1_units']
                pair2_units = -pos['pair2_units']
            else:
                print(f"[TRADE] No position to close for {spread_name}")
                return False
        else:
            return False
        
        now = datetime.utcnow().strftime('%H:%M')
        print(f"\n[TRADE {now}] {'=' * 50}")
        print(f"[TRADE {now}] Signal: {signal.signal} on {spread_name}")
        print(f"[TRADE {now}] Z-Score: {signal.z_score:.4f}")
        print(f"[TRADE {now}] {signal.pair1}: {pair1_units:+.0f} units")
        print(f"[TRADE {now}] {signal.pair2}: {pair2_units:+.0f} units")
        
        if DRY_RUN:
            print(f"[TRADE {now}] DRY RUN - No actual orders placed")
            success = True
        else:
            # Execute the trades with stop-loss protection
            result1 = self.client.place_market_order(signal.pair1, pair1_units, stop_loss_pips=STOP_LOSS_PIPS)
            result2 = self.client.place_market_order(signal.pair2, pair2_units, stop_loss_pips=STOP_LOSS_PIPS)
            success = result1 is not None and result2 is not None
            
            if success:
                print(f"[TRADE {now}] {signal.pair1} filled at {result1['price']}")
                print(f"[TRADE {now}] {signal.pair2} filled at {result2['price']}")
            else:
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
    
    def run_once(self) -> None:
        """Run a single iteration of the bot logic"""
        # Get current prices
        prices = self.get_current_prices()
        if len(prices) != len(INSTRUMENTS):
            print(f"[WARN] Only got prices for {list(prices.keys())}, expected {INSTRUMENTS}")
            return
        
        # Get signals for all spreads
        signals = self.analyzer.get_all_signals(prices)
        
        # Print status
        self.print_status(prices, signals)
        
        # Act on signals
        for sig in signals:
            if sig.signal in ['LONG_SPREAD', 'SHORT_SPREAD', 'CLOSE']:
                self.execute_spread_trade(sig)
    
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