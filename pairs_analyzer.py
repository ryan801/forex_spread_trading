"""
Pairs Analyzer
Calculates ratios and z-scores for currency pair spreads
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SpreadSignal:
    """Represents a trading signal from spread analysis"""
    pair1: str
    pair2: str
    ratio: float
    z_score: float
    mean: float
    std: float
    signal: str  # 'LONG_SPREAD', 'SHORT_SPREAD', 'NEUTRAL', 'CLOSE'
    timestamp: datetime


class PairsAnalyzer:
    """
    Analyzes the spread between two currency pairs and generates z-score signals
    """
    
    def __init__(
        self,
        pair1: str,
        pair2: str,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        mode: str = "ratio",
        hedge_ratio: float | None = None,
    ):
        """
        Args:
            pair1: First currency pair (e.g., "EUR_USD")
            pair2: Second currency pair (e.g., "GBP_USD")
            lookback: Number of periods for moving average/std calculation
            entry_z: Z-score threshold to enter a trade
            exit_z: Z-score threshold to exit a trade (close to mean)
            mode: "ratio" (price1/price2) or "hedged_spread" (price1 - beta*price2)
            hedge_ratio: beta to use when mode == "hedged_spread"
        """
        self.pair1 = pair1
        self.pair2 = pair2
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.mode = mode
        self.hedge_ratio = hedge_ratio if hedge_ratio is not None else 1.0
        
        # Store historical ratios
        self.ratio_history = []
        self.max_history = lookback * 3  # Keep some buffer
        
        # Track current position state
        self.in_position = False
        self.position_side = None  # 'LONG_SPREAD' or 'SHORT_SPREAD'
    
    @property
    def name(self) -> str:
        """Human readable name for this spread"""
        return f"{self.pair1}/{self.pair2}"
    
    def calculate_ratio(self, price1: float, price2: float) -> float:
        """Calculate ratio or hedge-adjusted spread"""
        if self.mode == "hedged_spread":
            return price1 - self.hedge_ratio * price2
        if price2 == 0:
            return 0
        return price1 / price2
    
    def add_ratio(self, ratio: float) -> None:
        """Add a new ratio to history"""
        self.ratio_history.append(ratio)
        
        # Trim history if too long
        if len(self.ratio_history) > self.max_history:
            self.ratio_history = self.ratio_history[-self.max_history:]
    
    def calculate_z_score(self) -> Tuple[float, float, float]:
        """
        Calculate current z-score based on historical ratios
        
        Returns:
            Tuple of (z_score, mean, std)
        """
        if len(self.ratio_history) < self.lookback:
            return 0.0, 0.0, 0.0
        
        recent = self.ratio_history[-self.lookback:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return 0.0, mean, 0.0
        
        current_ratio = self.ratio_history[-1]
        z_score = (current_ratio - mean) / std
        
        return z_score, mean, std
    
    def get_signal(self, price1: float, price2: float) -> SpreadSignal:
        """
        Calculate the current signal based on prices
        
        Args:
            price1: Current mid price of pair1
            price2: Current mid price of pair2
        
        Returns:
            SpreadSignal with current analysis
        """
        ratio = self.calculate_ratio(price1, price2)
        self.add_ratio(ratio)
        
        z_score, mean, std = self.calculate_z_score()
        
        # Determine signal
        signal = "NEUTRAL"
        
        if not self.in_position:
            # Looking for entry
            if z_score >= self.entry_z:
                # Ratio is high - pair1 is expensive relative to pair2
                # Short pair1, long pair2 (short the spread)
                signal = "SHORT_SPREAD"
            elif z_score <= -self.entry_z:
                # Ratio is low - pair1 is cheap relative to pair2
                # Long pair1, short pair2 (long the spread)
                signal = "LONG_SPREAD"
        else:
            # In a position, looking for exit
            if self.position_side == "LONG_SPREAD" and z_score >= -self.exit_z:
                signal = "CLOSE"
            elif self.position_side == "SHORT_SPREAD" and z_score <= self.exit_z:
                signal = "CLOSE"
        
        return SpreadSignal(
            pair1=self.pair1,
            pair2=self.pair2,
            ratio=ratio,
            z_score=z_score,
            mean=mean,
            std=std,
            signal=signal,
            timestamp=datetime.utcnow()
        )
    
    def update_position_state(self, entered: bool = False, exited: bool = False, side: str = None) -> None:
        """Update internal position tracking"""
        if entered and side:
            self.in_position = True
            self.position_side = side
        elif exited:
            self.in_position = False
            self.position_side = None
    
    def load_historical_ratios(self, candles1: list, candles2: list) -> int:
        """
        Pre-load historical ratios from candle data
        
        Args:
            candles1: List of candles for pair1
            candles2: List of candles for pair2
        
        Returns:
            Number of ratios loaded
        """
        # Match candles by time - they should align if we requested same count
        min_len = min(len(candles1), len(candles2))
        
        for i in range(min_len):
            ratio = self.calculate_ratio(candles1[i]['close'], candles2[i]['close'])
            self.ratio_history.append(ratio)
        
        print(f"[{self.name}] Loaded {len(self.ratio_history)} historical ratios")
        return len(self.ratio_history)
    
    def get_status(self) -> dict:
        """Get current analyzer status for logging/display"""
        z_score, mean, std = self.calculate_z_score()
        
        return {
            'spread_name': self.name,
            'history_count': len(self.ratio_history),
            'current_ratio': self.ratio_history[-1] if self.ratio_history else None,
            'z_score': round(z_score, 4),
            'mean': round(mean, 6),
            'std': round(std, 6),
            'in_position': self.in_position,
            'position_side': self.position_side,
            'ready': len(self.ratio_history) >= self.lookback
        }


class MultiPairAnalyzer:
    """Manages multiple spread analyzers"""
    
    def __init__(self):
        self.analyzers: dict[str, PairsAnalyzer] = {}
    
    def add_spread(self, pair1: str, pair2: str, **kwargs) -> PairsAnalyzer:
        """Add a new spread to track"""
        analyzer = PairsAnalyzer(pair1, pair2, **kwargs)
        key = analyzer.name
        self.analyzers[key] = analyzer
        return analyzer
    
    def get_all_signals(self, prices: dict) -> list[SpreadSignal]:
        """
        Get signals for all tracked spreads
        
        Args:
            prices: Dict mapping instrument to mid price, e.g., {"EUR_USD": 1.1234}
        
        Returns:
            List of SpreadSignal for each spread
        """
        signals = []
        
        for analyzer in self.analyzers.values():
            if analyzer.pair1 in prices and analyzer.pair2 in prices:
                signal = analyzer.get_signal(prices[analyzer.pair1], prices[analyzer.pair2])
                signals.append(signal)
        
        return signals
    
    def get_all_status(self) -> list[dict]:
        """Get status of all analyzers"""
        return [a.get_status() for a in self.analyzers.values()]
