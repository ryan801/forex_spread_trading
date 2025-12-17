"""
OANDA API Client Wrapper
Handles authentication, price fetching, and order execution
"""

import os
import requests
from typing import Optional


class OandaClient:
    """Client for interacting with OANDA's REST API"""
    
    def __init__(self, api_key: str = None, account_id: str = None, practice: bool = True):
        self.api_key = api_key or os.environ.get('OANDA_API_KEY')
        self.account_id = account_id or os.environ.get('OANDA_ACCOUNT_ID')
        
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_current_price(self, instrument: str) -> Optional[dict]:
        """
        Get current bid/ask price for an instrument
        
        Args:
            instrument: Currency pair like "EUR_USD" (OANDA uses underscore)
        
        Returns:
            dict with 'bid', 'ask', 'mid' prices or None on error
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('prices') and len(data['prices']) > 0:
                price_data = data['prices'][0]
                bid = float(price_data['bids'][0]['price'])
                ask = float(price_data['asks'][0]['price'])
                mid = (bid + ask) / 2
                
                return {
                    'instrument': instrument,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'time': price_data['time']
                }
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get price for {instrument}: {e}")
        
        return None
    
    def get_candles(self, instrument: str, granularity: str = "D", count: int = 50) -> Optional[list]:
        """
        Get historical candlestick data
        
        Args:
            instrument: Currency pair like "EUR_USD"
            granularity: Candle timeframe - S5, M1, M5, M15, H1, H4, D, W, M
            count: Number of candles to retrieve (max 5000)
        
        Returns:
            List of candle dicts with 'time', 'open', 'high', 'low', 'close', 'volume'
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"  # Midpoint prices
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for c in data.get('candles', []):
                if c['complete']:  # Only use completed candles
                    candles.append({
                        'time': c['time'],
                        'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']),
                        'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })
            
            return candles
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get candles for {instrument}: {e}")
        
        return None
    
    def get_account_summary(self) -> Optional[dict]:
        """Get account balance and summary info"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            account = data.get('account', {})
            return {
                'balance': float(account.get('balance', 0)),
                'nav': float(account.get('NAV', 0)),
                'unrealized_pl': float(account.get('unrealizedPL', 0)),
                'open_positions': int(account.get('openPositionCount', 0)),
                'open_trades': int(account.get('openTradeCount', 0))
            }
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get account summary: {e}")
        
        return None
    
    def place_market_order(self, instrument: str, units: int, stop_loss_pips: float = None) -> Optional[dict]:
        """
        Place a market order with optional stop-loss
        
        Args:
            instrument: Currency pair like "EUR_USD"
            units: Positive for buy, negative for sell
            stop_loss_pips: Optional stop-loss distance in pips
        
        Returns:
            Order response dict or None on error
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or kill
                "positionFill": "DEFAULT"
            }
        }
        
        # Add stop-loss if specified
        if stop_loss_pips is not None and stop_loss_pips > 0:
            # Get current price to calculate stop-loss level
            price_data = self.get_current_price(instrument)
            if price_data:
                current_price = price_data['mid']
                
                # Calculate pip value (most pairs are 0.0001, JPY pairs are 0.01)
                if 'JPY' in instrument:
                    pip_value = 0.01
                else:
                    pip_value = 0.0001
                
                # Calculate stop-loss price
                # Buy order: stop-loss below current price
                # Sell order: stop-loss above current price
                if units > 0:  # Buy
                    stop_price = current_price - (stop_loss_pips * pip_value)
                else:  # Sell
                    stop_price = current_price + (stop_loss_pips * pip_value)
                
                # Round to appropriate precision
                if 'JPY' in instrument:
                    stop_price = round(stop_price, 3)
                else:
                    stop_price = round(stop_price, 5)
                
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_price)
                }
                print(f"[ORDER] Setting stop-loss at {stop_price} ({stop_loss_pips} pips)")
        
        try:
            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            data = response.json()
            
            if 'orderFillTransaction' in data:
                fill = data['orderFillTransaction']
                return {
                    'id': fill['id'],
                    'instrument': fill['instrument'],
                    'units': float(fill['units']),
                    'price': float(fill['price']),
                    'pl': float(fill.get('pl', 0)),
                    'time': fill['time']
                }
            elif 'orderCancelTransaction' in data:
                cancel = data['orderCancelTransaction']
                print(f"[WARN] Order cancelled: {cancel.get('reason')}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to place order for {instrument}: {e}")
        
        return None
    
    def get_open_positions(self) -> Optional[list]:
        """Get all open positions"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            positions = []
            for pos in data.get('positions', []):
                long_units = float(pos['long'].get('units', 0))
                short_units = float(pos['short'].get('units', 0))
                
                positions.append({
                    'instrument': pos['instrument'],
                    'long_units': long_units,
                    'short_units': short_units,
                    'net_units': long_units + short_units,
                    'unrealized_pl': float(pos.get('unrealizedPL', 0))
                })
            
            return positions
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get open positions: {e}")
        
        return None
    
    def close_position(self, instrument: str) -> Optional[dict]:
        """Close all units of a position for an instrument"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"
        
        close_data = {
            "longUnits": "ALL",
            "shortUnits": "ALL"
        }
        
        try:
            response = requests.put(url, headers=self.headers, json=close_data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[ERROR] Failed to close position for {instrument}: {e}")
        
        return None
    
    def close_all_positions(self) -> int:
        """
        Close all open positions
        
        Returns:
            Number of positions closed
        """
        positions = self.get_open_positions()
        closed = 0
        
        if not positions:
            print("[INFO] No open positions to close")
            return 0
        
        for pos in positions:
            if pos['net_units'] != 0:
                print(f"[CLOSE] Closing {pos['instrument']}: {pos['net_units']} units")
                result = self.close_position(pos['instrument'])
                if result:
                    closed += 1
        
        print(f"[CLOSE] Closed {closed} positions")
        return closed