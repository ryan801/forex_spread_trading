#!/usr/bin/env python3
"""
Quick script to close all open positions on OANDA
Run locally when you need to emergency-close everything

Usage:
    export OANDA_API_KEY="your_key"
    export OANDA_ACCOUNT_ID="your_account"
    python close_all.py
"""

import os
import sys

# Add parent directory to path if running from scripts folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oanda_client import OandaClient


def main():
    # Check for required env vars
    if not os.environ.get('OANDA_API_KEY'):
        print("ERROR: OANDA_API_KEY environment variable not set")
        sys.exit(1)
    
    if not os.environ.get('OANDA_ACCOUNT_ID'):
        print("ERROR: OANDA_ACCOUNT_ID environment variable not set")
        sys.exit(1)
    
    client = OandaClient()
    
    # Show account info
    acct = client.get_account_summary()
    if acct:
        print(f"Account balance: ${acct['balance']:.2f}")
        print(f"Unrealized P/L: ${acct['unrealized_pl']:.2f}")
        print(f"Open positions: {acct['open_positions']}")
        print()
    
    # Get and display open positions
    positions = client.get_open_positions()
    
    if not positions:
        print("No open positions to close.")
        return
    
    print("Current open positions:")
    for pos in positions:
        if pos['net_units'] != 0:
            print(f"  {pos['instrument']}: {pos['net_units']:+.0f} units, P/L: ${pos['unrealized_pl']:.2f}")
    
    print()
    
    # Confirm before closing
    confirm = input("Close all positions? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Close all
    closed = client.close_all_positions()
    print(f"\nDone. Closed {closed} positions.")


if __name__ == "__main__":
    main()