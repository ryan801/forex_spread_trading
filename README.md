# Forex Pairs Trading Bot

![License](https://img.shields.io/badge/license-MIT-blue)

A statistical cointegration-based spread trading bot that trades currency pair spreads on mean reversion using z-scores.

## Strategy Overview

Instead of betting on directional moves in a single currency pair, this bot trades the *relationship* between two cointegrated pairs. When the spread between them deviates significantly from its historical average (measured in z-score), the bot enters a hedged position betting on mean reversion.

**Active spreads:**

| Spread | Timeframe | Entry Z | Exit Z | Hedge Ratio | Description |
|--------|-----------|---------|--------|-------------|-------------|
| EUR/JPY vs GBP/JPY | Weekly | ±1.5 | 0.2 | 0.8188 | EUR vs GBP via JPY crosses |
| AUD/USD vs NZD/USD | Daily | ±1.5 | 0.2 | 0.8715 | AUD vs NZD via USD crosses |

Each spread uses a **backtest-derived hedge ratio** to size position legs so the combined spread is stationary. The hedge ratio also accounts for cross-currency conversion, for JPY-quoted pairs, the bot fetches USD/{base} rates to correctly size each leg in USD-equivalent terms.

## How It Works

1. On startup, historical candles are loaded per spread based on its timeframe granularity (`D` daily, `W` weekly).
2. Each poll cycle, the bot computes the hedged spread (`pair1 - hedge_ratio × pair2`) and z-scores it against a rolling lookback window.
3. When z-score exceeds the entry threshold:
   - **Z > +1.5**: Spread is high → Long pair1, Short pair2 (scaled by hedge ratio)
   - **Z < -1.5**: Spread is low → Short pair1, Long pair2 (scaled by hedge ratio)
4. Position closes when z-score reverts below the exit threshold (0.2).
5. A spread-level stop-loss (`SPREAD_MAX_LOSS_USD`) closes both legs if combined unrealised P&L breaches the limit — no per-leg stops are used, as they would break the hedge.

### Updating Spread Configurations

Run the cointegration analyzer locally first:

```bash
pip install statsmodels --break-system-packages
python cointegration_analyzer.py
```

Then update `SPREAD_CONFIGS` in `main.py` with the resulting pairs and hedge ratios:

```python
SpreadConfig(
    pair1='EUR_JPY',
    pair2='GBP_JPY',
    granularity='W',
    entry_z=1.5,
    exit_z=0.2,
    lookback=20,
    hedge_ratio=0.8188,
    description='EUR vs GBP (JPY crosses) - Weekly'
),
```

## Project Structure

```
forex-pairs-bot/
├── main.py                   # Bot entry point, spread configs, and main loop
├── oanda_client.py           # OANDA API wrapper
├── pairs_analyzer.py         # Z-score and hedged spread calculation
├── cointegration_analyzer.py # One-time local analysis to find spread candidates
├── requirements.txt          # Python dependencies
├── close_all.py              # Emergency close script
├── Containerfile             # Container image definition
├── build-and-deploy.sh       # OpenShift build/deploy helper
└── k8s/
    ├── spread-trading-configmap.yaml   # Bot configuration
    ├── spread-trading-secret.yaml      # OANDA credentials (EDIT THIS)
    └── spread-trading-deployment.yaml  # Kubernetes deployment
```

## Setup Instructions

### Prerequisites

- OpenShift cluster (CRC works fine)
- Another platform like fly.io is also fine, please change the commands accordingly
- OANDA practice account with API access
- `oc` CLI logged into your cluster

### 1. Get OANDA API Credentials

1. Log into your OANDA account
2. Go to **Manage API Access** in account settings
3. Generate a new API token
4. Note your Account ID (format: `XXX-XXX-XXXXXXXX-XXX`)

### 2. Configure Secrets

Edit `k8s/spread-trading-secret.yaml` and replace the placeholders:

```yaml
stringData:
  OANDA_API_KEY: "your-actual-api-key"
  OANDA_ACCOUNT_ID: "your-account-id"
```

### 3. Deploy to OpenShift

```bash
chmod +x build-and-deploy.sh

oc apply -f k8s/spread-trading-secret.yaml

./build-and-deploy.sh
# Or with a custom project name:
./build-and-deploy.sh my-trading-project
```

### 4. Monitor the Bot

```bash
# Watch logs
oc logs -f deploy/forex-pairs-bot

# Check pod status
oc get pods

# View config
oc get configmap spread-trading-config -o yaml
```

## Configuration

All runtime configuration via environment variables (set in `k8s/spread-trading-configmap.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `POLL_INTERVAL` | 3600 | Seconds between price checks (1 hour) |
| `TRADE_UNITS` | 10000 | Units for leg 1; leg 2 scaled by hedge ratio |
| `SPREAD_MAX_LOSS_USD` | 200 | Combined spread P&L stop-loss in USD |
| `DRY_RUN` | true | Set to `"false"` for live trading |
| `CLOSE_ON_SHUTDOWN` | true | Close all positions on SIGTERM |

Spread-specific parameters (entry/exit z-score, lookback, granularity, hedge ratio) are set directly in `SPREAD_CONFIGS` in `main.py`.

To apply config changes:

```bash
oc edit configmap spread-trading-config
oc rollout restart deploy/forex-pairs-bot
```

## Running Locally

```bash
pip install -r requirements.txt

export OANDA_API_KEY="your-api-key"
export OANDA_ACCOUNT_ID="your-account-id"
export DRY_RUN="true"

python main.py
```

## Safety Notes

- **DRY_RUN is enabled by default** — signals are logged but no orders are placed
- No per-leg stop-losses: individual stops break the hedge by closing one leg while leaving the other open. A **spread-level stop** (`SPREAD_MAX_LOSS_USD`) closes both legs together
- On restart, the bot reconciles existing OANDA positions before entering new trades
- If one leg is closed externally (e.g. margin call), the surviving leg is immediately closed to eliminate unhedged exposure
- Always test on a practice account first

---

*This is for educational purposes and does not claim any ROI or model completeness nor accuracy.*

## License

MIT License

Copyright (c) 2025-2026 Ryan Shaw

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.