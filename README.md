# Forex Pairs Trading Bot

A statistical arbitrage bot that trades currency pair spreads based on z-score mean reversion.

## Strategy Overview

Instead of betting on directional moves in a single currency pair, this bot trades the *relationship* between two pairs:

1. **EUR/USD vs GBP/USD** - Highly correlated pairs (essentially trading EUR/GBP synthetically)
2. **EUR/USD vs AUD/USD** - Softer correlation based on shared "risk-on" behavior

When the ratio between pairs deviates significantly from its historical average (measured in standard deviations / z-score), the bot enters a hedged position betting on mean reversion.

## How It Works

1. Calculates the ratio: `price(EUR/USD) / price(GBP/USD)`
2. Tracks the rolling mean and standard deviation of this ratio
3. Computes z-score: `(current_ratio - mean) / std_dev`
4. When z-score exceeds threshold (e.g., ±2.0):
   - **Z > +2.0**: Ratio is high → Short EUR/USD, Long GBP/USD
   - **Z < -2.0**: Ratio is low → Long EUR/USD, Short GBP/USD
5. Closes position when z-score returns near zero

## Project Structure

```
forex-pairs-bot/
├── main.py              # Bot entry point and main loop
├── oanda_client.py      # OANDA API wrapper
├── pairs_analyzer.py    # Z-score calculation logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image definition
├── build-and-deploy.sh  # OpenShift build/deploy helper
└── k8s/
    ├── configmap.yaml   # Bot configuration
    ├── secret.yaml      # OANDA credentials (EDIT THIS)
    └── deployment.yaml  # Kubernetes deployment
```

## Setup Instructions

### Prerequisites

- OpenShift cluster (CRC works fine)
- OANDA practice account with API access
- `oc` CLI logged into your cluster

### 1. Get OANDA API Credentials

1. Log into your OANDA account
2. Go to **Manage API Access** in account settings
3. Generate a new API token
4. Note your Account ID (format: `XXX-XXX-XXXXXXXX-XXX`)

### 2. Configure Secrets

Edit `k8s/secret.yaml` and replace the placeholders:

```yaml
stringData:
  OANDA_API_KEY: "your-actual-api-key"
  OANDA_ACCOUNT_ID: "101-001-12345678-001"
```

### 3. Deploy to OpenShift

```bash
# Make the script executable
chmod +x build-and-deploy.sh

# Create the secret first
oc apply -f k8s/secret.yaml

# Build and deploy (creates project 'forex-bot' by default)
./build-and-deploy.sh

# Or specify a custom project name
./build-and-deploy.sh my-trading-project
```

### 4. Monitor the Bot

```bash
# Watch the logs
oc logs -f deploy/forex-pairs-bot

# Check pod status
oc get pods

# View current config
oc get configmap forex-bot-config -o yaml
```

## Configuration

All configuration is in `k8s/configmap.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `POLL_INTERVAL` | 60 | Seconds between price checks |
| `LOOKBACK_PERIODS` | 20 | Periods for mean/std calculation |
| `ENTRY_Z_SCORE` | 2.0 | Z-score threshold to enter trade |
| `EXIT_Z_SCORE` | 0.5 | Z-score threshold to exit trade |
| `TRADE_UNITS` | 1000 | Units per leg of spread |
| `GRANULARITY` | H1 | Candle size for historical data |
| `DRY_RUN` | true | Set to "false" for real trades |

To update config:

```bash
# Edit the configmap
oc edit configmap forex-bot-config

# Restart the bot to pick up changes
oc rollout restart deploy/forex-pairs-bot
```

## Running Locally (for development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OANDA_API_KEY="your-api-key"
export OANDA_ACCOUNT_ID="your-account-id"
export DRY_RUN="true"

# Run the bot
python main.py
```

## Safety Notes

- **DRY_RUN is enabled by default** - the bot will log signals but not execute trades
- The bot runs a single replica to prevent duplicate trades
- Always test with a practice account first
- This is for educational purposes - use at your own risk

## Next Steps / Ideas

- Add a health endpoint for Kubernetes liveness probes
- Persist trade history to a database
- Add Prometheus metrics for monitoring
- Create a simple web UI to view status
- Implement position sizing based on account balance
- Add stop-loss logic for protection
