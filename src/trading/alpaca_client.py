import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()  # This will let Alpaca pick up the correct vars automatically
LOG_PATH = "trade_logs.jsonl"
api = tradeapi.REST(api_version='v2')

def test_connection():
    try:
        account = api.get_account()
        print("✅ Connected to Alpaca")
        print(f"Status: {account.status}, Buying Power: {account.buying_power}")
    except Exception as e:
        print("❌ Connection failed:", e)

def log_trade(ticker, action, quantity, confidence):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"📝 Logged trade: {entry}")
    except Exception as e:
        print(f"❌ Failed to log trade: {e}")


def place_order(action: str, ticker: str, quantity: int,confidence: float=None):
    if action == "HOLD" or quantity == 0:
        print(f"🟡 HOLD: No order placed for {ticker}")
        return

    try:
        order = api.submit_order(
            symbol=ticker,
            qty=quantity,
            side=action.lower(),  # "buy" or "sell"
            type='market',
            time_in_force='gtc'
        )
        print(f"✅ {action.upper()} order placed: {quantity} shares of {ticker}")

        # Only log successful orders
        log_trade(ticker, action.upper(), quantity, confidence or 0.0)
    except Exception as e:
        print(f"❌ Failed to place order for {ticker}: {e}")

def get_current_position(ticker):
    try:
        position = api.get_position(ticker)
        qty = int(float(position.qty))
        print(f"📊 Current position in {ticker}: {qty} shares")
        return qty
    except tradeapi.rest.APIError as e:
        if "position does not exist" in str(e):
            print(f"📊 No position in {ticker}")
            return 0
        else:
            raise


def should_place_trade(action, ticker, desired_qty):
    current_qty = get_current_position(ticker)

    if action == "HOLD":
        print(f"🟡 HOLD decision for {ticker}")
        return False

    if action == "BUY" and current_qty >= desired_qty:
        print(f"⚠️ Already long on {ticker} ({current_qty}), skipping BUY of {desired_qty}")
        return False

    if action == "SELL" and current_qty <= 0:
        print(f"⚠️ No long position in {ticker} to SELL, skipping")
        return False

    if action == "SHORT" and current_qty != 0:
        print(f"⚠️ Can't SHORT {ticker} while holding a position ({current_qty}), skipping")
        return False

    if action == "BUY":
        price = api.get_latest_trade(ticker).price
        cash = get_available_cash()
        max_allowed = 0.2 * cash
        max_shares = int(max_allowed // price)

        if desired_qty > max_shares:
            if max_shares > 0:
                print(f"⚠️ Scaling down BUY for {ticker}: requested {desired_qty}, allowed {max_shares}")
                # we mutate quantity here
                # optional: return the scaled qty if you want to control upstream
                return max_shares
            else:
                print(f"🚫 Cannot afford even 1 share of {ticker}, skipping")
                return False

    return True


def get_available_cash():
    try:
        account = api.get_account()
        cash = float(account.cash)
        print(f"💰 Available cash: ${cash:.2f}")
        return cash
    except Exception as e:
        print(f"❌ Failed to fetch account cash: {e}")
        return 0.0




