# claritybot.py
# Full runnable ClarityBot: keeps your content, fixes structure, adds SQLite persistence,
# replaces LunarCrush with CoinGecko sentiment fallback, and polishes UX.

import numpy as np
import requests
from io import BytesIO
from telegram import InputFile, Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from websocket import WebSocketApp
import json
import threading
import random
import sqlite3
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import pandas as pd

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# -------------------------
# CONFIG - Replace keys if needed
# -------------------------
TOKEN = "8233021006:AAHM_4fHryu8ToFwhc69j8XheVxrM_E06LA"  # your real Telegram bot token
CRYPTO_PANIC_API_KEY = "3f14d218aa3c1e1f140e27a59136e8462c00977b"
NFA_TEXT = "\n\n*⚠️ Not financial advice — do your own research.*"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_tradingview_chart(symbol, interval):
    url_symbol = symbol.upper().replace("/", "")
    url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{url_symbol}&interval={interval}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Wait for chart to load
        await page.wait_for_timeout(2500)

        # Select the chart canvas
        chart = await page.locator("canvas").first.screenshot()

        await browser.close()

        buffer = BytesIO(chart)
        buffer.name = f"{symbol}_{interval}.png"
        return buffer

# -------------------------
# SQLite persistence
# -------------------------
DB_PATH = "claritybot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # watchlists table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS watchlists (
        user_id INTEGER,
        coin TEXT,
        PRIMARY KEY (user_id, coin)
    )
    """)
    # portfolios table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        user_id INTEGER,
        coin TEXT,
        amount REAL,
        PRIMARY KEY (user_id, coin)
    )
    """)
    conn.commit()
    conn.close()

def add_watchlist_item(user_id: int, coin: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR IGNORE INTO watchlists(user_id, coin) VALUES(?, ?)", (user_id, coin))
        conn.commit()
    finally:
        conn.close()

def remove_watchlist_item(user_id: int, coin: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM watchlists WHERE user_id = ? AND coin = ?", (user_id, coin))
        conn.commit()
    finally:
        conn.close()

def get_watchlist(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT coin FROM watchlists WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()

def add_portfolio_item(user_id: int, coin: str, amount: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR REPLACE INTO portfolios(user_id, coin, amount) VALUES(?, ?, ?)", (user_id, coin, amount))
        conn.commit()
    finally:
        conn.close()

def remove_portfolio_item(user_id: int, coin: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM portfolios WHERE user_id = ? AND coin = ?", (user_id, coin))
        conn.commit()
    finally:
        conn.close()

def get_portfolio(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT coin, amount FROM portfolios WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()

# Initialize DB on startup
init_db()

# -------------------------
# Helpers
# -------------------------
def coin_gecko_search_id(name_or_symbol: str):
    """Return CoinGecko id (slug) for a given name or symbol. Returns None if not found."""
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/search?query={name_or_symbol}", timeout=10).json()
        coins = r.get("coins", [])
        if coins:
            return coins[0].get("id")
    except Exception as e:
        logger.exception("CoinGecko search error")
    return None

def fetch_coingecko_market_chart(coin_id: str, days: int = 3, interval: str = "hourly"):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": interval}
        r = requests.get(url, params=params, timeout=10).json()
        return r
    except Exception as e:
        logger.exception("CoinGecko market_chart error")
        return {}

def safe_get_prices_from_market_chart(r):
    if not r or "prices" not in r:
        return []
    return [float(p[1]) for p in r.get("prices", []) if len(p) >= 2]

def coingecko_coin_data(coin_id: str):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {"localization": "false", "tickers": "false", "market_data": "false", "community_data": "true",
                  "developer_data": "false", "sparkline": "false"}
        r = requests.get(url, params=params, timeout=10).json()
        return r
    except Exception as e:
        logger.exception("CoinGecko coin data error")
        return {}

def nfa_and_footer():
    return NFA_TEXT

# -------------------------
# WebSocket (kept as optional live feed)
# -------------------------
def on_message(ws, message):
    try:
        data = json.loads(message)
        price = float(data.get('p', 0))
        if price:
            logger.debug(f"Live price: {price}")
        # You can expand here to check dynamic levels and send alerts via Telegram Bot API
    except Exception as e:
        logger.exception("Websocket message handling error")

def on_open(ws):
    logger.info("WebSocket opened")

def start_ws(symbol="btcusdt"):
    try:
        url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        ws = WebSocketApp(url, on_message=on_message, on_open=on_open)
        ws.run_forever()
    except Exception as e:
        logger.exception("Websocket start error")

# Run websocket thread non-blocking
threading.Thread(target=start_ws, daemon=True).start()

# ---------------------------------------------------------------------
# CoinGecko OHLC Fetcher (Replaces Binance)
# ---------------------------------------------------------------------
def fetch_coingecko_ohlc(symbol: str, interval: str, limit=50):
    """
    Fetch OHLC data from CoinGecko and resample it to match standard intervals.
    Returns a DataFrame with columns: open, high, low, close, volume
    """

    from datetime import datetime
    import pandas as pd

    # Map symbol to CoinGecko coin IDs
    # You can expand this dictionary with the coins you need
    COIN_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple",
        "BNB": "binancecoin",
    }

    symbol = symbol.upper().replace("USDT", "")
    if symbol not in COIN_MAP:
        return None

    coin_id = COIN_MAP[symbol]

    # Determine how many days of data CoinGecko should return
    interval_days = {
        "1m": 1,
        "5m": 1,
        "15m": 1,
        "30m": 1,
        "1h": 1,
        "4h": 7,
        "1d": 90,
    }.get(interval, 1)

    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            f"?vs_currency=usd&days={interval_days}"
        )
        r = requests.get(url).json()

        if not isinstance(r, list):
            return None

        # CoinGecko returns: [timestamp, open, high, low, close]
        df = pd.DataFrame(r, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = 0  # CG doesn't provide volume in this endpoint

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Set timestamp index for resampling
        df = df.set_index("timestamp")

        # Resampling rules
        resample_map = {
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        if interval in resample_map:
            df = df.resample(resample_map[interval]).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

        df = df.tail(limit)
        return df

    except Exception as e:
        print("Error fetching COINGECKO OHLC:", e)
        return None

# -------------------------
# Your content (EXPLANATIONS) - kept exactly as provided
# -------------------------
EXPLANATIONS = {
    "halving": (
        "🧠 *Bitcoin Halving Explained*\n\n"
        "Every ~4 years, the reward miners receive for creating new Bitcoin is cut in half. "
        "This slows the rate of new supply entering the market.\n\n"
        "Less supply = harder to get = historically price tends to rise over time."
    ),
    "marketcap": (
        "🏛️ *Market Cap Explained*\n\n"
        "Market cap = Price × Total coins.\n\n"
        "Bigger market cap = more established, usually safer but slower.\n"
        "Smaller market cap = more volatile, higher pump potential."
    ),
    "liquidity": (
        "🌊 *Liquidity Explained*\n\n"
        "Liquidity is how easily you can buy or sell without changing the price.\n\n"
        "High liquidity = safe trading.\n"
        "Low liquidity = your trade can move the chart."
    ),
    "gas": (
        "⛽ *Gas Fees Explained*\n\n"
        "Gas fees are the cost to use the blockchain.\n"
        "More network traffic = higher cost.\n"
        "It's like paying a toll to use a busy highway."
    ),
    "defi": (
        "🏦 *DeFi Explained*\n\n"
        "DeFi stands for Decentralized Finance.\n"
        "It lets you lend, borrow, trade, and earn yield without banks.\n"
        "Code replaces the middleman."
    ),
}

# -------------------------
# Command handlers (all top-level)
# -------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Onboarding + menu with inline buttons and quick onboarding copy"""
    user = update.effective_user
    text = (
        f"Yo {user.first_name or 'there'} 👋\n\n"
        "Welcome to *ClarityBot* — your compact Web3 assistant.\n"
        "Let's get you tracking your first 3 coins 👇\n\n"
        "Commands you can try:\n"
        "/price <coin> — current price\n"
        "/chart <coin> — 24h chart\n"
        "/watchlist add <coin> — add coin to watchlist\n"
        "/portfolio add <coin> <amount> — track holdings\n"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 View Chart", callback_data="btn_chart"),
        InlineKeyboardButton("💡 Learn", callback_data="btn_learn"),
        InlineKeyboardButton("🔔 Set Alert", callback_data="btn_alert")
    ]])
    await update.message.reply_text(text + nfa_and_footer(), parse_mode="Markdown", reply_markup=keyboard)

# Price command (unchanged logic, UX polished)
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Usage: `/price bitcoin` (type full name for best results)", parse_mode="Markdown")
        return
    symbol = context.args[0].lower()
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": symbol, "vs_currencies": "usd", "include_24hr_change": "true"}
        r = requests.get(url, params=params, timeout=10).json()
    except Exception as e:
        logger.exception("Price API error")
        await update.message.reply_text("⚠️ Could not fetch price right now. Try again later.")
        return

    if symbol not in r:
        await update.message.reply_text("Token not found. Try something like: `/price bitcoin`", parse_mode="Markdown")
        return

    data = r[symbol]
    price_val = data.get("usd", 0)
    change = round(data.get("usd_24h_change", 0), 2)
    msg = f"💰 *{symbol.upper()} Price*\n\n${price_val:,} USD\n24h: {change}%{nfa_and_footer()}"
    # add inline buttons for quick actions
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 View Chart", callback_data=f"chart|{symbol}"),
        InlineKeyboardButton("💡 Explain", callback_data=f"explain|halving")  # example
    ]])
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=keyboard)

# Chart command (unchanged logic)
# --- NORMALIZED SYMBOL MAP ---
SYMBOL_MAP = {
    "btc": "BTCUSDT",
    "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT",
    "ethereum": "ETHUSDT",
    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "avax": "AVAXUSDT",
    "doge": "DOGEUSDT",
    "ton": "TONUSDT",
    "xrp": "XRPUSDT",
}

# --- TIMEFRAME MAP ---
TF_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W"
}

async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Example: /chart bitcoin 1h")
        return

    user_symbol = context.args[0].lower()
    timeframe = context.args[1].lower()

    # --- FIX #1: map common names to TradingView format ---
    if user_symbol in SYMBOL_MAP:
        tv_symbol = SYMBOL_MAP[user_symbol]
    else:
        # fallback: assume user already wrote a correct symbol
        tv_symbol = user_symbol.upper()

    # --- FIX #2: validate timeframe ---
    if timeframe not in TF_MAP:
        await update.message.reply_text("❌ Invalid timeframe. Try: 1m, 5m, 15m, 1h, 4h, 1d")
        return

    interval = TF_MAP[timeframe]

    msg = await update.message.reply_text("📊 Generating TradingView chart… please wait 🔄")

    try:
        # --- FIX #3: pass CLEAN symbol, no double prefixing ---
        image = await get_tradingview_chart(tv_symbol, interval)

        await update.message.reply_photo(
            photo=InputFile(image),
            caption=f"📈 *{tv_symbol} — {timeframe} TradingView Chart*",
            parse_mode="Markdown"
        )

        await msg.delete()

    except Exception as e:
        await update.message.reply_text(
            f"⚠️ Couldn't load chart for {tv_symbol}. Symbol might be unsupported.\n\nError: `{e}`"
        )

# Explain command (keeps EXPLANATIONS content)
# ======================
# /EXPLAIN — AI-powered mentor explanation
# ======================
async def explain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Example: /explain halving")
        return

    topic = " ".join(context.args)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are ClarityBot — a crypto mentor. "
                    "Explain any crypto topic in simple language. "
                    "Use analogies. Teach the user. Make it enjoyable. "
                    "No financial advice."
                )},
                {"role": "user", "content": topic}
            ]
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        await update.message.reply_text("⚠️ AI explanation unavailable right now.")

# ======================
# /TREND (SMA + Structure)
# ======================
async def trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Example: /trend solana")
        return

    symbol = context.args[0].lower()

    prices = None

    # --- Try CoinGecko first
    try:
        cg_url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days=2"
        cg = requests.get(cg_url).json()

        if "prices" in cg and len(cg["prices"]) > 20:
            prices = [p[1] for p in cg["prices"]]
    except:
        pass

    # --- Try DexScreener fallback
    if prices is None:
        try:
            url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
            ds = requests.get(url).json()

            if "pairs" in ds and len(ds["pairs"]) > 0:
                price_list = [p["priceUsd"] for p in ds["pairs"][0]["priceHistory"]]
                prices = [float(x) for x in price_list if x]
        except:
            pass

    # If still nothing:
    if not prices or len(prices) < 10:
        await update.message.reply_text(f"⚠️ Not enough trend data for {symbol.upper()}.")
        return

    # Calculate SMAs
    sma20 = sum(prices[-20:]) / 20
    sma50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else sma20

    last_price = prices[-1]

    if last_price > sma20 > sma50:
        trend_text = "📈 Uptrend — higher lows + SMA20 above SMA50"
    elif last_price < sma20 < sma50:
        trend_text = "📉 Downtrend — lower highs + SMA20 below SMA50"
    else:
        trend_text = "➖ Sideways — consolidation structure"

    msg = (
        f"📊 *TREND for {symbol.upper()}*\n\n"
        f"Last Price: ${last_price:.4f}\n"
        f"SMA20: ${sma20:.4f}\n"
        f"SMA50: ${sma50:.4f}\n\n"
        f"{trend_text}"
    )

    await update.message.reply_text(msg, parse_mode="Markdown")

# Signals command (RSI + dynamic levels)
async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Example: `/signals bitcoin`", parse_mode="Markdown")
        return
    symbol = context.args[0].lower()
    coin_id = coin_gecko_search_id(symbol) or symbol
    r = fetch_coingecko_market_chart(coin_id, days=3, interval="hourly")
    prices = safe_get_prices_from_market_chart(r)
    # Fallback to DexScreener quick value if empty
    if not prices:
        try:
            ds_search = requests.get(f"https://api.dexscreener.io/latest/dex/search/?q={symbol}", timeout=10).json()
            if ds_search.get("pairs"):
                pair = ds_search["pairs"][0]
                price = float(pair.get("priceUsd", 0))
                if price > 0:
                    prices = [price] * 14
        except Exception:
            pass
    if not prices or len(prices) < 6:
        await update.message.reply_text(f"⚠️ Not enough price data for {symbol.upper()}. Try another token.", parse_mode="Markdown")
        return
    # RSI
    changes = np.diff(prices)
    gains = np.maximum(changes, 0)
    losses = np.abs(np.minimum(changes, 0))
    avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 1 else 0
    avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 1 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 50.0
    recent = prices[-20:]
    support = min(recent)
    resistance = max(recent)
    last_price = prices[-1]
    signal = ""
    if rsi > 70:
        signal = "🔴 *Overbought* — possible pullback soon."
    elif rsi < 30:
        signal = "🟢 *Oversold* — could bounce soon."
    else:
        signal = "⚪ *Neutral* — no strong signal right now."
    if last_price <= support * 1.02:
        signal += f"\n🟢 *Near Support*: ${support:.2f}"
    elif last_price >= resistance * 0.98:
        signal += f"\n🔴 *Near Resistance*: ${resistance:.2f}"
    trend_dir = "📈 *Uptrend*" if prices[-1] > np.mean(prices[-5:]) else "📉 *Downtrend*"
    msg = (
        f"📊 *{symbol.upper()} Technical Signals*\n\n"
        f"Last Price: ${last_price:.4f}\n"
        f"RSI: {rsi:.2f}\n"
        f"{trend_dir}\n"
        f"{signal}"
    )
    await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")

# News command (CryptoPanic integration if key provided; fallback to CoinGecko status)
async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].lower() if context.args else None
    try:
        if CRYPTO_PANIC_API_KEY:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&kind=news&public=true"
            r = requests.get(url, timeout=10).json()
            posts = r.get("results", [])
            if symbol:
                posts = [p for p in posts if symbol in p.get("title", "").lower() or symbol in p.get("slug", "").lower()]
            posts = posts[:5]
            if not posts:
                await update.message.reply_text(f"ℹ️ No recent news found for {symbol.upper()}.")
                return
            msg = f"📰 *Latest Crypto News{' for ' + symbol.upper() if symbol else ''}*\n\n"
            for p in posts:
                title = p.get("title", "No title")
                url_link = p.get("url", "")
                published = p.get("published_at", "").split("T")[0]
                msg += f"• [{title}]({url_link}) ({published})\n\n"
            await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown", disable_web_page_preview=True)
            return
        # Fallback: CoinGecko status_updates
        url = "https://api.coingecko.com/api/v3/status_updates"
        r = requests.get(url, timeout=10).json()
        updates = r.get("status_updates", [])[:5]
        if symbol:
            updates = [u for u in updates if symbol in (u.get("project", {}).get("name", "").lower())]
        if not updates:
            await update.message.reply_text("ℹ️ No news found at the moment.")
            return
        msg = f"📰 *Latest Crypto News{' for ' + symbol.upper() if symbol else ''}*\n\n"
        for u in updates:
            title = u.get("description", "No description")
            date = u.get("created_at", "").split("T")[0]
            msg += f"• {title} ({date})\n\n"
        await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")
    except Exception as e:
        logger.exception("News fetch error")
        await update.message.reply_text("⚠️ Could not fetch news at the moment.")

# Subscribe command (simple)
subscribers = {}  # in-memory mapping user_id -> [coins]

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    if len(context.args) == 0:
        await update.message.reply_text("Example: `/subscribe bitcoin`", parse_mode="Markdown")
        return
    token = context.args[0].lower()
    subscribers.setdefault(user_id, [])
    if token not in subscribers[user_id]:
        subscribers[user_id].append(token)
    await update.message.reply_text(f"✅ You are now subscribed to alerts for {token.upper()}")

# Learn command (keeps content)
async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lessons = [
        {
            "title": "Liquidity Traps 💧",
            "text": "Smart money often pushes price above a key high to trigger breakout traders, then dumps back inside the range. Watch for wicks and fading volume near highs."
        },
        {
            "title": "Smart Money Reversal 🧠",
            "text": "Look for three-tap bottoms or tops near major liquidity zones. Volume often spikes on the last trap candle — then reverses hard."
        },
        {
            "title": "Breakout Retests 🚀",
            "text": "After breaking a key resistance, strong assets retest that level and bounce. Weak ones lose it and dump — simple but powerful."
        },
        {
            "title": "Funding Rates 📉📈",
            "text": "When funding is heavily positive, longs are overcrowded — watch for squeeze down. Negative funding often marks short traps before pumps."
        },
        {
            "title": "High Time Frame Compression ⏳",
            "text": "When volatility shrinks and highs/lows compress, expect a violent move. The longer it builds, the bigger the breakout."
        }
    ]
    lesson = random.choice(lessons)
    msg = f"🎓 *{lesson['title']}*\n\n{lesson['text']}"
    await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")

# Heatmap command (CoinGecko top gainers/losers)
async def heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
        }
        data = requests.get(url, params=params, timeout=10).json()
        if not isinstance(data, list) or len(data) == 0:
            await update.message.reply_text("⚠️ Couldn't fetch market data right now. Try again later.")
            return
        gainers = sorted([c for c in data if c.get("price_change_percentage_24h") is not None],
                         key=lambda x: x["price_change_percentage_24h"], reverse=True)[:5]
        losers = sorted([c for c in data if c.get("price_change_percentage_24h") is not None],
                        key=lambda x: x["price_change_percentage_24h"])[:5]
        msg = "🔥 *Top Gainers (24h)*\n"
        for i, coin in enumerate(gainers, start=1):
            change = round(coin["price_change_percentage_24h"], 2)
            msg += f"{i}. {coin['symbol'].upper()} +{change}%\n"
        msg += "\n❄️ *Top Losers (24h)*\n"
        for i, coin in enumerate(losers, start=1):
            change = round(coin["price_change_percentage_24h"], 2)
            msg += f"{i}. {coin['symbol'].upper()} {change}%\n"
        await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")
    except Exception as e:
        logger.exception("Heatmap error")
        await update.message.reply_text(f"⚠️ Error: {e}")

# ======================
# /ALPHA  (Real-Time Narratives)
# ======================
async def alpha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        prompt = """
        Give me the **top 3 real-time crypto narratives RIGHT NOW**.
        Use only fresh live market themes such as:
        - AI tokens activity
        - Meme rotations
        - Gaming pumps
        - New listings
        - On-chain volume trends
        - Funding rates
        - Liquidity inflow/outflow

        Format like this:

        🔥 AI Narrative
        - FET, AGIX showing strength as...

        🐸 Meme Narrative
        - PEPE/BONK rotating as...

        🎮 Gaming Narrative
        - GALA, IMX gaining due to...
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real-time crypto analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        text = completion.choices[0].message.content
        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text("⚠️ Couldn't fetch live alpha right now.")

# Sentiment command - REPLACEMENT that uses CoinGecko community_data (free) as fallback
async def sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Example: `/sentiment bitcoin`", parse_mode="Markdown")
        return
    symbol = context.args[0].lower()
    coin_id = coin_gecko_search_id(symbol) or symbol
    try:
        # Try CoinGecko coin endpoint for community metrics
        data = coingecko_coin_data(coin_id)
        if not data:
            await update.message.reply_text("⚠️ No sentiment data available.", parse_mode="Markdown")
            return
        community = data.get("community_data", {})
        sentiment_up = data.get("sentiment_votes_up_percentage", None)
        sentiment_down = data.get("sentiment_votes_down_percentage", None)
        twitter_followers = community.get("twitter_followers", "N/A")
        reddit_subs = community.get("reddit_subscribers", "N/A")
        msg = (
            f"📊 *Sentiment for {symbol.upper()}* (CoinGecko snapshot)\n\n"
            f"Price: ${data.get('market_data', {}).get('current_price', {}).get('usd', 'N/A')}\n"
            f"Twitter Followers: {twitter_followers}\n"
            f"Reddit Subscribers: {reddit_subs}\n"
            f"Sentiment Up: {sentiment_up if sentiment_up is not None else 'N/A'}%\n"
            f"Sentiment Down: {sentiment_down if sentiment_down is not None else 'N/A'}%"
        )
        await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")
    except Exception as e:
        logger.exception("Sentiment error")
        await update.message.reply_text("⚠️ Error fetching sentiment data.")

# Watchlist command (uses SQLite)
async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if len(context.args) == 0:
        await update.message.reply_text(
            "Usage:\n"
            "/watchlist add <symbol> - Add a coin\n"
            "/watchlist remove <symbol> - Remove a coin\n"
            "/watchlist show - Show your watchlist"
        )
        return
    action = context.args[0].lower()
    if action == "add" and len(context.args) > 1:
        coin = context.args[1].lower()
        add_watchlist_item(user_id, coin)
        await update.message.reply_text(f"✅ Added {coin.upper()} to your watchlist.")
    elif action == "remove" and len(context.args) > 1:
        coin = context.args[1].lower()
        remove_watchlist_item(user_id, coin)
        await update.message.reply_text(f"❌ Removed {coin.upper()} from your watchlist.")
    elif action == "show":
        coins = get_watchlist(user_id)
        if not coins:
            await update.message.reply_text("Your watchlist is empty. Add coins using `/watchlist add <symbol>`", parse_mode="Markdown")
        else:
            watchlist_text = "\n".join([c.upper() for c in coins])
            await update.message.reply_text(f"📋 Your Watchlist:\n{watchlist_text}")
    else:
        await update.message.reply_text("Invalid usage. Try /watchlist add/remove/show")

# Compare command
async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/compare <coin1> <coin2>`\nExample: `/compare bitcoin ethereum`", parse_mode="Markdown")
        return
    coin1 = context.args[0].lower()
    coin2 = context.args[1].lower()
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "ids": f"{coin1},{coin2}"}
        data = requests.get(url, params=params, timeout=10).json()
        if len(data) < 2:
            await update.message.reply_text("⚠️ Couldn't fetch data for one or both coins. Try again.")
            return
        c1 = data[0]; c2 = data[1]
        msg = f"📊 *Comparison: {c1['name']} vs {c2['name']}*\n\n"
        msg += f"{c1['symbol'].upper()} | ${c1['current_price']:,} | 24h: {c1['price_change_percentage_24h']:.2f}% | Market Cap: ${c1['market_cap']:,}\n"
        msg += f"{c2['symbol'].upper()} | ${c2['current_price']:,} | 24h: {c2['price_change_percentage_24h']:.2f}% | Market Cap: ${c2['market_cap']:,}\n"
        await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")
    except Exception as e:
        logger.exception("Compare error")
        await update.message.reply_text(f"⚠️ Error fetching comparison data: {e}")

# Portfolio command (uses SQLite)
async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if len(context.args) == 0:
        await update.message.reply_text(
            "Usage:\n"
            "/portfolio add <symbol> <amount> - Add holdings\n"
            "/portfolio remove <symbol> - Remove holdings\n"
            "/portfolio show - Show your portfolio"
        )
        return
    action = context.args[0].lower()
    if action == "add" and len(context.args) > 2:
        coin = context.args[1].lower()
        try:
            amount = float(context.args[2])
        except ValueError:
            await update.message.reply_text("⚠️ Amount must be a number. Example: `/portfolio add bitcoin 0.5`", parse_mode="Markdown")
            return
        add_portfolio_item(user_id, coin, amount)
        await update.message.reply_text(f"✅ Added {amount} {coin.upper()} to your portfolio.")
    elif action == "remove" and len(context.args) > 1:
        coin = context.args[1].lower()
        remove_portfolio_item(user_id, coin)
        await update.message.reply_text(f"❌ Removed {coin.upper()} from your portfolio.")
    elif action == "show":
        portfolio_data = get_portfolio(user_id)
        if not portfolio_data:
            await update.message.reply_text("Your portfolio is empty. Add holdings using `/portfolio add <symbol> <amount>`", parse_mode="Markdown")
            return
        coins = ",".join(portfolio_data.keys())
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": coins, "vs_currencies": "usd"}
            prices = requests.get(url, params=params, timeout=10).json()
        except Exception:
            prices = {}
        total_value = 0
        msg = "💼 *Your Portfolio*\n\n"
        for coin, amt in portfolio_data.items():
            price = float(prices.get(coin, {}).get("usd", 0) or 0)
            value = price * amt
            total_value += value
            msg += f"{coin.upper()}: {amt} × ${price:,} = ${value:,.2f}\n"
        msg += f"\n💰 *Total Value:* ${total_value:,.2f}"
        await update.message.reply_text(msg + nfa_and_footer(), parse_mode="Markdown")
    else:
        await update.message.reply_text("Invalid usage. Try /portfolio add/remove/show")

# Health check command and simple help
async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is running.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "ClarityBot commands:\n"
        "/price <coin>\n/chart <coin>\n/explain <term>\n/trend <coin>\n/signals <coin>\n"
        "/news [coin]\n/learn\n/heatmap\n/alpha\n/sentiment <coin>\n"
        "/watchlist add/remove/show\n/compare <a> <b>\n/portfolio add/remove/show\n"
    )
    await update.message.reply_text(msg + nfa_and_footer())

# Callback query handler for inline buttons
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("chart|"):
        _, sym = data.split("|", 1)

        # create dummy context with args for chart
        class DummyContext:
            args = [sym, "1h"]  # default timeframe

        await chart(update, DummyContext())
        await query.edit_message_text(f"Requested chart for {sym.upper()} (use /chart {sym} 1h)")        # fallback (chart expects context.args) # Note: Inline callbacks may require more complex handling to pass args; this is simple placeholder
        await query.edit_message_text(f"Requested chart for {sym.upper()} (use /chart {sym})")
    elif data == "btn_learn":
        await learn(update, context)
    elif data == "btn_chart":
        await query.edit_message_text("To view a chart, type `/chart <coin>` (e.g. `/chart bitcoin`)", parse_mode="Markdown")
    elif data == "btn_alert":
        await query.edit_message_text("To set alerts, use `/subscribe <coin>`", parse_mode="Markdown")
    else:
        await query.edit_message_text("Button action not implemented yet.")

# ============================
# /ANALYZE — AI Market Breakdown (CoinGecko)
# ============================
async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Example: /analyze btc 1h")
        return

    symbol = context.args[0].upper()
    timeframe = context.args[1].lower() if len(context.args) > 1 else "1h"

    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    if timeframe not in valid_timeframes:
        await update.message.reply_text("❌ Invalid timeframe. Try: 1m 5m 15m 30m 1h 4h 1d")
        return

    msg = await update.message.reply_text("🧠 Analyzing market structure… Give me 2–3 sec 🔍")

    try:
        # Fetch OHLC from CoinGecko (sync → async wrapper)
        loop = asyncio.get_event_loop()
        ohlc = await loop.run_in_executor(None, fetch_coingecko_ohlc, symbol, timeframe, 120)

        if ohlc is None or len(ohlc) < 20:
            await msg.edit_text("⚠️ Not enough price data to analyze that token.")
            return

        # Build readable OHLC list for LLM
        ohlc_list = []
        for ts, row in ohlc.iterrows():
            ohlc_list.append(
                f"{ts.strftime('%Y-%m-%d %H:%M')} | "
                f"O:{round(row['open'], 4)} "
                f"H:{round(row['high'], 4)} "
                f"L:{round(row['low'], 4)} "
                f"C:{round(row['close'], 4)}"
            )

        last_prices = "\n".join(ohlc_list[-80:])

        # ---- AI Request ----
        prompt = f"""
You are a professional crypto market analyst.
Analyze the following OHLC data for {symbol} on the {timeframe} timeframe.

DATA (last 80 candles):
{last_prices}

Provide:

1. **Trend Direction** (bullish / bearish / ranging)
2. **Key Market Structure** (HH/HL or LH/LL sequences)
3. **Support & Resistance Levels**
4. **Liquidity Zones**
5. **Squeeze, volatility shift, or breakout risk**
6. **Smart Money / order flow perspective**
7. **2-sentence summary**
8. **Actionable trade ideas** (levels & bias only, no signals)

Tone: sharp, trader-focused, no emojis, no disclaimers.
"""

        ai_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=650
        )

        output_text = ai_response.choices[0].message.content.strip()

        await msg.edit_text(
            f"📊 *{symbol} — AI Market Analysis ({timeframe})*\n\n{output_text}",
            parse_mode="Markdown"
        )

    except Exception as e:
        await msg.edit_text("⚠️ Couldn't generate analysis right now.")
        print("AI ERROR:", e)

# -------------------------
# Register handlers and start bot
# ---------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # core commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("explain", explain))
    app.add_handler(CommandHandler("trend", trend))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("learn", learn))
    app.add_handler(CommandHandler("heatmap", heatmap))
    app.add_handler(CommandHandler("alpha", alpha))
    app.add_handler(CommandHandler("sentiment", sentiment))
    app.add_handler(CommandHandler("watchlist", watchlist))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Starting ClarityBot...")
    app.run_polling()

if __name__ == "__main__":
    main()
