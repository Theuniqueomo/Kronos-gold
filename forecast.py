import pandas as pd
import numpy as np
import yfinance as yf
import torch
import requests
import sys
import os
from datetime import datetime, timedelta

# --- INSTALL KRONOS ---
os.system("git clone -q https://github.com/shiyu-coder/Kronos.git")
sys.path.append("/content/Kronos")
sys.path.append("./Kronos")

from model import Kronos, KronosTokenizer, KronosPredictor

# --- TELEGRAM SETTINGS ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing Telegram credentials. Printing to log instead.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        print("Telegram sent:", r.status_code)
    except Exception as e:
        print("Telegram failed:", e)

def run_forecast():
    try:
        print("📥 Downloading Gold data (15-min candles)...")
        # --- CHANGED: period="1mo", interval="15m" ---
        df = yf.download("GC=F", period="1mo", interval="15m", progress=False)
        if df.empty:
            send_telegram("❌ Error: No data downloaded from Yahoo.")
            return

        # --- Data Cleaning ---
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.lower() for col in df.columns]
        
        date_cols = [col for col in df.columns if 'date' in col or 'datetime' in col or 'time' in col]
        if date_cols:
            df = df.rename(columns={date_cols[0]: 'timestamps'})
        else:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'timestamps'})
            
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in req_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].dropna()
        df['amount'] = 0.0

        print(f"✅ Loaded {len(df)} rows of 15-min data.")

        # --- Load Kronos Model ---
        print("🧠 Loading Kronos AI...")
        TOK = "NeoQuasar/Kronos-Tokenizer-base"
        MODEL = "NeoQuasar/Kronos-small"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = KronosTokenizer.from_pretrained(TOK)
        model = Kronos.from_pretrained(MODEL)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

        # --- Predict the Future ---
        lookback = min(400, len(df) - 120)
        # --- CHANGED: pred_len=20 (20 candles * 15min = 5 hours) ---
        pred_len = 20  # 5 hours total
        
        x_df = df.loc[:lookback-1, ["open", "high", "low", "close", "volume", "amount"]].fillna(0)
        x_timestamp = pd.Series(df.loc[:lookback-1, "timestamps"])
        
        last_time = df['timestamps'].iloc[-1]
        # --- CHANGED: minutes=15, freq='15min' ---
        future_times = pd.date_range(start=last_time + timedelta(minutes=15), periods=pred_len, freq='15min')
        y_timestamp = pd.Series(future_times)

        print("🔮 Running prediction...")
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.6,
            top_p=0.7,
            sample_count=1
        )

        # --- Format Results ---
        latest = pred_df.iloc[-1]
        first = pred_df.iloc[0]
        avg_price = pred_df["close"].mean()
        direction = "📈 UP" if latest["close"] > first["close"] else "📉 DOWN"
        change_pct = ((latest["close"] / first["close"]) - 1) * 100
        current_price = df['close'].iloc[-1]

        message = (
            f"<b>🚀 KRONOS GOLD FORECAST (15-min candles)</b>\n"
            f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"─────────────────\n"
            f"💰 Current Gold: <b>${current_price:.2f}</b>\n"
            f"📊 Forecast Direction (5hr): <b>{direction}</b>\n"
            f"📈 Predicted Range: ${first['close']:.2f} → ${latest['close']:.2f}\n"
            f"📉 Avg Predicted: ${avg_price:.2f}\n"
            f"🔄 Expected Move: {change_pct:+.2f}%\n"
            f"─────────────────\n"
            f"<i>First 3 predictions (15min intervals):</i>\n"
            f"1. {future_times[0].strftime('%H:%M')}: ${pred_df.iloc[0]['close']:.2f}\n"
            f"2. {future_times[1].strftime('%H:%M')}: ${pred_df.iloc[1]['close']:.2f}\n"
            f"3. {future_times[2].strftime('%H:%M')}: ${pred_df.iloc[2]['close']:.2f}\n"
            f"─────────────────\n"
            f"<b>⚠️ Not financial advice. AI research only.</b>"
        )

        send_telegram(message)
        print("✅ Forecast sent successfully!")

    except Exception as e:
        error_msg = f"❌ KRONOS CRASHED:\n{str(e)}"
        send_telegram(error_msg)
        print(error_msg)

if __name__ == "__main__":
    run_forecast()
