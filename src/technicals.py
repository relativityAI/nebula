import yfinance as yf
import talib
import pandas as pd
from utils.score import *

# -----------------------------
# Thresholds for technical indicators
# -----------------------------
TECH_THRESHOLDS = {
                "rsi":{"type":"sweet","center":50,"width":15,"weight":1.0},
                "stoch_k":{"type":"sweet","center":50,"width":20,"weight":0.8},
                "stoch_d":{"type":"sweet","center":50,"width":20,"weight":0.8},
                "macd_hist":{"type":"higher","midpoint":0.0,"weight":1.1},
                "macd":{"type":"higher","midpoint":0.0,"weight":0.9},
                "adx":{"type":"higher","midpoint":25,"weight":1.0},
                "plus_di_minus_di":{"type":"higher","midpoint":0.0,"weight":0.9},
                "sma_20_50_diff":{"type":"higher","midpoint":0.0,"weight":1.0},
                "sma_50_200_diff":{"type":"higher","midpoint":0.0,"weight":1.2},
                "ema_12_26_diff":{"type":"higher","midpoint":0.0,"weight":1.0},
                "bb_percent_b":{"type":"sweet","center":0.5,"width":0.25,"weight":0.8},
                "cci":{"type":"sweet","center":0.0,"width":100,"weight":0.7},
                "atr_pct":{"type":"lower","midpoint":0.03,"weight":0.7},
                "volatility_20d":{"type":"lower","midpoint":0.04,"weight":0.6},
                "momentum_10d":{"type":"higher","midpoint":0.0,"weight":0.9},
                "roc_10d":{"type":"higher","midpoint":0.0,"weight":0.9}
            }


def technical_analysis_talib(symbol: str, price_df: pd.DataFrame, thresholds=TECH_THRESHOLDS):
    df = price_df.copy()
    close = df["Close"].to_numpy(dtype=float).flatten()
    high = df["High"].to_numpy(dtype=float).flatten()
    low = df["Low"].to_numpy(dtype=float).flatten()
    open_ = df["Open"].to_numpy(dtype=float).flatten()

    if len(close) < 50:
        raise ValueError("Not enough data to calculate indicators")

    df["rsi"] = talib.RSI(close, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(high, low, close)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    macd, macd_signal, macd_hist = talib.MACD(close)
    df["macd"] = macd
    df["macd_hist"] = macd_hist

    df["adx"] = talib.ADX(high, low, close)
    plus_di = talib.PLUS_DI(high, low, close)
    minus_di = talib.MINUS_DI(high, low, close)
    df["plus_di_minus_di"] = plus_di - minus_di

    sma_20 = talib.SMA(close, timeperiod=20)
    sma_50 = talib.SMA(close, timeperiod=50)
    sma_200 = talib.SMA(close, timeperiod=200)
    df["sma_20_50_diff"] = (sma_20 - sma_50) / sma_50
    df["sma_50_200_diff"] = (sma_50 - sma_200) / sma_200
    ema_12 = talib.EMA(close, timeperiod=12)
    ema_26 = talib.EMA(close, timeperiod=26)
    df["ema_12_26_diff"] = ema_12 - ema_26

    upper, middle, lower = talib.BBANDS(close)
    df["bb_percent_b"] = (close - lower) / (upper - lower)
    df["cci"] = talib.CCI(high, low, close)
    atr = talib.ATR(high, low, close)
    df["atr_pct"] = atr / close
    df["momentum_10d"] = talib.MOM(close, timeperiod=10)
    df["roc_10d"] = talib.ROC(close, timeperiod=10)
    df["volatility_20d"] = pd.Series(close).pct_change().rolling(20).std().to_numpy()

    df.dropna(inplace=True)
    latest = df.iloc[-1]
    all_metrics = {k: float(latest[k]) for k in thresholds.keys() if k in latest}

    return {
        "symbol": symbol,
        "metrics": all_metrics,
        "score": score_metrics(all_metrics, thresholds)
    }

def get_price_data(symbol: str, period="6mo", interval="1d"):
    return yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)

if __name__ == "__main__":
    from pprint import pprint
    symbol = "MUTHOOTFIN.NS"
    price_df = get_price_data(symbol, "3y")
    analysis = technical_analysis_talib(symbol, price_df)
    pprint(analysis)    