"""Technical indicators for OHLCV bars.

We compute a curated set rather than dumping everything `ta` offers,
because (a) more features ≠ better model — collinear features hurt,
and (b) we want to know what each column means when SHAP attributes
to it later.

The set covers four categories:
  - **Momentum**:   RSI, ROC
  - **Trend**:      MACD, EMA spread
  - **Volatility**: ATR, Bollinger band width, realised vol
  - **Volume**:     OBV, volume vs. its moving average
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def add_technical_features(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """Append technical indicators to an OHLCV DataFrame.

    Expects columns: open, high, low, close, adj_close, volume.
    Indicators are computed on raw `close` (volume + price action),
    *not* adj_close — splits already corrected by yfinance for OHLC.
    """
    out = df if inplace else df.copy()
    close, high, low, vol = out["close"], out["high"], out["low"], out["volume"]

    # --- Momentum ---
    out["rsi_14"] = RSIIndicator(close, window=14).rsi()
    out["roc_10"] = ROCIndicator(close, window=10).roc()

    # --- Trend (MACD) ---
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # EMA spread: % distance of price above/below long-term EMA
    ema_50 = EMAIndicator(close, window=50).ema_indicator()
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    out["ema_50_spread"] = (close - ema_50) / ema_50
    out["ema_200_spread"] = (close - ema_200) / ema_200
    out["golden_cross"] = (ema_50 > ema_200).astype(float)

    # --- Volatility ---
    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    out["bb_pct_b"] = bb.bollinger_pband()  # 0=lower band, 1=upper band

    # Realised volatility (annualised, log-return std)
    log_ret = np.log(close / close.shift(1))
    out["vol_20d_ann"] = log_ret.rolling(20).std() * np.sqrt(252)
    out["vol_60d_ann"] = log_ret.rolling(60).std() * np.sqrt(252)

    # --- Volume ---
    out["obv"] = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["vol_ma_20"] = vol.rolling(20).mean()
    out["vol_ratio"] = vol / out["vol_ma_20"]

    # --- Past returns (multiple horizons) ---
    for h in (1, 5, 20, 60):
        out[f"ret_{h}d"] = np.log(close / close.shift(h))

    return out


FEATURE_COLUMNS: list[str] = [
    "rsi_14", "roc_10",
    "macd", "macd_signal", "macd_diff",
    "ema_50_spread", "ema_200_spread", "golden_cross",
    "atr_14", "bb_width", "bb_pct_b",
    "vol_20d_ann", "vol_60d_ann",
    "obv", "vol_ratio",
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
]
