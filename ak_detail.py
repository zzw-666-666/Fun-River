"""
A股技术信号策略模块（供 backtest.py 调用）

backtest.py 需要本文件提供：
    - RISK_PER_TRADE
    - ACCOUNT_SIZE
    - fetch_daily_akshare(code, start, end)
    - generate_signals_for_df_enhanced(df)

本文件不包含回测、绘图、下一买点估算等逻辑，只负责：
    - 从 akshare 获取日线数据
    - 计算技术指标
    - 生成买入/卖出信号和止损/目标价
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from typing import Dict

# ================== 基础配置（给 backtest 用） ==================
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

# 这两个是 backtest.py 直接 import 的参数
RISK_PER_TRADE = 0.05       # 每笔风险占本金比例，例如 5%
ACCOUNT_SIZE = 20000        # 账户初始资金

# 拉取历史数据时可能会用到的参数（可按需修改）
HISTORY_START = "20240101"
HISTORY_END = None
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.5         # 请求间隔，避免被封


# ================== 技术指标实现 ==================
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


# ================== 信号确认 & 止损/仓位等 ==================
def confirm_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    信号确认逻辑，避免假信号；生成 signal_score 和 confirmed_signal
    """
    df = df.copy()

    # 成交量确认
    df["volume_sma"] = sma(df["volume"], 20)
    df["volume_confirm"] = df["volume"] > df["volume_sma"] * 1.2

    # 价格突破确认（突破近期高低点）
    df["recent_high"] = df["high"].rolling(20).max()
    df["recent_low"] = df["low"].rolling(20).min()
    df["breakout_confirm"] = df["close"] > df["recent_high"].shift(1)
    df["breakdown_confirm"] = df["close"] < df["recent_low"].shift(1)

    # 信号强度评分
    df["signal_score"] = 0

    buy_conditions = [
        (df["ma_cross_up"] & df["volume_confirm"]),  # 金叉 + 放量
        (df["rsi"] < 30) & df["volume_confirm"],     # 超卖 + 放量
        df["breakout_confirm"],                      # 突破前高
        (df["close"] > df["sma_long"]) & (df["sma_short"] > df["sma_long"]),  # 趋势确认
    ]

    for condition in buy_conditions:
        df.loc[condition, "signal_score"] += 1

    # 只有评分达到阈值才确认买入
    df["confirmed_signal"] = "hold"
    df.loc[(df["signal"] == "buy") & (df["signal_score"] >= 1), "confirmed_signal"] = "buy"
    df.loc[df["signal"] == "sell", "confirmed_signal"] = "sell"

    return df


def calculate_stop_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    多种止损策略组合：生成 suggest_stop, suggest_target, risk_reward_ratio
    """
    df = df.copy()

    for i in range(len(df)):
        if df.at[i, "confirmed_signal"] == "buy":
            entry = df.at[i, "close"]
            atr_value = df.at[i, "atr"]

            stops = []

            # 1. ATR 止损
            stops.append(entry - 2 * atr_value)

            # 2. 百分比止损
            stops.append(entry * 0.95)

            # 3. 支撑位止损（近期低点）
            recent_low = df["low"].iloc[max(0, i - 20) : i + 1].min()
            stops.append(recent_low * 0.98)

            # 4. 均线止损（长均线）
            stops.append(df.at[i, "sma_long"] * 0.97)

            valid_stops = [s for s in stops if s > 0]
            if valid_stops:
                final_stop = max(valid_stops)
                df.at[i, "suggest_stop"] = final_stop

                risk = entry - final_stop
                if risk > 0:
                    market_trend = (
                        "bullish"
                        if df.at[i, "sma_short"] > df.at[i, "sma_long"]
                        else "bearish"
                    )
                    reward_multiplier = 3 if market_trend == "bullish" else 2
                    df.at[i, "suggest_target"] = entry + risk * reward_multiplier
                    df.at[i, "risk_reward_ratio"] = (
                        df.at[i, "suggest_target"] - entry
                    ) / risk
            else:
                df.at[i, "suggest_stop"] = entry * 0.95
                df.at[i, "suggest_target"] = entry * 1.06
                df.at[i, "risk_reward_ratio"] = 1.0

    return df


def assess_market_environment(df: pd.DataFrame) -> Dict[str, str]:
    """
    判断整体市场环境：trend, volatility, suggestion
    """
    if len(df) < 100:
        return {"trend": "neutral", "volatility": "medium", "suggestion": "数据不足"}

    recent_data = df.iloc[-50:]

    price_trend = (
        "bullish"
        if recent_data["close"].iloc[-1] > recent_data["close"].iloc[0]
        else "bearish"
    )
    ma_trend = (
        "bullish"
        if recent_data["sma_short"].iloc[-1] > recent_data["sma_long"].iloc[-1]
        else "bearish"
    )

    avg_atr = recent_data["atr"].mean()
    avg_price = recent_data["close"].mean()
    atr_percent = avg_atr / avg_price if avg_price > 0 else 0

    if atr_percent > 0.03:
        volatility = "high"
    elif atr_percent < 0.015:
        volatility = "low"
    else:
        volatility = "medium"

    if price_trend == "bullish" and ma_trend == "bullish":
        trend_strength = "strong_bull"
        suggestion = "适合做多"
    elif price_trend == "bearish" and ma_trend == "bearish":
        trend_strength = "strong_bear"
        suggestion = "谨慎操作，建议观望"
    else:
        trend_strength = "neutral"
        suggestion = "震荡市，可轻仓操作"

    return {
        "trend": trend_strength,
        "volatility": volatility,
        "suggestion": suggestion,
    }


def advanced_position_sizing(df: pd.DataFrame, market_env: Dict[str, str]) -> pd.DataFrame:
    """
    基于信号强度和市场环境的动态仓位管理：
    生成 suggest_position_shares, suggest_position_percent, position_multiplier
    """
    df = df.copy()

    for i in range(len(df)):
        if df.at[i, "confirmed_signal"] == "buy":
            entry = (
                df.at[i, "suggest_entry"]
                if not np.isnan(df.at[i, "suggest_entry"])
                else df.at[i, "close"]
            )
            stop = (
                df.at[i, "suggest_stop"]
                if not np.isnan(df.at[i, "suggest_stop"])
                else entry * 0.95
            )
            risk_per_share = entry - stop
            if risk_per_share <= 0:
                continue

            score = (
                df.at[i, "signal_score"]
                if not np.isnan(df.at[i, "signal_score"])
                else 1
            )
            strength_multiplier = 0.5 + (score / 4)

            if market_env["trend"] == "strong_bull":
                market_multiplier = 1.2
            elif market_env["trend"] == "strong_bear":
                market_multiplier = 0.3
            else:
                market_multiplier = 0.8

            if market_env["volatility"] == "high":
                vol_multiplier = 0.7
            elif market_env["volatility"] == "low":
                vol_multiplier = 1.1
            else:
                vol_multiplier = 1.0

            total_multiplier = strength_multiplier * market_multiplier * vol_multiplier

            risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE * total_multiplier
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            position_value = shares * entry

            df.at[i, "suggest_position_shares"] = shares
            df.at[i, "suggest_position_percent"] = (
                position_value / ACCOUNT_SIZE if ACCOUNT_SIZE > 0 else 0
            )
            df.at[i, "position_multiplier"] = total_multiplier

    return df


# ================== 基础信号生成 ==================
def generate_signals_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：包含 daily OHLCV 的 df
    输出：加上技术指标 + 初步 signal + suggest_entry/stop/target
    """
    df = df.copy().reset_index(drop=False)

    # 统一列名 (akshare 返回可能是 '日期','开盘','最高' 等)
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ["日期", "date"]:
            colmap[c] = "date"
        if lc in ["开盘", "open", "openprice", "open_price"]:
            colmap[c] = "open"
        if lc in ["最高", "high", "highprice", "high_price"]:
            colmap[c] = "high"
        if lc in ["最低", "low", "lowprice", "low_price"]:
            colmap[c] = "low"
        if lc in ["收盘", "close", "closeprice", "close_price", "最新价", "最新"]:
            colmap[c] = "close"
        if lc in ["成交量", "volume", "vol"]:
            colmap[c] = "volume"
    df = df.rename(columns=colmap)

    for req in ["date", "open", "high", "low", "close", "volume"]:
        if req not in df.columns:
            raise ValueError(f"输入数据缺少列: {req}, 当前列: {df.columns.tolist()}")

    # 指标
    df["sma_short"] = sma(df["close"], SMA_SHORT)
    df["sma_long"] = sma(df["close"], SMA_LONG)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df[["high", "low", "close"]], ATR_PERIOD)

    # 均线金叉/死叉
    df["ma_diff"] = df["sma_short"] - df["sma_long"]
    df["ma_diff_prev"] = df["ma_diff"].shift(1)
    df["ma_cross_up"] = (df["ma_diff"] > 0) & (df["ma_diff_prev"] <= 0)
    df["ma_cross_down"] = (df["ma_diff"] < 0) & (df["ma_diff_prev"] >= 0)

    # 初步信号
    df["signal"] = "hold"
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        buy_cond = False
        # 1）金叉 + RSI 不过高 + 价格不弱于长均线
        if (
            row["ma_cross_up"]
            and (row["rsi"] < 70)
            and (row["close"] >= row["sma_long"] * 0.98)
        ):
            buy_cond = True
        # 2）RSI 超卖反弹
        if (row["rsi"] < 30) and (row["close"] > prev["close"]):
            buy_cond = True
        # 3）多头趋势中的回调买入
        pullback_buy = (
            (row["sma_short"] > row["sma_long"])
            and (row["close"] >= row["sma_long"] * 1.0)
            and (row["close"] <= row["sma_short"] * 1.03)
            and (row["rsi"] > 40)
            and (row["rsi"] < 70)
            and (row["close"] > prev["close"])
        )
        if pullback_buy:
            buy_cond = True

        sell_cond = False
        # 1）死叉 + 跌破短均线
        if row["ma_cross_down"] and (row["close"] < row["sma_short"]):
            sell_cond = True
        # 2）RSI 从超买掉头 + 价格走弱
        rsi_turn_down = (prev["rsi"] > 70) and (row["rsi"] < prev["rsi"])
        price_weak = (row["close"] < prev["close"]) and (
            row["close"] < row["sma_short"]
        )
        if rsi_turn_down and price_weak:
            sell_cond = True

        if buy_cond:
            df.at[i, "signal"] = "buy"
        elif sell_cond:
            df.at[i, "signal"] = "sell"
        else:
            df.at[i, "signal"] = "hold"

    # 对每个 buy 估算初始止损 & 目标价
    df["suggest_entry"] = np.nan
    df["suggest_stop"] = np.nan
    df["suggest_target"] = np.nan
    df["suggest_position_shares"] = np.nan
    df["suggest_position_percent"] = np.nan
    df["risk_reward_ratio"] = np.nan
    df["position_multiplier"] = 1.0

    for i in range(len(df)):
        if df.at[i, "signal"] == "buy":
            entry = df.at[i, "close"]
            atr_i = df.at[i, "atr"]
            stop = entry - 2.0 * atr_i if not np.isnan(atr_i) else entry * 0.97
            if stop <= 0:
                stop = entry * 0.97
            target = entry + 4.0 * atr_i if not np.isnan(atr_i) else entry * 1.06

            df.at[i, "suggest_entry"] = entry
            df.at[i, "suggest_stop"] = stop
            df.at[i, "suggest_target"] = target

    return df


def generate_signals_for_df_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强版信号：在 generate_signals_for_df 基础上增加：
        - confirmed_signal / signal_score
        - suggest_stop/target/risk_reward_ratio（增强版）
        - 建议仓位（suggest_position_shares/percent）
        - 市场环境（market_trend / market_volatility / market_suggestion）
    """
    df = generate_signals_for_df(df)
    df = confirm_signals(df)
    df = calculate_stop_loss(df)

    market_env = assess_market_environment(df)
    df = advanced_position_sizing(df, market_env)

    df["market_trend"] = market_env["trend"]
    df["market_volatility"] = market_env["volatility"]
    df["market_suggestion"] = market_env["suggestion"]

    return df


# ================== akshare 数据拉取 ==================
def fetch_daily_akshare(code: str, start: str = HISTORY_START, end: str = HISTORY_END) -> pd.DataFrame:
    """
    用 akshare 拉取 A 股日线数据：
        ak.stock_zh_a_daily(symbol=code, start_date, end_date)
    尝试常见几种写法（'sh'/'sz' 前缀），返回按时间升序的 DataFrame。
    """
    code_str = str(code).strip()
    tries = [code_str]
    if code_str.isdigit() and len(code_str) == 6:
        tries += [f"sh{code_str}", f"sz{code_str}", f"sh.{code_str}", f"sz.{code_str}"]
    tries = list(dict.fromkeys(tries))

    for t in tries:
        for _ in range(MAX_RETRIES):
            try:
                df = ak.stock_zh_a_daily(symbol=t, start_date=start, end_date=end)
                if df is None or df.empty:
                    time.sleep(SLEEP_BETWEEN)
                    continue

                if "日期" in df.columns:
                    df = df.rename(
                        columns={
                            "日期": "date",
                            "开盘": "open",
                            "最高": "high",
                            "最低": "low",
                            "收盘": "close",
                            "成交量": "volume",
                        }
                    )
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                except Exception:
                    pass
                return df
            except Exception as e:
                print(f"尝试代码 {t} 失败: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

    raise RuntimeError(f"无法通过 akshare 获取 日线: {code}. 尝试过: {tries}")