"""
A股技术信号生成器（买入/卖出点）增强版 + 下一次买入位置估算
依赖：akshare, pandas, numpy, tqdm
功能：
 - 直接在代码中指定候选股票列表
 - 拉取日线历史数据（akshare）
 - 计算 SMA(short/long), RSI(14), ATR(14)
 - 生成买入/卖出信号，推荐止损/目标价与仓位建议
 - 包含信号确认、动态止损、市场环境判断和智能仓位管理
 - 新增：根据历史数据和当前指标估算“下一次买入位置和区间”
"""
import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from typing import List, Tuple, Dict, Any
from datetime import datetime

# ---------------- 配置 ----------------
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

HISTORY_START = "20240101"   # 起始历史（格式 YYYYMMDD），按需修改
HISTORY_END = None           # None 表示到最近
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.5          # 增加请求间隔，避免被封

# 风险管理 / 建议
RISK_PER_TRADE = 0.05        # 每笔风险占本金比例，比如 5%
ACCOUNT_SIZE = 20000         # 假设账户大小（用于仓位建议），可按实际调整

# ----------------- 候选股票列表 -----------------
# 在这里直接指定要分析的股票代码
CANDIDATE_STOCKS = [
    "600595"
]

# ----------------- 手动价格即时评估 -----------------
def analyze_manual_price(df_ind: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    基于当前手动输入的价格，对最新一根K线重新计算信号，给出 buy/sell/hold 建议
    输入:
        df_ind: 已经过 generate_signals_for_df_enhanced 处理过的完整历史数据
        current_price: 当前时刻你看到的价格（比如盘中价）
    输出:
        一个 dict，包含推荐操作和关键指标
    """
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in base_cols if c not in df_ind.columns]
    if missing:
        raise ValueError(f"历史数据缺少必要列: {missing}")

    # 只取原始OHLCV，重新走一遍指标逻辑
    df_raw = df_ind[base_cols].copy()
    last_idx = df_raw.index[-1]

    # 更新最后一根K线的收盘价
    df_raw.at[last_idx, 'close'] = current_price
    # 保守处理：高/低价至少包含当前价
    if current_price > df_raw.at[last_idx, 'high']:
        df_raw.at[last_idx, 'high'] = current_price
    if current_price < df_raw.at[last_idx, 'low']:
        df_raw.at[last_idx, 'low'] = current_price

    # 用增强版信号生成重新计算一遍
    df_new = generate_signals_for_df_enhanced(df_raw)
    last = df_new.iloc[-1]

    result = {
        'signal': last['signal'],
        'confirmed_signal': last['confirmed_signal'],
        'signal_score': last.get('signal_score', np.nan),
        'current_price': current_price,
        'rsi': last['rsi'],
        'sma_short': last['sma_short'],
        'sma_long': last['sma_long'],
        'atr': last['atr'],
        'suggest_entry': last.get('suggest_entry', np.nan),
        'suggest_stop': last.get('suggest_stop', np.nan),
        'suggest_target': last.get('suggest_target', np.nan),
        'risk_reward_ratio': last.get('risk_reward_ratio', np.nan),
        'position_shares': last.get('suggest_position_shares', np.nan),
        'position_percent': last.get('suggest_position_percent', np.nan),
        'market_trend': last.get('market_trend', 'unknown'),
        'market_volatility': last.get('market_volatility', 'unknown'),
        'market_suggestion': last.get('market_suggestion', 'unknown'),
    }
    return result

# ----------------- 技术指标实现 -----------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# ----------------- 信号确认机制 -----------------
def confirm_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    信号确认逻辑，避免假信号
    """
    df = df.copy()
    
    # 1. 成交量确认
    df['volume_sma'] = sma(df['volume'], 20)
    df['volume_confirm'] = df['volume'] > df['volume_sma'] * 1.2
    
    # 2. 价格突破确认（突破近期高低点）
    df['recent_high'] = df['high'].rolling(20).max()
    df['recent_low'] = df['low'].rolling(20).min()
    df['breakout_confirm'] = df['close'] > df['recent_high'].shift(1)
    df['breakdown_confirm'] = df['close'] < df['recent_low'].shift(1)
    
    # 3. 信号强度评分
    df['signal_score'] = 0
    
    # 买入信号强度
    buy_conditions = [
        (df['ma_cross_up'] & df['volume_confirm']),  # 金叉+放量
        (df['rsi'] < 30) & df['volume_confirm'],     # 超卖+放量
        df['breakout_confirm'],                      # 突破前高
        (df['close'] > df['sma_long']) & (df['sma_short'] > df['sma_long']),  # 趋势确认
    ]
    
    for condition in buy_conditions:
        df.loc[condition, 'signal_score'] += 1
    
    # 修正信号：只有评分达到阈值才确认买入
    df['confirmed_signal'] = 'hold'
    df.loc[(df['signal'] == 'buy') & (df['signal_score'] >= 1), 'confirmed_signal'] = 'buy'
    df.loc[df['signal'] == 'sell', 'confirmed_signal'] = 'sell'  # 卖出信号通常更紧急
    
    return df

# ----------------- 止损策略优化 -----------------
def calculate_stop_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    多种止损策略组合
    """
    df = df.copy()
    
    for i in range(len(df)):
        if df.at[i, 'confirmed_signal'] == 'buy':
            entry = df.at[i, 'close']
            atr_value = df.at[i, 'atr']
            
            # 多种止损方法
            stops = []
            
            # 1. ATR止损
            stops.append(entry - 2 * atr_value)
            
            # 2. 百分比止损
            stops.append(entry * 0.95)  # 5%固定止损
            
            # 3. 支撑位止损（使用近期低点）
            recent_low = df['low'].iloc[max(0, i-20):i+1].min()
            stops.append(recent_low * 0.98)
            
            # 4. 均线止损
            stops.append(df.at[i, 'sma_long'] * 0.97)
            
            # 选择最保守的止损（保护本金）
            valid_stops = [s for s in stops if s > 0]
            if valid_stops:
                final_stop = max(valid_stops)
                df.at[i, 'suggest_stop'] = final_stop
                
                # 动态计算风险回报比
                risk = entry - final_stop
                if risk > 0:
                    # 根据市场状况调整目标
                    market_trend = "bullish" if df.at[i, 'sma_short'] > df.at[i, 'sma_long'] else "bearish"
                    reward_multiplier = 3 if market_trend == "bullish" else 2
                    df.at[i, 'suggest_target'] = entry + risk * reward_multiplier
                    
                    # 计算风险回报比
                    df.at[i, 'risk_reward_ratio'] = (df.at[i, 'suggest_target'] - entry) / risk
            else:
                df.at[i, 'suggest_stop'] = entry * 0.95
                df.at[i, 'suggest_target'] = entry * 1.06
                df.at[i, 'risk_reward_ratio'] = 1.0
    
    return df

# ----------------- 市场环境判断 -----------------
def assess_market_environment(df: pd.DataFrame) -> dict:
    """
    判断整体市场环境，避免逆势操作
    """
    if len(df) < 100:
        return {"trend": "neutral", "volatility": "medium", "suggestion": "数据不足"}
    
    recent_data = df.iloc[-50:]  # 最近50个交易日
    
    # 趋势判断
    price_trend = "bullish" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "bearish"
    ma_trend = "bullish" if recent_data['sma_short'].iloc[-1] > recent_data['sma_long'].iloc[-1] else "bearish"
    
    # 波动性判断
    avg_atr = recent_data['atr'].mean()
    avg_price = recent_data['close'].mean()
    atr_percent = avg_atr / avg_price if avg_price > 0 else 0
    
    if atr_percent > 0.03:
        volatility = "high"
    elif atr_percent < 0.015:
        volatility = "low"
    else:
        volatility = "medium"
    
    # 综合判断
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
        "suggestion": suggestion
    }

# ----------------- 仓位管理增强 -----------------
def advanced_position_sizing(df: pd.DataFrame, market_env: dict) -> pd.DataFrame:
    """
    基于信号强度和市场环境的动态仓位管理
    """
    df = df.copy()
    
    for i in range(len(df)):
        if df.at[i, 'confirmed_signal'] == 'buy':
            # 基础风险计算
            entry = df.at[i, 'suggest_entry'] if not np.isnan(df.at[i, 'suggest_entry']) else df.at[i, 'close']
            stop = df.at[i, 'suggest_stop'] if not np.isnan(df.at[i, 'suggest_stop']) else entry * 0.95
            risk_per_share = entry - stop
            
            if risk_per_share <= 0:
                continue
                
            # 信号强度系数 (0.5-1.5)
            score = df.at[i, 'signal_score'] if not np.isnan(df.at[i, 'signal_score']) else 1
            strength_multiplier = 0.5 + (score / 4)  # 假设score最高4分
            
            # 市场环境系数
            if market_env['trend'] == "strong_bull":
                market_multiplier = 1.2
            elif market_env['trend'] == "strong_bear":
                market_multiplier = 0.3  # 熊市大幅降低仓位
            else:
                market_multiplier = 0.8
                
            # 波动性调整
            if market_env['volatility'] == "high":
                vol_multiplier = 0.7
            elif market_env['volatility'] == "low":
                vol_multiplier = 1.1
            else:
                vol_multiplier = 1.0
                
            # 综合仓位调整
            total_multiplier = strength_multiplier * market_multiplier * vol_multiplier
            
            # 计算最终仓位
            risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE * total_multiplier
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            position_value = shares * entry
            
            df.at[i, 'suggest_position_shares'] = shares
            df.at[i, 'suggest_position_percent'] = position_value / ACCOUNT_SIZE if ACCOUNT_SIZE > 0 else 0
            df.at[i, 'position_multiplier'] = total_multiplier
    
    return df

# ----------------- 新增：下一次买入位置估算 -----------------
def calculate_next_buy_level(df_ind: pd.DataFrame) -> Dict[str, Any]:
    """
    基于完整历史 + 当前指标，估算“下一次理想买入价”及买入区间。

    设计思路：
    - 如果当前已经有 confirmed buy 信号：认为已经/刚刚建仓，这里给的是“回调加仓”的理想买点；
    - 如果当前没有 buy 信号：结合市场趋势、ATR、均线和近期低点，给出下一次建议买入区间；
    - 强势熊市（strong_bear）下直接建议等待，不主动给买点。

    返回字段：
        mode           : 计算模式（add_on_pullback / pullback_in_uptrend / pullback_or_range / wait）
        reason         : 文本说明
        next_buy_price : 建议关注/挂单的核心价格
        buy_zone_low   : 建议买入区间下沿
        buy_zone_high  : 建议买入区间上沿
        ref_close      : 当前收盘价
        ref_sma_short  : 当前短均线
        ref_sma_long   : 当前长均线
        ref_recent_low : 最近支撑位参考（最近20日低点或 recent_low）
    """
    if df_ind is None or df_ind.empty:
        return {
            'mode': 'unknown',
            'reason': '数据为空，无法计算',
            'next_buy_price': np.nan,
            'buy_zone_low': np.nan,
            'buy_zone_high': np.nan,
            'ref_close': np.nan,
            'ref_sma_short': np.nan,
            'ref_sma_long': np.nan,
            'ref_recent_low': np.nan,
        }

    df = df_ind.copy()
    last = df.iloc[-1]

    close = float(last['close'])
    atr_val = float(last['atr']) if not np.isnan(last['atr']) else 0.0
    sma_s = float(last['sma_short']) if not np.isnan(last['sma_short']) else close
    sma_l = float(last['sma_long']) if not np.isnan(last['sma_long']) else close

    # 近期低点作为支撑位参考
    recent_low = last.get('recent_low', np.nan)
    try:
        recent_low_val = float(recent_low)
        if np.isnan(recent_low_val):
            raise ValueError
    except Exception:
        # 如果没有 recent_low 列（或者是 NaN），退化为最近 20 根 K 线的最低价
        recent_low_val = float(df['low'].tail(20).min())

    market_trend = last.get('market_trend', 'neutral')
    confirmed = last.get('confirmed_signal', 'hold')

    result = {
        'mode': 'neutral',
        'reason': '',
        'next_buy_price': np.nan,
        'buy_zone_low': np.nan,
        'buy_zone_high': np.nan,
        'ref_close': close,
        'ref_sma_short': sma_s,
        'ref_sma_long': sma_l,
        'ref_recent_low': recent_low_val,
    }

    # 1）明显熊市：不主动给买点，防止硬抄底
    if market_trend == 'strong_bear':
        result['mode'] = 'wait'
        result['reason'] = '市场为强势下跌趋势，暂不建议主动寻找买点'
        return result

    # 2）当前已经是买入信号：给“回调加仓”的下一买入点
    if confirmed == 'buy':
        base_entry = last.get('suggest_entry', np.nan)
        if np.isnan(base_entry):
            base_entry = close
        base_entry = float(base_entry)

        # 期望在 base_entry 下方 0.5~1 个 ATR 处做回调加仓（至少 1%）
        down_buffer = max(0.5 * atr_val, base_entry * 0.01)
        ideal = base_entry - down_buffer
        zone_low = ideal - 0.5 * atr_val
        zone_high = ideal + 0.5 * atr_val

        result.update({
            'mode': 'add_on_pullback',
            'reason': '当前已有买入信号，给出回调加仓区间',
            'next_buy_price': max(ideal, 0.01),
            'buy_zone_low': max(zone_low, 0.01),
            'buy_zone_high': max(zone_high, 0.01),
        })
        return result

    # 3）当前没有买入信号：基于支撑位 + 均线给出下一买入价
    # 支撑位：长期均线 & 近期低点略上方
    support_level = max(sma_l, recent_low_val * 1.01) if recent_low_val > 0 else sma_l

    if market_trend == 'strong_bull':
        # 强势多头：优先在短均线/支撑位附近接回调
        base = max(sma_s, support_level)
        mode = 'pullback_in_uptrend'
        reason = '强势多头趋势，优先考虑回调到短均线/支撑位附近买入'
    else:
        # 非强势多头：参考中长期支撑位
        base = support_level
        mode = 'pullback_or_range'
        reason = '非强势多头，以中长期支撑位作为买入参考'

    # 稍微高于支撑位 1% 作为核心参考价，区间 ±2%
    ideal = base * 1.01
    zone_low = base * 0.99
    zone_high = base * 1.03

    result.update({
        'mode': mode,
        'reason': reason,
        'next_buy_price': max(ideal, 0.01),
        'buy_zone_low': max(zone_low, 0.01),
        'buy_zone_high': max(zone_high, 0.01),
    })
    return result

# ----------------- 基础信号生成 -----------------
def generate_signals_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：包含 daily OHLCV 的 df（index 或列包含日期，列名必须为: open, high, low, close, volume）
    输出：原 df 加上指标列及 signal 列（'buy'/'sell'/'hold'）和建议 stop/take/position
    """
    df = df.copy().reset_index(drop=False)
    
    # 统一列名 (akshare 返回可能是 '日期','开盘','最高' 等)
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ['日期','date']: colmap[c] = 'date'
        if lc in ['开盘','open','openprice','open_price']: colmap[c] = 'open'
        if lc in ['最高','high','highprice','high_price']: colmap[c] = 'high'
        if lc in ['最低','low','lowprice','low_price']: colmap[c] = 'low'
        if lc in ['收盘','close','closeprice','close_price','最新价','最新']: colmap[c] = 'close'
        if lc in ['成交量','volume','vol']: colmap[c] = 'volume'
    df = df.rename(columns=colmap)
    
    # ensure required cols
    for req in ['date','open','high','low','close','volume']:
        if req not in df.columns:
            raise ValueError(f"输入数据缺少列: {req}, 当前列: {df.columns.tolist()}")

    # 指标
    df['sma_short'] = sma(df['close'], SMA_SHORT)
    df['sma_long'] = sma(df['close'], SMA_LONG)
    df['rsi'] = rsi(df['close'], RSI_PERIOD)
    df['atr'] = atr(df[['high','low','close']], ATR_PERIOD)

    # 均线金叉/死叉（当天短线上穿/下穿长线）
    df['ma_diff'] = df['sma_short'] - df['sma_long']
    df['ma_diff_prev'] = df['ma_diff'].shift(1)
    df['ma_cross_up'] = (df['ma_diff'] > 0) & (df['ma_diff_prev'] <= 0)
    df['ma_cross_down'] = (df['ma_diff'] < 0) & (df['ma_diff_prev'] >= 0)

    # 信号判定
    df['signal'] = 'hold'
    for i in range(1, len(df)):
        row = df.iloc[i]
        # 默认条件：
        # 买入：出现金叉 & RSI < 70 & 收盘价在长均线上方（表示趋势方向） 或 RSI < 30（超卖反弹机会）
        buy_cond = False
        if row['ma_cross_up'] and (row['rsi'] < 70) and (row['close'] >= row['sma_long'] * 0.98):
            buy_cond = True
        if row['rsi'] < 30 and (row['close'] > df.iloc[i-1]['close']):  # RSI 超卖且价格有反弹迹象
            buy_cond = buy_cond or True

        # 卖出：出现死叉 或 RSI > 70（超买）
        sell_cond = False
        if row['ma_cross_down']:
            sell_cond = True
        if row['rsi'] > 70:
            sell_cond = True

        if buy_cond:
            df.at[i, 'signal'] = 'buy'
        elif sell_cond:
            df.at[i, 'signal'] = 'sell'
        else:
            df.at[i, 'signal'] = 'hold'

    # 对每个 buy signal 估算止损 & 目标价（用 ATR）
    df['suggest_entry'] = np.nan
    df['suggest_stop'] = np.nan
    df['suggest_target'] = np.nan
    df['suggest_position_shares'] = np.nan
    df['suggest_position_percent'] = np.nan
    df['risk_reward_ratio'] = np.nan
    df['position_multiplier'] = 1.0

    for i in range(len(df)):
        if df.at[i, 'signal'] == 'buy':
            entry = df.at[i, 'close']
            stop = entry - 2.0 * df.at[i, 'atr'] if not np.isnan(df.at[i, 'atr']) else entry * 0.97
            if stop <= 0:
                stop = entry * 0.97
            target = entry + 4.0 * df.at[i, 'atr'] if not np.isnan(df.at[i, 'atr']) else entry * 1.06
            
            df.at[i, 'suggest_entry'] = entry
            df.at[i, 'suggest_stop'] = stop
            df.at[i, 'suggest_target'] = target

    return df

# ----------------- 增强版信号生成 -----------------
def generate_signals_for_df_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强版的信号生成函数
    """
    # 原有计算
    df = generate_signals_for_df(df)
    
    # 新增功能
    df = confirm_signals(df)
    df = calculate_stop_loss(df)
    
    # 市场环境评估
    market_env = assess_market_environment(df)
    df = advanced_position_sizing(df, market_env)
    
    # 添加市场环境信息
    df['market_trend'] = market_env['trend']
    df['market_volatility'] = market_env['volatility']
    df['market_suggestion'] = market_env['suggestion']
    
    return df

# ---------------- akshare 数据拉取封装 ----------------
def fetch_daily_akshare(code: str, start: str = HISTORY_START, end: str = HISTORY_END) -> pd.DataFrame:
    """
    用 akshare 拉取 A 股日线：ak.stock_zh_a_daily(symbol=code, start_date, end_date)
    尝试几种常见写法（'sh' 和 'sz' 前缀），并返回 DataFrame（按时间升序）
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
                
                if '日期' in df.columns:
                    df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                except Exception:
                    pass
                return df
            except Exception as e:
                print(f"尝试代码 {t} 失败: {str(e)}")
                time.sleep(SLEEP_BETWEEN)
                continue
    
    raise RuntimeError(f"无法通过 akshare 获取 日线: {code}. 尝试过: {tries}")

# ----------------- 回测验证框架 -----------------
def backtest_signal(df: pd.DataFrame, days_hold: int = 10) -> pd.DataFrame:
    """
    简单回测：买入后持有N天的表现
    """
    results = []
    for i in range(len(df) - days_hold):
        if df.iloc[i]['confirmed_signal'] == 'buy':
            buy_price = df.iloc[i]['close']
            sell_price = df.iloc[i + days_hold]['close']
            returns = (sell_price - buy_price) / buy_price
            results.append({
                'buy_date': df.iloc[i]['date'],
                'return': returns,
                'hold_days': days_hold
            })
    return pd.DataFrame(results)

# ----------------- 批量处理候选池 -----------------
def analyze_candidates(codes: List[str], start: str = HISTORY_START, end: str = HISTORY_END) -> Tuple[pd.DataFrame, dict]:
    """
    对候选股票列表进行分析，返回：
      - df_summary: 每只股票最新一条的信号摘要
      - detailed: dict[code] -> full dataframe with indicators & signals
    """
    summary_rows = []
    detailed = {}
    
    for code in tqdm(codes, desc="分析候选股票"):
        try:
            time.sleep(SLEEP_BETWEEN)
            df = fetch_daily_akshare(code, start=start, end=end)
            df_ind = generate_signals_for_df_enhanced(df)
            detailed[code] = df_ind
            
            last = df_ind.iloc[-1]
            name = f"股票{code}"  # 默认名称

            # 新增：计算下一次买入价
            next_buy_info = calculate_next_buy_level(df_ind)

            summary_rows.append({
                'code': code,
                'name': name,
                'last_date': last['date'],
                'last_close': last['close'],
                'signal': last['signal'],
                'confirmed_signal': last['confirmed_signal'],
                'signal_score': last['signal_score'],
                'entry': last['suggest_entry'],
                'stop': last['suggest_stop'],
                'target': last['suggest_target'],
                'risk_reward_ratio': last.get('risk_reward_ratio', np.nan),
                'position_shares': last['suggest_position_shares'],
                'position_percent': last['suggest_position_percent'],
                'position_multiplier': last.get('position_multiplier', 1.0),
                'rsi': last['rsi'],
                'sma_short': last['sma_short'],
                'sma_long': last['sma_long'],
                'atr': last['atr'],
                'market_trend': last.get('market_trend', 'unknown'),
                'market_suggestion': last.get('market_suggestion', 'unknown'),
                # 下一次买入位置
                'next_buy_price': next_buy_info['next_buy_price'],
                'next_buy_low': next_buy_info['buy_zone_low'],
                'next_buy_high': next_buy_info['buy_zone_high'],
                'next_buy_mode': next_buy_info['mode'],
            })
            
        except Exception as e:
            print(f"分析股票 {code} 时出错: {str(e)}")
            summary_rows.append({
                'code': code,
                'name': None,
                'last_date': None,
                'last_close': None,
                'signal': 'error',
                'confirmed_signal': 'error',
                'signal_score': 0,
                'entry': None,
                'stop': None,
                'target': None,
                'risk_reward_ratio': None,
                'position_shares': None,
                'position_percent': None,
                'position_multiplier': None,
                'rsi': None,
                'sma_short': None,
                'sma_long': None,
                'atr': None,
                'market_trend': 'unknown',
                'market_suggestion': '数据获取失败',
                'next_buy_price': None,
                'next_buy_low': None,
                'next_buy_high': None,
                'next_buy_mode': 'error',
            })
            continue

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(['confirmed_signal', 'position_percent'], 
                                       ascending=[True, False]).reset_index(drop=True)
    return df_summary, detailed

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    
    # 使用预设的候选股票列表
    candidates = CANDIDATE_STOCKS
    print(f"开始分析 {len(candidates)} 只候选股票: {candidates}")
    
    try:
        # 分析候选股票
        summary, details = analyze_candidates(candidates, start=HISTORY_START, end=HISTORY_END)
        
        # 输出结果
        print("\n" + "="*80)
        print("信号摘要：")
        print("="*80)
        
        # 只显示有买入信号的股票
        buy_signals = summary[summary['confirmed_signal'] == 'buy']
        if len(buy_signals) > 0:
            print(f"发现 {len(buy_signals)} 只买入信号股票：")
            display_columns = [
                'code', 'name', 'last_close', 'signal_score',
                'entry', 'stop', 'target', 'risk_reward_ratio',
                'position_percent', 'rsi',
                'next_buy_price', 'next_buy_low', 'next_buy_high'
            ]
            print(buy_signals[display_columns].to_string(index=False))
        else:
            print("当前未发现买入信号")
        
        # 额外输出：所有标的的下一次买入价参考
        print("\n各标的下一次买入价参考：")
        cols_next_buy = [
            'code', 'name', 'last_close',
            'market_trend', 'next_buy_mode',
            'next_buy_price', 'next_buy_low', 'next_buy_high'
        ]
        print(summary[cols_next_buy].to_string(index=False))
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"C:\\Users\\zzw\\Desktop\\py代码\\trade_signals_summary_{timestamp}.csv"
        summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n已保存信号摘要到: {summary_file}")
        
        # 保存每只股票的详细信号到 csv（可选）
        detail_dir = f"C:\\Users\\zzw\\Desktop\\py代码\\signal_details_{timestamp}"
        os.makedirs(detail_dir, exist_ok=True)
        for code, df in details.items():
            try:
                if code in [c for c in candidates if not pd.isna(summary[summary['code'] == c]['last_date'].iloc[0])]:
                    df.to_csv(f"{detail_dir}/signal_detail_{code}.csv", index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"保存详细数据 {code} 时出错: {str(e)}")
        print(f"已保存详细信号数据到目录: {detail_dir}")
        
        # 回测评估（可选）
        print("\n回测结果:")
        valid_codes = [code for code in details.keys() if len(details[code]) > 20]
        for code in valid_codes[:10]:  # 只回测前10个
            if code in buy_signals['code'].values:
                backtest_results = backtest_signal(details[code])
                if not backtest_results.empty:
                    win_rate = (backtest_results['return'] > 0).mean()
                    avg_return = backtest_results['return'].mean()
                    print(f"{code}: 胜率 {win_rate:.2%}, 平均收益 {avg_return:.2%}")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

    # ============= 手动输入当前价的即时分析（新功能） =============
    try:
        print("\n" + "="*80)
        print("手动输入当前价格，即时给出买入/卖出/持有建议（可选）")
        print("="*80)

        for code in candidates:
            if code not in details:
                continue

            user_input = input(f"\n如需评估 {code} 当前价格，请输入价格（直接回车跳过）: ").strip()
            if not user_input:
                continue

            try:
                current_price = float(user_input)
            except ValueError:
                print("输入的价格无效，跳过该标的。")
                continue

            # 基于手动输入价格进行重新评估
            try:
                eval_result = analyze_manual_price(details[code], current_price)
            except Exception as e:
                print(f"{code} 基于手动价格评估失败: {e}")
                continue

            rec = eval_result['confirmed_signal']
            rec_map = {
                'buy':  "建议买入",
                'sell': "建议卖出",
                'hold': "建议观望/持有"
            }

            print(f"\n{code} 基于当前价格 {current_price:.2f} 的即时评估结果：")
            print(f"操作建议：{rec_map.get(rec, rec)}")
            print(f"信号强度：{eval_result['signal_score']:.2f} | RSI：{eval_result['rsi']:.2f}")
            print(f"短均线(SMA{SMA_SHORT})：{eval_result['sma_short']:.2f} | 长均线(SMA{SMA_LONG})：{eval_result['sma_long']:.2f}")
            print(f"ATR：{eval_result['atr']:.4f}")
            if not np.isnan(eval_result['suggest_stop']):
                print(f"建议止损价：{eval_result['suggest_stop']:.2f}")
            if not np.isnan(eval_result['suggest_target']):
                print(f"建议目标价：{eval_result['suggest_target']:.2f} | 预期风险回报比：{eval_result['risk_reward_ratio']:.2f}")
            if not np.isnan(eval_result['position_percent']):
                print(f"建议仓位占比（基于当前价）：{eval_result['position_percent']*100:.2f}%")
            print(f"市场环境：{eval_result['market_trend']} | 波动：{eval_result['market_volatility']} | 说明：{eval_result['market_suggestion']}")

    except Exception as e:
        print(f"手动价格即时分析模块出错: {e}")
