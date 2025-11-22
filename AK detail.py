"""
A股技术信号生成器（买入/卖出点）增强版
依赖：akshare, pandas, numpy, tqdm
功能：
 - 直接在代码中指定候选股票列表
 - 拉取日线历史数据（akshare）
 - 计算 SMA(short/long), RSI(14), ATR(14)
 - 生成买入/卖出信号，推荐止损/目标价与仓位建议
 - 包含信号确认、动态止损、市场环境判断和智能仓位管理
"""
import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from typing import List, Tuple, Dict
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
RISK_PER_TRADE = 0.05        # 每笔风险占本金比例，比如 2%
ACCOUNT_SIZE = 20000       # 假设账户大小（用于仓位建议），可按实际调整

# ----------------- 候选股票列表 -----------------
# 在这里直接指定要分析的股票代码
CANDIDATE_STOCKS = [
    "002957", 
    "600595", 
    "000818", 
]

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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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
    df.loc[(df['signal'] == 'buy') & (df['signal_score'] >= 2), 'confirmed_signal'] = 'buy'
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

    # 信号判定（基于最新一行）
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

        # 卖出：出现死叉 或 RSI > 70（超买） 或 跌破止损（由持仓时判定，下面提供止损价）
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

    # 从账户规模与风险预算估算仓位（基于 risk per trade 与 ATR 计算止损距离）
    for i in range(len(df)):
        if df.at[i, 'signal'] == 'buy':
            entry = df.at[i, 'close']
            # 止损设为 entry - 2 * ATR（可调整）
            stop = entry - 2.0 * df.at[i, 'atr'] if not np.isnan(df.at[i, 'atr']) else entry * 0.97
            if stop <= 0: stop = entry * 0.97
            # 目标： entry + 4 * ATR（风险回报比 1:2）
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
    akshare 接口对代码格式可能需要 'sh.600000' 或 '600000'，不同版本行为不同。
    尝试几种常见写法（'sh' 和 'sz' 前缀），并返回 DataFrame（按时间升序）
    """
    # 规范 code（若传入带 .SH/.SZ）
    code_str = str(code).strip()
    # try original
    tries = [code_str]
    # if only numeric, add prefixes
    if code_str.isdigit() and len(code_str) == 6:
        tries += [f"sh{code_str}", f"sz{code_str}", f"sh.{code_str}", f"sz.{code_str}"]
    # deduplicate
    tries = list(dict.fromkeys(tries))
    
    for t in tries:
        for r in range(MAX_RETRIES):
            try:
                df = ak.stock_zh_a_daily(symbol=t, start_date=start, end_date=end)
                if df is None or df.empty:
                    time.sleep(SLEEP_BETWEEN)
                    continue
                
                # akshare 通常返回按日期降序，列名中文（'日期','开盘','收盘'等）
                # convert colnames to English lower
                # ensure ascending by date
                if '日期' in df.columns:
                    df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
                # ensure asc
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
    
    # 若均失败，抛出异常
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
                'market_suggestion': last.get('market_suggestion', 'unknown')
            })
            
        except Exception as e:
            print(f"分析股票 {code} 时出错: {str(e)}")
            # 记录失败并继续（不要中断批量分析）
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
                'market_suggestion': '数据获取失败'
            })
            continue

    df_summary = pd.DataFrame(summary_rows)
    # 保存 summary - 按确认信号和仓位比例排序
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
            display_columns = ['code', 'name', 'last_close', 'signal_score', 'entry', 'stop', 'target', 
                             'risk_reward_ratio', 'position_percent', 'rsi']
            print(buy_signals[display_columns].to_string(index=False))
        else:
            print("当前未发现买入信号")
        
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