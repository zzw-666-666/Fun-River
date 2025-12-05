"""
A股技术信号回测独立脚本（只回测 + 绘图，不导出表格文件）
使用方法：
1. 确保当前目录下有 ak_detail.py（你的策略逻辑文件）
   其中应包含：
      - fetch_daily_akshare(code, start, end)
      - generate_signals_for_df_enhanced(df)
      - RISK_PER_TRADE
      - ACCOUNT_SIZE
2. 修改最下面的 CODES / START / END，直接运行：
      python backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from ak_detail import (
    fetch_daily_akshare,
    generate_signals_for_df_enhanced,
    RISK_PER_TRADE,
    ACCOUNT_SIZE,
)


def backtest_trading(
    df: pd.DataFrame,
    initial_capital: float = ACCOUNT_SIZE,
    risk_per_trade: float = RISK_PER_TRADE,
    fee_rate: float = 0.0005,
    slippage: float = 0.0005,
) -> dict:
    """
    基于 generate_signals_for_df_enhanced 输出的 df，做单标的回测。
    需要 df 至少包含：
      - date, open, high, low, close
      - confirmed_signal（'buy'/'sell'/'hold'）
      - suggest_stop, suggest_target（可选，没有时会自动兜底）
    """
    df = df.copy().reset_index(drop=True)
    if "date" not in df.columns:
        raise ValueError("df 中必须包含 'date' 列")

    cash = initial_capital
    position = 0
    entry_price = np.nan
    entry_commission = 0.0
    stop_price = np.nan
    target_price = np.nan
    entry_index = None

    equity_curve = []
    trades = []

    max_equity = initial_capital
    equity = initial_capital

    for i, row in df.iterrows():
        date = row["date"]
        close_price = float(row["close"])
        high_price = float(row["high"])
        low_price = float(row["low"])

        # ===== 1）先检查是否需要平仓 =====
        if position > 0:
            exit_reason = None
            exit_price = None

            hit_stop = not np.isnan(stop_price) and (low_price <= stop_price)
            hit_target = not np.isnan(target_price) and (high_price >= target_price)
            signal_sell = row.get("confirmed_signal", "hold") == "sell"

            if hit_stop:
                exit_price = stop_price * (1 - slippage)
                exit_reason = "stop"
            elif hit_target:
                exit_price = target_price * (1 - slippage)
                exit_reason = "target"
            elif signal_sell:
                exit_price = close_price * (1 - slippage)
                exit_reason = "signal"

            if exit_reason is not None:
                gross = position * exit_price
                commission = gross * fee_rate
                cash += gross - commission

                pnl = (exit_price - entry_price) * position - entry_commission - commission
                ret_pct = pnl / (entry_price * position) if position > 0 else 0.0

                trades.append(
                    {
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_date": df.loc[entry_index, "date"],
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": position,
                        "pnl": pnl,
                        "return_pct": ret_pct,
                        "reason": exit_reason,
                        "hold_days": i - entry_index,
                    }
                )

                position = 0
                entry_price = np.nan
                entry_commission = 0.0
                stop_price = np.nan
                target_price = np.nan
                entry_index = None

        # ===== 2）再考虑是否开新仓 =====
        if position == 0:
            cs = row.get("confirmed_signal", row.get("signal", "hold"))
            if cs == "buy":
                entry_price_today = close_price * (1 + slippage)

                stop_today = row.get("suggest_stop", np.nan)
                target_today = row.get("suggest_target", np.nan)

                # 没给止损/目标就兜底
                if np.isnan(stop_today) or stop_today <= 0 or stop_today >= entry_price_today:
                    stop_today = entry_price_today * 0.95
                if np.isnan(target_today) or target_today <= entry_price_today:
                    target_today = entry_price_today * 1.06

                risk_per_share = entry_price_today - stop_today
                if risk_per_share <= 0:
                    risk_per_share = entry_price_today * 0.02  # 再兜底

                equity = cash  # 此时无持仓
                max_risk_amount = equity * risk_per_trade

                shares_by_risk = int(max_risk_amount / risk_per_share)
                max_shares_by_cash = int(cash / (entry_price_today * (1 + fee_rate)))
                shares = max(0, min(shares_by_risk, max_shares_by_cash))

                if shares > 0:
                    cost = shares * entry_price_today
                    commission = cost * fee_rate

                    cash -= cost + commission
                    position = shares
                    entry_price = entry_price_today
                    entry_commission = commission
                    stop_price = stop_today
                    target_price = target_today
                    entry_index = i

        # ===== 3）记录每日账户情况 =====
        equity = cash + position * close_price
        max_equity = max(max_equity, equity)
        max_drawdown = (equity - max_equity) / max_equity if max_equity > 0 else 0.0

        equity_curve.append(
            {
                "date": date,
                "close": close_price,
                "cash": cash,
                "position_shares": position,
                "position_value": position * close_price,
                "equity": equity,
                "max_drawdown": max_drawdown,
            }
        )

    # ===== 4）回测结束时如果还有仓位，全部平掉 =====
    if position > 0:
        last_row = df.iloc[-1]
        exit_price = float(last_row["close"]) * (1 - slippage)
        gross = position * exit_price
        commission = gross * fee_rate
        cash += gross - commission

        pnl = (exit_price - entry_price) * position - entry_commission - commission
        ret_pct = pnl / (entry_price * position) if position > 0 else 0.0
        last_index = len(df) - 1

        trades.append(
            {
                "entry_index": entry_index,
                "exit_index": last_index,
                "entry_date": df.loc[entry_index, "date"],
                "exit_date": last_row["date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": position,
                "pnl": pnl,
                "return_pct": ret_pct,
                "reason": "end_of_backtest",
                "hold_days": last_index - entry_index,
            }
        )

        equity = cash
        max_equity = max(max_equity, equity)
        max_drawdown = (equity - max_equity) / max_equity if max_equity > 0 else 0.0
        equity_curve[-1].update(
            {
                "cash": cash,
                "position_shares": 0,
                "position_value": 0.0,
                "equity": equity,
                "max_drawdown": max_drawdown,
            }
        )

    equity_curve_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    final_equity = equity_curve_df["equity"].iloc[-1]
    total_return = final_equity / initial_capital - 1.0

    if not trades_df.empty:
        win_rate = (trades_df["pnl"] > 0).mean()
        avg_trade_return = trades_df["return_pct"].mean()
        max_dd = equity_curve_df["max_drawdown"].min()
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        max_dd = 0.0

    stats = {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return * 100,
        "num_trades": int(len(trades_df)),
        "win_rate_pct": win_rate * 100,
        "avg_trade_return_pct": avg_trade_return * 100,
        "max_drawdown_pct": max_dd * 100,
    }

    return {
        "equity_curve": equity_curve_df,
        "trades": trades_df,
        "stats": stats,
    }


def plot_backtest(code: str, df_ind: pd.DataFrame, bt_result: dict) -> None:
    """
    绘制：
      - 上图：价格 + 实际交易的买入/卖出点
      - 下图：账户净值曲线
    """
    df = df_ind.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    ec = bt_result["equity_curve"].copy()
    ec["date"] = pd.to_datetime(ec["date"])

    trades_df = bt_result["trades"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # 价格曲线
    ax1.plot(df["date"], df["close"], label="收盘价")

    # 标出实际开平仓点（按回测产生的交易）
    if not trades_df.empty:
        entry_idx = trades_df["entry_index"].values
        exit_idx = trades_df["exit_index"].values

        entries = df.loc[entry_idx]
        exits = df.loc[exit_idx]

        ax1.scatter(entries["date"], entries["close"], marker="^", color="g", label="买入", zorder=3)
        ax1.scatter(exits["date"], exits["close"], marker="v", color="r", label="卖出", zorder=3)

    ax1.set_title(f"{code} 价格与交易点")
    ax1.set_ylabel("价格")
    ax1.legend(loc="best")

    # 权益曲线
    ax2.plot(ec["date"], ec["equity"], label="账户净值")
    ax2.set_title("账户权益曲线")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("净值")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def backtest_codes(
    codes,
    start="20240101",
    end=None,
    initial_capital: float = ACCOUNT_SIZE,
    risk_per_trade: float = RISK_PER_TRADE,
) -> None:
    """
    对一组股票代码做逐只回测，并打印结果 + 绘图。
    不生成任何文件。
    """
    for code in codes:
        print(f"\n====== 回测 {code} ({start} ~ {end or '最新'}) ======")
        try:
            df_raw = fetch_daily_akshare(code, start=start, end=end)
            if df_raw is None or df_raw.empty:
                print("  无数据，跳过。")
                continue

            # 加指标 & 信号（使用你策略里的函数）
            df_ind = generate_signals_for_df_enhanced(df_raw)

            # 回测
            bt = backtest_trading(
                df_ind,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
            )

            # 打印简单指标（在终端看一眼就行）
            print("  回测指标：")
            for k, v in bt["stats"].items():
                if k.endswith("_pct"):
                    print(f"    {k}: {v:.2f}%")
                else:
                    print(f"    {k}: {v}")

            # 绘图（价格 + 信号 + 净值曲线）
            plot_backtest(code, df_ind, bt)

        except Exception as e:
            print(f"  回测 {code} 出错：{e}")


if __name__ == "__main__":
    # 在这里配置你要回测的股票列表和时间区间
    CODES = ["600595"]   # 想测哪些股票就填哪些
    START = "20240101"
    END = None           # None 表示到最近数据

    backtest_codes(
        CODES,
        start=START,
        end=END,
        initial_capital=ACCOUNT_SIZE,
        risk_per_trade=RISK_PER_TRADE,
    )
