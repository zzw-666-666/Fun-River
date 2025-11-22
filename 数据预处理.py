import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data  = "附件2.xlsx"
sheet_name = 0
out_name = "波峰波谷"

mwf = 0.06        # 基线窗口占比
sw = 9            # SG窗长
sp = 3            # SG阶数
prominence = 1.0    # 附件二的阈值
#prominence = 1.5    # 附件一的阈值

def roll(y: np.ndarray, frac: float) -> np.ndarray:
    n = len(y)
    win = max(5, int(np.floor(frac * n)))
    if win % 2 == 0:
        win += 1
    s = pd.Series(y)
    b = s.rolling(win, center=True).median().bfill().ffill()
    return b.to_numpy()

def main():
    df = pd.read_excel(data, sheet_name=sheet_name)
    col_k = next((c for c in df.columns if "波数" in str(c)), df.columns[0])
    col_R = next((c for c in df.columns if "反射率" in str(c)), df.columns[1])

    nu = df[col_k].to_numpy(dtype=float)
    R  = df[col_R].to_numpy(dtype=float)

    good = (R > 0.0) & (R < 100.0)
    nu, R = nu[good], R[good]

    base = roll(R, mwf)
    det  = R - base

    n = len(det)
    win = sw if (sw % 2 == 1) else (sw + 1)
    min_win = max(5, (sp + 2) if ((sp + 2) % 2 == 1) else (sp + 3))
    win = max(win, min_win)
    win = min(win, n - 1 if ((n - 1) % 2 == 1) else (n - 2))
    sm = savgol_filter(det, window_length=win, polyorder=sp, mode="interp")


    pk_idx, _  = find_peaks(sm, prominence=prominence)
    vl_idx, _  = find_peaks(-sm, prominence=prominence)

    pk_nu, pk_R = nu[pk_idx], R[pk_idx]
    vl_nu, vl_R = nu[vl_idx], R[vl_idx]

    raw = pd.DataFrame({"波数 (cm-1)": nu, "反射率 (%)": R})
    proc = pd.DataFrame({
        "波数 (cm-1)": nu,
        "基线 (%)": base,
        "去趋势后反射率 (%)": det,
        "去趋势+平滑 (%)": sm
    })
    
    df_pk = pd.DataFrame({
        "波数 (cm-1)": pk_nu, 
        "反射率 (%)": pk_R
    })
    
    df_vl = pd.DataFrame({
        "波数 (cm-1)": vl_nu, 
        "反射率 (%)": vl_R
    })

    xls = f"{out_name}.xlsx"
    with pd.ExcelWriter(xls, engine="xlsxwriter") as w:
        raw.to_excel(w, sheet_name="raw", index=False)
        proc.to_excel(w, sheet_name="processed", index=False)
        df_pk.to_excel(w, sheet_name="波峰", index=False)
        df_vl.to_excel(w, sheet_name="波谷", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(nu, sm,  'b-', lw=1.2, label='去趋势+平滑(%)')
    
    if len(pk_idx):
        plt.plot(pk_nu, sm[pk_idx], 'ro', ms=5, label='波峰(处理后)')
        for x, y in zip(pk_nu, sm[pk_idx]):
            plt.annotate(f'{x:.1f}', (x, y), xytext=(0, 10), 
                         textcoords='offset points', ha='center', fontsize=8)
    
    if len(vl_idx):
        plt.plot(vl_nu, sm[vl_idx], 'go', ms=5, label='波谷(处理后)')
        for x, y in zip(vl_nu, sm[vl_idx]):
            plt.annotate(f'{x:.1f}', (x, y), xytext=(0, -15), 
                         textcoords='offset points', ha='center', fontsize=8)
    
    plt.xlabel('波数 (cm$^{-1}$)')
    plt.ylabel('反射率 (%)')
    plt.title('去趋势+平滑后的光谱及峰谷')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_name}_图1_去趋势平滑_峰谷.png"
    plt.savefig(p1, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(nu, R, 'b-', lw=1, alpha=0.8, label='原始光谱')
    plt.plot(nu, base, 'k--', lw=1, label='基线')

    if len(pk_idx):
        plt.plot(pk_nu, pk_R, 'ro', ms=5, label='波峰(原始)')
        for x, y in zip(pk_nu, pk_R):
            plt.annotate(f'{x:.1f}', (x, y), xytext=(0, 10), 
                         textcoords='offset points', ha='center', fontsize=8)
    if len(vl_idx):
        plt.plot(vl_nu, vl_R, 'go', ms=5, label='波谷(原始)')
        for x, y in zip(vl_nu, vl_R):
            plt.annotate(f'{x:.1f}', (x, y), xytext=(0, -15), 
                         textcoords='offset points', ha='center', fontsize=8)
    
    plt.xlabel('波数 (cm$^{-1}$)')
    plt.ylabel('反射率 (%)')
    plt.title('原始光谱上的峰谷与基线')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    p2 = f"{out_name}_图2_原始光谱峰谷_含基线.png"
    plt.savefig(p2, dpi=300)
    plt.close()

    print(f"找到 {len(pk_idx)} 个波峰 和 {len(vl_idx)} 个波谷")
    print(f"Excel：{os.path.abspath(xls)}")
    print(f"图像：{os.path.abspath(p1)}")
    print(f"图像：{os.path.abspath(p2)}")

if __name__ == "__main__":
    main()