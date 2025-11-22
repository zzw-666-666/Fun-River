import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path  = "附件二优化后波峰波谷.xlsx"
sheet_p = "波峰"
sheet_v = "波谷"
out = "色散优化"

thta = 15.0   # 入射角
A0, B0, C0, d0 = 2.6, 5e-9, 0.0, 5e-4

lb = [1.01, -np.inf, -np.inf, 1e-6]
ub = [4.00,  np.inf,  np.inf, 1e-2]

def pick_clo(df: pd.DataFrame) -> np.ndarray:
    c_d = [c for c in df.columns if "波数" in str(c)]
    col = c_d[0] if c_d else df.columns[0]
    sig = pd.to_numeric(df[col], errors="coerce").dropna().values
    sig = np.unique(sig)
    sig.sort()
    return sig

def make_p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.empty((0,2))
    x = np.unique(x)
    x.sort()
    return np.column_stack([x[:-1], x[1:]])

def n_kexi(sig, A, B, C):
    return A + B*sig**2 + C*sig**4

def cos_t(sig, A, B, C, sin_t):
    n = n_kexi(sig, A, B, C)
    arg = 1.0 - (sin_t / np.maximum(n, 1e-9))**2
    return np.sqrt(np.clip(arg, 0.0, 1.0))

def re_p(pairs, p, sin_t):
    A, B, C, d = p
    s0 = pairs[:,0]; s1 = pairs[:,1]
    t1 = n_kexi(s1, A,B,C) * s1 * cos_t(s1, A,B,C, sin_t)
    t0 = n_kexi(s0, A,B,C) * s0 * cos_t(s0, A,B,C, sin_t)
    return 1.0 - 2.0 * d * (t1 - t0)

def fit_d(pairs, thta, p0=None, bounds=None):
    sin_t = np.sin(np.deg2rad(thta))
    if p0 is None: 
        p0 = np.array([A0, B0, C0, d0], dtype=float)
    kw = {}
    if bounds is not None:
        kw["bounds"] = bounds
    res = least_squares(lambda p: re_p(pairs, p, sin_t),
                        p0, loss='soft_l1', f_scale=1.0, **kw)
    return res.x, True, "OK"

def local_p(pairs, A,B,C, thta):
    sin_t = np.sin(np.deg2rad(thta))
    s0 = pairs[:,0]; s1 = pairs[:,1]
    t1 = n_kexi(s1, A,B,C) * s1 * cos_t(s1, A,B,C, sin_t)
    t0 = n_kexi(s0, A,B,C) * s0 * cos_t(s0, A,B,C, sin_t)
    denom = 2.0 * (t1 - t0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_loc = 1.0 / denom
    return d_loc[np.isfinite(d_loc)]

def main():
    xls = pd.ExcelFile(file_path)
    df_peak   = pd.read_excel(xls, sheet_name=sheet_p)
    df_valley = pd.read_excel(xls, sheet_name=sheet_v)

    pk_nu = pick_clo(df_peak)
    vl_nu = pick_clo(df_valley)

    pr_p   = make_p(pk_nu)
    pr_v = make_p(vl_nu)

    rows = []
    fit = {}
    for tag, pairs in [("仅峰", pr_p), ("仅谷", pr_v)]:
        params, ok, msg = fit_d(pairs, thta, p0=np.array([A0,B0,C0,d0]), bounds=(lb,ub))
        A,B,C,d = params
        fit[tag] = (A,B,C,d)
        d_um = d * 1e4
        d_loc = local_p(pairs, A,B,C, thta)
        std_um  = np.nanstd(d_loc) * 1e4 if d_loc.size else np.nan

        print(f"{tag}：d = {d_um:.4f} μm,  A={A:.6f}, B={B:.3e}, C={C:.3e}, 条纹对数={len(pairs)}")
        if d_loc.size:
            print(f"局部厚度标准差 = {std_um:.4f} μm")
        rows.append({
            "拟合类型": tag,
            "条纹对数": len(pairs),
            "厚度 d (μm)": d_um,
            "A": A, "B": B, "C": C,
            "局部厚度标准差 (μm)": std_um
        })

        out_xlsx = f"{out}_结果.xlsx"
        pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    print(f"\n已导出色散优化结果：{os.path.abspath(out_xlsx)}")

    tag_to_plot = "峰+谷" if "峰+谷" in fit else (list(fit.keys())[0] if fit else None)

    A,B,C,d = fit[tag_to_plot]
    sig_all = np.unique(np.concatenate([pk_nu, vl_nu])) if pk_nu.size and vl_nu.size \
              else (pk_nu if pk_nu.size else vl_nu)

    s_min, s_max = sig_all.min(), sig_all.max()
    s_lin = np.linspace(s_min, s_max, 500)
    n_lin = n_kexi(s_lin, A,B,C)

    n_pk = n_kexi(pk_nu, A,B,C) if pk_nu.size else np.array([])
    n_vl = n_kexi(vl_nu, A,B,C) if vl_nu.size else np.array([])

    plt.figure(figsize=(10,6))
    plt.plot(s_lin, n_lin, lw=1.8, label=f'{tag_to_plot} 拟合的 n(v)')
    if pk_nu.size:
        plt.scatter(pk_nu, n_pk, s=18, marker='o', label='峰位 n(v)', alpha=0.8)
    if vl_nu.size:
        plt.scatter(vl_nu, n_vl, s=18, marker='s', label='谷位 n(v)', alpha=0.8)
    plt.xlabel('波数 v (cm$^{-1}$)')
    plt.ylabel('折射率 n(v)')
    plt.title(f'折射率色散曲线（{tag_to_plot}）')
    plt.grid(True, alpha=0.45, linestyle='--')
    plt.legend()
    plt.tight_layout()

    out_png = f"{out}_折射率曲线_{tag_to_plot}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"已保存折射率曲线：{os.path.abspath(out_png)}")

if __name__ == "__main__":
    main()