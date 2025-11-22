import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def read_data(file):
    data = pd.read_excel(file)
    wave = data.iloc[:, 0].values.astype(float)
    ref = data.iloc[:, 1].values.astype(float)
    mask = np.isfinite(wave) & np.isfinite(ref)
    return wave[mask], ref[mask]

def s_ac(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))

def r_s(n_i, n_j, thta_i, thta_j):
    ci, cj = np.cos(thta_i), np.cos(thta_j)
    return (n_i*ci - n_j*cj) / (n_i*ci + n_j*cj)

def r_p(n_i, n_j, thta_i, thta_j):
    ci, cj = np.cos(thta_i), np.cos(thta_j)
    return (n_j*ci - n_i*cj) / (n_j*ci + n_i*cj)

def thin_ref(wave_cm1, d_um, n0, n1, n2, thta_rad):

    sigma = np.asarray(wave_cm1, dtype=float)
    thta1 = s_ac(n0*np.sin(thta_rad)/n1)
    thta2 = s_ac(n1*np.sin(thta1)/n2)

    beta = 2.0*np.pi * n1 * d_um * np.cos(thta1) * sigma / 1.0e4
    expi = np.exp(2j*beta)

    s01 = r_s(n0, n1, thta_rad, thta1)
    s12 = r_s(n1, n2, thta1, thta2)
    r_s = (s01 + s12*expi) / (1.0 + s01*s12*expi)
    Rs = np.abs(r_s)**2

    p01 = r_p(n0, n1, thta_rad, thta1)
    p12 = r_p(n1, n2, thta1, thta2)
    r_p = (p01 + p12*expi) / (1.0 + p01*p12*expi)
    Rp = np.abs(r_p)**2

    R = 0.5*(Rs + Rp)
    return R * 100.0

def bean_2(wave_cm1, d_um, n0, n1, n2, thta_rad):
    sigma = np.asarray(wave_cm1, dtype=float)
    thta1 = s_ac(n0*np.sin(thta_rad)/n1)
    thta2 = s_ac(n1*np.sin(thta1)/n2)
    def R_unpol(n_i, n_j, thta_i, thta_j):
        rs = r_s(n_i, n_j, thta_i, thta_j)
        rp = r_p(n_i, n_j, thta_i, thta_j)
        return 0.5*(np.abs(rs)**2 + np.abs(rp)**2)

    R01 = R_unpol(n0, n1, thta_rad, thta1)
    R12 = R_unpol(n1, n2, thta1, thta2)

    beta = 2.0*np.pi * n1 * d_um * np.cos(thta1) * sigma / 1.0e4
    R = R01 + R12 + 2.0*np.sqrt(R01*R12)*np.cos(2.0*beta)
    return np.clip(R, 0, 1)*100.0

def fx(wave, ref, prominence=1.0):
    peaks, _ = find_peaks(ref, prominence=prominence)
    valleys, _ = find_peaks(-ref, prominence=prominence)

    if len(peaks) > 0 and len(valleys) > 0:
        mean_peak = float(np.mean(ref[peaks]))
        mean_valley = float(np.mean(ref[valleys]))
        contrast = mean_peak - mean_valley
    else:
        contrast = 0.0
    return contrast, peaks, valleys

def em(wave, ref, n_layer, thta_rad, prominence=1.0):
    contrast, peaks, valleys = fx(
        wave, ref, prominence=prominence
    )
    use_idx = peaks if len(peaks) >= 2 else valleys
    print(len(peaks))
    print(len(valleys))
    if len(use_idx) >= 2:
        delta_sigma = float(np.mean(np.diff(wave[use_idx])))
        thta_t = s_ac(np.sin(thta_rad)/n_layer)
        d_um = 1.0e4 / (2.0 * n_layer * delta_sigma * np.cos(thta_t))
        return d_um, contrast, peaks, valleys
    else:
        return None, contrast, peaks, valleys

def et(
    wave, ref, thta_rad, n_epi, n_sub,
    method='multi_beam', prominence=1.0, d_bounds=(1e-3, 300.0)
):

    d0, contrast, peaks, valleys = em(
        wave, ref, n_epi, thta_rad, prominence=prominence
    )
    if method == '.':
        def fit_func(w, d_um):
            return thin_ref(w, d_um, 1.0, n_epi, n_sub, thta_rad)
        method_used = 'multi_beam'
    else:
        def fit_func(w, d_um):
            return bean_2(w, d_um, 1.0, n_epi, n_sub, thta_rad)
        method_used = 'two_beam'

    ydata = ref
    p0 = [max(d_bounds[0], min(d_bounds[1], float(d0)))]
    bounds = ([d_bounds[0]], [d_bounds[1]])
    popt, pcov = curve_fit(fit_func, wave, ydata, p0=p0, bounds=bounds, maxfev=20000)
    d_fit = float(popt[0])
    d_err = float(np.sqrt(np.diag(pcov))[0]) if (pcov is not None and np.all(np.isfinite(pcov))) else 0.0
    return d_fit, d_err, contrast, len(peaks), len(valleys), method_used

def main():
    n_sic_epi, n_sic_sub = 2.6, 2.6
    n_si_epi,  n_si_sub  = 3.4, 3.4

    mbt = 15.0  # 峰谷对比度
    prominence = 1.0  # 峰谷显著性
    files = {
        'SiC_10°': ("附件1.xlsx", np.deg2rad(10.0), n_sic_epi, n_sic_sub),
        'SiC_15°': ("附件2.xlsx", np.deg2rad(15.0), n_sic_epi, n_sic_sub),
        'Si_10°':  ("附件3.xlsx", np.deg2rad(10.0), n_si_epi,  n_si_sub ),
        'Si_15°':  ("附件4.xlsx", np.deg2rad(15.0), n_si_epi,  n_si_sub ),
    }
    results = {}

    for name, (file, theta, n_epi, n_sub) in files.items():
        wave, ref = read_data(file)
        contrast, peaks, valleys = fx(
            wave, ref, prominence=prominence
        )
        has_n = (contrast > mbt)

        method = 'multi_beam' if has_n else 'two_beam'
        d_fit, d_err, contrast2, n_peaks, n_valleys, method_used = et(
            wave, ref, theta, n_epi, n_sub,
            method=method, prominence=prominence, d_bounds=(1e-3, 300.0)
        )

        results[name] = dict(
            wave=wave,
            ref=ref,
            theta=theta,
            n_epi=n_epi,
            n_sub=n_sub,
            has_n_beam=has_n,
            contrast=contrast2,
            n_peaks=n_peaks,
            n_valleys=n_valleys,
            thickness=d_fit,
            error=d_err,
            method=method_used
        )

    for name, r in results.items():
        print(f"{name}:")
        print(f"  多光束干涉: {'是' if r['has_n_beam'] else '否'}")
        print(f"  对比度: {r['contrast']:.2f}%  (peaks={r['n_peaks']}, valleys={r['n_valleys']})")
        print(f"  厚度: {r['thickness']:.3f} ± {r['error']:.3f} μm")
        print(f"  使用模型: {r['method']}")
        print()

    print("生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (name, r) in zip(axes, results.items()):
        w, R = r['wave'], r['ref']
        theta = r['theta']
        n_epi, n_sub = r['n_epi'], r['n_sub']

        ax.plot(w, R, '-', lw=1.2, label='原始数据')

        peaks, _ = find_peaks(R, prominence=prominence)
        valleys, _ = find_peaks(-R, prominence=prominence)
        ax.plot(w[peaks],  R[peaks],  'ro', ms=3, label='峰')
        ax.plot(w[valleys], R[valleys], 'go', ms=3, label='谷')

        if r['method'] == 'multi_beam':
            fit = thin_ref(w, r['thickness'], 1.0, n_epi, n_sub, theta)
            fit_label = '多光束(转移矩阵) 拟合'
            fit_color = 'orange'
        else:
            fit = bean_2(w, r['thickness'], 1.0, n_epi, n_sub, theta)
            fit_label = '两束近似 拟合'
            fit_color = 'purple'

        ax.plot(w, fit, color=fit_color, lw=2.0, label=fit_label)
        ax.set_xlabel('波数 ')
        ax.set_ylabel('反射率 (%)')
        ax.set_title(f'{name} 对比度: {r["contrast"]:.2f}% | 多光束: {"是" if r["has_n_beam"] else "否"}')
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('厚度分析结果.png', dpi=300, bbox_inches='tight')

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(results.keys())
    contrasts = [results[k]['contrast'] for k in names]
    colors = ['red' if results[k]['has_n_beam'] else 'blue' for k in names]

    bars = ax1.bar(names, contrasts, color=colors, alpha=0.75)
    ax1.axhline(y=mbt, color='r', ls='--', label='多光束阈值')
    ax1.set_ylabel('对比度 (%)')
    ax1.set_title('干涉对比度')
    ax1.legend()
    for b, c in zip(bars, contrasts):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height(), f'{c:.1f}', ha='center', va='bottom')

    thicks = [results[k]['thickness'] for k in names]
    errs   = [results[k]['error'] for k in names]
    bars2 = ax2.bar(names, thicks, yerr=errs, capsize=5, color=colors, alpha=0.75, ecolor='black')
    ax2.set_ylabel('厚度 (μm)')
    ax2.set_title('厚度拟合结果')
    for b, d in zip(bars2, thicks):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height(), f'{d:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('对比度和厚度比较.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
