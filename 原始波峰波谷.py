import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("附件1.xlsx")

wave = df['波数 (cm-1)'].values
fsl = df['反射率 (%)'].values
prominence=1.5  #附件一取1.5，附件二取1.0

fs, _ = find_peaks(fsl, prominence=prominence) 
peak_wave = wave[fs]
peak_fsl = fsl[fs]

gs, _ = find_peaks(-fsl, prominence=prominence)
g_wave = wave[gs]
g_fsl = fsl[gs]

p_df = pd.DataFrame({
    '波数 (cm-1)': peak_wave,
    '反射率 (%)': peak_fsl
})

v_df = pd.DataFrame({
    '波数 (cm-1)': g_wave,
    '反射率 (%)': g_fsl
})

with pd.ExcelWriter('波峰波谷分析结果.xlsx') as writer:
    p_df.to_excel(writer, sheet_name='波峰', index=False)
    v_df.to_excel(writer, sheet_name='波谷', index=False)

print(f"找到 {len(fs)} 个波峰和 {len(gs)} 个波谷")
print("结果已保存到 '波峰波谷分析结果.xlsx'")

plt.figure(figsize=(12, 6))

plt.plot(wave, fsl, 'b-', label='原始光谱', linewidth=1, alpha=0.7)

plt.plot(peak_wave, peak_fsl, 'ro', markersize=5, label='波峰')

plt.plot(g_wave, g_fsl, 'go', markersize=5, label='波谷')

plt.xlabel('波数')
plt.ylabel('反射率 (%)')
plt.title('光谱数据波峰波谷分析')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('光谱波峰波谷分析图.png', dpi=300)
plt.show()
print("图表已保存为 '光谱波峰波谷分析图.png'")