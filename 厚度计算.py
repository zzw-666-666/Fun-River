import pandas as pd
import numpy as np
data = "附件一原始波峰波谷分析结果.xlsx"

data1 = pd.read_excel(data, sheet_name='波峰')
print("波峰数据:")
print(data1.head())
data2 = pd.read_excel(data, sheet_name='波谷')
print("\n波谷数据:")
print(data2.head())

def calculate(num, n1, thta):
    thta = np.deg2rad(thta)
    sin_thta = np.sin(thta)
    
    delta_v = np.diff(num)
    avg= np.mean(delta_v)
    
    fm = 2 * np.sqrt(n1**2 - sin_thta**2) * avg
    d = 1 / fm
    return d

p_wave = data1['波数 (cm-1)'].values
v_wave = data2['波数 (cm-1)'].values

n1 = 2.6  # 折射率
thta = 10  # 入射角度

d_peaks = calculate(p_wave, n1, thta)
print(f"\n使用波峰数据计算出的薄膜厚度: {d_peaks*1e4:.4f} μm")
d_valleys = calculate(v_wave, n1, thta)
print(f"\n使用波谷数据计算出的薄膜厚度: {d_valleys*1e4:.4f} μm")