# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
# import statsmodels.formula.api as smf  # 备选：公式法
from matplotlib.font_manager import FontProperties
import warnings

# ========= 基本设置 =========
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ========= 读取数据 =========
file_path = r"C:\Users\zzw\Desktop\交通治理调查问卷数据61.xlsx"
df = pd.read_excel(file_path)

# ========= 标签映射 =========
gender_labels = {1: '男', 2: '女'}
age_labels = {1: '不到18岁', 2: '18-30岁', 3: '31-40岁', 4: '41-50岁', 5: '51-59岁', 6: '60岁以上'}
edu_labels = {1: '初中以下', 2: '初中', 3: '高中/中专', 4: '大专', 5: '本科', 6: '研究生及以上'}
job_labels = {1: '企业工作人员', 2: '机关事业单位', 3: '科教文体卫', 4: '个体及私营业主',
              5: '离退休人员', 6: '在校学生', 7: '自由职业者', 8: '无业人员', 9: '务农人员', 10: '其他人员'}
income_labels = {1: '5万以下', 2: '5-10万', 3: '10-30万', 4: '30万以上'}
residence_labels = {1: '常住', 2: '暂住', 3: '出差/旅游'}
car_labels = {0: '0辆', 1: '1辆', 2: '2辆', 3: '3辆及以上'}

transport_labels = {
    1: '步行', 2: '公交', 3: '自行车(小红车)', 4: '电瓶车',
    5: '出租车/网约车', 6: '私家车(含搭车)', 7: '其他方式', 8: '未曾坐过地铁'
}

# ========= 应用标签映射（保留原编码列，以免建模丢失数值） =========
df['gender'] = df['Q19'].map(gender_labels)
df['age'] = df['Q20'].map(age_labels)
df['education'] = df['Q21'].map(edu_labels)
df['occupation'] = df['Q22'].map(job_labels)
df['income'] = df['Q23'].map(income_labels)
df['residence'] = df['Q24'].map(residence_labels)
df['cars'] = df['Q18'].map(car_labels)
df['transfer_mode'] = df['Q6'].map(transport_labels)

# ========= 1) 卡方检验 =========
print("="*60)
print("卡方检验：分类变量间的独立性")
print("="*60)

def safe_crosstab(a, b):
    # 丢 NA，避免 chi2 报错
    tmp = pd.DataFrame({'a': a, 'b': b}).dropna()
    if tmp.empty:
        return None
    ct = pd.crosstab(tmp['a'], tmp['b'])
    if (ct.values.sum() == 0) or (ct.shape[0] < 2) or (ct.shape[1] < 2):
        return None
    return ct

# 示例1: 性别 与 M线知晓(Q3)
ct1 = safe_crosstab(df['gender'], df['Q3'])
if ct1 is not None:
    chi2, p, dof, expected = chi2_contingency(ct1)
    print("性别与M线知晓情况的卡方检验:")
    print(f"卡方值: {chi2:.3f}, p值: {p:.3f}")
    print("原假设：性别与M线知晓情况独立")
    print("结论：", "拒绝原假设（存在关联）" if p < 0.05 else "不能拒绝原假设（独立）")
else:
    print("性别×M线知晓 交叉表无效（可能是全空或类别不足）。")

# 示例2: 年龄 与 接驳方式(Q6)
ct2 = safe_crosstab(df['age'], df['transfer_mode'])
if ct2 is not None:
    chi2, p, dof, expected = chi2_contingency(ct2)
    print("\n年龄与接驳方式选择的卡方检验:")
    print(f"卡方值: {chi2:.3f}, p值: {p:.3f}")
    print("原假设：年龄与接驳方式选择独立")
    print("结论：", "拒绝原假设（存在关联）" if p < 0.05 else "不能拒绝原假设（独立）")
else:
    print("\n年龄×接驳方式 交叉表无效（可能是全空或类别不足）。")

# ========= 2) t检验 / ANOVA =========
print("\n" + "="*60)
print("t检验/ANOVA：不同组别在满意度等方面的差异")
print("="*60)

# 满意度分值
satisfaction_mapping = {1:1, 2:2, 3:3, 4:4, 5:5}
df['satisfaction_score'] = df['Q8'].map(satisfaction_mapping)

# t检验：性别差异（用原始编码列过滤，避免映射缺失）
male_satisfaction = df.loc[df['Q19'] == 1, 'satisfaction_score'].dropna()
female_satisfaction = df.loc[df['Q19'] == 2, 'satisfaction_score'].dropna()
if len(male_satisfaction) > 1 and len(female_satisfaction) > 1:
    t_stat, p_value_t = stats.ttest_ind(male_satisfaction, female_satisfaction, equal_var=False)
    print(f"性别对满意度的t检验: t={t_stat:.3f}, p={p_value_t:.3f}")
    print("原假设：男女满意度无显著差异")
    if p_value_t < 0.05:
        print("结论：拒绝原假设，存在显著差异")
        print(f"男性均值={male_satisfaction.mean():.2f}，女性均值={female_satisfaction.mean():.2f}")
    else:
        print("结论：不能拒绝原假设，差异不显著")
else:
    print("性别分组样本不足，无法做 t 检验。")

# ANOVA：年龄组
age_groups = [df.loc[df['Q20'] == a, 'satisfaction_score'].dropna() for a in range(1, 7)]
age_groups_valid = [g for g in age_groups if len(g) > 1]
if len(age_groups_valid) >= 2:
    f_stat_age, p_value_age = f_oneway(*age_groups_valid)
    print(f"\n年龄对满意度的ANOVA: F={f_stat_age:.3f}, p={p_value_age:.3f}")
    if p_value_age < 0.05:
        print("结论：各年龄组存在显著差异")
        for a in range(1, 7):
            mean_val = df.loc[df['Q20'] == a, 'satisfaction_score'].mean()
            print(f"{age_labels[a]}：均值={mean_val:.2f}")
    else:
        print("结论：各年龄组差异不显著")
else:
    print("\n年龄组有效样本不足，无法做 ANOVA。")

# ========= 3) Spearman 相关 =========
print("\n" + "="*60)
print("Spearman相关系数分析态度题之间的关联")
print("="*60)

attitude_columns = [c for c in [f'Q17.{i}' for i in range(1, 12)] if c in df.columns]
if len(attitude_columns) >= 2:
    attitude_df = df[attitude_columns].copy()
    # 强制为数值（若为李克特1-5，通常本就为数值；遇到非数值则置 NaN）
    for c in attitude_columns:
        attitude_df[c] = pd.to_numeric(attitude_df[c], errors='coerce')
    spearman_corr = attitude_df.corr(method='spearman')

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .6})
    plt.title('态度题Spearman相关系数矩阵')
    plt.tight_layout()
    plt.savefig('态度题相关系数矩阵.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("强相关关系 (|r| > 0.5):")
    cols = spearman_corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = spearman_corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.5:
                print(f"{cols[i]} 与 {cols[j]}: r={r:.3f}")
else:
    print("未找到足够的 Q17.* 列进行相关分析。")

# ========= 4) 回归：影响满意度的因素 =========
print("\n" + "="*60)
print("回归分析：影响满意度的关键因素（OLS）")
print("="*60)

# 选择可能影响满意度的变量（保留原问卷编码列，便于 one-hot）
reg_cols = ['satisfaction_score', 'Q19', 'Q20', 'Q21', 'Q23', 'Q18', 'Q3', 'Q5', 'Q9', 'Q10', 'Q11']
regression_df = df[reg_cols].copy()

# 丢缺失（OLS 需要完整行）
regression_df = regression_df.dropna(subset=['satisfaction_score'])
# 将所有自变量先转为整数/类别（避免字符串混入）
cat_cols = ['Q19', 'Q20', 'Q21', 'Q23', 'Q18', 'Q3', 'Q5', 'Q9', 'Q10', 'Q11']
for c in cat_cols:
    if c in regression_df.columns:
        regression_df[c] = pd.to_numeric(regression_df[c], errors='coerce')

# 丢掉自变量的 NA 行
regression_df = regression_df.dropna(subset=cat_cols)

# one-hot（drop_first 避免虚拟变量陷阱）
reg_dum = pd.get_dummies(regression_df, columns=cat_cols, drop_first=True)

y = reg_dum['satisfaction_score'].astype(float)
X = reg_dum.drop(columns=['satisfaction_score'])

# **关键修复**：将 X 强制转成 float，避免 object dtype
X = X.apply(pd.to_numeric, errors='coerce').astype(float)

# 再次对齐与去 NA（任何列转不成数值都会变 NaN）
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# 添加常数项
X = sm.add_constant(X, has_constant='add')

# 运行 OLS
if X.shape[0] > 0 and X.shape[1] > 1:
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 提取显著变量
    sig = model.pvalues[model.pvalues < 0.05]
    print("\n显著影响满意度的变量 (p < 0.05):")
    for var, p_val in sig.items():
        if var != 'const':
            coef = model.params[var]
            print(f"{var}: 系数={coef:.3f}, p值={p_val:.3f}")

    # 可视化前 10 个显著变量的系数
    sig_no_const = sig.drop('const', errors='ignore').sort_values()
    top_vars = sig_no_const.head(10).index
    if len(top_vars) > 0:
        coefs = model.params[top_vars]
        colors = ['red' if c < 0 else 'green' for c in coefs]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_vars)), coefs.values, color=colors)
        plt.yticks(range(len(top_vars)), top_vars)
        plt.xlabel('系数大小')
        plt.title('影响满意度的关键因素（前10显著）')
        plt.axvline(x=0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig('满意度影响因素.png', dpi=300, bbox_inches='tight')
        plt.show()
else:
    print("可用于 OLS 的数据不足（样本或特征为空）。")

# ========= 5) 接驳方式满意度差异 =========
print("\n" + "="*60)
print("不同接驳方式的满意度差异")
print("="*60)

transfer_satisfaction = df.groupby('transfer_mode')['satisfaction_score'].agg(['mean', 'std', 'count'])
print(transfer_satisfaction)

# ANOVA（只取样本量>1 的组）
groups = [g['satisfaction_score'].dropna() for _, g in df.groupby('transfer_mode')]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    f_stat_tm, p_value_tm = f_oneway(*groups)
    print(f"\n不同接驳方式满意度的ANOVA: F={f_stat_tm:.3f}, p={p_value_tm:.3f}")
    if p_value_tm < 0.05:
        print("结论：各接驳方式满意度存在显著差异")
        # 事后检验
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(endog=df['satisfaction_score'].dropna(),
                                  groups=df['transfer_mode'].dropna(),
                                  alpha=0.05)
        print(tukey)
    else:
        print("结论：各接驳方式满意度差异不显著")
else:
    print("各接驳方式有效样本不足，无法做 ANOVA。")

# ========= 6) 结果导出 =========
with pd.ExcelWriter('高级统计分析结果.xlsx') as writer:
    # 卡方
    chi_rows, chi_vals, chi_ps = [], [], []
    if ct1 is not None:
        chi1 = chi2_contingency(ct1)
        chi_rows.append('性别×M线知晓'); chi_vals.append(chi1[0]); chi_ps.append(chi1[1])
    if ct2 is not None:
        chi2r = chi2_contingency(ct2)
        chi_rows.append('年龄×接驳方式'); chi_vals.append(chi2r[0]); chi_ps.append(chi2r[1])
    if len(chi_rows) > 0:
        pd.DataFrame({'检验项目': chi_rows, '卡方值': chi_vals, 'p值': chi_ps}).to_excel(writer, sheet_name='卡方检验', index=False)

    # t/ANOVA 汇总（根据上面计算到的变量是否存在）
    rows, stats_list, p_list = [], [], []
    if 'p_value_t' in locals():
        rows.append('性别对满意度(t)'); stats_list.append(float(t_stat)); p_list.append(float(p_value_t))
    if 'p_value_age' in locals():
        rows.append('年龄对满意度(ANOVA)'); stats_list.append(float(f_stat_age)); p_list.append(float(p_value_age))
    if 'p_value_tm' in locals():
        rows.append('接驳方式对满意度(ANOVA)'); stats_list.append(float(f_stat_tm)); p_list.append(float(p_value_tm))
    if rows:
        pd.DataFrame({'检验项目': rows, '统计量': stats_list, 'p值': p_list}).to_excel(writer, sheet_name='t检验_ANOVA', index=False)

    # 相关矩阵
    if 'spearman_corr' in locals():
        spearman_corr.to_excel(writer, sheet_name='Spearman相关系数')

    # 回归
    if 'model' in locals():
        reg_out = pd.DataFrame({
            '变量': model.params.index,
            '系数': model.params.values,
            '标准误': model.bse.values,
            't值': model.tvalues.values,
            'p值': model.pvalues.values
        })
        reg_out.to_excel(writer, sheet_name='回归分析', index=False)

print("\n分析完成：图表已保存到当前目录，Excel汇总为《高级统计分析结果.xlsx》")
