import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

name = ['附件一峰', '附件一谷', '附件一均值', '附件二峰', '附件二谷', '附件二均值']
raw_data = [14.5138, 13.5211, 14.01745, 12.5809, 11.9372, 12.25905]
processed_data = [13.1409, 13.511, 13.32595, 12.1698, 12.622, 12.3959]
optimized_data = [9.3528, 6.1325, 7.74265, 7.7186, 7.3417, 7.53015]

x = np.arange(len(name))
width = 0.25 

fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#FFB6C1', '#90EE90', '#ADD8E6']

rects1 = ax.bar(x - width, raw_data, width, label='原始', color=colors[0])
rects2 = ax.bar(x, processed_data, width, label='数据处理', color=colors[1])
rects3 = ax.bar(x + width, optimized_data, width, label='数据处理+色散优化', color=colors[2])

ax.set_xlabel('数据类型', fontsize=12)
ax.set_ylabel('数值', fontsize=12)
ax.set_title('不同数据处理方法下的附件一和附件二数据比较', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(name)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

name = ['原始', '处理后', '处理+色散']
attachment1_diff = [0.9927, 0.3701, 3.2203]
attachment2_diff = [0.6437, 0.4522, 0.3769]  
mean_diff = [1.7584, 0.93005, 0.2125]

x = np.arange(len(name)) 
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#FF9999', '#99CCFF', '#99FF99']
rects1 = ax.bar(x - width, attachment1_diff, width, label='附件一峰谷差', color=colors[0])
rects2 = ax.bar(x, attachment2_diff, width, label='附件二峰谷差', color=colors[1])
rects3 = ax.bar(x + width, mean_diff, width, label='两附件均值差', color=colors[2])

ax.set_xlabel('处理阶段', fontsize=12)
ax.set_ylabel('差值', fontsize=12)
ax.set_title('不同处理阶段的峰谷差和均值差比较', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(name)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()