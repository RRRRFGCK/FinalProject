# 绘制符合工字形的柱状图
import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 设置宽度和位置
width = 0.4
x = np.arange(3)  # 3个柱子的位置

# 绘制时延的柱状图，包含工字形的表示
for i, (latency_min, latency_max, latency_mean) in enumerate(zip([0.026, 0.077, 0.126], [0.157, 0.214, 0.269], [0.108, 0.157, 0.198])):
    # 绘制顶部和底部的水平线（工字形的上下部分）
    ax[0].plot([x[i] - width / 2, x[i] + width / 2], [latency_min, latency_min], color='blue', lw=2)
    ax[0].plot([x[i] - width / 2, x[i] + width / 2], [latency_max, latency_max], color='blue', lw=2)
    # 绘制矩形柱体
    ax[0].add_patch(plt.Rectangle((x[i] - width / 2, latency_min), width, latency_max - latency_min, color='blue', alpha=0.3))
    # 绘制中间的实线（表示平均值）
    ax[0].plot([x[i] - width / 2, x[i] + width / 2], [latency_mean, latency_mean], color='red', lw=2)

# 设置标题和标签
ax[0].set_title('Latency Comparison Across Pareto Fronts')
ax[0].set_xlabel('Latency')
ax[0].set_ylabel('Time (s)')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Pareto Front 1', 'Pareto Front 2', 'Pareto Front 3'])

# 绘制能耗的柱状图，包含工字形的表示
for i, (energy_min, energy_max, energy_mean) in enumerate(zip([1.897, 2.487, 3.103], [0.218, 0.504, 0.736], [1.08, 1.485, 2.036])):
    # 绘制顶部和底部的水平线（工字形的上下部分）
    ax[1].plot([x[i] - width / 2, x[i] + width / 2], [energy_min, energy_min], color='green', lw=2)
    ax[1].plot([x[i] - width / 2, x[i] + width / 2], [energy_max, energy_max], color='green', lw=2)
    # 绘制矩形柱体
    ax[1].add_patch(plt.Rectangle((x[i] - width / 2, energy_min), width, energy_max - energy_min, color='green', alpha=0.3))
    # 绘制中间的实线（表示平均值）
    ax[1].plot([x[i] - width / 2, x[i] + width / 2], [energy_mean, energy_mean], color='red', lw=2)

# 设置标题和标签
ax[1].set_title('Energy Comparison Across Pareto Fronts')
ax[1].set_xlabel('Energy')
ax[1].set_ylabel('Energy (J)')
ax[1].set_xticks(x)
ax[1].set_xticklabels(['Pareto Front 1', 'Pareto Front 2', 'Pareto Front 3'])

# 显示图表
plt.tight_layout()
plt.show()
