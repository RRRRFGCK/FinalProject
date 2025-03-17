import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建一个图形
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制各个矩形框
box_props = {'facecolor': 'lightgray', 'edgecolor': 'black', 'boxstyle': 'round,pad=0.5'}

# 1. 开始
ax.text(0.5, 0.9, '开始', ha='center', va='center', fontsize=12, bbox=box_props)

# 2. 设置初始化参数
ax.text(0.5, 0.8, '设置初始化参数', ha='center', va='center', fontsize=12, bbox=box_props)

# 3. 生成初代父代种群，选择代数t=1
ax.text(0.5, 0.7, '生成初代父代种群，选择代数t=1', ha='center', va='center', fontsize=12, bbox=box_props)

# 4. 快速非支配排序
ax.text(0.5, 0.6, '快速非支配排序', ha='center', va='center', fontsize=12, bbox=box_props)

# 5. 遗传算法操作（选择、交叉、变异）
ax.text(0.5, 0.5, '遗传算法操作（选择、交叉、变异）', ha='center', va='center', fontsize=12, bbox=box_props)

# 6. 合并父代和子代种群
ax.text(0.5, 0.4, '合并父代和子代种群', ha='center', va='center', fontsize=12, bbox=box_props)

# 7. 选择拥挤度最大的个体进入新父代
ax.text(0.5, 0.3, '选择拥挤度最大的个体进入新父代', ha='center', va='center', fontsize=12, bbox=box_props)

# 8. 是否生成新父代种群
ax.text(0.5, 0.2, '是否生成新父代种群？', ha='center', va='center', fontsize=12, bbox=box_props)

# 9. 是否达到最大代数G
ax.text(0.5, 0.1, '是否达到最大代数G？', ha='center', va='center', fontsize=12, bbox=box_props)

# 10. 输出帕累托前沿解集
ax.text(0.5, 0, '输出帕累托前沿解集', ha='center', va='center', fontsize=12, bbox=box_props)

# 绘制箭头
arrowprops = {'arrowstyle': '->', 'lw': 1.5, 'color': 'black'}

# 箭头连接每个步骤
ax.annotate('', xy=(0.5, 0.83), xytext=(0.5, 0.88), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.73), xytext=(0.5, 0.78), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.63), xytext=(0.5, 0.68), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.53), xytext=(0.5, 0.58), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.43), xytext=(0.5, 0.48), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.33), xytext=(0.5, 0.38), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.23), xytext=(0.5, 0.28), arrowprops=arrowprops)
ax.annotate('', xy=(0.5, 0.13), xytext=(0.5, 0.18), arrowprops=arrowprops)

# 添加结束箭头
ax.annotate('', xy=(0.5, -0.05), xytext=(0.5, 0), arrowprops=arrowprops)

# 隐藏坐标轴
ax.set_axis_off()

# 调整布局，避免文本重叠
plt.tight_layout()
plt.show()
