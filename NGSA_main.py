import numpy as np
import matplotlib.pyplot as plt


from compute_average_energy_and_latency import compute_average_energy_and_latency_with_CV

# 定义遗传算法相关参数
NP = 100  # 种群大小
G = 100  # 最大迭代次数
cross_prob = 0.9  # 交叉概率
mut_prob = 0.1  # 变异概率
A = 500  # 总的用户数量
M = 4  # 总的边缘服务器数量


# 初始化种群
def initialize_population():
    population = []
    for _ in range(NP - 2):
        individual = np.round(np.random.rand(A), 2)  # 随机生成染色体，并保留两位小数
        population.append(individual)

    # 添加全1和全0的染色体
    population.append(np.ones(A))  # 全1染色体
    population.append(np.zeros(A))  # 全0染色体

    return np.array(population)

# 计算适应度函数
def fitness(individual, U_m_values):
    # PR_m_values 应该是一个 M x A 的二维数组，其中每个边缘服务器有多个用户的本地执行概率
    PR_m_values = np.array([individual for _ in range(M)])  # 为每个边缘服务器复制个体

    avg_energy, avg_latency, local_data, mec_data, CV_total = compute_average_energy_and_latency_with_CV(A, M,
                                                                                                         U_m_values,
                                                                                                         PR_m_values)

    # 目标是最小化能量和时延
    return avg_energy, avg_latency


# 非支配排序
def non_dominated_sorting(population, fitness_values):
    # fitness_values 是一个包含每个个体的两个目标值的数组
    # S 用于存储每个个体支配的其他个体的列表
    # front 用于存储所有前沿的列表
    S = [[] for _ in range(len(population))]
    front = [[]]  # 初始化第一个前沿

    # 每个个体的支配数目
    n = [0] * len(population)
    rank = [0] * len(population)  # 用于存储每个个体的排名

    # 对所有个体进行非支配排序
    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(fitness_values[p], fitness_values[q]):
                S[p].append(q)  # p 支配 q
            elif dominates(fitness_values[q], fitness_values[p]):
                n[p] += 1  # p 被 q 支配，增加 p 的支配数目

        # 如果个体p的支配数为0，说明是一个非支配解
        if n[p] == 0:
            rank[p] = 0  # 初始化为第一个前沿
            front[0].append(p)

    # 逐个计算其他前沿
    k = 0
    while len(front[k]) > 0:
        next_front = []
        for p in front[k]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = k + 1
                    next_front.append(q)
        k += 1
        front.append(next_front)  # 添加到下一前沿

    # 返回所有非支配解（所有前沿）
    return front


# 获取所有非支配解
def get_all_non_dominated_solutions(population, fitness_values):
    # 获取所有的非支配解的前沿
    fronts = non_dominated_sorting(population, fitness_values)

    all_non_dominated_solutions = []
    for front in fronts:
        all_non_dominated_solutions.append([population[i] for i in front])

    return all_non_dominated_solutions


def dominates(fit1, fit2):
    # 判断fit1是否支配fit2
    return fit1[0] <= fit2[0] and fit1[1] < fit2[1]


# 交叉操作
def crossover(parent1, parent2):
    if np.random.rand() < cross_prob:
        crossover_point = np.random.randint(1, A - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2


# 变异操作
def mutation(child):
    if np.random.rand() < mut_prob:
        mutation_point = np.random.randint(0, A)
        child[mutation_point] = np.round(np.random.rand(), 2)  # 变异为一个新的随机值
    return child


def plot_pareto_front(all_non_dominated_solutions, population, U_m_values):
    # 提取所有解的时延和功耗
    times = []  # 存储时延
    energies = []  # 存储能耗

    for front in all_non_dominated_solutions:
        for individual in front:
            # 获取当前解对应的适应度（时延和功耗）
            fitness_values = fitness(individual, U_m_values)
            avg_energy, avg_latency = fitness_values  # 只获取能量和时延

            # 将时延和功耗添加到列表
            energies.append(avg_energy)
            times.append(avg_latency)

    # 绘制帕累托前沿
    plt.figure(figsize=(8, 6))
    plt.scatter(times, energies, color='b', label='Pareto Front')

    plt.title('Pareto Front: Average Latency vs Average Energy')
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Average Energy (J)')
    plt.grid(True)
    plt.legend()
    plt.show()

def NGSA2():
    population = initialize_population()

    # 三组U_m_values
    U_m_values_sets = [
        np.random.randint(41, 61, size=(M)),  # 第一组
        np.random.randint(61, 81, size=(M)),  # 第二组
        np.random.randint(81, 101, size=(M))  # 第三组
    ]

    # 创建一个图形，准备在同一图上绘制多个Pareto前沿
    plt.figure(figsize=(8, 6))

    # 对每组U_m_values进行处理
    for idx, U_m_values in enumerate(U_m_values_sets):
        # 遍历代数进行进化
        for generation in range(G):
            fitness_values = np.array([fitness(individual, U_m_values) for individual in population])

            # 非支配排序
            fronts = non_dominated_sorting(population, fitness_values)

            # 获取所有非支配解
            all_non_dominated_solutions = get_all_non_dominated_solutions(population, fitness_values)

            # 打印当前代的非支配解
            print(f"Generation {generation + 1}/{G} for range {idx + 1} completed.")
            for front_num, front in enumerate(all_non_dominated_solutions):
                print(f"Front {front_num + 1}:")
                for solution in front:
                    print(solution)

            # 创建新的种群
            new_population = []

            # 从非支配前沿开始生成新种群
            for front in fronts:
                for i in range(0, len(front), 2):
                    parent1 = population[front[i]]
                    parent2 = population[front[i + 1]] if i + 1 < len(front) else parent1  # 如果是奇数个体则自交叉
                    child1, child2 = crossover(parent1, parent2)
                    new_population.append(mutation(child1))
                    new_population.append(mutation(child2))

            population = np.array(new_population[:NP])  # 保证种群大小为NP

        # 获取所有非支配解并绘制
        times = []  # 存储时延
        energies = []  # 存储能耗

        for front in all_non_dominated_solutions:
            for individual in front:
                # 获取当前解对应的适应度（时延和功耗）
                fitness_values = fitness(individual, U_m_values)
                avg_energy, avg_latency = fitness_values  # 只获取能量和时延

                # 将时延和功耗添加到列表
                energies.append(avg_energy)
                times.append(avg_latency)

        # 绘制当前组的Pareto前沿，使用不同的颜色区分
        plt.scatter(times, energies, label=f'Range {idx + 1} ({U_m_values[0]}-{U_m_values[-1]})')

    plt.title('Pareto Front: Average Latency vs Average Energy')
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Average Energy (J)')
    plt.grid(True)
    plt.legend()
    plt.show()

# 运行NGSA-II
NGSA2()


# # 在NGSA2函数结束时调用此函数来绘制帕累托前沿
# def NGSA2():
#     population = initialize_population()
#     U_m_values = np.random.randint(81, 101, size=(M))  # 假设每个边缘服务器的用户数在[1, 50]之间
#
#     for generation in range(G):
#         fitness_values = np.array([fitness(individual, U_m_values) for individual in population])
#
#         # 非支配排序
#         fronts = non_dominated_sorting(population, fitness_values)
#
#         # 获取所有非支配解
#         all_non_dominated_solutions = get_all_non_dominated_solutions(population, fitness_values)
#
#         # 打印当前代的非支配解
#         print(f"Generation {generation + 1}/{G} completed.")
#         for front_num, front in enumerate(all_non_dominated_solutions):
#             print(f"Front {front_num + 1}:")
#             for solution in front:
#                 print(solution)
#
#         # 创建新的种群
#         new_population = []
#
#         # 从非支配前沿开始生成新种群
#         for front in fronts:
#             for i in range(0, len(front), 2):
#                 parent1 = population[front[i]]
#                 parent2 = population[front[i + 1]] if i + 1 < len(front) else parent1  # 如果是奇数个体则自交叉
#                 child1, child2 = crossover(parent1, parent2)
#                 new_population.append(mutation(child1))
#                 new_population.append(mutation(child2))
#
#         population = np.array(new_population[:NP])  # 保证种群大小为NP
#
#     # 最终绘制帕累托前沿图
#     plot_pareto_front(all_non_dominated_solutions, population, U_m_values)
#
#
# # 运行NGSA-II
# NGSA2()
