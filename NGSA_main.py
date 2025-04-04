import numpy as np
import matplotlib.pyplot as plt
from compute_average_energy_and_latency import compute_average_energy_and_latency_with_CV

# 定义遗传算法相关参数
NP = 100  # 种群大小
G = 100  # 最大迭代次数
cross_prob = 0.9  # 交叉概率
mut_prob = 0.1  # 变异概率
A = 500  # 染色体长度（用户总数）
M = 4    # 总的边缘服务器数量

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
    # PR_m_values 是一个 M x A 的二维数组，其中每个边缘服务器复制相同的个体
    PR_m_values = np.array([individual for _ in range(M)])
    avg_energy, avg_latency, D_local_total, D_rsu_total, CV_total = compute_average_energy_and_latency_with_CV(A, M,
                                                                                                               U_m_values,
                                                                                                               PR_m_values)
    # 目标是最小化能量和时延
    return avg_energy, avg_latency

# 判断fit1是否支配fit2
def dominates(fit1, fit2):
    return fit1[0] <= fit2[0] and fit1[1] < fit2[1]

# 非支配排序
def non_dominated_sorting(population, fitness_values):
    S = [[] for _ in range(len(population))]  # 每个个体支配的解集合
    front = [[]]  # 存储每一前沿
    n = [0] * len(population)  # 每个个体被支配的次数
    rank = [0] * len(population)  # 个体排名

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(fitness_values[p], fitness_values[q]):
                S[p].append(q)
            elif dominates(fitness_values[q], fitness_values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

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
        front.append(next_front)
    return front

# 获取所有非支配解（用于绘图和打印）
def get_all_non_dominated_solutions(population, fitness_values):
    fronts = non_dominated_sorting(population, fitness_values)
    all_non_dominated_solutions = []
    for front in fronts:
        all_non_dominated_solutions.append([population[i] for i in front])
    return all_non_dominated_solutions

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

def NGSA2():
    population = initialize_population()

    # 三次模拟：总用户数分别为200, 280, 360，每个RSU均分
    U_m_values_sets = [
        np.full((M,), 200 // M),  # 第一组：每个RSU获得50个用户
        np.full((M,), 280 // M),  # 第二组：每个RSU获得70个用户
        np.full((M,), 360 // M)   # 第三组：每个RSU获得90个用户
    ]

    plt.figure(figsize=(8, 6))
    # 针对每组U_m_values分别进行演化
    for idx, U_m_values in enumerate(U_m_values_sets):
        for generation in range(G):
            fitness_values = np.array([fitness(individual, U_m_values) for individual in population])
            fronts = non_dominated_sorting(population, fitness_values)
            all_non_dominated_solutions = get_all_non_dominated_solutions(population, fitness_values)

            print(f"Generation {generation + 1}/{G} for range {idx + 1} completed.")
            for front_num, front in enumerate(all_non_dominated_solutions):
                print(f"Front {front_num + 1}:")
                for solution in front:
                    print(solution)

            # 精英保留：保留第一前沿（非支配最优）的个体
            elite_indices = fronts[0]
            elite_individuals = population[elite_indices]

            offspring = []
            # 使用非支配前沿顺序生成子代
            for front in fronts:
                for i in range(0, len(front), 2):
                    parent1 = population[front[i]]
                    parent2 = population[front[i + 1]] if i + 1 < len(front) else parent1
                    child1, child2 = crossover(parent1, parent2)
                    offspring.append(mutation(child1))
                    offspring.append(mutation(child2))

            # 合并精英与子代，并截断保证种群大小为 NP
            new_population = list(elite_individuals) + offspring
            population = np.array(new_population[:NP])

        # 收集当前模拟中所有非支配解对应的适应度数据
        times = []   # 存储时延
        energies = []  # 存储能耗
        for front in all_non_dominated_solutions:
            for individual in front:
                avg_energy, avg_latency = fitness(individual, U_m_values)
                energies.append(avg_energy)
                times.append(avg_latency)
        plt.scatter(times, energies, label=f'Range {idx + 1} ({U_m_values[0]*M})')

    plt.title('Pareto Front: Average Latency vs Average Energy')
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Average Energy (J)')
    plt.grid(True)
    plt.legend()
    plt.show()

# 运行NGSA-II
NGSA2()
