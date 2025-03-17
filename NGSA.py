import numpy as np
import matplotlib.pyplot as plt
import json
# 常量参数
A = 200  # 总的移动用户数量
M = 4  # 总的边缘服务器数量
pop_size = 100  # 种群大小 NP
G_max = 100  # 最大代数 G
CR = 0.9  # 交叉概率 CR
PM = 0.1  # 变异概率 PM
f_m_j = 1e9  # 车辆的本地计算能力 (1 GHz)
F_m = 10e9  # 边缘服务器的计算能力 (10 GHz)
p_m_j = 0.5  # 用户的传输功率 (W)

# 模拟的任务参数
D_m_j_min = 100e3  # 最小子任务数据量 (100 kbit)
D_m_j_max = 300e3  # 最大子任务数据量 (300 kbit)
v_m_j_min = 50  # 最小车辆速度 (50 km/h)
v_m_j_max = 80  # 最大车辆速度 (80 km/h)
d_m_j_min = 1  # 最小距离 (1 m)
d_m_j_max = 500  # 最大距离 (500 m)

# 需要定义的缺失常量
noise_power = -154  # 噪声功率 (dBm)
B = 20e6  # 带宽 (20 MHz)
N = 20  # 信道数
h = 4  # 信道衰减因子


# 计算通信速率 (基于香农定理)
def compute_communication_rate(p_m_j, d_m_j):
    noise_power_linear = 10 ** (noise_power / 10)  # 从dBm转换到线性单位
    R_m_j = B / N * np.log2(1 + (p_m_j * h) / (noise_power_linear * d_m_j ** 2))  # 香农定理
    return R_m_j


# 计算传输时延
def compute_transmission_time(R_m_j, D_m_j, PR_m_j):
    t_trans = (1 - PR_m_j) * D_m_j / R_m_j  # 传输时延 (PR_m_j暂时为0)
    return t_trans


# 计算传输能耗
def compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j):
    t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)
    e_trans = p_m_j * t_trans  # 传输能耗
    return e_trans


# 计算本地计算时延
def compute_local_computation_time(f_m_j, D_m_j, PR_m_j):
    t_local = PR_m_j * D_m_j / f_m_j  # 本地计算时延 (PR_m_j暂时为0)
    return t_local


# 计算本地计算能耗
def compute_local_computation_energy(f_m_j, D_m_j, PR_m_j):
    eta = 30e-27  # 假设计算功率系数
    e_local = eta * f_m_j ** 2 * PR_m_j * D_m_j  # 本地计算能耗
    return e_local


# 计算边缘计算时延
def compute_edge_computation_time(F_m, D_m_j, PR_m_j):
    t_mec = (1 - PR_m_j) * D_m_j / F_m  # 边缘计算时延 (PR_m_j暂时为0)
    return t_mec


# 计算约束违规度
def compute_CV_local(D_local, Q_c):
    if D_local <= Q_c:
        return 0
    else:
        return (D_local - Q_c) / Q_c


def compute_CV_mec(D_mec, Q_d):
    if D_mec <= Q_d:
        return 0
    else:
        return (D_mec - Q_d) / Q_d


def compute_CV(M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d):
    total_CV = 0
    for m in range(M):
        for j in range(U_m_values[m]):
            D_local = D_local_values[m][j]  # 本地任务数据量
            D_mec = D_mec_values[m][j]  # 边缘计算任务数据量
            # 计算本地计算和边缘计算的约束违规程度
            CV_local = compute_CV_local(D_local, Q_c)
            CV_mec = compute_CV_mec(D_mec, Q_d)
            total_CV += CV_local + CV_mec  # 累加约束违规程度
    return total_CV


# 适应度计算（考虑目标函数与CV）
def compute_fitness(population, M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d, p_m_j):
    fitness = []
    for individual in population:
        PR_m_j = individual  # 这里假设每个个体的值对应于PRm,j的值
        total_energy = 0
        total_latency = 0
        # 计算能耗和时延
        for m in range(M):
            for j in range(U_m_values[m]):
                # 计算每个用户的任务数据量 D_m_j
                C_m_j = U_m_values[m] - 1  # 计算子任务数量 (Cm,j = Um,j - 1)
                D_m_j = np.random.uniform(D_m_j_min, D_m_j_max) * C_m_j  # 任务数据量由多个子任务构成

                D_local = D_local_values[m][j]
                D_mec = D_mec_values[m][j]

                # 计算通信、传输、计算时延和能耗
                R_m_j = compute_communication_rate(p_m_j, D_m_j)
                e_local = compute_local_computation_energy(f_m_j, D_local, PR_m_j)
                e_trans = compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j)
                t_local = compute_local_computation_time(f_m_j, D_local, PR_m_j)
                t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)
                t_mec = compute_edge_computation_time(F_m, D_mec, PR_m_j)

                total_energy += e_local + e_trans
                total_latency += t_local + t_trans + t_mec

        # 计算约束违反度
        CV_value = compute_CV(M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d)
        # 加入CV值来调整适应度（如果CV > 0，则需要惩罚）
        fitness.append([total_energy + CV_value, total_latency + CV_value])

    return np.array(fitness)


# 锦标赛选择
def tournament_selection(population, fitness):
    ranks = non_dominated_sort(fitness)  # 获取每个个体的支配等级
    selected = []
    for i in range(pop_size):
        competitors = np.random.choice(len(population), size=2, replace=False)  # 随机选择两个个体

        # 获取两个竞争者的支配等级的集合
        rank_1 = ranks[competitors[0]]
        rank_2 = ranks[competitors[1]]

        # 选择支配等级更小的个体，支配等级越小表示该个体在多目标优化中表现更好
        if len(rank_1) < len(rank_2):
            selected.append(population[competitors[0]])
        elif len(rank_1) > len(rank_2):
            selected.append(population[competitors[1]])
        else:
            # 如果支配等级相同，我们随机选择一个
            selected.append(population[competitors[np.random.randint(2)]])

    return np.array(selected)



# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring1, offspring2


# 变异操作
def mutation(offspring):
    mutation_point = np.random.randint(len(offspring))
    mutation_value = np.random.normal(0, 0.1)
    offspring[mutation_point] = offspring[mutation_point] + mutation_value
    return offspring


def initialize_population(A, pop_size):
    # 随机生成 NP-2 个个体
    population = np.random.uniform(0, 1, (pop_size - 2, A))

    # 添加全0的染色体（所有任务都由边缘服务器执行）
    all_zeros = np.zeros((1, A))

    # 添加全1的染色体（所有任务都本地执行）
    all_ones = np.ones((1, A))

    # 合并全0，全1和随机生成的染色体
    population = np.vstack((all_zeros, all_ones, population))

    return population

def compute_fitness(population, M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d, p_m_j):
    fitness = []
    latency_values = []  # 用来存储时延
    energy_values = []  # 用来存储能耗

    for individual in population:
        PR_m_j = individual  # 假设每个个体的值对应于PRm,j的值
        total_energy = 0
        total_latency = 0

        # 计算能耗和时延
        for m in range(M):
            for j in range(U_m_values[m]):
                # 计算每个用户的任务数据量 D_m_j
                C_m_j = U_m_values[m] - 1  # 计算子任务数量 (Cm,j = Um,j - 1)
                D_m_j = np.random.uniform(D_m_j_min, D_m_j_max) * C_m_j  # 任务数据量由多个子任务构成

                D_local = D_local_values[m][j]
                D_mec = D_mec_values[m][j]

                # 计算通信、传输、计算时延和能耗
                R_m_j = compute_communication_rate(p_m_j, D_m_j)
                e_local = compute_local_computation_energy(f_m_j, D_local, PR_m_j)
                e_trans = compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j)
                t_local = compute_local_computation_time(f_m_j, D_local, PR_m_j)
                t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)
                t_mec = compute_edge_computation_time(F_m, D_mec, PR_m_j)

                total_energy += e_local + e_trans
                total_latency += t_local + t_trans + t_mec

        # 计算约束违反度
        CV_value = compute_CV(M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d)

        fitness.append([total_energy + CV_value, total_latency + CV_value])
        latency_values.append(total_latency + CV_value)  # 添加时延数据
        energy_values.append(total_energy + CV_value)  # 添加能耗数据

    return np.array(fitness), latency_values, energy_values

def dominates_individual(f1, f2):
    """
    判断个体 f1 是否支配个体 f2
    f1 和 f2 都是目标函数值的向量，例如 [能耗, 时延]
    """
    # f1 支配 f2，当 f1 在所有目标上都不差于 f2，并且至少有一个目标上 f1 严格优于 f2
    dominate = False
    for i in range(len(f1)):
        # 比较目标函数值，确保它们是标量而不是数组
        if np.any(f1[i] > f2[i]):  # f1 在第 i 个目标上比 f2 差
            return False
        elif np.any(f1[i] < f2[i]):  # f1 在第 i 个目标上严格优于 f2
            dominate = True
    return dominate


def non_dominated_sort(fitness):
    population_size = len(fitness)

    # 存储每个个体支配的个体列表
    dominated_count = np.zeros(population_size)  # 每个个体被支配的个体数
    dominates = [[] for _ in range(population_size)]  # 存储每个个体支配的个体
    fronts = [[]]  # 用来存储每个前沿的个体集合，fronts[i]表示第i个前沿

    for p in range(population_size):
        for q in range(population_size):
            if dominates_individual(fitness[p], fitness[q]):  # 如果 p 支配 q
                dominates[p].append(q)
            elif dominates_individual(fitness[q], fitness[p]):  # 如果 q 支配 p
                dominated_count[p] += 1

        if dominated_count[p] == 0:
            fronts[0].append(p)

    front_index = 0
    while len(fronts[front_index]) > 0:
        next_front = []

        for p in fronts[front_index]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)

        front_index += 1
        fronts.append(next_front)

    return fronts

def tournament_selection(population, fitness):
    # 获取种群的非支配排序
    ranks = non_dominated_sort(fitness)  # ranks 是非支配排序后的前沿，每个前沿是一个支配的个体集合
    selected = []

    for i in range(pop_size):
        # 随机选择两个个体
        competitors = np.random.choice(len(population), size=2, replace=False)

        # 获取这两个竞争者的支配等级
        rank_1 = get_individual_rank(ranks, competitors[0])
        rank_2 = get_individual_rank(ranks, competitors[1])

        # 比较支配等级，选择支配等级较小的个体
        if rank_1 < rank_2:
            selected.append(population[competitors[0]])
        elif rank_1 > rank_2:
            selected.append(population[competitors[1]])
        else:
            # 如果支配等级相同，随机选择一个
            selected.append(population[competitors[np.random.randint(2)]])

    return np.array(selected)


# 获取个体在非支配排序中的排名
def get_individual_rank(ranks, individual_index):
    for rank_index, rank in enumerate(ranks):
        if individual_index in rank:
            return rank_index  # 返回该个体所在的支配等级的排名
    return len(ranks)  # 如果没找到，返回一个较大的值（表示该个体没有找到对应的支配等级）

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring1, offspring2

def mutation(offspring):
    mutation_point = np.random.randint(len(offspring))
    mutation_value = np.random.normal(0, 0.1)
    offspring[mutation_point] = offspring[mutation_point] + mutation_value
    return offspring

# def save_population_to_json(population, fitness, generation):
#     # 获取帕累托前沿个体
#     ranks = non_dominated_sort(fitness)
#     pareto_front = ranks[0]  # 假设第一个前沿即为帕累托前沿
#
#     # 获取帕累托前沿个体的基因，保留小数点后三位
#     pareto_genes = [list(map(lambda x: round(x, 3), population[i])) for i in pareto_front]
#
#     # 保存每代的基因和帕累托前沿
#     generation_data = {
#         'generation': generation,
#         'population': [list(map(lambda x: round(x, 3), individual)) for individual in population],  # 每个个体基因四舍五入到小数点后三位
#         'pareto_front': pareto_genes  # 保存帕累托前沿个体的基因
#     }
#
#     # 将数据保存到文件
#     with open(f'population_generation_{generation}.json', 'w') as f:
#         json.dump(generation_data, f, indent=4)

def save_first_front_to_json(population, fitness, generation):
    # 获取种群的非支配排序
    ranks = non_dominated_sort(fitness)  # 获取非支配排序的前沿
    first_front = ranks[0]  # 第一前沿

    # 获取第一前沿个体的基因、时延和能耗
    pareto_genes = [population[i].tolist() for i in first_front]  # 转换为列表方便存储
    pareto_latency = []  # 存储第一前沿的时延
    pareto_energy = []  # 存储第一前沿的能耗

    for i in first_front:
        # 获取个体的时延和能耗
        total_latency = fitness[i][1]  # 从fitness中获取时延
        total_energy = fitness[i][0]  # 从fitness中获取能耗

        pareto_latency.append(total_latency)
        pareto_energy.append(total_energy)

    # 计算平均时延和能耗
    avg_latency = np.mean(pareto_latency)
    avg_energy = np.mean(pareto_energy)

    # 保存到JSON文件
    generation_data = {
        'generation': generation,
        'pareto_front': {
            'genes': pareto_genes,  # 第一前沿的个体基因
            'average_latency': avg_latency,  # 平均时延
            'average_energy': avg_energy  # 平均能耗
        }
    }

    # 将数据保存到文件
    with open(f'first_front_generation_{generation}.json', 'w') as f:
        json.dump(generation_data, f, indent=4)


def nsga2():
    population = initialize_population(A, pop_size)  # 初始化种群
    Q_c = 1e5  # 本地任务最大数据量 (500 kbit)
    Q_d = 5e6  # 边缘计算任务最大数据量 (2 Mbit)

    D_local_values = np.random.uniform(200e3, 600e3, (M, max([50 for _ in range(M)])))
    D_mec_values = np.random.uniform(1e6, 3e6, (M, max([50 for _ in range(M)])))
    U_m_values = [50, 50, 50, 50]  # 每个边缘服务器的用户数

    latency_data = []  # 用来存储每代的时延数据
    energy_data = []  # 用来存储每代的能耗数据

    for generation in range(G_max):
        fitness, latency_values, energy_values = compute_fitness(population, M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d, p_m_j)

        # 保存每代的时延和能耗
        latency_data.append(latency_values)
        energy_data.append(energy_values)

        ranks = non_dominated_sort(fitness)
        selected_population = tournament_selection(population, fitness)

        offspring_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring_population.append(mutation(offspring1))
            offspring_population.append(mutation(offspring2))

        # 合并父代和子代，选择较优的个体
        combined_population = np.concatenate((population, offspring_population))
        population = combined_population[:pop_size]

        # 每代保存第一前沿信息到 JSON 文件
        save_first_front_to_json(population, fitness, generation)

        if generation % 10 == 0:
            print(f"Generation {generation}, Best fitness: {np.min(fitness)}")

    return population

final_population = nsga2()


# def nsga2():
#     population = initialize_population(A, pop_size)  # 初始化种群
#     Q_c = 1e5  # 本地任务最大数据量 (500 kbit)
#     Q_d = 5e6  # 边缘计算任务最大数据量 (2 Mbit)
#
#     D_local_values = np.random.uniform(200e3, 600e3, (M, max([50 for _ in range(M)])))
#     D_mec_values = np.random.uniform(1e6, 3e6, (M, max([50 for _ in range(M)])))
#     U_m_values = [50, 50, 50, 50]  # 每个边缘服务器的用户数
#
#     latency_data = []  # 用来存储每代的时延数据
#     energy_data = []  # 用来存储每代的能耗数据
#
#     for generation in range(G_max):
#         fitness, latency_values, energy_values = compute_fitness(population, M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d, p_m_j)
#
#         # 保存每代的时延和能耗
#         latency_data.append(latency_values)
#         energy_data.append(energy_values)
#
#         ranks = non_dominated_sort(fitness)
#         selected_population = tournament_selection(population, fitness)
#
#         offspring_population = []
#         for i in range(0, len(selected_population), 2):
#             parent1, parent2 = selected_population[i], selected_population[i + 1]
#             offspring1, offspring2 = crossover(parent1, parent2)
#             offspring_population.append(mutation(offspring1))
#             offspring_population.append(mutation(offspring2))
#
#         # 合并父代和子代，选择较优的个体
#         combined_population = np.concatenate((population, offspring_population))
#         population = combined_population[:pop_size]
#
#         # 每代保存种群基因和帕累托前沿信息到 JSON 文件
#         save_population_to_json(population, fitness, generation)
#
#         if generation % 10 == 0:
#             print(f"Generation {generation}, Best fitness: {np.min(fitness)}")
#
#     return population
#
# final_population = nsga2()


#
# def nsga2():
#     population = initialize_population(A, pop_size)  # 初始化种群
#     Q_c = 1e6  # 本地任务最大数据量 (500 kbit)
#     Q_d = 4e6  # 边缘计算任务最大数据量 (2 Mbit)
#
#     D_local_values = np.random.uniform(200e3, 600e3, (M, max([50 for _ in range(M)])))
#     D_mec_values = np.random.uniform(1e6, 3e6, (M, max([50 for _ in range(M)])))
#     U_m_values = [50, 50, 50, 50]  # 每个边缘服务器的用户数
#
#     latency_data = []  # 用来存储每代的时延数据
#     energy_data = []  # 用来存储每代的能耗数据
#
#     for generation in range(G_max):
#         fitness, latency_values, energy_values = compute_fitness(population, M, U_m_values, D_local_values, D_mec_values, Q_c, Q_d, p_m_j)
#
#         # 保存每代的时延和能耗
#         latency_data.append(latency_values)
#         energy_data.append(energy_values)
#
#         ranks = non_dominated_sort(fitness)
#         selected_population = tournament_selection(population, fitness)
#
#         offspring_population = []
#         for i in range(0, len(selected_population), 2):
#             parent1, parent2 = selected_population[i], selected_population[i + 1]
#             offspring1, offspring2 = crossover(parent1, parent2)
#             offspring_population.append(mutation(offspring1))
#             offspring_population.append(mutation(offspring2))
#
#         # 合并父代和子代，选择较优的个体
#         combined_population = np.concatenate((population, offspring_population))
#         population = combined_population[:pop_size]
#
#         if generation % 10 == 0:
#             print(f"Generation {generation}, Best fitness: {np.min(fitness)}")
#
#     # 绘制帕累托前沿图
#     plt.scatter(energy_data[-1], latency_data[-1], label='Pareto Front', color='blue')
#     plt.title("Pareto Front (Energy vs Latency)")
#     plt.xlabel("Energy (J)")
#     plt.ylabel("Latency (s)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#     return population
#
# final_population = nsga2()