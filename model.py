import numpy as np

# 常量参数
A = 400  # 总的移动用户数量
M = 4  # 总的边缘服务器数量
noise_power = -154  # 噪声功率 (dBm)
h = 4  # 信道衰减因子
B = 20e6  # 带宽 (20 MHz)
N = 20  # 信道数
p_m_j = 0.5  # 用户的传输功率 (W)
F_m = 10e9  # 边缘服务器的计算能力 (10 GHz)
f_m_j = 1e9  # 车辆的本地计算能力 (1 GHz)

# 模拟的任务参数
D_m_j_min = 100e3  # 最小子任务数据量 (100 kbit)
D_m_j_max = 300e3  # 最大子任务数据量 (300 kbit)
v_m_j_min = 50  # 最小车辆速度 (50 km/h)
v_m_j_max = 80  # 最大车辆速度 (80 km/h)
d_m_j_min = 1  # 最小距离 (1 m)
d_m_j_max = 500  # 最大距离 (500 m)


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


# 计算能量和时延的平均值
def compute_average_energy_and_latency(A, M, U_m_values):
    total_energy = 0
    total_latency = 0

    for m in range(M):
        for j in range(U_m_values[m]):
            # 每个用户的子任务数量 Cm,j
            C_m_j = U_m_values[m] - 1  # 计算子任务的数量 (Cm,j = Um,j - 1)

            # 计算任务的数据量 D_m_j
            D_m_j = 0
            for k in range(C_m_j):
                D_m_j_k = np.random.uniform(D_m_j_min, D_m_j_max)  # 每个子任务的数据量在[100kbit, 300kbit]之间
                D_m_j += D_m_j_k  # 累加子任务数据量得到总的任务数据量

            v_m_j = np.random.uniform(v_m_j_min, v_m_j_max)  # 车辆速度在50km/h到80km/h之间
            d_m_j = np.random.uniform(d_m_j_min, d_m_j_max)  # 每辆车与边缘服务器的距离在1m到500m之间

            R_m_j = compute_communication_rate(p_m_j, d_m_j)  # 计算通信速率
            PR_m_j = 0  # 暂时假设PR_m_j为0，即所有任务都本地执行

            # 计算每个用户的能量和时延
            e_local = compute_local_computation_energy(f_m_j, D_m_j, PR_m_j)
            e_trans = compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j)
            t_local = compute_local_computation_time(f_m_j, D_m_j, PR_m_j)
            t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)
            t_mec = compute_edge_computation_time(F_m, D_m_j, PR_m_j)

            # 累计能量和时延
            total_energy += e_local + e_trans
            total_latency += t_local + t_trans + t_mec

    # 计算平均能量和时延
    avg_energy = total_energy / A
    avg_latency = total_latency / A
    return avg_energy, avg_latency


# 假设每个边缘服务器上的移动用户数量
U_m_values = [50, 50, 50, 50]  # 每个边缘服务器上有100个用户

# 计算平均能量和时延
avg_energy, avg_latency = compute_average_energy_and_latency(A, M, U_m_values)

print(f"Average Energy Consumption: {avg_energy} J")
print(f"Average Latency: {avg_latency} s")
