import numpy as np

# 常量参数
A = 500  # 总的移动用户数量
M = 4  # 总的RSU服务器数量
noise_power = -154  # 噪声功率 (dBm)
h = 4  # 信道衰减因子
B = 20e6  # 带宽 (20 MHz)
N = 20  # 信道数
p_m_j = 0.5  # 用户的传输功率 (W)
F_m = 5e8  # RSU服务器的计算能力 (10 GHz)
F_0 = 10e8
f_m_j = 1e8  # 车辆的本地计算能力 (1 GHz)

# 模拟的任务参数
S_m_j = 250e6  # 子任务数据量 (250 MB)
v_m_j_min = 50  # 最小车辆速度 (50 km/h)
v_m_j_max = 80  # 最大车辆速度 (80 km/h)
d_m_j_min = 1  # 最小距离 (1 m)
d_m_j_max = 500  # 最大距离 (500 m)
gamma_1 = 0.1
gamma_2 = 0.65
eta = 30e-27  # 假设计算功率系数
alpha =0.6

# 计算通信速率 (基于香农定理)
def compute_preprocess_time(S_m_j, gamma_1, f_m_j):
    t_pre = gamma_1*S_m_j/f_m_j
    return t_pre

def compute_preprocess_energy(S_m_j, gamma_1, f_m_j, eta):
    e_pre = eta * f_m_j ** 2 * gamma_1 * S_m_j
    return e_pre

D_m_j = (1-gamma_1)*gamma_2*S_m_j
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

# 计算RSU服务器计算时延
def compute_rsu_computation_time(F_m, D_m_j, PR_m_j, alpha):
    t_rsu = (1 - PR_m_j) * D_m_j * (1-alpha) / F_m  # RSU服务器计算时延 (PR_m_j暂时为0)
    return t_rsu

def compute_edge_computation_time(F_m, D_m_j, PR_m_j, alpha, F_0):
    t_edge = (1 - PR_m_j) * D_m_j * alpha / F_0
    return t_edge

# 计算本地任务的数据量（D_local_m_j）和RSU服务器计算任务的数据量（D_rsu_m）
def compute_local_and_rsu_data(U_m_values, PR_m_values, D_m_j):
    D_local_total = 0  # 本地任务总数据量
    D_rsu_total = 0  # RSU服务器计算任务总数据量
    CV_total = 0  # 违反程度总和
    Qc = 100e3
    Qd = 2e6
    Qe = 4e6

    # 遍历每个RSU服务器和其对应的用户
    for m in range(M):
        D_rsu_m = 0  # 当前RSU服务器的RSU计算任务数据量
        D_local_m_j_total = 0  # 当前RSU服务器的本地计算任务数据量

        for j in range(U_m_values[m]):
            # 获取每个用户的PR_m_j（本地执行概率）
            PR_m_j = PR_m_values[m][j]


            D_local_m_j = D_m_j * PR_m_j  # 本地任务的数据量
            D_rsu_m_j = D_m_j * (1 - PR_m_j)  # RSU计算任务的数据量

            # 累加本地和RSU计算数据量
            D_local_m_j_total += D_local_m_j
            D_rsu_m += D_rsu_m_j

            # 计算CV^1_m_j
            if D_local_m_j <= Qc:
                CV_1_m_j = 0
            else:
                CV_1_m_j = (D_local_m_j - Qc) / Qc

            # 计算CV^2_m
            if D_rsu_m_j <= Qd:
                CV_2_m = 0
            else:
                CV_2_m = (D_rsu_m_j - Qd) / Qd

            # 累加CV
            CV_total += CV_1_m_j + CV_2_m

        # 累加所有RSU服务器的总数据量
        D_local_total += D_local_m_j_total
        D_rsu_total += D_rsu_m

    return D_local_total, D_rsu_total, CV_total

# 修改后的主函数部分
def compute_average_energy_and_latency_with_CV(A, M, U_m_values, PR_m_values):
    total_energy = 0
    total_latency = 0
    local_data = 0  # 本地执行的数据量总和
    rsu_data = 0  # RSU服务器计算的数据量总和

    # 计算任务数据量 D_m_j，统一为250MB
    D_m_j = 250 * 0.9 * 0.65  # 任务数据量

    # 计算本地和RSU计算的数据量及CV
    D_local_total, D_rsu_total, CV_total = compute_local_and_rsu_data(U_m_values, PR_m_values, D_m_j)

    # 遍历所有RSU服务器及其用户
    for m in range(M):
        for j in range(U_m_values[m]):
            # 每个用户的子任务数量 C_m_j
            C_m_j = U_m_values[m] - 1  # 计算子任务的数量 (Cm,j = Um,j - 1)

            v_m_j = np.random.uniform(v_m_j_min, v_m_j_max)  # 车辆速度在50km/h到80km/h之间
            d_m_j = np.random.uniform(d_m_j_min, d_m_j_max)  # 每辆车与RSU服务器的距离在1m到500m之间

            # 计算通信速率 (基于香农定理)
            R_m_j = compute_communication_rate(p_m_j, d_m_j)  # 计算通信速率

            # 获取每个用户的PR_m_j（本地执行概率）
            PR_m_j = PR_m_values[m][j]

            # 计算每个用户的能量和时延
            e_local = compute_local_computation_energy(f_m_j, D_m_j, PR_m_j)  # 本地计算能耗
            e_trans = compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j)  # 传输能耗
            e_pre = compute_preprocess_energy(S_m_j, gamma_1, f_m_j, eta)  # 预处理能耗
            t_local = compute_local_computation_time(f_m_j, D_m_j, PR_m_j)  # 本地计算时延
            t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)  # 传输时延
            t_rsu = compute_rsu_computation_time(F_m, D_m_j, PR_m_j, alpha)  # RSU服务器计算时延
            t_edge = compute_edge_computation_time(F_m, D_m_j, PR_m_j, alpha, F_0)  # 边缘计算时延
            t_pre = compute_preprocess_time(S_m_j, gamma_1, f_m_j)  # 预处理时延

            # 累计能量和时延
            total_energy += e_local + e_trans + e_pre  # 加上预处理能耗
            total_latency += t_local + t_trans + t_rsu + t_edge + t_pre  # 将t_edge和t_pre加到时延中

            # 计算本地执行和RSU计算的数据量
            local_data += D_m_j * PR_m_j  # 本地执行的数据量
            rsu_data += D_m_j * (1 - PR_m_j)  # RSU服务器计算的数据量

    # 计算平均能量和时延
    avg_energy = total_energy / A
    avg_latency = total_latency / A

    return avg_energy, avg_latency, D_local_total, D_rsu_total, CV_total
