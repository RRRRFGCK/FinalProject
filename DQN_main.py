import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# ===============================
# 1. 模拟函数与常量参数定义
# ===============================
# 常量参数
A_total = 500  # 总的移动用户数量
M = 4  # 总的RSU服务器数量
noise_power = -154  # 噪声功率 (dBm)
h = 4  # 信道衰减因子
B = 20e7  # 带宽 (20 MHz)
N = 20  # 信道数
p_m_j = 0.5  # 用户的传输功率 (W)
F_m = 5e10  # RSU服务器的计算能力 (10 GHz)
F_0 = 10e10
f_m_j = 1e9  # 车辆的本地计算能力 (1 GHz)

# 模拟的任务参数
v_m_j_min = 50  # 最小车辆速度 (50 km/h)
v_m_j_max = 80  # 最大车辆速度 (80 km/h)
d_m_j_min = 1  # 最小距离 (1 m)
d_m_j_max = 500  # 最大距离 (500 m)
gamma_1 = 0.1
gamma_2 = 0.65
eta = 30e-27  # 假设计算功率系数
alpha = 0.6
# 随机任务数据量（150MB - 300MB），用于后续模拟时各用户随机抽取
S_m_j_sample = np.random.uniform(150e6, 300e6)
D_m_j_sample = (1 - gamma_1) * gamma_2 * S_m_j_sample


def compute_preprocess_time(S_m_j, gamma_1, f_m_j):
    t_pre = gamma_1 * S_m_j / f_m_j
    return t_pre


def compute_preprocess_energy(S_m_j, gamma_1, f_m_j, eta):
    e_pre = eta * f_m_j ** 2 * gamma_1 * S_m_j
    return e_pre


def compute_communication_rate(p_m_j, d_m_j):
    noise_power_linear = 10 ** (noise_power / 10)  # dBm转换为线性单位
    R_m_j = B / N * np.log2(1 + (p_m_j * h) / (noise_power_linear * d_m_j ** 2))
    return R_m_j


def compute_transmission_time(R_m_j, D_m_j, PR_m_j):
    t_trans = (1 - PR_m_j) * D_m_j / R_m_j
    return t_trans


def compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_m_j):
    t_trans = compute_transmission_time(R_m_j, D_m_j, PR_m_j)
    e_trans = p_m_j * t_trans
    return e_trans


def compute_local_computation_time(f_m_j, D_m_j, PR_m_j):
    t_local = PR_m_j * D_m_j / f_m_j
    return t_local


def compute_local_computation_energy(f_m_j, D_m_j, PR_m_j):
    e_local = eta * f_m_j ** 2 * PR_m_j * D_m_j
    return e_local


def compute_rsu_computation_time(F_m, D_m_j, PR_m_j, alpha):
    t_rsu = (1 - PR_m_j) * D_m_j * (1 - alpha) / F_m
    return t_rsu


def compute_edge_computation_time(F_m, D_m_j, PR_m_j, alpha, F_0):
    t_edge = (1 - PR_m_j) * D_m_j * alpha / F_0
    return t_edge


def compute_local_and_rsu_data(U_m_values, PR_m_values, D_m_j):
    D_local_total = 0
    D_rsu_total = 0
    CV_total = 0
    Qc = 100e3
    Qd = 2e6
    Qe = 4e6
    for m in range(M):
        D_rsu_m = 0
        D_local_m_j_total = 0
        for j in range(U_m_values[m]):
            PR_val = PR_m_values[m][j]
            D_local_m_j = D_m_j * PR_val
            D_rsu_m_j = D_m_j * (1 - PR_val)
            D_local_m_j_total += D_local_m_j
            D_rsu_m += D_rsu_m_j
            # 计算违约程度（CV）
            CV_1_m_j = 0 if D_local_m_j <= Qc else (D_local_m_j - Qc) / Qc
            CV_2_m = 0 if D_rsu_m_j <= Qd else (D_rsu_m_j - Qd) / Qd
            CV_total += CV_1_m_j + CV_2_m
        D_local_total += D_local_m_j_total
        D_rsu_total += D_rsu_m
    return D_local_total, D_rsu_total, CV_total


def compute_average_energy_and_latency_with_CV(A, M, U_m_values, PR_m_values):
    total_energy = 0
    total_latency = 0
    for m in range(M):
        for j in range(U_m_values[m]):
            S_m_j = np.random.uniform(150e6, 300e6)  # 每个用户任务数据量随机
            D_m_j = (1 - gamma_1) * gamma_2 * S_m_j
            d_m_j = np.random.uniform(d_m_j_min, d_m_j_max)
            R_m_j = compute_communication_rate(p_m_j, d_m_j)
            PR_val = PR_m_values[m][j]
            e_local = compute_local_computation_energy(f_m_j, D_m_j, PR_val)
            e_trans = compute_transmission_energy(p_m_j, R_m_j, D_m_j, PR_val)
            e_pre = compute_preprocess_energy(S_m_j, gamma_1, f_m_j, eta)
            t_local = compute_local_computation_time(f_m_j, D_m_j, PR_val)
            t_trans = compute_transmission_time(R_m_j, D_m_j, PR_val)
            t_rsu = compute_rsu_computation_time(F_m, D_m_j, PR_val, alpha)
            t_edge = compute_edge_computation_time(F_m, D_m_j, PR_val, alpha, F_0)
            t_pre = compute_preprocess_time(S_m_j, gamma_1, f_m_j)
            total_energy += e_local + e_trans + e_pre
            total_latency += t_local + t_trans + t_rsu + t_edge + t_pre
    avg_energy = total_energy / A
    avg_latency = total_latency / A
    # 计算数据量和违约程度（仅供参考，不参与 reward 计算）
    D_local_total, D_rsu_total, CV_total = compute_local_and_rsu_data(U_m_values, PR_m_values, D_m_j)
    return avg_energy, avg_latency, D_local_total, D_rsu_total, CV_total


# ===============================
# 2. 定义 IoV 环境 (利用上述模拟函数)
# ===============================
class IoVEnv:
    def __init__(self, A=500, M=4, max_steps=50):
        self.A = A
        self.M = M
        # 初始每个RSU的用户数量（这里假设均分）
        self.U_m_values = [A // M] * M
        self.state_dim = M  # 状态：各RSU用户数归一化（除以125）
        self.action_dim = 5  # 离散动作数，对应本地执行概率：[0.0, 0.25, 0.5, 0.75, 1.0]
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.U_m_values = [125] * self.M
        state = np.array(self.U_m_values, dtype=np.float32) / 125.0
        return state

    def step(self, action):
        self.current_step += 1
        discrete_actions = [0.0, 0.25, 0.5, 0.75, 1.0]
        pr = discrete_actions[action]
        # 对每个RSU中所有用户采用相同的本地执行概率 pr
        PR_m_values = [[pr] * self.U_m_values[m] for m in range(self.M)]
        avg_energy, avg_latency, D_local_total, D_rsu_total, CV_total = compute_average_energy_and_latency_with_CV(
            self.A, self.M, self.U_m_values, PR_m_values
        )
        # 奖励函数：目标是最小化能耗 T 和时延 E，故用负的加权和
        reward = - (0.16 * avg_energy + 1.67 * avg_latency)
        # 模拟用户数量波动（可看作环境状态的随机变化）
        new_U_m_values = [max(100, 125 + np.random.randint(-10, 10)) for _ in range(self.M)]
        self.U_m_values = new_U_m_values
        next_state = np.array(self.U_m_values, dtype=np.float32) / 125.0
        done = (self.current_step >= self.max_steps)
        info = {
            'CV_total': CV_total,
            'D_local_total': D_local_total,
            'D_rsu_total': D_rsu_total,
            'avg_energy': avg_energy,
            'avg_latency': avg_latency
        }
        return next_state, reward, done, info


# ===============================
# 3. 定义 Q 网络 (DQN)
# ===============================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# ===============================
# 4. 定义 ReplayBuffer
# ===============================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor([1.0 if d else 0.0 for d in dones])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ===============================
# 5. 定义 DQN Agent（包含 ε-贪心策略与 Q 网络更新）
# ===============================
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 target_update_freq=50):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.total_steps = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            action_index = int(q_values.argmax().item())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        return action_index

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        q_values = self.q_network(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(state_action_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


# ===============================
# 6. 定义训练循环
# ===============================
def run_experiment(env, agent, replay_buffer, num_episodes=100, max_steps_per_episode=50, batch_size=64):
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            agent.update(replay_buffer, batch_size)
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    return episode_rewards


# ===============================
# 7. 定义评估函数（评估训练后在环境下的平均能耗和时延）
# ===============================
def evaluate_agent(agent, env, eval_steps=100):
    total_energy = 0.0
    total_latency = 0.0
    state = env.reset()
    for _ in range(eval_steps):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self_q = agent.q_network(state_t)
        action = int(q_vals.argmax().item())
        next_state, _, done, info = env.step(action)
        total_energy += info['avg_energy']
        total_latency += info['avg_latency']
        state = env.reset() if done else next_state
    avg_energy = total_energy / eval_steps
    avg_latency = total_latency / eval_steps
    return avg_energy, avg_latency


# ===============================
# 8. 主函数：训练并评估
# ===============================
def main():
    A_value = 500
    env = IoVEnv(A=A_value, M=4, max_steps=50)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ReplayBuffer(capacity=10000)
    num_episodes = 100  # 训练回合数，可根据需要调整
    run_experiment(env, agent, replay_buffer, num_episodes=num_episodes, max_steps_per_episode=50, batch_size=64)
    avg_energy, avg_latency = evaluate_agent(agent, env, eval_steps=100)
    print(f"Evaluation: Average Energy = {avg_energy:.2f}, Average Latency = {avg_latency:.2f}")

    # 可选：绘制每回合奖励曲线（如记录 episode_rewards 进行绘图）


if __name__ == '__main__':
    main()
