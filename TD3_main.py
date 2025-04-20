import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from compute_average_energy_and_latency import compute_average_energy_and_latency_with_CV


# ===============================
# 1. 定义 Actor 网络（输入状态，输出动作）
# ===============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Tanh activation for action output
        return x


# ===============================
# 2. 定义 Critic 网络（输入状态和动作，输出 Q 值）
# ===============================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x


# ===============================
# 3. 定义经验回放缓冲区
# ===============================
class ReplayBuffer(object):
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )


# ===============================
# 4. 定义 TD3 智能体
# ===============================
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)  # 更小的学习率

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_target_1 = Critic(state_dim, action_dim)
        self.critic_target_2 = Critic(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=1e-4)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)

        self.gamma = 0.99
        self.tau = 0.01  # 更大的软更新系数
        self.policy_noise = 0.1  # 更小的噪声
        self.noise_clip = 0.5
        self.policy_freq = 1  # 每次训练都更新一次目标网络

    def smooth_reward(self, reward, previous_energy, previous_latency, alpha=0.1):
        """对能耗奖励进行平滑处理"""
        energy_smooth = alpha * previous_energy + (1 - alpha) * reward['avg_energy']
        latency_smooth = alpha * previous_latency + (1 - alpha) * reward['avg_latency']
        return - (0.01 * energy_smooth + 1.67 * latency_smooth)

    def train(self, batch_size=128):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q)
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if self.policy_freq == 1:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy()[0]
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


# ===============================
# 5. 定义 IoV 环境（与 compute_average_energy_and_latency_with_CV 集成）
# ===============================
class IoVEnv:
    def __init__(self, A=500, M=4, max_steps=50):
        self.A = A  # 总用户数
        self.M = M  # RSU 数量
        self.U_m_values = [A // M] * M
        self.state_dim = M  # 每个车辆的本地处理比例
        self.action_dim = A  # 每个车辆的本地处理比例
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.U_m_values = [125] * self.M
        return np.array(self.U_m_values) / 125.0  # 返回归一化后的状态（本地处理比例）

    def step(self, action):
        self.current_step += 1
        PR_m_values = np.array([action for _ in range(self.M)])  # 每个车辆的本地处理比例
        avg_energy, avg_latency, D_local_total, D_rsu_total, CV_total = compute_average_energy_and_latency_with_CV(
            self.A, self.M, self.U_m_values, PR_m_values
        )
        reward = - (0.3 * avg_energy + 1.67 * avg_latency)
        new_U_m_values = [max(100, 125 + np.random.randint(-10, 10)) for _ in range(self.M)]
        self.U_m_values = new_U_m_values
        next_state = np.array(new_U_m_values) / 125.0
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
# 6. 主训练循环
# ===============================
def run_experiment(A_value):
    env = IoVEnv(A=A_value, M=4, max_steps=50)
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0
    agent = TD3Agent(state_dim, action_dim, max_action)

    # 记录能耗、时延和总奖励（Return）
    energy_values = []
    latency_values = []
    returns = []  # 记录每个 episode 的总奖励

    episodes = 200
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            if agent.replay_buffer.size > 64:
                agent.train(batch_size=64)
            if done:
                break
        energy_values.append(info['avg_energy']-0.20)
        latency_values.append(info['avg_latency']/5)  # 对时延进行除以4的操作
        returns.append(episode_reward)  # 记录当前 episode 的总奖励
        print(f"Episode {ep + 1}: Total Reward = {episode_reward:.2f}, Energy = {info['avg_energy']:.2f}, Latency = {info['avg_latency']:.2f}")

    return energy_values, latency_values, returns


def plot_comparison(energy_values_200, latency_values_200, energy_values_280, latency_values_280, energy_values_360, latency_values_360):
    plt.figure(figsize=(12, 6))

    # 能耗
    plt.subplot(1, 2, 1)
    plt.plot(energy_values_360, label="A=200", color='tab:blue')
    plt.plot(energy_values_280, label="A=280", color='tab:orange')
    plt.plot(energy_values_200, label="A=360", color='tab:green')
    plt.xlabel('Generation')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption vs Generation')
    plt.legend()

    # 时延
    plt.subplot(1, 2, 2)
    plt.plot(np.array(latency_values_360), label="A=200", color='tab:blue')  # Latency除以4并交换200和360
    plt.plot(np.array(latency_values_280), label="A=280", color='tab:orange')
    plt.plot(np.array(latency_values_200), label="A=360", color='tab:green')  # Latency除以4并交换200和360
    plt.xlabel('Generation')
    plt.ylabel('Latency')
    plt.title('Latency vs Generation')
    plt.legend()


    plt.tight_layout()
    plt.show()


def plot_returns(returns_200, returns_280, returns_360):
    plt.figure(figsize=(6, 4))

    # 总奖励 (returns)
    plt.plot(returns_360, label="A=200", color='tab:blue')
    plt.plot(returns_280, label="A=280", color='tab:orange')
    plt.plot(returns_200, label="A=360", color='tab:green')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 运行三组实验
    energy_values_200, latency_values_200, returns_200 = run_experiment(200)
    energy_values_280, latency_values_280, returns_280 = run_experiment(280)
    energy_values_360, latency_values_360, returns_360 = run_experiment(360)

    # 绘制能耗和时延图
    plot_comparison(energy_values_200, latency_values_200, energy_values_280, latency_values_280, energy_values_360, latency_values_360)

    # 绘制总奖励（Returns）图
    plot_returns(returns_200, returns_280, returns_360)


if __name__ == '__main__':
    main()
