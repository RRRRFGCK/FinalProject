import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# ===============================
# 1. 定义 Actor 网络（输入状态，输出动作）
# ===============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 使用 Sigmoid 保证输出在 [0, 1]，后面乘上最大动作值
        x = torch.sigmoid(self.fc3(x))
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

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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
# 4. 定义 DDPG 智能体
# ===============================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # 初始化演员和评论家及其目标网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)

        self.gamma = 0.99
        self.tau = 0.005  # 目标网络软更新系数

    # 【步骤1】输入：状态 → 输出：动作（决策结果）
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy()[0]
        # 将输出动作缩放到实际动作范围
        return action * self.max_action

    # 【步骤5】训练：从经验池采样，更新评论家和演员网络
    def train(self, batch_size=64):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # 计算目标 Q 值（目标 = reward + γ * Q(next_state, actor_target(next_state))）
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 演员更新：最大化当前状态下演员网络输出动作对应的 Q 值
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络参数
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # 辅助函数：存储当前经验
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


# ===============================
# 5. 定义一个简单环境（示例环境）
# ===============================
# 该环境仅作为示例，实际应用时请替换为 IoV 的卸载决策环境
class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        # 返回随机初始化状态
        return np.random.uniform(0, 1, size=self.state_dim)

    def step(self, action):
        self.current_step += 1
        # 【步骤3】环境执行动作，返回：下一状态、奖励、是否结束
        next_state = np.random.uniform(0, 1, size=self.state_dim)
        # 模拟：成本与动作有关，这里简单设定 cost = sum(action) + 随机噪声
        cost = np.sum(action) + np.random.uniform(0, 0.1)
        reward = -cost  # 奖励为负成本（目标是最小化成本）
        done = (self.current_step >= self.max_steps)
        return next_state, reward, done, {}

    def state_dimension(self):
        return self.state_dim

    def action_dimension(self):
        return self.action_dim


# ===============================
# 6. 主训练循环（整个 DRL 训练流程，分步骤说明）
# ===============================
def main():
    # 输入：定义环境状态和动作的维度
    state_dim = 10  # 示例状态维度
    action_dim = 2  # 示例动作维度（例如分别表示卸载比例）
    max_action = 1.0  # 动作范围 [0, 1]
    env = DummyEnv(state_dim, action_dim)
    agent = DDPGAgent(state_dim, action_dim, max_action)

    episodes = 200  # 总训练回合数
    for ep in range(episodes):
        # 【步骤1】重置环境，获得初始状态
        state = env.reset()
        episode_reward = 0
        while True:
            # 【步骤2】根据当前状态，智能体选择动作（输入 state → 输出 action）
            action = agent.select_action(state)
            # 【步骤3】将动作作用于环境，获得下一个状态、奖励和结束标志
            next_state, reward, done, _ = env.step(action)
            # 【步骤4】将 (state, action, reward, next_state, done) 经验存入经验回放缓冲区
            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            # 【步骤5】当经验足够时，进行网络训练更新
            if agent.replay_buffer.size > 64:
                agent.train(batch_size=64)
            if done:
                break
        print(f"Episode {ep + 1}: Total Reward = {episode_reward:.2f}")


if __name__ == '__main__':
    main()
