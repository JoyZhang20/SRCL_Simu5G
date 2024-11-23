import os
import pandas as pd
import gym
import torch.nn.functional as F
import torch
import numpy as np
import random
import collections
import matplotlib.pyplot as plt


# 绘制图像
def draw(x, x_label, y, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列（类似于列表list）

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        # transitions = [(s1, a1, r1, n_s1, d1), (s2, a2,...), ...]
        # *transitions = (s1, a1, r1, n_s1, d1) (s2, a2,...) ...
        # zip(*transitions) = (s1, s2, ...) (a1, a2,...) (r1, r2, ...)...
        states, actions, rewards, next_states, dones = zip(*transitions)
        # 直接从list转变变为tensor会很慢，因此先将list转为np.array()
        return np.array(states), np.array(actions), rewards, np.array(next_states), dones

    def size(self):  # 当前buffer中数据的数量
        return len(self.buffer)


# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # self.action_abound = action_abound  # 环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.sigmoid(x)
        return F.tanh(x)  # 小于0则取0，大于等于0则取1


# 价值网络
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)  # 输入的是状态-动作对
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 最终输出该状态-动作对的价值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc2(x)


# 深度确定性策略
class DDPG:
    '''
    sigma: 高斯噪声标准差
    tau: 软更新参数
    gamma: 折扣因子
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 使目标策略网络与策略网络有相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        # 使目标价值网络与价值网络有相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 使用Adam优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_dim = action_dim
        self.sigma = sigma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.device = device

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        offload_action = []
        action = self.actor(state).detach()  # 得到一维张量
        # 给动作添加噪声，增加探索
        action += self.sigma * np.abs(np.random.randn(self.action_dim))
        for j in range(self.action_dim):
            if action[j].item() <= 0:
                offload_action.append(0)
            elif action[j].item() > 0:
                offload_action.append(1)
        return action, offload_action

    # 软更新参数
    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param * self.tau)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        # 训练价值网络
        y_critic = self.critic(torch.cat([states, actions], dim=1))  # torch.cat([A,B],dim=1)将A和B按指定维度dim进行拼合，dim=1表示横向操作列（横向拼接）
        y_critic_target = rewards + self.gamma * self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1)) * (1 - dones)
        critic_loss = F.mse_loss(y_critic, y_critic_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 训练策略网络

        # 注意，pytorch只能对标量求导，不能对张量求导，所以这里取平均值
        actor_loss = - torch.mean(self.critic(torch.cat([states, self.actor(states)], dim=1)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新网络参数
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)


def load_agent(state_dim, action_dim):
    actor_lr = 5e-5
    critic_lr = 5e-4
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    agent = DDPG(state_dim, hidden_dim, action_dim, sigma, actor_lr, critic_lr, tau, gamma, device)
    agent.critic = torch.load('./networkModel/critic_net.pth')
    agent.actor = torch.load(f'./networkModel/actor_net.pth')
    return agent


def generate_state(slot_num):
    state = [random.randint(0, 40) for _ in range(10)]  # 10个ue的count
    state.append(slot_num)
    return state


def generate_decision(epoch, state_dim, action_dim):
    agent = load_agent(state_dim, action_dim)
    slot_sum = 15
    states = []
    decisions = []
    train_decisions = []
    rewards = [0 for _ in range(15)]
    dones = [False for _ in range(14)]  # 前14个时隙的done设为False
    dones.append(True)  # 最后一个时隙的done设为True

    for slot_num in range(slot_sum):
        state = generate_state(slot_num)
        decision = agent.choose_action(state)
        train_decision = decision[0].tolist()
        current_decision = decision[1]
        states.append(state)
        decisions.append(current_decision)
        train_decisions.append(train_decision)
    # 写入 decisions.txt 文件
    with open('D:/Simu5G/decision.txt', 'w') as f:
        for decision in decisions:
            f.write(','.join(map(str, decision)) + '\n')

    # 写入 states 和 decisions 到 Excel 文件
    states = [','.join(map(str, state)) for state in states]
    decisions = [','.join(map(str, decision)) for decision in decisions]
    train_decisions = [','.join(map(str, decision)) for decision in train_decisions]
    data = zip(states, decisions, train_decisions, rewards, dones)
    df = pd.DataFrame(data, columns=['state', 'decision', 'action', 'reward', 'done'])
    df.to_excel(f'./data/states_decisions_{epoch}.xlsx', engine='openpyxl', index=False)


def train_DDPG(epoch, state_dim, action_dim):
    agent = load_agent(state_dim, action_dim)
    training_data = pd.read_excel(f'./data/states_decisions_{epoch}.xlsx', engine='openpyxl')
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    for i in range(training_data.shape[0]):
        states.append(list(map(int, training_data.loc[i, 'state'].split(','))))
        actions.append(list(map(float, training_data.loc[i, 'action'].split(','))))
        rewards.append(float(training_data.loc[i, 'reward']))
        dones.append(bool(training_data.loc[i, 'done']))
        if i < training_data.shape[0] - 1:
            next_states.append(list(map(int, training_data.loc[i + 1, 'state'].split(','))))
        else:
            next_states.append(list(map(int, training_data.loc[0, 'state'].split(','))))
    agent.update(states, actions, rewards, next_states, dones)
    net_save_dir_path = r'.\networkModel'
    os.makedirs(net_save_dir_path, exist_ok=True)
    torch.save(agent.actor, os.path.join(net_save_dir_path, 'actor_net.pth'))
    torch.save(agent.critic, os.path.join(net_save_dir_path, 'critic_net.pth'))





