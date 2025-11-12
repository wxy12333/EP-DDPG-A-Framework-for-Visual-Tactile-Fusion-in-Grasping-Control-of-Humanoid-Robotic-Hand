import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import math
import torch.nn.functional as F
import prioritized_memory as Memory

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 100000
batch_size = 64


class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.1, decay_rate=0.9):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.decay_rate = decay_rate  # 引入衰减率
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state

    def decay_sigma(self):
        self.sigma = max(0.3, self.sigma * self.decay_rate)


# 在动作上添加 OU 噪声
ou_noise = OUNoise(action_dimension=1, mu=0.0, theta=0.15, sigma=0.2, decay_rate=0.9)  # 初始化OU噪声


def add_noise(action):
    noise = ou_noise.noise()
    noisy_action = action.cpu().detach().numpy() + noise
    noisy_action = np.clip(noisy_action, 0.5, 5)  # 动作范围限制
    noisy_action = torch.tensor(noisy_action, dtype=torch.float32).to(device)
    return noisy_action


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.min_action = 0.5  # 动作最小值
        self.max_action = 5

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)  # 转换为浮点型的张量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 动作选择在[0, 3]之间
        x = (x + 1) / 2 * (self.max_action - self.min_action) + self.min_action
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)  # 转换为浮点型的张量
        x = x.unsqueeze(0) if x.dim() == 1 else x  # 转换为 [1, 11]
        a = a.unsqueeze(0) if a.dim() == 1 else a  # 转换为 [1, 1]

        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ContrastiveAdaptiveReward:
    def __init__(self):
        self.base_reward_weight = 1.0
        self.entropy_weight = 0.1
        self.initial_entropy_weight = 0.05
        self.decay_rate = 0.01
        self.contrastive_decay_rate = 0.005

    def calculate_contrastive_reward(self, reward, entropy_bonus, episode):
        self.base_reward_weight = 1.0  # 或者根据任务需求稍微调大
        self.entropy_weight = max(0.0, self.initial_entropy_weight * math.exp(-self.decay_rate * episode))
        total_reward = (self.base_reward_weight * reward +
                        self.entropy_weight * entropy_bonus
                        )
        return total_reward


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, initial_alpha=0.5, decay_rate=0.1):
        self.actor = Actor(state_dim, action_dim).to(device)  # move nn to device
        self.actor_target = Actor(state_dim, action_dim).to(device)  # same structure as actor
        self.actor_target.load_state_dict(self.actor.state_dict())  # copy the current nn's weights of actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)  # retrieves the parameters

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 正则化熵
        self.alpha = initial_alpha  # 初始熵系数
        self.decay_rate = decay_rate  # 衰减率
        self.entropy_bonus = 0

        self.error = None

        self.replay_buffer = Memory.Memory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # unsqueeze(0) add a dimension from (3,) to (1,3)
        action = self.actor(state)
        action = add_noise(action)
        return action.detach().cpu().numpy()[0]  

    def update_alpha(self, episode):
        # 动态调整熵系数，随着episode的增加逐渐衰减
        self.alpha = max(0.1, self.alpha * math.exp(-self.decay_rate * episode))

    def compute_error(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        # actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor([rewards]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor([dones]).unsqueeze(1).to(device)

        # Critic update
        next_actions = self.actor_target(next_states)
        # print(next_states.shape, next_actions.detach().shape)
        target_Q = self.critic_target(next_states,
                                      next_actions.detach())  # .detach() means the gradient won't be backpropagated to the actor
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        # target_Q = target_Q.view_as(current_Q)  # 将 target_Q 调整为 current_Q 的形状
        # critic_loss = nn.MSELoss()(current_Q, target_Q.detach())  # nn.MSELoss() means Mean Squared Error 均方差误差的方法
        td_errors = (target_Q - current_Q).detach().cpu().numpy()  # 计算 TD 误差
        td_error = target_Q - current_Q
        return abs(td_errors), current_Q, target_Q

    def update(self):
        if len(self.replay_buffer) < batch_size:
            return # skip the update if the replay buffer is not filled enough

        (states, actions, rewards, next_states, dones, is_weights, indices) = self.replay_buffer.sample(batch_size)  #PER
        # states = torch.FloatTensor(states).to(device)

        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(device)

        td_errors, current_Q, target_Q = self.compute_error(states, actions, rewards, next_states, dones)
        td_errors = td_errors.mean(axis=1)
        critic_loss = (is_weights * nn.SmoothL1Loss(reduction='none')(current_Q, target_Q)).mean()
        # critic_loss = nn.SmoothL1Loss()(current_Q, target_Q.detach()) #Smooth L1的损失值更新方法
        self.critic_optimizer.zero_grad()  # .zero_grad() clears old gradients from the last step
        critic_loss.backward()  # .backward() computes the derivative of the loss
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()  # .step() is to update the parameters

        # 更新优先级，基于 TD 误差
        # print(td_errors)
        new_priorities = np.abs(td_errors) + self.replay_buffer.e  # e 为小的正数，避免优先级为 0
        self.replay_buffer.update(indices, new_priorities)

        # 熵正则化
        policy_actions = self.actor(states)
        policy_entropy = -torch.mean(torch.sum(policy_actions * torch.log(policy_actions + 1e-10), dim=1))
        self.entropy_bonus = self.alpha * policy_entropy

        # Actor update
        actor_loss = (self.critic(states, self.actor(states)).mean() + self.entropy_bonus)  # .mean() is to calculate the mean of the tensor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        # print(critic_loss, actor_loss)

        return critic_loss, actor_loss, policy_entropy

