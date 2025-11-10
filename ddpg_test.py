import cv2
import numpy as np
import json
from os import path
from environment_zmq import Robot
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import torch
import torch.nn as nn
import os
import struct
import keyboard
from ddpg_agent import DDPGAgent
from ddpg_agent import ContrastiveAdaptiveReward


client = RemoteAPIClient()
sim = client.require('sim')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

self = Robot()
Agent = DDPGAgent(11, 1)
Reward = ContrastiveAdaptiveReward()

state_dim = 11
action_dim = 1


class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2, decay_rate=0.9):
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
    noise = ou_noise.noise()  # 获取 OU 噪声
    noisy_action = action.detach().cpu().numpy() + noise
    # 噪声衰减（例如每个 episode 结束后调用此函数）
    ou_noise.decay_sigma()
    return torch.tensor(np.clip(noisy_action, 0.9, 2.4)).to(device)  # 将动作限制在范围内


# Actor Network
class Actor(nn.Module):
    def __init__(self, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)  # 转换为浮点型的张量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 动作选择在[0, 3]之间
        x = (x + 1) * 1.5
        return x


# Test phase
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ddpg_actor.pth'

actor = Actor().to(device)
actor.load_state_dict(torch.load(actor_path))

transform_params_path = 'transform_params.json'
if path.exists(transform_params_path):
    with open(transform_params_path, 'r') as openfile:
        transform_params = json.load(openfile)

NUM_EPISODE = 2
NUM_STEP = 50
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)



def getVisionPos():
    actor_losses, critic_losses, reward_list = [], [], []
    out = False
    success_num = 0
    for episode_i in range(NUM_EPISODE):
        # self.object_reset()
        self.image()
        state = self.reset()  # state: n darray, others: dict
        episode_reward = 0
        step_i = 0
        actor_loss_list, critic_loss_list = [], []
        # time.sleep(3)
        while True:
            self.image()

            action = add_noise(actor(torch.FloatTensor(state).unsqueeze(0).to(device))).detach().cpu().numpy()[0]
            next_state, reward, done, terminate = self.step(action)
            total_reward = Reward.calculate_contrastive_reward(reward, state, next_state, Agent.entropy_bonus,
                                                               episode_i)
            # print(total_reward)
            total_reward = total_reward.item()
            state = next_state
            step_i = step_i + 1
            if step_i == NUM_STEP:
                arm_pos = sim.getObjectPosition(self.end_effector_handle, sim.handle_world)
                arm_height = arm_pos[2]
                if arm_height < 0.13:
                    episode_reward += total_reward
                    episode_reward = episode_reward * 0.5
                else:
                    episode_reward += total_reward
            else:
                episode_reward += total_reward
            # print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}, Critic_loss: {critic_loss}"
            #       f", Actor_loss: {actor_loss}")
            print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
            # print(next_state, reward, done)
            if done and terminate:
                episode_reward = episode_reward * (1 + 1.5 * np.cos(np.pi * step_i / (2 * NUM_STEP)))
                success_num += 1
                break
            if done:
                break
            if terminate:
                break
            # print(step_i)
            if step_i == NUM_STEP:
                break
            if keyboard.is_pressed('esc'):
                out = True
                break
        REWARD_BUFFER[episode_i] = episode_reward

        if out:
            break
        self.object_reset()
    return success_num

if __name__ == "__main__":
    success = getVisionPos()
    print(success)
