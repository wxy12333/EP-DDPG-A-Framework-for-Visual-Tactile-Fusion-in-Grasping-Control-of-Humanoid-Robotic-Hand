import time

import cv2
import numpy as np
from environment_zmq import Robot
from ddpg_agent import DDPGAgent
from ddpg_agent import ContrastiveAdaptiveReward
import random
import os
import torch
import keyboard
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self = Robot()
Agent = DDPGAgent(11, 1)
Reward = ContrastiveAdaptiveReward()

NUM_EPISODE = 4000
NUM_STEP = 30
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)


def getVisionPos():
    # plt.ion()  # 开启交互模式
    # fig, ax = plt.subplots()
    actor_losses, critic_losses, reward_list, policy_entropies, step, z_list = [], [], [], [], [], []
    out = False
    for episode_i in range(NUM_EPISODE):
        if episode_i < 330:
            self.z = episode_i // 30
        else:
            self.z = random.randint(0, 10)
        self.image()
        state = self.reset()  # state: n darray, others: dict
        episode_reward = 0
        episode_entropy = 0
        step_i = 0
        actor_loss_list, critic_loss_list = [], []
        # time.sleep(3)
        print(self.z)
        while True:
            self.image()
            # cv2.imshow('img', self.img)
            # cv2.imshow('depth', self.depth)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 q 退出
            #     break
            epsilon = np.interp(episode_i * NUM_STEP + step_i, [0, EPSILON_DECAY],
                                [EPSILON_START, EPSILON_END])  # interpolation
            random_sample = random.random()
            if random_sample <= epsilon:
                action = np.random.uniform(low=0, high=2, size=1)
            else:
                action = Agent.get_action(state)
            # action = agent.get_action(state)
            next_state, reward, done, terminate, done_episode = self.step(action)
            # print(reward)
            total_reward = Reward.calculate_contrastive_reward(reward, Agent.entropy_bonus, episode_i)
            total_reward = total_reward.item()
            error, _, _ = Agent.compute_error(state, action, total_reward, next_state, done_episode)
            Agent.replay_buffer.add_memo(state, action, total_reward, next_state, done_episode, error)
            state = next_state
            # print(state)
            # if reward < 0 or terminate:
            #     episode_reward = -episode_reward
            # else:
            step_i = step_i + 1
            episode_reward += total_reward
            # episode_reward = np.log(1 + episode_reward)
            if len(Agent.replay_buffer) >= 640:
                critic_loss, actor_loss, policy_entropy = Agent.update()
                actor_loss = actor_loss.cpu().detach().numpy()  # Move tensor to CPU and convert to numpy
                critic_loss = critic_loss.cpu().detach().numpy()
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                episode_entropy += policy_entropy
            # print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}, Critic_loss: {critic_loss}"
            #       f", Actor_loss: {actor_loss}")
            # print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}, Action: {action}")
            # print(next_state, reward, done)
            # print(episode_reward)
            if done and terminate:
                episode_reward -= 0.02 * step_i
                break
            if done:
                episode_reward -= 0.02 * step_i
                break
            if terminate:
                episode_reward -= 0.02 * step_i
                break
            # print(step_i)
            if step_i == NUM_STEP:
                episode_reward -= 0.02 * step_i
                break
            if keyboard.is_pressed('esc'):
                out = True
                break




        # REWARD_BUFFER[episode_i] = episode_reward / step_i
        # episode_reward = np.log(1 + episode_reward)

        actor_losses.append(np.mean(actor_loss_list) if actor_loss_list else 0.)
        critic_losses.append(np.mean(critic_loss_list) if critic_loss_list else 0.)
        reward_list.append(episode_reward)
        # reward_list_average.append(episode_reward / step_i)
        policy_entropies.append(episode_entropy / step_i)
        step.append(step_i)
        z_list.append(self.z)
        print(f"Episode: {episode_i + 1}"
              f"\tTotal Reward: {episode_reward}"
              f"\tActor Loss: {np.mean(actor_loss_list) if actor_loss_list else 0:.2e}"
              f"\tCritic Loss: {np.mean(critic_loss_list) if critic_loss_list else 0.:.2e}")
        if out:
            break

    return actor_losses, critic_losses, reward_list, policy_entropies, step, z_list


if __name__ == "__main__":
    actor_loss, critic_loss, reward, policy_entropy, step, z = getVisionPos()
    current_path = os.path.dirname(os.path.realpath(__file__))
    model = current_path + '/models/'
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Save models
    torch.save(Agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
    torch.save(Agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')

    # Close environment
    sim.stopSimulation()
    print('Program end')

    # Save the rewards as txt file
    np.savetxt(current_path + f'/reward/ddpg_reward_00.txt', reward)
    np.savetxt(current_path + f'/reward/ddpg_actor_loss_00.txt', actor_loss)
    np.savetxt(current_path + f'/reward/ddpg_critic_loss_00.txt', critic_loss)
    # np.savetxt(current_path + f'/reward/ddpg_step_1-11_sj3.txt', step)
    # policy_entropy_cpu = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
    #                       for item in policy_entropy]
    # policy_entropy_array = np.array(policy_entropy_cpu)
    # np.savetxt(current_path + f'/reward/ddpg_policy_entropy_1-11_sj3.txt', policy_entropy_array)
    # np.savetxt(current_path + f'/reward/ddpg_z_1-11_sj3.txt', z)

    # self.reset()
