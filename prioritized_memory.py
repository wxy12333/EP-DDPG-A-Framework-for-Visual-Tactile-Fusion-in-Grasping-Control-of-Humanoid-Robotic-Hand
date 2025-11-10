import random
import numpy as np
from SumTree import SumTree


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.5
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add_memo(self, state, action, reward, next_state, done, error):
        if not self.validate_sample(state, action, reward, next_state, done):
            print("Invalid sample detected, skipping addition to replay buffer.")
            return
        """添加样本及其对应的优先级到 SumTree"""
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        priority = self._get_priority(error)
        self.tree.add(priority, (state, action, reward, next_state, done))

    def validate_sample(self, state, action, reward, next_state, done):
        """校验样本数据的完整性和合法性"""
        if state is None or next_state is None:
            return False
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            return False
        if np.isnan(state).any() or np.isnan(next_state).any():
            return False
        if not isinstance(action, np.ndarray) or np.isnan(action).any():
            return False
        if not isinstance(reward, (int, float)) or np.isnan(reward):
            return False
        if not isinstance(done, bool):
            return False
        return True

    def sample(self, batch_size):
        """从 SumTree 中采样 n 个样本"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1., self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if not isinstance(data, tuple) or len(data) != 5 or not self.validate_batch_sample(data):
                print(f"Invalid data sampled: {data}, skipping this sample.")
                continue  # 跳过无效样本
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * (sampling_probabilities + 1e-8), -self.beta)
        is_weight /= is_weight.max()  # 归一化
        # print(f"batch = {batch}")
        # if not all(isinstance(x, tuple) and len(x) == 5 for x in batch):
        #     raise ValueError(f"Invalid batch sampled: {batch}")
        # for i in range(len(batch)):
            # if not isinstance(batch[i], (list, tuple)):
            #     print(batch)
            #     batch[i] = [batch[i]]  # 如果是整数或其他类型，转换为列表
        state, action, reward, next_state, done = zip(*batch)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, is_weight, idxs

    def validate_batch_sample(self, sample):
        """校验单个采样样本的完整性"""
        state, action, reward, next_state, done = sample
        return self.validate_sample(state, action, reward, next_state, done)

    def update(self, idx, error):
        """更新样本的优先级"""
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
