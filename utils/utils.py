import torch
import gym
import numpy as np
import random, time, os
from collections import deque


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def make_log_directory(domain, task, seed, agent_name, batch_size,  work_dir):
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H-%M", ts)
    env_name = domain + '-' + task + '-' + agent_name
    exp_name = env_name + '-' + ts + '-b' + str(batch_size) + '-s' + str(seed)
    result_dir = work_dir + '/' + exp_name

    make_dir(result_dir)
    video_dir = make_dir(os.path.join(result_dir, 'video'))
    model_dir = make_dir(os.path.join(result_dir, 'model'))
    buffer_dir = make_dir(os.path.join(result_dir, 'buffer'))

    return result_dir, video_dir, model_dir, buffer_dir


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)