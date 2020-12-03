import numpy as np
import time

from utils.logger import Logger
from curl import center_crop_image
from utils.video_recorder import VideoRecorder


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def run_evaluation(env, agent, model, n_episodes,
                   logger: Logger, env_step,
                   video_recorder: VideoRecorder, record_video=False, video_name='',
                   crop_size=84,
                   sample_action_stochastically=False
                   ):

    PREPROCESS_OBSERVATION_TECHNIQUE = {
        'sac': lambda x: x,
        'sacae': lambda x: x,
        'curl_sac': lambda obs: center_crop_image(obs, crop_size)
    }

    episode_rewards = []
    prefix = 'stochastic_' if sample_action_stochastically else ''

    start = time.time()
    for episode in range(1, n_episodes+1):
        video_recorder.init(enabled=True if record_video and episode==1 else False)
        observation = env.reset()
        episode_reward = 0
        done = False

        while not done:
            observation = PREPROCESS_OBSERVATION_TECHNIQUE[model](observation)

            with eval_mode(agent):
                action_policy = agent.select_action if not sample_action_stochastically else agent.sample_action
                action = action_policy(observation)

                observation, reward, done, _ = env.step(action)
                episode_reward += reward
                video_recorder.record(env)

        episode_rewards.append(episode_reward)
        file_name = video_name if video_name != '' else '{0}_step_eval.mp4'.format(env_step)
        video_recorder.save(file_name)

    evaluation_time = time.time() - start
    reward_mean = np.mean(episode_rewards)
    reward_std = np.std(episode_rewards)
    reward_max = np.max(episode_rewards)

    logger.log('eval/' + prefix + 'eval_time', evaluation_time, env_step)
    logger.log('eval/' + prefix + 'mean_episode_reward', reward_mean, env_step)
    logger.log('eval/' + prefix + 'best_episode_reward', reward_max, env_step)
    logger.log('eval/' + prefix + 'std_episode_reward', reward_std, env_step)
    logger.log('eval/' + prefix + 'episode_reward', reward_mean, env_step)
    logger.dump(env_step)
