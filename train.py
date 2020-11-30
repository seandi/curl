import time

from utils import utils
from replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.video_recorder import VideoRecorder
from eval import run_evaluation, eval_mode


def run_training_loop(env, agent, agent_type, agent_action_repeated, replay_buffer: ReplayBuffer,
                      num_env_steps, init_train_steps, eval_freq, num_eval_episodes,
                      L: Logger, video_recorder: VideoRecorder, save_video, crop_size,
                      backup_every_n_episodes, model_dir, buffer_dir
                      ):
    # ----------------- TRAINING LOOP ----------------- #
    episode, episode_step, episode_reward, done = 0, 0, 0, True
    env_step, train_step = 0, 0
    training_start_time = time.time()

    while env_step < num_env_steps:
        train_step += 1
        env_step += agent_action_repeated

        if done:
            episode_start_time = time.time()
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        if train_step < init_train_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(observation)

        if train_step >= init_train_steps:
            experience_tuple = replay_buffer.sample_proprio() if agent_type == 'sac' else replay_buffer.sample_cpc()
            agent.update(*experience_tuple, env_step=env_step, train_step=train_step, logger=L)

        next_observation, reward, done, _ = env.step(action)

        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(
            obs=observation,
            action=action,
            reward=reward,
            next_obs=next_observation,
            done=done_bool
        )

        episode_reward += reward
        episode_step += 1
        observation = next_observation

        # At the end of each episode log training stat and create model backup
        if done:
            L.log('train/duration', time.time() - episode_start_time, env_step)
            L.log('train/episode_reward', episode_reward, env_step)
            L.log('train/episode', episode, env_step)
            L.dump(env_step)

            if backup_every_n_episodes > 0 and episode % backup_every_n_episodes == 0:
                agent.save(model_dir=model_dir, env_step=env_step)
                replay_buffer.save(save_dir=buffer_dir)

        if env_step % eval_freq == 0:
            L.log('eval/episode', episode, env_step)
            run_evaluation(
                env, agent, agent_type, num_eval_episodes,
                L, env_step, video_recorder, record_video=save_video,
                crop_size=crop_size
            )

    return env_step, episode
