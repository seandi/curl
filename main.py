import numpy as np
import torch
import dmc2gym

from train import run_training_loop
from eval import run_evaluation
from utils.parse_args import parse_args
from utils.logger import Logger
from utils.video_recorder import VideoRecorder
from utils.utils import set_seed_everywhere, make_log_directory, FrameStack
from agents.make_agent import make_agent
from replay_buffer import ReplayBuffer


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1,1000000)
    set_seed_everywhere(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {0}".format(device))

    log_directory, video_dir, model_dir, buffer_dir = make_log_directory(
        domain=args.domain_name,
        task=args.task_name,
        seed=args.seed,
        agent_name=args.agent,
        work_dir=args.work_dir
    )

    L = Logger(log_directory, use_tb=args.save_tb)
    video_recorder = VideoRecorder(video_dir)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=False if args.agent == 'sac' else True,
        height=args.pre_transform_image_size if args.agent != 'sacae' else args.image_size,
        width=args.pre_transform_image_size if args.agent != 'sacae' else args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.agent != 'sac':
        env = FrameStack(env, k=args.frame_stack)

    action_shape = env.action_space.shape
    if args.agent == 'curl_sac':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size, args.pre_transform_image_size)
    elif args.agent == 'sacae':
        pre_aug_obs_shape = env.observation_space.shape
        obs_shape = pre_aug_obs_shape
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    env_steps, episodes = run_training_loop(
        env=env, agent=agent, agent_type=args.agent, agent_action_repeated=args.action_repeat, replay_buffer=replay_buffer,
        num_env_steps=args.env_steps_training, init_train_steps=args.init_train_steps,
        eval_freq=args.eval_every_n_env_steps, num_eval_episodes=args.num_eval_episodes,
        L=L, video_recorder=video_recorder, save_video=args.save_video, crop_size=args.image_size,
        backup_every_n_episodes=args.backup_every_n_episodes, model_dir=model_dir, buffer_dir=buffer_dir
    )

    # Evaluate after training always recording the final model
    L.log('eval/episode', episodes, env_steps)
    run_evaluation(
        env, agent, args.agent, args.num_eval_episodes,
        L, env_steps, video_recorder, record_video=True,
        crop_size=args.image_size
    )




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()