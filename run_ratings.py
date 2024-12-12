import gymnasium as gym
from ppo_reward import PPO
import torch
from reward_predictor import RewardModel

from dmc2gym import make
from common.env_util import make_vec_dmcontrol_env
from common.vec_env import VecNormalize
from typing import Callable,  Union

import imageio
import os

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

seeds = [12345, 23451, 34512, 45123, 51234, 67890, 78906, 89067, 90678, 6789]
num_ratings_total = [2, 3, 4, 5, 6]
for seed in seeds:
    for num_ratings in num_ratings_total:
        # ENV PARAMS
        env_name = 'walker'
        task_name = 'walk'
        num_envs = 32

        # REWARD PREDICTOR PARAMS
        max_reward = 50
        seg_size = 50
        k = 30
        batch = 100
        reward_lr = 0.0003

        # POLICY PARAMS
        lr = 0.00005
        policy_batch = 64
        ent_coef = 0.0
        num_layers = 3
        num_hidden_dim = 256
        use_sde = True
        sde_freq = 4
        gae_lambda = 0.9
        clip_range = linear_schedule(0.4)
        update_every = 32000
        max_num_feedback = 1000
        n_steps = 500

        env = make_vec_dmcontrol_env(env_name, task_name, n_envs=num_envs, seed=seed)

        print('\n\nStarting Test for ' + env_name + '_' + task_name + '\n\n')

        input_dim_a = env.action_space.shape[0]
        input_dim_obs = env.observation_space.shape[0]

        reward_model = RewardModel(input_dim_obs, input_dim_a, mb_size=batch, size_segment=seg_size, teacher_num_ratings=num_ratings, max_reward=max_reward, k=k)

        # network arch
        net_arch = dict(
            pi=[num_hidden_dim] * num_layers,
            vf=[num_hidden_dim] * num_layers
        )
        policy_kwargs = dict(net_arch=net_arch)

        env= VecNormalize(env, norm_reward=False)

        model = PPO(reward_model, "MlpPolicy", env, learning_rate=lr, n_steps=n_steps ,seed=seed, batch_size=policy_batch, ent_coef=ent_coef, policy_kwargs=policy_kwargs, use_sde=use_sde, sde_sample_freq=sde_freq, gae_lambda=gae_lambda, clip_range=clip_range, update_every=update_every, num_ratings=num_ratings, num_feedback=max_num_feedback, segment_len=seg_size, verbose=1, tensorboard_log="./tests/" + env_name + "_" + task_name + "_" + str(num_ratings) +"_seed_" +str(seed) + "/")

        model.learn(total_timesteps=4_000_000)

        model.save("./tests/" + env_name + "_" + task_name + "_" + str(num_ratings) +"_seed_" +str(seed) + "/")

        reward_model.save("./tests/" + env_name + "_" + task_name + "_" + str(num_ratings) +"_seed_" +str(seed) + "/", step=4000000)

        # Initialize environment and variables
        obs = env.reset()
        dones = [0, 0]
        frames = []

        # Main loop
        while not int(sum(dones)):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

            # Render the environment and collect the frame
            img = env.env_method('render')[0]
            frames.append(img)

        # Ensure the 'videos' directory exists
        if not os.path.exists('./videos/'):
            os.makedirs('./videos/')

        # Save the frames as a video
        print('Saving Rollout...')
        video_filename = "./videos/" + env_name + "_" + task_name + "_" + str(num_ratings) + '_video_seed_' +str(seed) + '.mp4'
        imageio.mimsave(video_filename, frames, fps=30)  # Adjust fps as needed

        print(f"Video saved as {video_filename}")

        del model
        del reward_model