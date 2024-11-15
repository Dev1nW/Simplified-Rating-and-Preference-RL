import gymnasium as gym
from ppo_reward import PPO
import torch
from pref_reward_predictor import RewardModel

from dmc2gym import make
from common.env_util import make_vec_dmcontrol_env

import imageio
import os

env_name = 'cheetah'
task_name = 'run'

env = make_vec_dmcontrol_env(env_name, task_name, n_envs=16, seed=12345)

print('\n\nStarting Test for ' + env_name + '_' + task_name + '\n\n')

input_dim_a = env.action_space.shape[0]
input_dim_obs = env.observation_space.shape[0]

num_ratings = 6

reward_model = RewardModel(input_dim_obs, input_dim_a, mb_size=100, size_segment=50, max_size=100)

model = PPO(reward_model, "MlpPolicy", env, verbose=1, tensorboard_log="./tests/" + env_name + "_" + task_name + "_pref/")

model.learn(total_timesteps=4_000_000)

model.save("./tests/" + env_name + "_" + task_name + "_pref/")

reward_model.save("./tests/" + env_name + "_" + task_name + "_pref/", step=4000000)

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
video_filename = "./videos/" + env_name + "_" + task_name + "_pref_video.mp4"
imageio.mimsave(video_filename, frames, fps=30)  # Adjust fps as needed

print(f"Video saved as {video_filename}")