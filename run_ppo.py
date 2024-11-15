import gymnasium as gym
from common.env_util import make_vec_dmcontrol_env
from ppo.ppo import PPO
import dmc2gym
import matplotlib.pyplot as plt
import imageio
import os

env_name = 'cheetah'
task_name = 'run'

# Parallel environments
env = make_vec_dmcontrol_env(env_name, task_name, n_envs=16, seed=12345)

print('\n\nStarting Test for ' + env_name + '_' + task_name + '\n\n')

input_dim_a = env.action_space.shape[0]
input_dim_obs = env.observation_space.shape[0]

#env = dmc2gym.make(domain_name='walker', task_name='walk', seed=1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tests/" + env_name + "_" + task_name + "_ppo/")
model.learn(total_timesteps=4_000_000)
model.save("./tests/" + env_name + "_" + task_name + "_ppo/")

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
video_filename = "./videos/" + env_name + "_" + task_name + '_ppo_video.mp4'
imageio.mimsave(video_filename, frames, fps=30)  # Adjust fps as needed

print(f"Video saved as {video_filename}")