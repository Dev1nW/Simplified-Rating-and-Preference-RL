import gymnasium as gym
from gymnasium.envs.registration import registry
from gymnasium import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import registry
import matplotlib.pyplot as plt


def make(
        domain_name,
        task_name,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=True
):
    env_id = 'dmc_%s_%s_%s-v1' % (domain_name, task_name, seed)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # Shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    # Check if the environment is already registered
    if env_id not in gym.envs.registry:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        gym.register(
            id=env_id,
            entry_point='dmc2gym:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if isinstance(s, specs.Array):
            # Default bound for unbounded action space
            bound = 1.0 * np.ones(dim, dtype=np.float32)  # Assuming default bounds of -1 to 1
            return -bound, bound
        elif isinstance(s, specs.BoundedArray):
            zeros = np.zeros(dim, dtype=np.float32)
            minimum = np.where(np.isfinite(s.minimum), s.minimum, -1.0)  # Replace -inf with -1.0
            maximum = np.where(np.isfinite(s.maximum), s.maximum, 1.0)    # Replace inf with 1.0
            return minimum + zeros, maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True, 
        render_mode=None
    ):
        
        self.render_mode = render_mode
        
        self.metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        }
        
        if render_mode and render_mode not in self.metadata['render_modes']:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = 'rgb_array'

        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # Create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # True and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )
        self.action_space = self._norm_action_space  # Set the action space

        # Create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self.observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )
        self._state_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )
        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        
        # Ensure no infinite deltas
        true_delta = np.where(np.isfinite(true_delta), true_delta, 1.0)
        norm_delta = np.where(np.isfinite(norm_delta), norm_delta, 1.0)
        
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        
        return action

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        done = False  # Start with done=False

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()  # Check if the environment signals done
            if done:
                break

        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra = {'internal_state': self._env.physics.get_state().copy()}

        # Debugging print

        return obs, reward, done, False, extra

    def reset(self, *, seed=None, options=None):
        # Handle the seed by setting it in the environment reset
        time_step = self._env.reset()
        if seed is not None:
            np.random.seed(seed)
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
    
if __name__ == '__main__':
    env = make(domain_name='point_mass', task_name='easy', seed=1)

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        # Render the environment and display it using matplotlib
        img = env.render()
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show()  # Display the image
        
