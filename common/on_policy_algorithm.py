import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from common.base_class import BaseAlgorithm
from common.buffers import DictRolloutBuffer, RolloutBuffer, EntReplayBuffer
from common.callbacks import BaseCallback
from common.policies import ActorCriticPolicy
from common.type_aliases import GymEnv, MaybeCallback, Schedule
from common.utils import obs_as_tensor, safe_mean
from common.vec_env import VecEnv
from scipy.stats import pearsonr
import torch

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        reward_model,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        max_feedback = 2000,
        update_every = 100000,
        num_ratings = 2,
        segment_len = 50,
        max_ep_len = 1000,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}


        self.reward_model = reward_model
        self.update_every = update_every
        self.num_ratings = num_ratings
        self.segment_len = segment_len
        self.max_ep_len = max_ep_len
        self.max_feedback = max_feedback

        self.traj_obsact = None
        self.traj_reward = None
        self.traj_frames_1 = []
        self.traj_frames_2 = []
        self.predicted_reward_list = []
        self.real_reward_list = []
        self.best_reward = None
        self.best_corr = None
        self.correlation_reward = 0

        self.num_updates = 0 
        self.total_feedback = 0
        self.label_feedback = 0

        self.unsuper_step = 32000

        self.reward_batch = self.reward_model.mb_size
        self.segment_len = segment_len
        self.first_reward_train = 0

        self.re_update = 100

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space, 
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.unsuper_buffer = EntReplayBuffer(
                self.unsuper_step+100,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=False,
            )
        
        self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=self.device)

        self.policy = self.policy_class( 
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def learn_reward(self):
        # Randomly select segments
        if self.first_reward_train == 0:
            labeled_queries = self.reward_model.uniform_sampling()
            self.first_reward_train += 1
        # Either randomly select segments or perform disagreement sampling
        else:
            labeled_queries = self.reward_model.disagreement_sampling()
        
        self.total_feedback += self.reward_model.mb_size
        self.label_feedback += labeled_queries
        
        # Train Reward Predictor
        for epoch in range(self.re_update):
            train_acc = self.reward_model.train_soft_reward()
            total_acc = np.mean(train_acc)
            
            if total_acc > 0.97:
                break;
                
        print("Reward function is updated!! ACC: " + str(total_acc))
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            obsact = np.concatenate((self._last_obs, clipped_actions), axis=-1) 
            obsact = np.expand_dims(obsact, axis=1) 

            new_obs, rewards, done, infos = env.step(clipped_actions)
            batch_reward = rewards.reshape(-1,1,1)
    
            pred_reward = self.reward_model.r_hat_batch(obsact)
            pred_reward = pred_reward.reshape(-1)
            
            
            self.predicted_reward_list.append(pred_reward)
            self.real_reward_list.append(rewards)

            # Create list of all state action pairs within a trajectory 
            if self.traj_obsact is None:
                self.traj_obsact = obsact
                self.traj_reward = batch_reward
            else:
                self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)

            self.num_updates += env.num_envs
            self.num_timesteps += env.num_envs

            num_dones = int(sum(done))
            
            if num_dones > 0:
                self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)

                flat_list1 = np.array([item for sublist in self.real_reward_list for item in sublist])
                flat_list2 = np.array([item for sublist in  self.predicted_reward_list for item in sublist])

                self.predicted_reward_list = []
                self.real_reward_list = []  
                self.correlation_reward = np.corrcoef(flat_list1, flat_list2,rowvar=False)
                
                # Reset trajectories
                self.traj_obsact, self.traj_reward, self.traj_frames_1, self.traj_frames_2 = None, None, [], []
                
                
                # Train Reward
                if self.num_updates >= self.update_every and self.total_feedback < self.max_feedback:
                    print('Training Reward Predictor')
                    self.num_updates = 0
                    self.learn_reward()

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, done)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(done):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    pred_reward[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                pred_reward,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = done

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def collect_rollouts_unsupervised(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        replay_buffer: EntReplayBuffer,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            obsact = np.concatenate((self._last_obs, clipped_actions), axis=-1) 
            obsact = np.expand_dims(obsact, axis=1) 

            obs_origin = env.get_original_obs()
            replay_buffer.add_obs(obs_origin) 
            state_entropy = replay_buffer.compute_state_entropy(obs_origin)
            self.s_ent_stats.update(state_entropy)
            norm_state_entropy = state_entropy / self.s_ent_stats.std

            new_obs, rewards, done, infos = env.step(clipped_actions)
            next_obs_origin = env.get_original_obs()            
            batch_reward = rewards.reshape(-1,1,1)
            
            pred_reward = norm_state_entropy.reshape(-1).data.cpu().numpy()

            # Create list of all state action pairs within a trajectory 
            if self.traj_obsact is None:
                self.traj_obsact = obsact
                self.traj_reward = batch_reward
            else:
                self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)

            self.num_updates += env.num_envs
            self.num_timesteps += env.num_envs

            num_dones = int(sum(done))
            
            if num_dones > 0:
                self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)
                
                # Reset trajectories
                self.traj_obsact, self.traj_reward, self.traj_frames_1, self.traj_frames_2 = None, None, [], []

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, done)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(done):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    pred_reward[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                pred_reward,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = done

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """

        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        self.logger.record("reward/total_feed", self.total_feedback)
        self.logger.record("reward/labeled_feedback", self.label_feedback)
        self.logger.record("reward/reward_correlation", self.correlation_reward[0][1])
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            if self.num_timesteps < self.unsuper_step:
                continue_training = self.collect_rollouts_unsupervised(
                    self.env, callback, self.rollout_buffer, 
                    n_rollout_steps=self.n_steps, replay_buffer=self.unsuper_buffer)
                flag=False
            else:

                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
                flag=True

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0 and flag:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []