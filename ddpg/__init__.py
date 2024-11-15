from ddpg.ddpg import DDPG
from ddpg.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

__all__ = ["CnnPolicy", "MlpPolicy", "MultiInputPolicy", "DDPG"]
