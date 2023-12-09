import gymnasium as gym
from gymnasium import spaces
from action import Tasks
import numpy as np 

MAP_SIZE = 30
MAX_ITEM_WEIGHT = 2

MAX_ITEMS = 5
MAX_ROBOTS = 1

class CustomEnv(gym.Env):
	def __init__(self):
		self.action_space = spaces.Dict(
            {
				"subtask": spaces.Discrete(len(Tasks)),
            }
        )
		self.observation_space = spaces.Dict(
            {
                "ego_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
                "objects_held": spaces.Discrete(3, start=-1),
				
                "num_items": spaces.Discrete(MAX_ITEMS),
                "item_info": spaces.Dict(
                    {
                        "weight": spaces.Discrete(MAX_ITEM_WEIGHT),
                        "isDangerous": spaces.Discrete(1),
                        "danger_confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                        "location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
                    }
                ),

                "num_neighbors": spaces.Discrete(MAX_ROBOTS),
                "neighbors_info": spaces.Dict(
                    {
                        "location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16)
                    }
                ),
				
                "strength": spaces.Discrete(len(self.map_config['all_robots']) + 2),
                "num_messages": spaces.Discrete(100)
            }
        )
	def get_obs(self):
		return {
			"ego_location": self.ego_location
        }
	def step(self, action):
		pass
	def render(self):
		pass
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		
		self.ego_location = (MAP_SIZE//2,MAP_SIZE//2)
		observation = self.get_obs()
		
		return observation


