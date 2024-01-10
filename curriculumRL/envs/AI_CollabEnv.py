import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/curriculumRL')

import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pygame
from action import Tasks

MAP_SIZE = 30
MAX_ITEM_WEIGHT = 2
MAX_ITEMS = 5
WINDOW_SIZE = 500
FRAMERATE = 60

def generateOccMap():
	map = np.zeros((MAP_SIZE,MAP_SIZE),dtype=np.uint8)
	return map

class AI_CollabEnv(gym.Env):
	def __init__(self):
		self.action_space = spaces.Dict(
            {
				"subtask": spaces.Discrete(len(Tasks)),
            }
        )
		self.observation_space = spaces.Dict(
            {
				# stores the X and Y position of robot
                "ego_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
				# stores the states of objects held (-1: No object, 0: Left Hand, 1: Right Hand)
                "objects_held": spaces.Discrete(3, start=-1),
				# stores the number items in view of the robot (0 - MAX_ITEMS)
                "num_items": spaces.Discrete(MAX_ITEMS+1), 
				# stores the info of the item that has been scanned
                "item_info": spaces.Dict(
                    {
                        "weight": spaces.Discrete(MAX_ITEM_WEIGHT),
                        "isDangerous": spaces.Discrete(1),
                        "danger_confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                        "location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
                    }
                ),
				# stores the locaton of the neighboring robot
                "neighbors_info": spaces.Dict(
                    {
                        "location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16)
                    }
                ),
				# stores the strength of the robot (0 - Maxiumum Possible Combined Strength)
                "strength": spaces.Discrete(len(self.map_config['all_robots']) + 2),
				# stores the current number of message received (0 - 99)
                "num_messages": spaces.Discrete(100)
            }
        )

	def get_obs(self):
		return {
			"ego_location": self.ego_location,
			"objects_held": self.objects_held,
			"num_items": self.num_items,
			"item_info": self.item_info,
			"neighbors_info": self.neighbors_info,
			"strength" : self.strength,
			"num_messages": self.num_messages
        }
	
	def step(self, action):
		self.render()
		
	def render(self):
		# if first time, initialize pygame and clock
		if self.window is None:
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode(
                (WINDOW_SIZE, WINDOW_SIZE)
            )
		if self.clock is None:
			self.clock = pygame.time.Clock()
			
		canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
		canvas.fill((255, 255, 255))
		px_size = (WINDOW_SIZE / MAP_SIZE)
		
        # Draw robot
		pygame.draw.circle(canvas, (0,0,255), (self.ego_location+0.5) * px_size, px_size/3)

        # Draw map grid lines
		for i in range(MAP_SIZE+1):
			pygame.draw.line(canvas, 0, (0, px_size * i),
					(WINDOW_SIZE, px_size * i), width=3)
			pygame.draw.line(canvas, 0, (px_size * i, 0),
					(px_size * i, WINDOW_SIZE), width=3)
			
		self.window.blit(canvas, canvas.get_rect())
		pygame.event.pump()
		pygame.display.update()
		self.clock.tick(FRAMERATE)
    
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.ego_location = (MAP_SIZE//2,MAP_SIZE//2)
		self.objects_held = -1
		self.num_items = 0
		self.item_info = None
		self.neighbors_info = None
		self.strength = 1
		self.num_messages = 0
		observation = self.get_obs()
		return observation

