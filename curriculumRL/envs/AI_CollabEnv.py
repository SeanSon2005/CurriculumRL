import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/curriculumRL')

import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pygame
from action import Action

MAP_SIZE = 50
MAX_ITEM_WEIGHT = 2
MAX_ITEMS = 40
MAX_ROBOTS = 1
WINDOW_SIZE = 1000
FRAMERATE = 60
VISION_DISTANCE = 8

MOVE_DICT = {
	0:(0,-1),
	1:(0,1),
	2:(-1,0),
	3:(1,0),
	4:(1,-1),
	5:(-1,-1),
	6:(1,1),
	7:(-1,1)
}

class AI_CollabEnv(gym.Env):
	def __init__(self):
		self.action_space = spaces.Discrete(len(Action))
		self.observation_space = spaces.Dict(
            {
				# Stores the current vision of the robot
				# "frame": spaces.Box(low=-2, high=3, shape=(MAP_SIZE,MAP_SIZE), dtype= np.int16),
				# stores the X and Y position of robot
                "ego_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype= np.int16),
				# stores the states of objects held (0: No object, 1: Object in hand)
                "objects_held": spaces.Discrete(2, start=0),
				# stores the number items in view of the robot (0 - MAX_ITEMS)
                "num_items": spaces.Discrete(MAX_ITEMS+1), 
				# stores the info of the item that has been scanned
                "item_info": spaces.Dict(
                    {
                        # "weight": spaces.MultiDiscrete(np.array([MAX_ITEM_WEIGHT]*MAX_ITEMS)),
                        # "isDangerous": spaces.MultiDiscrete(np.ones(MAX_ITEMS)),
                        # "danger_confidence": spaces.Box(low=0, high=1, shape=(MAX_ITEMS,), dtype=float),
                        "location": spaces.Box(low=-np.infty, high=np.infty, shape=(MAX_ITEMS,2), dtype= np.int16),
                    }
                ),
				# # stores the locaton of the neighboring robot
                # "neighbors_info": spaces.Dict(
                #     {
                #         "location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype= np.int16)
                #     }
                # ),
				# # stores the strength of the robot (0 - Maxiumum Possible Combined Strength)
                "strength": spaces.Discrete(MAX_ROBOTS + 2),
				# # stores the current number of message received (0 - 99)
                "num_messages": spaces.Discrete(100)
            }
        )
		self.window = None
		self.clock = None

		# Create Starter Map elements
		self.ego_location = np.array([MAP_SIZE//2,MAP_SIZE//2],dtype=np.int16)
		self.objects = {
			# "weight": np.ones(MAX_ITEMS),
            # "isDangerous": np.ones(MAX_ITEMS),
            # "danger_confidence": np.ones(MAX_ITEMS, dtype=float),
            "location": np.zeros((MAX_ITEMS,2), dtype=np.int16)
		}
		for i in range(MAX_ITEMS):
			r = np.random.randint(0,MAP_SIZE)
			c = np.random.randint(0,MAP_SIZE)
			self.objects["location"][i][0] = r
			self.objects["location"][i][1] = c

	def isInView(self, row, col):
		if row < 0 or col < 0:
			return False
		distance = np.sqrt((self.ego_location[0] - row)**2 + (self.ego_location[1] - col)**2)
		return distance < VISION_DISTANCE

	def get_obs(self):
		observed_objects = {
			# "weight": np.ones(MAX_ITEMS),
            # "isDangerous": np.ones(MAX_ITEMS),
            # "danger_confidence": np.ones(MAX_ITEMS, dtype=float),
            "location": np.zeros((MAX_ITEMS,2), dtype=np.int16)
		}
		object_count = 0
		for i in range(MAX_ITEMS):
			if self.isInView(self.objects["location"][i][0],self.objects["location"][i][1]):
				observed_objects["location"][object_count] = self.objects["location"][i]
				object_count += 1
				
		observation = {
			"ego_location": self.ego_location,
			"objects_held": self.objects_held,
			"num_items": object_count,
			"item_info": observed_objects,
			"strength" : self.strength,
			"num_messages": self.num_messages
        }
		return observation
	
	def move(self, adjustment):
		x, y = adjustment
		if (self.ego_location[0] + x) == -1 or (self.ego_location[0] + x) == MAP_SIZE:
			return True
		if (self.ego_location[1] + y) == -1 or (self.ego_location[1] + y) == MAP_SIZE:
			return True
		self.ego_location[0] += x
		self.ego_location[1] += y
		return False
	
	def step(self, action):
		# init reward
		reward = 0
		terminate = False

		# Move Robot
		if action < 8:
			if self.move(MOVE_DICT[action]):
				terminate = True
				reward = -1

		# Pick Object Up
		elif action == 8:
			for object_location in self.objects['location']:
				if object_location[0] == self.ego_location[0] and object_location[1] == self.ego_location[1] and self.objects_held == 0:
					# Denote that object has been picked up (-1 -1 is robot's "storage")
					object_location[0] = -1
					object_location[1] = -1
					self.objects_held = 1
					reward = 1

		# Drop Object
		elif action == 9:
			for object_location in self.objects['location']:
				# Place picked up object back on grid
				if object_location[0] == -1:
					object_location[0] = self.ego_location[0]
					object_location[1] = self.ego_location[1]

		self.render()

		return self.get_obs(), reward, terminate, False, {}
		
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

		# Draw Boxes
		frame = self.generateOccupancyMap()
		for r in range(MAP_SIZE):
			for c in range(MAP_SIZE):
				if frame[r][c] == -2:
					pygame.draw.rect(canvas, (100,100,100), pygame.Rect(r*px_size,c*px_size,px_size,px_size))
				elif frame[r][c] == 2:
					pygame.draw.rect(canvas, (255,0,0), pygame.Rect(r*px_size,c*px_size,px_size,px_size))

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
	
	def generateOccupancyMap(self):
		# Empty Map
		map = np.zeros((MAP_SIZE,MAP_SIZE),dtype=np.int16)-2
		# Robot Placement
		map[self.ego_location[0]][self.ego_location[1]] = 1
		# Vision
		for r in range(MAP_SIZE):
			for c in range(MAP_SIZE):
				if self.isInView(r, c):
					map[r][c] = 0
		# Box Placements
		for object in self.objects["location"]:
			if map[object[0]][object[1]] == 0:
				map[object[0]][object[1]] = 2
		return map
	
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Obversations
		self.objects_held = 0
		self.strength = 1
		self.num_messages = 0
		return self.get_obs(), {}

