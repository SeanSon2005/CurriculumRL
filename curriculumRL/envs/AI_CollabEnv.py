import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/curriculumRL')

import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pygame
from action import Action

MAP_SIZE = 15
MAX_ITEM_WEIGHT = 1
MAX_ITEMS = 3
MAX_ROBOTS = 1
WINDOW_SIZE = 1000
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
	def __init__(self, render_mode = True):
		self.render_mode = render_mode
		self.action_space = spaces.Discrete(len(Action))
		# self.observation_space = spaces.Dict(
        #     {
		# 		# Stores the current vision of the robot
		# 		"frame": spaces.Box(low=-2, high=3, shape=(VISION_DISTANCE*2,VISION_DISTANCE*2), dtype= np.int16),
		# 		# stores the X and Y position of robot
        #         "ego_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype= np.int16),
		# 		# stores the states of objects held (0: No object, 1: Object in hand)
        #         "objects_held": spaces.Discrete(2, start=0),
		# 		# stores the number items in view of the robot (0 - MAX_ITEMS)
        #         "num_items": spaces.Discrete(MAX_ITEMS+1), 
		# 		# stores the info of the item that has been scanned
		# 		"item_distance": spaces.Box(low=-np.infty, high=np.infty, shape=(MAX_ITEMS,2), dtype= np.int16),

		# 		# "item_weight": spaces.MultiDiscrete(np.array([MAX_ITEM_WEIGHT]*MAX_ITEMS)),
		# 		# "item_isDangerous": spaces.MultiDiscrete(np.ones(MAX_ITEMS)),
		# 		# "item_danger_confidence": spaces.Box(low=0, high=1, shape=(MAX_ITEMS,), dtype=float),

		# 		# # stores the strength of the robot (0 - Maxiumum Possible Combined Strength)
        #         "strength": spaces.Discrete(MAX_ROBOTS + 2),
		# 		# # stores the current number of message received (0 - 99)
        #         "num_messages": spaces.Discrete(100)
        #     }
        # )
		self.observation_space = spaces.Box(low=-2, high=3, shape=(1,VISION_DISTANCE*2,VISION_DISTANCE*2), dtype= np.int16)
		self.window = None

		# Create Starter Map elements
		self.ego_location = np.array([MAP_SIZE//2,MAP_SIZE//2],dtype=np.int16)
		self.item_location = np.zeros((MAX_ITEMS,2), dtype=np.int16)
		self.item_weight = np.ones(MAX_ITEMS)
		self.item_isDangerous = np.ones(MAX_ITEMS)
		self.item_danger_confidence = np.ones(MAX_ITEMS, dtype=float)

		# Generate info for each item
		for i in range(MAX_ITEMS):
			r = np.random.randint(0,MAP_SIZE)
			c = np.random.randint(0,MAP_SIZE)
			self.item_location[i][0] = r
			self.item_location[i][1] = c
			self.item_isDangerous[i] = np.random.randint(0,2)

	def isInView(self, row, col):
		if row < 0 or col < 0:
			return False
		distance = np.sqrt((self.ego_location[0] - row)**2 + (self.ego_location[1] - col)**2)
		return distance < VISION_DISTANCE

	def get_obs(self):
		item_distance = np.zeros((MAX_ITEMS,2), dtype=np.int16)
		object_count = 0
		for i in range(MAX_ITEMS):
			if self.isInView(self.item_location[i][0],self.item_location[i][1]):
				x_dist = self.item_location[i][0] - self.ego_location[0]
				y_dist = self.item_location[i][1] - self.ego_location[1]
				item_distance[object_count][0] = x_dist
				item_distance[object_count][1] = y_dist
				object_count += 1
				
		# observation = {
		# 	"frame": self.generateInfoMap(),
		# 	"ego_location": self.ego_location,
		# 	"objects_held": self.objects_held,
		# 	"num_items": object_count,
		# 	"item_distance": item_distance,
		# 	"strength" : self.strength,
		# 	"num_messages": self.num_messages
        # }
		observation = self.generateInfoMap()
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
			# Reduce reward if left object without picking it up
			for i in range(MAX_ITEMS):
				if not self.item_isDangerous[i]:
					if self.item_location[i][0] == self.ego_location[0] and self.item_location[i][1] == self.ego_location[1]:
						reward -= 5
			# Perform move and reward calc if ran into wall
			if self.move(MOVE_DICT[action]):
				reward -= 2

		# Reward for being on object
		if not terminate:
			for i in range(MAX_ITEMS):
				object_location = self.item_location[i]
				if object_location[0] == self.ego_location[0] and object_location[1] == self.ego_location[1]:
					if not self.item_isDangerous[i]:
						reward += 5
				break

		# Pick Object Up
		if action > 7:
			for i in range(MAX_ITEMS):
				object_location = self.item_location[i]
				if object_location[0] == self.ego_location[0] and object_location[1] == self.ego_location[1] and self.objects_held == 0:
					# Denote that object has been picked up (-1 -1 is robot's "storage")
					object_location[0] = -1
					object_location[1] = -1
					self.objects_held = 1
					if self.item_isDangerous[i]:
						reward = -10
					else:
						reward = 20
						terminate = True
					break

		# # Drop Object
		# elif action == 9:
		# 	for object_location in self.item_location:
		# 		# Place picked up object back on grid
		# 		if object_location[0] == -1:
		# 			object_location[0] = self.ego_location[0]
		# 			object_location[1] = self.ego_location[1]

		if self.render_mode:
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
			
		canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
		canvas.fill((255, 255, 255))
		px_size = (WINDOW_SIZE / MAP_SIZE)

		# Draw Vision limit
		frame = self.generateOccupancyMap()
		for r in range(MAP_SIZE):
			for c in range(MAP_SIZE):
				if frame[r][c] == -2:
					pygame.draw.rect(canvas, (100,100,100), pygame.Rect(r*px_size,c*px_size,px_size,px_size))

		# Draw objects
		for i in range(MAX_ITEMS):
			r, c = self.item_location[i][1], self.item_location[i][0]
			if self.item_isDangerous[i]:
				pygame.draw.rect(canvas, (255,0,0), pygame.Rect(r*px_size,c*px_size,px_size,px_size))
			else:
				pygame.draw.rect(canvas, (0,255,0), pygame.Rect(r*px_size,c*px_size,px_size,px_size))

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
		for object in self.item_location:
			if map[object[0]][object[1]] == 0:
				map[object[0]][object[1]] = 2
		return map
	
	def generateInfoMap(self):
		# Empty map
		map = np.zeros((2,VISION_DISTANCE*2, VISION_DISTANCE*2),dtype=np.int16)
		# Wall Placements
		for r in range(VISION_DISTANCE*2):
			for c in range(VISION_DISTANCE*2):
				relative_r = r + self.ego_location[1] - VISION_DISTANCE
				relative_c = c + self.ego_location[0] - VISION_DISTANCE
				if relative_r >= MAP_SIZE or \
				  relative_c >= MAP_SIZE or \
				  relative_r < 0 or relative_c < 0:
					map[0,r,c] = -1
		# Box Placements
		for i in range(MAX_ITEMS):
			# get object location relative to robot's view
			relative_x = self.item_location[i][0] - self.ego_location[0] + VISION_DISTANCE
			relative_y = self.item_location[i][1] - self.ego_location[1] + VISION_DISTANCE
			# bound checks
			if relative_x >= VISION_DISTANCE*2 or \
				  relative_y >= VISION_DISTANCE*2 or \
				  relative_x < 0 or relative_y < 0:
				continue
			map[0, relative_y, relative_x] = 2
			# If item is not dangerous, represent as 1 otherwise -1
			if self.item_isDangerous[i]:
				map[1, relative_y, relative_x] = -1
			else:
				map[1, relative_y, relative_x] = 1
		return map
	
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Reset Robot Values
		self.ego_location = np.array([MAP_SIZE//2,MAP_SIZE//2],dtype=np.int16)
		self.objects_held = 0
		self.strength = 1
		self.num_messages = 0
		# Regenerate Objects
		for i in range(MAX_ITEMS):
			r = np.random.randint(0,MAP_SIZE)
			c = np.random.randint(0,MAP_SIZE)
			self.item_location[i][0] = r
			self.item_location[i][1] = c
			self.item_isDangerous[i] = np.random.randint(0,2)

		return self.get_obs(), {}

