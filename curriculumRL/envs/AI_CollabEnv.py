import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/curriculumRL')

import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pygame
from action import Action

MAP_SIZE = 15
MAX_ITEM_WEIGHT = 1
MAX_ITEMS = 4
MIDDLE_SPAWN_BAN_RADIUS = MAP_SIZE // 4
MAX_OTHER_ROBOTS = 1
ROBOT_SPAWN_RADIUS = 5
WINDOW_SIZE = 1000
VISION_DISTANCE = 8
VISION_LENGTH = VISION_DISTANCE*2+1

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
		self.observation_space = spaces.Box(low=-2, high=3, shape=(1,VISION_LENGTH*2,VISION_LENGTH*2), dtype= np.int16)
		self.window = None

		# Create Starter Map elements
		self.ego_location = np.array([MAP_SIZE//2,MAP_SIZE//2],dtype=np.int16)
		self.robot_location = np.zeros((MAX_OTHER_ROBOTS,2), dtype=np.int16)
		self.item_location = np.zeros((MAX_ITEMS,2), dtype=np.int16)
		self.item_weight = np.ones(MAX_ITEMS)
		self.item_isDangerous = np.ones(MAX_ITEMS)
		self.item_danger_confidence = np.ones(MAX_ITEMS, dtype=float)

	def get_obs(self):
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
		# init reward as -1 (penalize more steps taken)
		reward = -1
		terminate = False

		# Update Other Robot positions
		# for robot in self.robot_location:
		# 	robot[0] += np.random.randint(-1,2)
		# 	robot[1] += np.random.randint(-1,2)

		# Move Robot
		if action < 8:
			# Reduce reward if left object without picking it up
			for i in range(MAX_ITEMS):
				if not self.item_isDangerous[i]:
					if self.item_location[i][0] == self.ego_location[0] and self.item_location[i][1] == self.ego_location[1]:
						reward = -20
			# Perform move and reward calc if ran into wall
			if self.move(MOVE_DICT[action]):
				terminate = True
				reward -= 5
			# Check if ran into robot
			for robot in self.robot_location:
				if robot[0] == self.ego_location[0] and robot[1] == self.ego_location[1]:
					terminate = True
					reward -= 30

		# Reward for being on object
		if not terminate:
			for i in range(MAX_ITEMS):
				object_location = self.item_location[i]
				if object_location[0] == self.ego_location[0] and object_location[1] == self.ego_location[1]:
					if not self.item_isDangerous[i]:
						reward = 25
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
					terminate = True
					if self.item_isDangerous[i]:
						reward = -40
					else:
						reward = 40
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

		# Draw Vision limit adn Robots
		frame = self.generateOccupancyMap()
		for r in range(MAP_SIZE):
			for c in range(MAP_SIZE):
				if frame[r][c] == -2:
					pygame.draw.rect(canvas, (100,100,100), pygame.Rect(c*px_size,r*px_size,px_size,px_size))
				elif frame[r][c] == 5:
					pygame.draw.rect(canvas, (0,0,0), pygame.Rect(c*px_size,r*px_size,px_size,px_size))

		# Draw objects
		for i in range(MAX_ITEMS):
			r, c = self.item_location[i][1], self.item_location[i][0]
			if self.item_isDangerous[i]:
				pygame.draw.rect(canvas, (255,0,0), pygame.Rect(r*px_size,c*px_size,px_size,px_size))
			else:
				pygame.draw.rect(canvas, (0,255,0), pygame.Rect(r*px_size,c*px_size,px_size,px_size))

		# Draw robot
		pygame.draw.circle(canvas, (0,0,255), 
					 ((self.ego_location[1]+0.5) * px_size, 
	   (self.ego_location[0]+0.5) * px_size), px_size/3)

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
		for x in range(MAP_SIZE):
			for y in range(MAP_SIZE):
				if abs(self.ego_location[0]-x) < VISION_DISTANCE and abs(self.ego_location[1]-y) < VISION_DISTANCE:
					map[x][y] = 0
		# Box Placements
		for object in self.item_location:
			map[object[0]][object[1]] = 2
		# Other Robot Placement
		for robot in self.robot_location:
			map[robot[0]][robot[1]] = 5
		return map
	
	def generateInfoMap(self):

		# Empty map
		map = np.zeros((2,VISION_LENGTH, VISION_LENGTH),dtype=np.int16)
		# Wall Placements
		for y in range(VISION_LENGTH):
			for x in range(VISION_LENGTH):
				relative_y = y - VISION_DISTANCE + self.ego_location[1] 
				relative_x = x - VISION_DISTANCE +  self.ego_location[0]
				if relative_y >= MAP_SIZE or \
				  relative_x >= MAP_SIZE or \
				  relative_y < 0 or relative_x < 0:
					map[0,x,y] = -1
		# Robot Placements
		for i in range(MAX_OTHER_ROBOTS):
			# get object location relative to robot's view
			relative_x = self.robot_location[i][0] - self.ego_location[0] + VISION_DISTANCE
			relative_y = self.robot_location[i][1] - self.ego_location[1] + VISION_DISTANCE
			# bound checks
			if relative_x >= VISION_LENGTH or \
				  relative_y >= VISION_LENGTH or \
				  relative_x < 0 or relative_y < 0:
				continue
			map[0, relative_x, relative_y] = 5
		# Box Placements
		for i in range(MAX_ITEMS):
			# get object location relative to robot's view
			relative_x = self.item_location[i][0] - self.ego_location[0] + VISION_DISTANCE
			relative_y = self.item_location[i][1] - self.ego_location[1] + VISION_DISTANCE
			# bound checks
			if relative_x >= VISION_LENGTH or \
				  relative_y >= VISION_LENGTH or \
				  relative_x < 0 or relative_y < 0:
				continue
			map[0, relative_x, relative_y] = 2
			# If item is not dangerous, represent as 1 otherwise -1
			if self.item_isDangerous[i]:
				map[1, relative_x, relative_y] = -1
			else:
				map[1, relative_x, relative_y] = 1
		return map
	
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Reset Robot Values
		self.ego_location = np.array([MAP_SIZE//2,MAP_SIZE//2],dtype=np.int16)
		self.objects_held = 0
		self.strength = 1
		self.num_messages = 0
		# Generate Other Robots
		for i in range(MAX_OTHER_ROBOTS):
			self.robot_location[i][0] = MAP_SIZE//2 + 3
			self.robot_location[i][1] = MAP_SIZE//2 + 3
		# Regenerate Objects
		for i in range(MAX_ITEMS):
			if np.random.random() > 0.5:
				r = np.random.randint(0,MAP_SIZE//2 - MIDDLE_SPAWN_BAN_RADIUS)
			else:
				r = np.random.randint(MAP_SIZE//2 + MIDDLE_SPAWN_BAN_RADIUS, MAP_SIZE)
			if np.random.random() > 0.5:
				c = np.random.randint(0,MAP_SIZE//2 - MIDDLE_SPAWN_BAN_RADIUS)
			else:
				c = np.random.randint(MAP_SIZE//2 + MIDDLE_SPAWN_BAN_RADIUS, MAP_SIZE)
			self.item_location[i][0] = r
			self.item_location[i][1] = c
			self.item_isDangerous[i] = np.random.randint(0,2)

		return self.get_obs(), {}

