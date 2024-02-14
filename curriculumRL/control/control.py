import gymnasium as gym
import sys
import time
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/')
from gymnasium.envs.registration import register
register(
     id="AI_CollabEnv-v0",
     entry_point="curriculumRL.envs:AI_CollabEnv",
     max_episode_steps=1000,
)


env = gym.make('AI_CollabEnv-v0')

for i in range(1):
    observation, info = env.reset(options={})    
    done = False
    while not done:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.2)


env.close()