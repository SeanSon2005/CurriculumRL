import gymnasium as gym
env = gym.make('AI-Collab-v0')

for i in range(100):
    observation, info = env.reset(options={})    
    done = False
    while not done:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

env.close()