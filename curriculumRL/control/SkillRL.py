import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import math
import numpy as np
import random
from collections import deque, namedtuple
import json
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from itertools import count
import os

import gymnasium as gym
import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/')
from gymnasium.envs.registration import register
register(
     id="AI_CollabEnv-v0",
     entry_point="curriculumRL.envs:AI_CollabEnv",
     max_episode_steps=200,
)
from taskModel import TaskModel

# takes in the observation matrix
# returns which task model should focus on

SAVE_FOLDER = "curriculumRL/taskRuns/"
LOAD_PREV_WEIGHTS = True

index = len(os.listdir(SAVE_FOLDER))
WEIGHTS_FOLDER = SAVE_FOLDER+"run"+str(index+1)+"/"
os.mkdir(WEIGHTS_FOLDER)

class Action_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TaskModel()
    
    def forward(self, frame):        
        return self.model(frame)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
class DeepQControl:

    def __init__(self):
        # define device
        self.device = 'cuda'

        # define constants
        self.BATCH_SIZE = 128
        self.GAMMA = 0.9
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.episode_durations = []

        # define models
        self.policy_net = Action_Model().to(self.device)
        self.target_net = Action_Model().to(self.device)

        # load previous weights if requested
        if LOAD_PREV_WEIGHTS:
            print("loaded weights")
            self.policy_net.load_state_dict(torch.load("curriculumRL/runs/WEIGHTS/intermediate.pt"))

        # transfer weights from policy net to the target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # declare the optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # initialize replay memory
        self.memory_replay = ReplayMemory(10000)
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        plt.ion()
                
    def select_action(self, state):
        global steps_done
        # epsilon threshold to decide if action should use model or be randomly sampled
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

        
    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory_replay) < self.BATCH_SIZE:
            return
        # get batch from memory replay
        transitions = self.memory_replay.sample(self.BATCH_SIZE)

        # format batch
        batch = Transition(*zip(*transitions))

        # get mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

if __name__ == '__main__':
    #seed_everything(2024)

    env = gym.make('AI_CollabEnv-v0')

    device = 'cuda'
    steps_done = 0

    q_control = DeepQControl()

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # train
        for t in count():
            action = q_control.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            q_control.memory_replay.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            q_control.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = q_control.target_net.state_dict()
            policy_net_state_dict = q_control.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*q_control.TAU + target_net_state_dict[key]*(1-q_control.TAU)
            q_control.target_net.load_state_dict(target_net_state_dict)

            if done:
                q_control.episode_durations.append(t + 1)
                q_control.plot_durations()
                break
        # Save weights per episode
        file_name = WEIGHTS_FOLDER + "intermediate.pt"
        torch.save(q_control.policy_net.state_dict(), file_name)

    print('Complete')
    q_control.plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    # Save Ending Weights
    file_name = WEIGHTS_FOLDER + "best.pt"
    torch.save(q_control.policy_net.state_dict(), file_name)

