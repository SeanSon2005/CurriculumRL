import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from convModel import ConvModel
import sys
sys.path.insert(0, '/home/nesl/julian/CurriculumLearningTest/')
from curriculumRL.action import Action, Tasks

TASK_WEIGHTS_FOLDER = "curriculumRL/weights/"
FREEZE_TASK_MODEL = True

class TaskModel (nn.Module):
    def __init__(self, num_tasks=len(Tasks), 
                 num_actions=len(Action), 
                 num_inputs = 3) -> None:
        super().__init__()
        # Define model for choosing task
        self.task_decider = ConvModel(num_actions=num_tasks, 
                                      num_inputs=num_inputs)
        # Create models for each Task
        self.task_models = {}
        for task in Tasks:
            self.task_models[task.value] = ConvModel(num_actions=num_actions, 
                                                    num_inputs=num_inputs)
            weights_file = TASK_WEIGHTS_FOLDER + task.name + ".pt"
            self.task_models[task.value].load_state_dict(torch.load("curriculumRL/runs/WEIGHTS/intermediate.pt"))
            # Freeze task model
            if FREEZE_TASK_MODEL:
                for param in self.task_models[task.value].parameters():
                    param.requires_grad = False
    
    def forward(self, x) -> Tensor:
        # Decide which skill to utilize
        task_id = self.task_decider(x).max(1).indices.view(1, 1)
        # Run the skill
        out = self.task_models[task_id](x)
        return out