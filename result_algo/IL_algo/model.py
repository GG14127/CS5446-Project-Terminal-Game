import torch
import torch.nn as nn
import json
import numpy as np

class ILAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ILAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action = torch.sigmoid(self.fc4(x))
        return action