import torch
import torch.nn as nn

class LoanApprovalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
