import torch
from torch import nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
    
print("feed_forward.py loaded")