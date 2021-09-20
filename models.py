import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, sinusoids):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 75),
            nn.ReLU(),
            nn.Linear(75, 25),
            nn.ReLU(),
            nn.Linear(25, sinusoids)
        )

    def forward(self, x):
        return self.model(x)