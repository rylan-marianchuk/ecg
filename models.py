import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, sinusoids):
        """
        Simple pytorch Multi-Layered feed forward Neural Network
        :param input_size: input dimension
        :param sinusoids: output dimension, predicting some coefficient for each sinusoid in the signal
        """
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
