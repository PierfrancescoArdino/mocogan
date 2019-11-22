import torch
import torch.nn as nn


class ContentMotionEncoder(nn.Module):
    def __init__(self):
        super(ContentMotionEncoder, self).__init__()

        self.fc0 = nn.Linear(4096, 2048, bias=True)
        self.fc1 = nn.Linear(2048, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)

        self.fc4 = nn.Linear(256, 2, bias=True)
        self.fc5 = nn.Linear(256, 2, bias=True)

    def forward(self, x):
        x = x.view(-1, 4096)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        h1 = F.leaky_relu(self.fc3(x))
        return self.fc4(h1), self.fc5(h1)
