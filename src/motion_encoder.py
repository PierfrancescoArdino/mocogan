import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(self, dim_z_content, dim_z_motion):
        super(MotionEncoder, self).__init__()

        self.fc0 = nn.Linear(12288, 4096, bias=True)
        self.fc1 = nn.Linear(4096, 2048, bias=True)
        self.fc2 = nn.Linear(2048, 1024, bias=True)
        self.fc3 = nn.Linear(1024, 512, bias=True)
        self.fc4 = nn.Linear(512, 256, bias=True)

        self.fc5 = nn.Linear(256, dim_z_content, bias=True)
        self.fc6 = nn.Linear(256, dim_z_motion, bias=True)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        h1 = F.leaky_relu(self.fc4(x))
        return self.fc5(h1), self.fc6(h1)
