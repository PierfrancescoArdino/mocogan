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
        x = x.reshape(-1, 12288)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        h1 = F.leaky_relu(self.fc4(x))
        return self.fc5(h1), self.fc6(h1)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MotionConvEncoder(nn.Module):
    def __init__(self, dim_z_content, dim_z_motion):
        super(MotionConvEncoder, self).__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True)
        )
        self.z_m = nn.Sequential(View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, dim_z_motion))

        self.z_c = nn.Sequential(View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, dim_z_content))

    def forward(self, x):
        x = self.enc(x)
        return self.z_c(x), self.z_m(x)