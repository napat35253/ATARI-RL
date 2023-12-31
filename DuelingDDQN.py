import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDDQN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DuelingDDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)
        
        linear_input_size = convw * convh * 64
        
        # Action Value Stream
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value Stream
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2, padding=0):
        next_w = ((w - kernel_size + 2*padding) // stride) + 1
        next_h = ((h - kernel_size + 2*padding) // stride) + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        # Action Value Calculation
        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)

        # State Value Calculation
        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)

        q = Vx + (Ax - Ax.mean(dim=1, keepdim=True))
        return q
