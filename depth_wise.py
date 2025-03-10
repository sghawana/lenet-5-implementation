import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_d(nn.Module):
    def __init__(self, num_channels, dtp, dvc, depth_wise=False):
        super(LeNet5_d, self).__init__()
        
        self.device = dvc
        self.dtype  = dtp
        self.C = num_channels
        self.depth_wise = depth_wise
        
        self.group = 1 if not self.depth_wise else self.C
        
        # BxCx32x32 --> Bx6x28x28 --> Bx6x14x14
        self.depthwise1 = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=5, stride=1, padding=0, groups=self.C,
                                    device=self.device, dtype=self.dtype)
        self.pointwise1 = nn.Conv2d(in_channels=self.C, out_channels=6, kernel_size=1, stride=1,
                                    device=self.device, dtype=self.dtype)
        
        # Bx6x14x14 --> Bx16x10x10 --> Bx16x5x5
        self.depthwise2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1, padding=0, groups=6,
                                    device=self.device, dtype=self.dtype)
        self.pointwise2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1,
                                    device=self.device, dtype=self.dtype)
        
        # Bx16x5x5 --> Bx120
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120,
                             device=self.device, dtype=self.dtype)
    
        # Bx120 --> Bx84
        self.fc2 = nn.Linear(in_features=120, out_features=84,
                             device=self.device, dtype=self.dtype)
        
        # Bx84 --> Bx10
        self.fc3 = nn.Linear(in_features=84, out_features=10,
                             device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = F.relu(self.pointwise1(self.depthwise1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.pointwise2(self.depthwise2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    model1 = LeNet5_d(1, torch.float32, torch.device('cpu'))
    model2 = LeNet5_d(1, torch.float32, torch.device('cpu'), depth_wise=True)
    x = torch.randn(64, 1, 32, 32, device='cpu', dtype=torch.float32)
    y = model1(x)
    z = model2(x)
    print(model1)
    print(model2)
    print('y: ', y.shape)
    print('z: ', z.shape)
    

