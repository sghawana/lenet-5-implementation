import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_channels, dtp, dvc):
        super(LeNet5, self).__init__()
        
        self.device = dvc
        self.dtype  = dtp
        self.C = num_channels
        
        # BxCx32x32 --> Bx6x28x28 --> Bx6x14x14
        self.conv1 = nn.Conv2d(in_channels=self.C, out_channels=6, kernel_size=5, stride=1, padding=0,
                               device=self.device, dtype=self.dtype)
        
        # Bx6x14x14 --> Bx16x10x10 --> Bx16x5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0,
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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    model1 = LeNet5(1, torch.float32, torch.device('cpu'))
    x = torch.randn(64, 1, 32, 32, device='cpu', dtype=torch.float32)
    y = model1(x)
    print(model1)
    print('y: ', y.shape)