import torch
from torch import nn

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        # convolution part
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.BatchNorm2d(32),  # batchnorm1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(64),  # batchnorm2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(64),  # batchnorm3
            nn.PReLU(), # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2), # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1), # conv4
            nn.BatchNorm2d(128),  # batchnorm4
            nn.PReLU() # prelu4
        )
        self.lin1 = nn.Linear(128*2*2*196, 256)  # lin1
        self.bn1 = nn.BatchNorm1d(256)  # batchnorm4
        self.prelu5 = nn.PReLU()  # prelu5
        # lanbmark localization
        self.lin2 = nn.Linear(256, 68*2)

    def forward(self, x) -> torch.Tensor:
        # conv part
        out = x.clone()
        out = self.pre_layer(out)
        # actvation map to vectors
        out = out.view(out.size(0), -1)
        out = self.bn1(self.lin1(out))
        out = self.prelu5(out)
        # landmark loaclization
        landmark = self.lin2(out).view(-1, 68, 2)
        return landmark