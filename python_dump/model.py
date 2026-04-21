import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        #  Layer 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        #  Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        #  Layer 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #  Layer 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        # Skip Connection 1
        self.skip1 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1, stride=4),
            nn.BatchNorm2d(128)
        )

        # Skip Connection 2
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        #  Dense Layers 
        self.drop1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256 * 4 * 4, 256) 
        self.drop2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        #  Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        identity1 = x

        #  Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        #  Block 3 + Skip 1
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        skip_out1 = self.skip1(identity1)
        x = self.relu(x + skip_out1)

        #  Block 4 + Skip 2
        identity2 = x

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool3(x)

        skip_out2 = self.skip2(identity2)
        x = self.relu(x + skip_out2)

        #  Classifier 
        x = self.drop1(x)

        x = x.view(x.size(0), -1)   # (B, 4096)

        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)

        return x