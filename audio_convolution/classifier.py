import torch.nn as nn
import torch
import math

class MusicClassifier(nn.Module):
    def __init__(self, input_dimensions):
        super().__init__()
        
        # Adjusted padding to maintain spatial dimensions after convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Adjusted kernel_size and stride for MaxPool2d to prevent spatial dimensions from becoming too small
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate the spatial dimensions after the max-pooling layers
        self.fc1 = nn.Linear(in_features=(32 * math.ceil(input_dimensions / 4) * math.ceil(input_dimensions / 4)), out_features=128)

        self.fc2 = nn.Linear(128, 1)  # Output: 2 classes (music and non-music)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Max-pooling with adjusted kernel_size and stride
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Max-pooling with adjusted kernel_size and stride
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x