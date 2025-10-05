import torch
import torch.nn as nn



class ModCNN(nn.Module):

    def __init__(self):

        super().__init__()
        # used activations
        self.relu = nn.ReLU()

        # ------------------------- Conv. Layers -------------------------

        # output 32x28x28 
        self.C1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # output 64x28x28
        self.C2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # output 64x14x14
        self.P2 = nn.MaxPool2d(kernel_size=2)
        # output 128x14x14        
        self.C3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # output 128x7x7
        self.P3 = nn.MaxPool2d(kernel_size=2)

        # ------------------------- Linear Layers -------------------------

        # output 6272
        self.flatten = nn.Flatten()
        
        # output 128
        self.FC1 = nn.Linear(in_features=6272, out_features=128)

        self.dropout = nn.Dropout(p=0.5)

        # output: classification layer
        self.FC2 = nn.Linear(in_features=128, out_features=10)
        




    def forward(self, x):

        # ------- convolution layer 1 -------
        x = self.relu(self.C1(x))

        # ------- convolution layer 2 -------
        x = self.relu(self.C2(x))
        x = self.P2(x)

        # ------- convolution layer 3 -------
        x = self.relu(self.C3(x))
        x = self.P3(x)

        # ------- flatten layer -------
        x = self.flatten(x)

        # ------- fully connected layers -------
        x = self.relu(self.FC1(x))

        x = self.dropout(x)

        x = self.FC2(x)

        return x

    


