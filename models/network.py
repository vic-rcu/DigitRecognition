
import torch
import torch.nn as nn

class LeNet1989(nn.Module):

    def __init__(self):
        # TODO: implement constructor with all layers

        super().__init__()
        self.H1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.H2 = nn.AvgPool2d(kernel_size=2)
        # not fully connected in paper, therefore splitting third layer into multiple modules that are connected as described in the paper
        # connection of feature map 1 & 2
        self.H3_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.H3_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.H3_12 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5)

        # connection of feature map 3 & 4
        self.H3_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.H3_4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.H3_34 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5)


        self.H4 = nn.AvgPool2d(kernel_size=2)

        # flattening the last layer and fully connecting to the output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(192, 10)

        # activation function
        self.tanh = nn.Tanh()


    def forward(self, x):
        # H1 forward pass
        x = self.tanh(self.H1(x))

        # H2 forward pass
        x = self.tanh(self.H2(x))

        # split the feauture maps 
        x1, x2, x3, x4 = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]

        # H3 forward pass
        h3_1 = self.tanh(self.H3_1(x1))
        h3_2 = self.tanh(self.H3_2(x2))
        h3_12 = self.tanh(self.H3_12(torch.cat([x1, x2], dim=1)))

        h3_3 = self.tanh(self.H3_1(x3))
        h3_4 = self.tanh(self.H3_2(x4))
        h3_34 = self.tanh(self.H3_12(torch.cat([x3, x4], dim=1)))

        # concat all feature maps in H3
        h3_out = torch.cat([h3_1, h3_2, h3_12, h3_3, h3_4, h3_34], dim=1)

        h4 = self.H4(h3_out)
        
    
        # flatten and pass through fully connected last layer
        x = self.flatten(h4)

        return self.fc(x)



