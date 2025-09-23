import torch
from models.network import LeNet1989

def test_LeNet1989():
    net = LeNet1989()

    dummy_picture = torch.randn(1, 1, 28, 28)

    output = net.forward(dummy_picture)

    print(output.shape)


test_LeNet1989()
