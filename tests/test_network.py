import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.LeNet1989 import LeNet1989
from models.mod_CNN import ModCNN

def test_LeNet1989():
    net = LeNet1989()

    dummy_picture = torch.randn(1, 1, 28, 28)

    output = net.forward(dummy_picture)

    print(output.shape)


def test_ModCNN():
    net  = ModCNN()

    dummy_picture = torch.randn(1, 1, 28, 28)

    output = net.forward(dummy_picture)

    print(output.shape)


test_ModCNN()
