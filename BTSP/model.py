import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(0, 0)
        self.classifier = nn.Linear(512, 512)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

model = U_Net()
print(model)
