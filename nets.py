import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.lin1 = nn.Linear(73,256)
        self.lin2 = nn.Linear(256, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, 512)
        self.lin6 = nn.Linear(512, 256)
        self.linf = nn.Linear(256,73)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = self.linf(x)
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.lin1 = nn.Linear(73, 256)
        self.drop1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(256, 512)
        self.drop2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(512, 512)
        self.drop3 = nn.Dropout(0.5)
        self.lin4 = nn.Linear(512, 512)
        self.drop4 = nn.Dropout(0.5)
        self.lin5 = nn.Linear(512, 512)
        self.drop5 = nn.Dropout(0.5)
        self.lin6 = nn.Linear(512, 256)
        self.drop6 = nn.Dropout(0.5)
        self.linf = nn.Linear(256, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.lin1(x)))
        x = self.drop2(F.relu(self.lin2(x)))
        x = self.drop3(F.relu(self.lin3(x)))
        x = self.drop4(F.relu(self.lin4(x)))
        x = self.drop5(F.relu(self.lin5(x)))
        x = self.drop6(F.relu(self.lin6(x)))
        x = self.linf(x)
        return x


