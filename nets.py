import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.lin1 = nn.Linear(73, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.lin4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.5)
        self.lin5 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.5)
        self.lin6 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.5)
        self.linf = nn.Linear(256, 73)

    def forward(self, x):
        x = self.drop1(self.bn1(F.relu(self.lin1(x))))
        x = self.drop2(self.bn2(F.relu(self.lin2(x))))
        x = self.drop3(self.bn3(F.relu(self.lin3(x))))
        x = self.drop4(self.bn4(F.relu(self.lin4(x))))
        x = self.drop5(self.bn5(F.relu(self.lin5(x))))
        x = self.drop6(self.bn6(F.relu(self.lin6(x))))
        x = self.linf(x)
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.lin1 = nn.Linear(73, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.lin4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.5)
        self.lin5 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.5)
        self.lin6 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.5)
        self.linf = nn.Linear(256, 1)

    def forward(self, x):
        x = self.drop1(self.bn1(F.relu(self.lin1(x))))
        x = self.drop2(self.bn2(F.relu(self.lin2(x))))
        x = self.drop3(self.bn3(F.relu(self.lin3(x))))
        x = self.drop4(self.bn4(F.relu(self.lin4(x))))
        x = self.drop5(self.bn5(F.relu(self.lin5(x))))
        x = self.drop6(self.bn6(F.relu(self.lin6(x))))
        x = self.linf(x)
        return x


