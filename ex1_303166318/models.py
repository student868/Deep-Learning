from torch import nn

ACTIVATION = nn.functional.relu
POOL = nn.AvgPool2d
DROPOUT = 0.2


class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Lenet5 Original'
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.S2 = POOL(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.S4 = POOL(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.S5 = nn.Linear(in_features=400, out_features=120)
        self.S6 = nn.Linear(in_features=120, out_features=84)
        self.S7 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.C1(x)
        x = ACTIVATION(x)
        x = self.S2(x)
        x = self.C3(x)
        x = ACTIVATION(x)
        x = self.S4(x)
        x = self.flatten(x)
        x = self.S5(x)
        x = ACTIVATION(x)
        x = self.S6(x)
        x = ACTIVATION(x)
        x = self.S7(x)
        x = self.softmax(x)
        return x


class Lenet5Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Lenet5 Dropout'
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.S2 = POOL(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.S4 = POOL(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.S5 = nn.Linear(in_features=400, out_features=120)
        self.D5 = nn.Dropout(DROPOUT)
        self.S6 = nn.Linear(in_features=120, out_features=84)
        self.D6 = nn.Dropout(DROPOUT)
        self.S7 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.C1(x)
        x = ACTIVATION(x)
        x = self.S2(x)
        x = self.C3(x)
        x = ACTIVATION(x)
        x = self.S4(x)
        x = self.flatten(x)
        x = self.S5(x)
        x = self.D5(x)
        x = ACTIVATION(x)
        x = self.S6(x)
        x = self.D6(x)
        x = ACTIVATION(x)
        x = self.S7(x)
        x = self.softmax(x)
        return x


class Lenet5BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Lenet5 Batch Normalization'
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.B1 = nn.BatchNorm2d(6)
        self.S2 = POOL(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.B3 = nn.BatchNorm2d(16)
        self.S4 = POOL(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.S5 = nn.Linear(in_features=400, out_features=120)
        self.B5 = nn.BatchNorm1d(120)
        self.S6 = nn.Linear(in_features=120, out_features=84)
        self.B6 = nn.BatchNorm1d(84)
        self.S7 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = ACTIVATION(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.B3(x)
        x = ACTIVATION(x)
        x = self.S4(x)
        x = self.flatten(x)
        x = self.S5(x)
        x = self.B5(x)
        x = ACTIVATION(x)
        x = self.S6(x)
        x = self.B6(x)
        x = ACTIVATION(x)
        x = self.S7(x)
        x = self.softmax(x)
        return x
