import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2,2,2], stride=[1,1,1]))
        self.layers.append(self.layer1)
        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
        self.layers.append(self.layer2)
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
        self.layers.append(self.layer3)
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
        self.layers.append(self.layer4)
        self.avgpool = nn.AvgPool3d(kernel_size=[2,1,1], stride=[1,1,1])
        self.layers.append(self.avgpool)
        self.drop_out = nn.Dropout(inplace=True)
        self.fc1 = nn.Linear(4992, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# self.layer1 = nn.Sequential(
#     nn.Conv3d(1, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#     nn.Conv3d(8, 8, kernel_size=[4, 4, 4], stride=[1, 1, 1]),
#     nn.ReLU(),
#     nn.Conv3d(8, 8, kernel_size=[2, 2, 2], stride=[1, 1, 1]),
#     nn.Conv3d(8, 16, kernel_size=[3,3,3], stride=[1,1,1]),
#     nn.Conv3d(16, 32, kernel_size=[3,3,3], stride=[1,1,1]),
#     nn.ReLU(),
#     nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
# self.layers.append(self.layer1)
# self.layer2 = nn.Sequential(
#     nn.Conv3d(32, 40, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#     nn.Conv3d(40, 40, kernel_size=[2, 2, 2], stride=[1, 1, 1]),
#     nn.ReLU(),
#     nn.Conv3d(40, 40, kernel_size=[4, 4, 4], stride=[1, 1, 1]),
#     nn.Conv3d(40, 48, kernel_size=[3,3,3], stride=[1,1,1]),
#     nn.Conv3d(48, 64, kernel_size=[3, 3, 3], stride=[2, 1, 1]),
#     nn.ReLU(),
#     nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
# self.layers.append(self.layer2)
# self.drop_out = nn.Dropout()
# self.fc1 = nn.Linear(2304, 1000)
# self.fc2 = nn.Linear(1000, num_classes)
