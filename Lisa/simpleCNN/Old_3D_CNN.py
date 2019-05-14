

import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        # self.avgpool = nn.AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2])
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(48000, 1000)
        self.fc2 = nn.Linear(1000, num_classes)



    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.drop_out(self.conv_layer2(out))
        out = self.conv_layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = nn.functional.softmax(out, dim=0)
        return out


## BEST ONE below



import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            # nn.BatchNorm3d(64),
            # nn.Conv3d(8, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(8, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=[3, 3, 3], stride=2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm3d(128),
            # nn.Conv3d(16, 16, kernel_size=[3,3,3], stride=[1,1,1]),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(16, 16, kernel_size=[3,3,3], stride=[1,1,1]),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.conv_layer3 = nn.Sequential(
            # nn.Conv3d(16, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            # nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=[3,3,3], stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            # nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
        self.avgpool = nn.AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2])
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(256, num_classes)
        # self.fc2 = nn.Linear(1000, num_classes)



    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.drop_out(self.conv_layer2(out))
        out = self.drop_out(self.avgpool(self.drop_out(self.conv_layer3(out))))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = nn.functional.softmax(out, dim=0)
        return out
