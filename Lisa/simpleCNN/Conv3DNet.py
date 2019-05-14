

import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=[3, 3, 3], stride=[1,1,1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2,2,2]))
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=[3,3,3], stride=[1,1,1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
        self.avgpool = nn.AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2])
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(512, num_classes)



    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.drop_out(self.conv_layer2(out))
        out = self.drop_out(self.conv_layer3(out))
        out = self.drop_out(self.avgpool(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = nn.functional.softmax(out, dim=0)
        return out
