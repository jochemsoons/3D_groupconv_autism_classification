import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.layers = []
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.layers.append(self.conv_layer1)
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.layers.append(self.conv_layer2)
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.layers.append(self.conv_layer3)
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[1, 1, 1]))
        self.layers.append(self.conv_layer4)
        self.avgpool = nn.AvgPool3d(kernel_size=[2,1,1], stride=[1,1,1])
        self.layers.append(self.avgpool)
        self.fc1 = nn.Sequential(
        nn.Linear(2380800, 128),
        nn.ReLU(inplace=True))
        self.batch_norm = nn.BatchNorm1d(2380800)
        self.drop_out = nn.Dropout(p=0.7, inplace=True)
        self.fc2 = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(64, num_classes)


    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.drop_out(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.batch_norm(out)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = nn.functional.softmax(out)
        return out
