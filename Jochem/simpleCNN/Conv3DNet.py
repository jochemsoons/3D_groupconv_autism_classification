import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[2, 2, 2]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.layers.append(self.layer1)
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[3, 3, 3]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.layers.append(self.layer2)
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2]),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4096, 2)
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        print(x.shape)
        out = self.layer1(x)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        # out = out.reshape(out.size(0), -1)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.drop_out(out)
        print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        return out