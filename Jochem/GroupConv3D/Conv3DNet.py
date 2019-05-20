import torch.nn as nn

# class Conv3DNet(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv3DNet, self).__init__()
#         self.conv_layer1 = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
#         self.conv_layer2 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=[3, 3, 3], stride=[1,1,1]),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
#         self.conv_layer3 = nn.Sequential(
#             nn.Conv3d(128, 256, kernel_size=[3,3,3], stride=[1,1,1]),
#             nn.LeakyReLU(inplace=True),
#             nn.BatchNorm3d(256),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
#         self.avgpool = nn.AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2])
#         self.drop_out = nn.Dropout()
#         self.fc1 = nn.Linear(512, num_classes)


#     def forward(self, x):
#         out = self.conv_layer1(x)
#         out = self.drop_out(self.conv_layer2(out))
#         out = self.drop_out(self.conv_layer3(out))
#         out = self.drop_out(self.avgpool(out))
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         out = nn.functional.softmax(out, dim=0)
#         return out

class Conv3DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv3DNet, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=[2,2,2])
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=[5, 5, 5]),
            nn.MaxPool3d(kernel_size=[2,2,2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=[3, 5, 3]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=[3, 3, 3]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128))
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=[3, 3, 3]),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))
        self.drop_out = nn.Dropout(inplace=True)
        self.fc1 = nn.Linear(1728, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.drop_out(self.layer1(out))
        out = self.drop_out(self.layer2(out))
        out = self.drop_out(self.layer3(out))
        out = self.drop_out(self.layer4(out))
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(self.fc1(out))
        out = self.drop_out(self.fc2(out))
        out = nn.functional.softmax(out, dim=1)
        return out