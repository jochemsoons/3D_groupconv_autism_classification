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


# class Conv3DNet(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv3DNet, self).__init__()
#         self.conv_layer1 = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
#         self.conv_layer2 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=[3, 3, 3], stride=[1,1,1]),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2,2,2]))
#         self.conv_layer3 = nn.Sequential(
#             nn.Conv3d(128, 128, kernel_size=[3,3,3], stride=[1,1,1]),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=2))
#         self.avgpool = nn.AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2])
#         self.drop_out = nn.Dropout()
#         self.fc1 = nn.Linear(256, num_classes)


#     def forward(self, x):
#         out = self.conv_layer1(x)
#         out = self.drop_out(self.conv_layer2(out))
#         out = self.drop_out(self.conv_layer3(out))
#         out = self.drop_out(self.avgpool(out))
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         out = nn.functional.softmax(out, dim=0)
#         return out

# class Conv3DNet(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv3DNet, self).__init__()
#         self.conv_layer1 = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=[5, 5, 5], stride=[2, 2, 2]),
#             nn.LeakyReLU(0.1))
#         self.conv_layer2 = nn.Sequential(
#             nn.Conv3d(32, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.LeakyReLU(0.1))
#         self.maxpool = nn.MaxPool3d(kernel_size=[2,2,2])
#         self.fc1 = nn.Sequential(
#         nn.Linear(114048, 128))

#         self.drop_out = nn.Dropout(p=0.5, inplace=True)
#         self.fc2 = nn.Sequential(
#         nn.Linear(128, 2))

#     def forward(self, x):
#         out = self.drop_out(self.conv_layer1(x))
#         out = self.drop_out(self.conv_layer2(out))
#         out = self.drop_out(self.maxpool(out))
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         # out = self.fc2(out)
#         out = nn.functional.softmax(out, dim=0)
#         return out

# # class Conv3DNet(nn.Module):
# #     def __init__(self, num_classes):
# #         super(Conv3DNet, self).__init__()
# #         self.layers = []
# #         self.layer1 = nn.Sequential(
# #             nn.Conv3d(1, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[2, 2, 2]),
# #             nn.ReLU(),
# #             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
# #         self.layers.append(self.layer1)
# #         self.layer2 = nn.Sequential(
# #             nn.Conv3d(32, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[3, 3, 3]),
# #             nn.ReLU(),
# #             nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
# #         self.layers.append(self.layer2)
# #         self.layer3 = nn.Sequential(
# #             nn.Conv3d(64, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2]),
# #             nn.ReLU(),
# #             nn.AvgPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]))
# #         self.drop_out = nn.Dropout()
# #         self.fc1 = nn.Linear(864, 1000)
# #         self.fc2 = nn.Linear(1000, num_classes)

# #     def forward(self, x):
# #         # print(x.shape)
# #         out = self.layer1(x)
# #         # print(out.shape)
# #         out = self.layer2(out)
# #         # print(out.shape)

# #         out = self.layer3(out)
# #         # print(out.shape)
# #         out = out.reshape(out.size(0), -1)
# #         # print(out.shape)
# #         out = self.drop_out(out)
# #         # print(out.shape)
# #         out = self.fc1(out)
# #         out = self.fc2(out)
# #         return out

# class Conv3DNet(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv3DNet, self).__init__()
#         self.layers = []
#         self.layer1 = nn.Sequential(
#             nn.Conv3d(1, 8, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=[2,2,2], stride=[1,1,1]))
#         self.layers.append(self.layer1)
#         self.layer2 = nn.Sequential(
#             nn.Conv3d(8, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.BatchNorm3d(16),
#             nn.Conv3d(16, 32, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
#         self.layers.append(self.layer2)
#         self.layer3 = nn.Sequential(
#             nn.Conv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(32, 64, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
#         self.layers.append(self.layer3)
#         self.layer4 = nn.Sequential(
#             nn.Conv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1]),
#             nn.BatchNorm3d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(64, 128, kernel_size=[2, 2, 2], stride=[1, 1, 1], padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2]))
#         self.layers.append(self.layer4)
#         self.avgpool = nn.AvgPool3d(kernel_size=[7,4,2], stride=[1,1,1])
#         self.layers.append(self.avgpool)
#         self.drop_out = nn.Dropout(inplace=True)
#         self.fc1 = nn.Linear(8448, 500)
#         self.fc2 = nn.Linear(500, num_classes)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.drop_out(out)
#         # print(out.shape)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out