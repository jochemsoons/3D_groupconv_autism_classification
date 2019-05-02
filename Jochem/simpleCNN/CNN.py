import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from AbideData import AbideDataset
from Conv3DNet import Conv3DNet

# Data paths for local run of code
DATA_PATH = '/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis'
MODEL_STORE_PATH = '/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/afstudeerproject_KI/Jochem/simpleCNN/Model_ckpts/'

# Data paths for run of code on GPU server
# DATA_PATH = '/home/jsoons/afstudeerproject_KI/Jochem'
# MODEL_STORE_PATH = '/home/jsoons/afstudeerproject_KI/Jochem/simpleCNN/Models/'

# Hyperparameters
num_epochs = 5
num_classes = 2
batch_size = 32
learning_rate = 0.001

# Create train and validation set
train_set = AbideDataset(DATA_PATH, "train", "T1")
val_set = AbideDataset(DATA_PATH, "validation", "T1")

# Initialise dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

# Initialise cuda GPU
# GPU = torch.device('cuda:0')

model = Conv3DNet(num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images   = images.to(GPU)
        # labels = labels.to(GPU)

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
            # break

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(len(val_set), (correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')