import numpy as np
import torch
import torch.nn as nn


from AbideData import AbideDataset
from Conv3DNet import Conv3DNet
from config import parse_opts, print_config

args = parse_opts()
print("#" * 60)
print_config(args)
print("#" * 60)

# Data paths for local run of code
DATA_PATH = args.data_path
MODEL_STORE_PATH = args.model_store_path

# Set parameters
num_epochs = args.epochs
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.lr

# Create train and validation set
print("Loading data...\n")
train_set = AbideDataset(DATA_PATH, "train", args.summary)
val_set = AbideDataset(DATA_PATH, "validation", args.summary)
print("#" * 60)

# Initialize dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

use_cuda = not args.no_cuda and torch.cuda.is_available()
print(use_cuda)
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
print(device)
if torch.cuda.is_available() and use_cuda: GPU = True
else: GPU = False
print(GPU)

print("Using {} device...".format(device))

# Initialize model
model = Conv3DNet(num_classes)
if GPU:
    model = model.cuda()

def test_evaluation(model, val_loader, GPU):
# Test the model
    print("\nStarting evaluation...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            if GPU:
                images = images.to(device)
                labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = (correct / total) * 100

        print('Test Accuracy of the model on the {} validation images: {} %\n'.format(len(val_loader.dataset), val_acc))
        return val_acc

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Starting training phase...\n")
total_step = len(train_loader)
loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if GPU:
            images = images.to(device)
            labels = labels.to(device)

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
        train_acc_list.append(correct / total)

        if (i + 1) % args.log_interval == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

    val_acc = test_evaluation(model, val_loader, GPU)
    val_acc_list.append(val_acc)
    if args.save_model:
        print("Saving model...")
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model_epoch{}.ckpt'.format(epoch+1))
        print()


# # Save the model and plot
# if args.save_model:
#     print("Saving model...")
#     torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')