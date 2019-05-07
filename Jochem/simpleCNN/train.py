import numpy as np
import h5py
import torch
import torch.nn as nn

from create_hdf5 import write_subset_files
from AbideData import AbideDataset
from Conv3DNet import Conv3DNet
from config import parse_opts, print_config

args = parse_opts()
print("#" * 60)
print_config(args)
print("#" * 60)

# Set parameters
num_epochs = args.epochs
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.lr

# Paths of data and model storage
DATA_PATH = args.data_path
MODEL_STORE_PATH = args.model_store_path

print("Splitting dataset into subsets...")
f = h5py.File(DATA_PATH  + 'fmri_summary_abideI_II.hdf5', 'r')
write_subset_files(f, DATA_PATH, args.summary, args.test_ratio, args.train_val_ratio)

# Create train and validation set
print("Loading data subsets...\n")
train_set = AbideDataset(DATA_PATH, "train", args.summary)
val_set = AbideDataset(DATA_PATH, "validation", args.summary)
test_set = AbideDataset(DATA_PATH, "test", args.summary)
print("#" * 60)

# Initialize dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
if torch.cuda.is_available() and use_cuda: GPU = True
else: GPU = False

print("Using {} device...".format(device))

# Initialize model
print("Initializing model...")
model = Conv3DNet(num_classes)
if GPU:
    model = model.cuda()
print("Model initialized.")
print("#" * 60)

def validation_acc(model, val_loader, GPU):
# Test the model
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
        return (correct / total) * 100

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Starting training phase...")
print("Training on {} train images".format(len(train_loader.dataset)))
print("Validating on {} validation images\n".format(len(val_loader.dataset)))

train_acc_list = []
train_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    total = len(train_loader.dataset)
    total_correct = 0
    total_loss = 0
    for images, labels in train_loader:
        if GPU:
            images = images.to(device)
            labels = labels.to(device)

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()

    # Calculate accuracy scores
    train_acc = (total_correct / total) * 100
    val_acc = validation_acc(model, val_loader, GPU)

    # Append values to lists and print epoch results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(total_loss)
    print("Epoch [{}/{}], Loss: {:.4f}, Train acc: {:.2f}%, Val. acc: {:.2f}%".format(epoch + 1, num_epochs, total_loss, train_acc, val_acc))

    if args.save_model:
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model_epoch{}.ckpt'.format(epoch+1))