import numpy as np
import h5py
import torch
import torch.nn as nn

from create_hdf5 import write_subset_files
from AbideData import AbideDataset
from Conv3DNet import Conv3DNet
import ResNet3D

from config import parse_opts, print_config
import matplotlib as mpl
import sklearn.metrics
mpl.use('Agg')
import matplotlib.pyplot as plt

# Fucntion for validation accuracy
def validation_acc(model, val_loader, criterion):
# Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for images, labels in val_loader:
            if use_cuda:
                images = images.to(device)
                labels = labels.to(device)
            # Run through model and calulate loss and accuracy
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100, (total_loss/(total/batch_size))

# Parse the input arguments
args = parse_opts()
print("#" * 60)
print_config(args)
print("#" * 60)

# Set parameters
num_epochs = args.epochs
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.lr
lr_d = args.lr_d

# Paths for data and model storage
DATA_PATH = args.data_path
MODEL_STORE_PATH = args.model_store_path

# Splitting
print("Splitting dataset into subsets...")
f = h5py.File(DATA_PATH  + 'fmri_summary_abideI_II.hdf5', 'r')
write_subset_files(f, DATA_PATH, args.summary, args.test_ratio, args.train_val_ratio)

# Create train, validation and test set
print("Loading data subsets...\n")
train_set = AbideDataset(DATA_PATH, "train", args.summary)
val_set = AbideDataset(DATA_PATH, "validation", args.summary)
print("#" * 60)

# Initialize dataloaders
train_loader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Check for GPU and set seeds for reproducable results
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device...".format(device))

# Initialize model Conv3d or ResNet
assert args.model in ['conv3d', 'resnet']
if args.model == 'conv3d':
    model = Conv3DNet(num_classes)
elif args.model == 'resnet':
    model = ResNet3D.resnet10(
        num_classes=num_classes,
        shortcut_type=args.resnet_shortcut)

# Transfer model to GPU
if use_cuda:
    model = model.cuda()
print('Model initialized')
print("#" * 60)

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
val_loss_list =[]
best_val_acc = 0

# Train the model for x epochs
for epoch in range(num_epochs):
    model.train()
    total = len(train_loader.dataset)
    total_correct = 0
    total_loss = 0.0
    for images, labels in train_loader:
        if use_cuda:
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
    train_acc = (total_correct /total) * 100
    val_acc, val_loss = validation_acc(model, val_loader, criterion)

    # Append values to lists and print epoch results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(total_loss / (total/batch_size))
    val_loss_list.append(val_loss)
    print("Epoch [{}/{}], Train loss: {:.4f}, Train acc: {:.2f}%, Val loss: {:.4f}, Val. acc: {:.2f}%".format(epoch + 1, num_epochs, total_loss/(total/batch_size), train_acc, val_loss, val_acc))

    # Save model if this is specified
    if args.save_model and best_val_acc <= val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_STORE_PATH + '{}_model_epoch{}_valloss{:.4f}_valacc{:.2f}.pt'.format(args.model, epoch+1, val_loss, val_acc))

# Summarize the training session
print("Done average val. acc: {:.2f}, best val. acc: {:.2f} ".format(sum(val_acc_list)/num_epochs, max(val_acc_list)))

# Find train and validation accuray max
t_acc_max = max(train_acc_list)
t_xpos = train_acc_list.index(t_acc_max)
t_epoch_max = range(num_epochs)[t_xpos]
v_acc_max = max(val_acc_list)
v_xpos = val_acc_list.index(v_acc_max)
v_epoch_max = range(num_epochs)[v_xpos]

# Plot the accuracy figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Training vs validation accuracy for {}".format(args.model))
ax.plot(range(num_epochs), train_acc_list, 'r', label='Train')
ax.plot(t_epoch_max, t_acc_max, color='r', marker=11, label='Train accuray max', markersize=10)
ax.plot(range(num_epochs), val_acc_list, 'b', label='Validation')
ax.plot(v_epoch_max, v_acc_max, color='b', marker=11, label='Validation accuray max', markersize=10)
ax.set_xlabel('Epochs')
ax.set_ylabel('Percentage correct')
ax.legend(loc='best')
fig.savefig('accuracy_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), args.lr))


# Plot the loss figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(range(num_epochs), train_loss_list, 'r')
ax2.plot(range(num_epochs), val_loss_list, 'b')
ax2.set_title("Training loss vs validation loss for the {} model".format(args.model))
ax2.set_xlabel('Epochs')
ax2.set_ylabel('CE loss')
fig2.savefig('loss_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), args.lr))
