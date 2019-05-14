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
mpl.use('Agg')
import matplotlib.pyplot as plt

# Fucntion for validation accuracy
def validation_acc(model, val_loader):
# Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            if use_cuda:
                images = images.to(device)
                labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return (correct / total) * 100

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
test_set = AbideDataset(DATA_PATH, "test", args.summary)
print("#" * 60)

# Initialize dataloaders
train_loader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Check for GPU
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
    val_acc = validation_acc(model, val_loader)
    # if epoch % 30 ==0 and epoch != 0:
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     lr = learning_rate / 2
    #     learning_rate = lr
    #     for param_group in optimizer.param_groups:
    #         print(param_group['lr'])
    #         param_group['lr'] = lr

    # Append values to lists and print epoch results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(total_loss / (total/batch_size))
    print("Epoch [{}/{}], Loss: {:.4f}, Train acc: {:.2f}%, Val. acc: {:.2f}%".format(epoch + 1, num_epochs, total_loss / (total/batch_size), train_acc, val_acc))

    # Save model if this is specified
    if args.save_model:
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model_epoch{}.ckpt'.format(epoch+1))
        torch.save(optimizer.state_dict(),MODEL_STORE_PATH + 'optimizer_epoch{}.ckpt'.format(epoch+1))

# Summarize the training session
print("Done average val. acc: {:.2f}, best val. acc: {:.2f} ".format(sum(val_acc_list)/num_epochs, max(val_acc_list)))

# Plot the accuracy figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("training vs validation accuracy for {}".format(args.model))
ax.plot(range(num_epochs), train_acc_list, 'r', label='train')
ax.plot(range(num_epochs), val_acc_list, 'b', label='validation')
ax.set_xlabel('epochs')
ax.set_ylabel('percentage correct')
ax.legend(loc='best')
fig.savefig('accuracy_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), args.lr))

# Plot the loss figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(range(num_epochs), train_loss_list, 'r')
ax2.set_title("Training loss for the {} model".format(args.model))
ax2.set_xlabel('epochs')
ax2.set_ylabel('MSE loss')
fig2.savefig('loss_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), args.lr))
