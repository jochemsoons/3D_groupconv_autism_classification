import numpy as np
import h5py
import torch
import torch.nn as nn
import sklearn.metrics

from AbideData import AbideDataset, write_subset_files, explore_data
from Conv3DNet import Conv3DNet
from config import parse_opts, print_config
from plot import plot_accuracy, plot_loss, plot_roc_auc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

args = parse_opts()

# If argument given, description of dataset is printed.
if args.explore_data:
    print("#" * 60)
    data_file = h5py.File(args.data_path  + 'fmri_summary_abideI_II.hdf5', 'r')
    explore_data(data_file)

print("#" * 60)
print_config(args)
print("#" * 60)

# Set parameters
num_epochs = args.epochs
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.lr

# Paths of data, model and plot storage
DATA_PATH = args.data_path
MODEL_STORE_PATH = args.model_store_path
PLOT_STORE_PATH = args.plot_store_path

print("Splitting dataset into subsets...")
data_file = h5py.File(DATA_PATH  + 'fmri_summary_abideI_II.hdf5', 'r')
write_subset_files(data_file, DATA_PATH, args.summary, args.test_ratio, args.train_val_ratio)

# Create train, validation and test set
print("Loading data subsets...\n")
train_set = AbideDataset(DATA_PATH, "train", args.summary)
val_set = AbideDataset(DATA_PATH, "validation", args.summary)
test_set = AbideDataset(DATA_PATH, "test", args.summary)
print("#" * 60)

# Initialize dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Check if cuda GPU is available
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
if torch.cuda.is_available() and use_cuda: GPU = True
else: GPU = False
print("Using {} device...".format(device))

# Initialize model
print("Initializing model...")
if args.model == "conv3d":
    model = Conv3DNet(num_classes)
# elif args.model = "groupconv3d":
#     model = GroupConv3D(num_classes)

if GPU:
    model = model.cuda()
print("Model initialized.")
print("#" * 60)

def validation_acc(model, val_loader, GPU, criterion, batch_size):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for images, labels in val_loader:
            if GPU:
                images = images.to(device)
                labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = (correct / total) * 100
        val_loss = total_loss / (total / batch_size)
        return val_acc, val_loss

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
val_loss_list = []
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
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
        correct = (predicted == labels).sum().item()
        total_correct += correct

    # Calculate accuracy scores
    train_acc = (total_correct / total) * 100
    val_acc, val_loss = validation_acc(model, val_loader, GPU, criterion, batch_size)

    train_loss = total_loss / (total / batch_size)

    # Append values to lists and print epoch results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print("Epoch [{}/{}], Train loss: {:.4f}, Train acc: {:.2f}%, Val. loss: {:.4f}, Val. acc: {:.2f}%".format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    # Save model if this is specified
    if args.save_model and best_val_acc <= val_acc:
        best_val_acc = val_acc
        OPTIMAL_PATH = MODEL_STORE_PATH + '{}_model_epoch{}_valloss{:.4f}_valacc{:.2f}.pt'.format(args.model, epoch+1, val_loss, val_acc)
        torch.save(model.state_dict(), OPTIMAL_PATH)


# Summarize the training session
print("Done with training: average val. acc: {:.2f}, best val. acc: {:.2f} (epoch: {}), average val. loss: {:.4f}, lowest val. loss: {:.4f} (epoch: {})".format(sum(val_acc_list)/num_epochs, max(val_acc_list), val_acc_list.index(max(val_acc_list)), sum(val_loss_list)/num_epochs, min(val_loss_list), val_loss_list.index(min(val_loss_list))))

# Create accuracy and loss plot figures
plot_accuracy(args, num_epochs, train_acc_list, val_acc_list, PLOT_STORE_PATH)
plot_loss(args, num_epochs, train_loss_list, val_loss_list, val_acc_list, PLOT_STORE_PATH)

print("#" * 60)
print("Starting testing phase...")

# Load model, Conv3D or GroupConv3D
assert args.model in ['conv3d', 'groupconv3d']
if args.model == 'conv3d':
    model = Conv3DNet(num_classes)
    model.load_state_dict(torch.load(OPTIMAL_PATH))

# elif args.model = 'groupconv3d':
    # model = GroupConv3DNet(num_classes)
    # model.load_state_dict(torch.load(OPTIMAL_PATH))

if use_cuda:
    model = model.cuda()

print("Optimal model from training is restored")
print("Testing on {} test images\n".format(len(test_loader.dataset)))
# Run the test samples through the model in the specified batch size
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    labels_list = np.array([])
    predicted_list = np.array([])
    for images, labels in test_loader:
        labels_list = np.append(labels_list, labels.numpy())
        if use_cuda:
            images = images.to(device)
            labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predicted_list = np.append(predicted_list, predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Values for ROC CURVE, accuracy and loss
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels_list, predicted_list)
    test_acc = (correct / total) * 100
    test_loss = total_loss/(total/batch_size)

# Plot and finish test phase
plot_roc_auc(args, test_acc, fpr, tpr, PLOT_STORE_PATH)
print("Done with testing: Accuracy: {:.2f}%, Loss: {:.4f}\n".format(test_acc, test_loss))
