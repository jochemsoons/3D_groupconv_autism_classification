import numpy as np
import h5py
import torch
import torch.nn as nn

from create_hdf5 import write_subset_files
from AbideData import AbideDataset
from Conv3DNet import Conv3DNet

from config import parse_opts, print_config
import matplotlib as mpl
import sklearn.metrics
mpl.use('Agg')
import matplotlib.pyplot as plt


# Parse the input arguments
args = parse_opts()
print("#" * 60)
print_config(args)
print("#" * 60)

# Set parameters
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.lr

# Paths for data and model storage
DATA_PATH = args.data_path
MODEL_STORE_PATH = args.model_store_path
PLOT_STORE_PATH = args.plot_store_path

# Retrieve test dataset
test_set = AbideDataset(DATA_PATH, "test", args.summary)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Check for GPU and set seed for reproducable results
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device...".format(device))


# Load stated model, Conv3d or ResNet
assert args.model in ['conv3d', 'groupconv3d']
if args.model == 'conv3d':
    model = Conv3DNet(num_classes)
    model.load_state_dict(torch.load(MODEL_STORE_PATH))
# elif args.model == 'groupconv3d':
    # model = ResNet3D.resnet10(
    #     num_classes=num_classes,
    #     shortcut_type=args.resnet_shortcut)
    # model.load_state_dict(torch.load(MODEL_STORE_PATH))


# Transfer model to GPU
if use_cuda:
    model = model.cuda()
print('Model loaded')
print("#" * 60)

# Initialize criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Starting testing phase...")
print("Testing on {} test images".format(len(test_loader.dataset)))

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

print("Done with testing. Accuracy: {:.2f}%, Loss: {:.4f}".format(test_acc, test_loss))

# Plot the ROC Curve
roc_auc = sklearn.metrics.auc(fpr, tpr)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, markevery=100)
ax3.plot(np.linspace(0,1,100), np.linspace(0,1,100), color='0.3', linestyle=':')
ax3.set_title('Receiver Operating Characteristics for {} model'.format(args.model))
ax3.set_xlabel('True Positive Rate')
ax3.set_ylabel('False Positive Rate')
ax3.set_xlim([0,1])
ax3.set_ylim([0,1])
ax3.legend(loc="best")
fig3.savefig(PLOT_STORE_PATH + 'roc_auc_curve_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary, test_acc, args.lr))
