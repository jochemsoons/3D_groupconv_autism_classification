import h5py
import numpy
import torch
from torch import nn
import abide
from torch.utils.data import DataLoader
import argparse

class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=[1,1,1], stride=(1,1,1), padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3], stride=(2,2,2)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=[1,1,1], stride=(1,1,1), padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[2,2,2], stride=(1,1,1)))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1160000, 1000)
        self.fc2 = nn.Linear(1000, 2)


    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = out.reshape(out.size(0), -1)
        out = self.fc2(self.fc1(self.drop_out(out)))
        return out

def train(args, model, device, train_loader, criterion, optimizer, epcoh, GPU):
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for i, (images, labels) in enumerate(train_loader):
        if GPU:
            images = images.to(device)
            labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        print(correct, total)
        acc_list.append(correct/total)
        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

def test(args, model, device, test_loader, GPU):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if GPU:
                images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='3D-CNN For autism detection')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--summary', type=str, help='give summary of data that needs to be trained on.')
    # parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    if torch.cuda.is_available(): GPU = True
    else: GPU = False
    print("Using GPU device {}".format(device))

    # DATA_PATH = '/home/lisasalomons/Desktop'
    DATA_PATH = '/home/lsalomons/eigenconvnet'
    # MODEL_STORE_PATH = '/home/lisasalomons/Desktop/models'

    training_data = abide.AbideDataset(root_path=DATA_PATH, subset='train',summarie=args.summary)
    test_data = abide.AbideDataset(root_path=DATA_PATH, subset='test', summarie=args.summary)

    train_loader = DataLoader(training_data, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = Conv3D()
    if GPU:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader,criterion, optimizer, epoch, args.summary)

    if args.save_model:
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

if __name__ == '__main__':
    main()
