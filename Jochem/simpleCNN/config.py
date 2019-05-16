import argparse

def parse_opts():

    parser = argparse.ArgumentParser(description='3D-CNN For autism detection')

    parser.add_argument('--explore_data', action='store_true', default=False, help='if given, no training is performed but an exploration of the dataset is given')

    parser.add_argument('--data_path', type=str, required=True, help='location path of data')

    parser.add_argument('--model_store_path', type=str, required=True, help='location path of model checkpoints')

    parser.add_argument('--plot_store_path', type=str, required=True, help='location path of results plots')

    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--test_ratio', type=float, default=0.3, help='ratio that defines size of test set')

    parser.add_argument('--train_val_ratio', type=float, default=0.7, help='ratio of train/val set sizes')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--summary', type=str, help='give summary of data that needs to be trained on.')

    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    return args

def print_config(args):
    print("PARAMETERS:\n")
    print("data path: \t \t {}".format(args.data_path))
    print("model storage path: \t {}".format(args.model_store_path))
    print("plots storage path: \t {}".format(args.plot_store_path))
    print("train/val ratio: \t {}".format(args.train_val_ratio))
    print("test ratio: \t \t {}".format(args.test_ratio))
    print("summary to train on: \t {}".format(args.summary))
    print("number of classes: \t {}".format(args.num_classes))
    print("number of epochs: \t {}".format(args.epochs))
    print("batch size: \t \t {}".format(args.batch_size))
    print("learning rate: \t \t {}".format(args.lr))
    print("momentum: \t \t {}".format(args.momentum))
    print("seed: \t \t \t {}".format(args.seed))
    print("cuda GPU: \t \t No\n") if args.no_cuda else print("cuda GPU: \t \t Yes\n")
