import argparse

def parse_opts():

    parser = argparse.ArgumentParser(description='3D-CNN For autism detection')

    parser.add_argument('--data_path', type=str, required=True, help='location path of data')

    parser.add_argument('--model_store_path', type=str, required=True, help='location path of model checkpoints')

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

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--summary', type=str, help='give summary of data that needs to be trained on.')

    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    return args

def print_config(args):
    print("PARAMETERS:")
    print("data path: \t \t {}".format(args.data_path))
    print("model storage path: \t {}".format(args.model_store_path))
    print("summary to train on: \t {}".format(args.summary))
    print("number of classes: \t {}".format(args.num_classes))
    print("number of epochs: \t {}".format(args.epochs))
    print("log interval: \t \t {}".format(args.log_interval))
    print("batch size: \t \t {}".format(args.batch_size))
    print("learning rate: \t \t {}".format(args.lr))
    print("momentum: \t \t {}".format(args.momentum))
    print("seed: \t \t \t {}".format(args.seed))
    print("cuda GPU: \t \t No") if args.no_cuda else print("cuda GPU: \t \t No")
