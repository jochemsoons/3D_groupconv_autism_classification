

def print_config(args):
    print("PARAMETERS:\n")
    print("train file: \t \t {}".format(args.train_file))
    print("val file: \t \t {}".format(args.val_file))
    print("test file: \t \t {}".format(args.test_file))
    print("model storage path: \t {}".format(args.model_path))
    print("plots storage path: \t {}\n".format(args.plot_store_path))
    print("model: \t \t \t {}".format(args.model))
    print("summary to train on: \t {}".format(args.summary))
    print("train/val ratio: \t {}".format(args.train_val_ratio))
    print("test ratio: \t \t {}".format(args.test_ratio))
    print("summary to train on: \t {}".format(args.summary))
    print("number of classes: \t {}".format(args.num_classes))
    print("number of epochs: \t {}".format(args.epochs))
    print("batch size: \t \t {}".format(args.batch_size))
    print("learning rate: \t \t {}".format(args.lr))
