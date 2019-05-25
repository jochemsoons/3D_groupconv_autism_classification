import sklearn.metrics
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Plot the accuracy figure
def plot_accuracy_train_val(args, num_epochs, train_acc_list, val_acc_list,
                    PLOT_STORE_PATH):

    t_acc_max = max(train_acc_list)
    t_epoch_max = train_acc_list.index(t_acc_max)
    v_acc_max = max(val_acc_list)
    v_epoch_max = val_acc_list.index(v_acc_max)

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
    fig.savefig(PLOT_STORE_PATH + 'accuracy_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), float(args.lr)))

def plot_accuracy_val(args, num_epochs, val_acc_list, PLOT_STORE_PATH):

    v_acc_max = max(val_acc_list)
    v_epoch_max = val_acc_list.index(v_acc_max)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Validation accuracy for {}".format(args.model))
    ax.plot(range(num_epochs), val_acc_list, 'b')
    ax.plot(v_epoch_max, v_acc_max, color='b', marker=11, label='Validation accuray max', markersize=10)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Percentage correct')
    ax.legend(loc='best')
    fig.savefig(PLOT_STORE_PATH + 'accuracy_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), float(args.lr)))


# Plot the loss figure
def plot_loss_train_val(args, num_epochs, train_loss_list, val_loss_list, val_acc_list,
               PLOT_STORE_PATH):

    v_loss_min = min(val_loss_list)
    v_epoch_min = val_loss_list.index(v_loss_min)
    t_loss_min = min(train_loss_list)
    t_epoch_min = train_loss_list.index(t_loss_min)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(num_epochs), train_loss_list, 'r', label='Train')
    ax.plot(range(num_epochs), val_loss_list, 'b', label='Validation')
    ax.plot(t_epoch_min, t_loss_min, color='r', marker=11, label='Train loss min', markersize=10)
    ax.plot(v_epoch_min, v_loss_min, color='b', marker=11, label='Validation loss min', markersize=10)
    ax.set_title("Training loss vs validation loss for the {} model".format(args.model))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('CE Loss')
    ax.legend(loc='best')
    fig.savefig(PLOT_STORE_PATH + 'loss_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), float(args.lr)))

def plot_loss_val(args, num_epochs, val_loss_list, val_acc_list, PLOT_STORE_PATH):
    fig = plt.figure()

    v_loss_min = min(val_loss_list)
    v_epoch_min = val_loss_list.index(v_loss_min)

    ax = fig.add_subplot(111)
    ax.plot(range(num_epochs), val_loss_list, 'b')
    ax.plot(v_epoch_min, v_loss_min, color='b', marker=11, label='Validation loss min', markersize=10)
    ax.set_title("Validation loss for the {} model".format(args.model))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('CE Loss')
    fig.savefig(PLOT_STORE_PATH + 'loss_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary,max(val_acc_list), float(args.lr)))


# Plot the ROC/AUC Curve
def plot_roc_auc(args, test_acc, fpr, tpr):
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
    fig3.savefig(args.plot_store_path + 'roc_auc_curve_{}_{}_{:.2f}_{:.5f}.png'.format(args.model, args.summary, test_acc, args.lr))