import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import convnet_helpers as ch
from convnet import Convnet
import os
import sys
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import utility_functions as uf
import matplotlib.pyplot as plt
import matplotlib
import batcher
matplotlib.use('GTK') 

def results(ckpt_dir, rolling):
    #The directory where the files are located
    data_dir = "checkpoints_directory/{}".format(ckpt_dir)

    #Get the latest parameter file
    param_files = [f for f in uf.ls_function(data_dir) if "npy" in f]
    max_ind = max(range(len(param_files)), key=lambda i:int(param_files[i].split(".")[0].split("_")[-1]))
    params = np.load("{}/{}".format(data_dir, param_files[max_ind])).item()

    rocs = []
    entropies = []
    accuracies = []
    step = params["bsize"]
    #Get this network's predictions and truth throughout the experiment
    train_preds, val_preds = uf.softmax(np.array(params["train_predictions"])), uf.softmax(np.array(params["val_predictions"]))
    train_labels, val_labels = np.array(params["train_truth"]), np.array(params["val_truth"])

    #Compute the roc, accuracy, and cross entropy for each batch
    for i in range(0,len(train_preds), step):
        rocs.append((roc_auc_score(np.int64(train_labels[i:i+step]), train_preds[i:i+step]), roc_auc_score(np.int64(val_labels[i:i+step]), val_preds[i:i+step])))
        entropies.append((log_loss(np.int64(train_labels[i:i+step]), train_preds[i:i+step]), log_loss(np.int64(val_labels[i:i+step]), val_preds[i:i+step])))
        accuracies.append((accuracy_score(np.argmax(train_labels[i:i+step], axis=1), np.argmax(train_preds[i:i+step], axis=1)), accuracy_score(np.argmax(val_labels[i:i+step], axis=1), np.argmax(val_preds[i:i+step], axis=1))))

    plt.subplot(3,1,1)
    plt.plot([a[0] for a in rocs], label="train roc")
    plt.plot([a[1] for a in rocs], label="val roc")
    plt.legend()
    plt.title('rocs for {}'.format(ckpt_dir.replace("_"," ")))

    plt.subplot(3,1,2)
    plt.plot([a[0] for a in entropies], label="train cross entropy")
    plt.plot([a[1] for a in entropies], label="val cross entropy")
    plt.legend()
    plt.title('entropies for {}'.format(ckpt_dir.replace("_"," ")))

    plt.subplot(3,1,3)
    plt.plot([a[0] for a in accuracies], label="train accuracy")
    plt.plot([a[1] for a in accuracies], label="val accuracy")
    plt.legend()
    plt.title('accuracies for {}'.format(ckpt_dir.replace("_"," ")))
    plt.savefig("{}.png".format(ckpt_dir))
    plt.show()
        

if __name__ == "__main__":
    results(sys.argv[1], True  if len(sys.argv) == 2 else bool(sys.argv[2]))
