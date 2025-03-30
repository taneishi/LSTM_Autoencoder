import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

import torch
from torch import nn

from main import load_dataset, create_dataset
from models import LSTMAutoencoder

def plot_loss_distribution(loss_train):
    plt.figure(figsize=(8, 4))
    plt.hist(loss_train['MAE loss'], bins=20, rwidth=0.9, color='blue', alpha=0.5)
    plt.xlim([0.0, 0.5])
    plt.grid(True, linestyle='--')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figure/loss_distribution.png', dpi=600/8)

def plot_test_loss(loss, threshold):
    plt.figure(figsize=(8, 4))
    loss['MAE loss'].plot(color='blue', alpha=0.7, ax=plt.gca())
    loss['Threshold'].plot(color='red', linestyle='--', alpha=0.7, ax=plt.gca())
    plt.ylim([1e-3, 1e2])
    plt.yscale('log')
    plt.grid(True, linestyle='--')
    plt.ylabel('MAE loss')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('figure/test_loss.png', dpi=600/8)

def plot_sensors(train, test):
    split_datetime = train.index.max()

    plt.figure(figsize=(8, 4))
    pd.concat([train, test]).plot(color=['blue', 'red', 'green', 'black'], linewidth=1, ax=plt.gca())
    plt.axvline(split_datetime, linestyle='--', color='gray')
    plt.grid(True, linestyle='--')
    plt.legend(loc='upper left')
    plt.ylabel('Vibration signal')
    plt.tight_layout()
    plt.savefig('figure/sensors.png', dpi=600/8)

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, test = load_dataset(args)

    plot_sensors(train, test)

    train_dataset, test_dataset = create_dataset(train, test, args)

    _, seq_len, n_features = train_dataset.shape

    net = LSTMAutoencoder(seq_len, n_features, args.embedding_dim)
    net = net.to(device)
    print(net)

    net.load_state_dict(torch.load(args.model_path, weights_only=True))

    criterion = nn.L1Loss(reduction='sum').to(device)

    # Plot the loss distribution of the training set.
    X_pred_train = []
    net = net.eval()
    with torch.no_grad():
        for seq_true in train_dataset:
            seq_true = seq_true.to(device)
            seq_pred = net(seq_true)
            X_pred_train.append(seq_pred.cpu().numpy())

    X_pred_train = np.asarray(X_pred_train)

    loss_train = pd.DataFrame(index=train.index)
    loss_train['MAE loss'] = np.mean(np.abs(X_pred_train - train_dataset.numpy()), axis=2)
    loss_train['Threshold'] = args.threshold

    plot_loss_distribution(loss_train)

    # Calculate the loss distribution of the test set.
    X_pred_test = []
    net = net.eval()
    with torch.no_grad():
        for seq_true in test_dataset:
            seq_true = seq_true.to(device)
            seq_pred = net(seq_true)
            X_pred_test.append(seq_pred.cpu().numpy())

    X_pred_test = np.asarray(X_pred_test)

    loss_test = pd.DataFrame(index=test.index)
    loss_test['MAE loss'] = np.mean(np.abs(X_pred_test - test_dataset.numpy()), axis=2)
    loss_test['Threshold'] = args.threshold

    # Merge train and test data and plot.
    loss = pd.concat([loss_train, loss_test])
    loss['Anomaly'] = loss['MAE loss'] > args.threshold
    print(loss)

    plot_test_loss(loss, args.threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--data_dir', default='data/bearing_data', type=str)
    parser.add_argument('--data_split', default=0.4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--embedding_dim', default=4, type=int)
    parser.add_argument('--threshold', default=0.275, type=float)
    parser.add_argument('--model_path', default='model.pth', type=str)
    args = parser.parse_args()

    main(args)
