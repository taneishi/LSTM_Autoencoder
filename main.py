import numpy as np
import pandas as pd
import argparse
import timeit
import os

import torch
from torch import nn

from models import LSTMAutoencoder

def create_dataset(train, test, args):
    # Min-max scaling the dataset.
    scaler = dict(max=train.max(), min=train.min())
    X_train = ((train - scaler['min']) / (scaler['max'] - scaler['min'])).values
    X_test = ((test - scaler['min']) / (scaler['max'] - scaler['min'])).values

    # Reshape inputs for LSTM as [samples, timepoints, features].
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print('Training set shape: {}'.format(X_train.shape))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print('Test set shape: {}'.format(X_test.shape))

    train_dataset = torch.FloatTensor(X_train)
    test_dataset = torch.FloatTensor(X_test)

    return train_dataset, test_dataset

def load_dataset(args):
    df, index = [], []
    columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

    # Load sample data.
    for filename in os.listdir(args.data_dir):
        index.append(filename)
        sample = pd.read_csv(os.path.join(args.data_dir, filename), sep='\t')
        sample_mean_abs = sample.abs().mean().values.T
        df.append(sample_mean_abs)

    # Merge the sensor samples.
    df = pd.DataFrame(df, index=index, columns=columns)

    # Transform sample index to datetime and sort in chronological order.
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    df = df.sort_index()
    print('Dataset shape: {}'.format(df.shape))

    split_datetime = df.index[int(df.shape[0] * args.data_split)]
    print('Training and test split: {}'.format(split_datetime))
    train = df[:split_datetime]
    test = df[split_datetime:]

    return train, test

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, test = load_dataset(args)

    train_dataset, test_dataset = create_dataset(train, test, args)

    _, seq_len, n_features = train_dataset.shape

    # Create the autoencoder model.
    net = LSTMAutoencoder(seq_len, n_features, args.embedding_dim)
    net = net.to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.L1Loss(reduction='sum').to(device)

    # Fit the model to the data.
    for epoch in range(1, args.epochs + 1):
        epoch_start = timeit.default_timer()

        net = net.train()
        train_losses = []
        for seq_true in train_dataset:
            seq_true = seq_true.to(device)
            seq_pred = net(seq_true)

            loss = criterion(seq_pred, seq_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_losses.append(train_loss)

        train_loss = np.mean(train_losses)
        print('epoch {:3d} train loss {:6.4f} {:4.1f}sec'.format(epoch, train_loss, timeit.default_timer() - epoch_start))

    # Save the model weights.
    torch.save(net.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--data_dir', default='data/bearing_data', type=str)
    parser.add_argument('--data_split', default=0.4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--embedding_dim', default=4, type=int)
    parser.add_argument('--model_path', default='model.pth', type=str)
    args = parser.parse_args()

    main(args)
