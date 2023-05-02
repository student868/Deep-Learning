import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from models import RNN

BATCH_SIZE = 20
PRINT_EVERY = 5
EPOCHS = 3  # 13
SEQUENCE_LENGTH = 35
HIDDEN_UNITS = 200
NUM_LAYERS = 2
EMBEDDING_LENGTH = 128
DATA_DIR = 'data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
LR = 0.002

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(root):
    def read_file(path):
        with open(path, encoding='utf-8') as f:
            words = f.read().replace('\n', '<eos>').split(' ')
        words = list(filter(None, words))
        return words

    train_data = read_file(os.path.join(root, 'ptb.train.txt'))
    valid_data = read_file(os.path.join(root, 'ptb.valid.txt'))
    test_data = read_file(os.path.join(root, 'ptb.test.txt'))

    train_vocabulary = sorted(set(train_data))
    wordToIndex = {word: index for index, word in enumerate(train_vocabulary)}

    train_data = [wordToIndex[word] for word in train_data]
    valid_data = [wordToIndex[word] for word in valid_data]
    test_data = [wordToIndex[word] for word in test_data]

    return train_data, valid_data, test_data, len(train_vocabulary)


class RNN(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_LENGTH)
        self.lstm = nn.LSTM(EMBEDDING_LENGTH, HIDDEN_UNITS, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_UNITS, vocabulary_size)

    def forward(self, x, h):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        y = self.fc(out)
        return y, (h, c)


def detach(states):
    return [state.detach() for state in states]


def train(model, data):
    num_batches = data.size(1) // SEQUENCE_LENGTH

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # Set initial hidden and cell states
        states = (torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS).to(device),
                  torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS).to(device))

        for i in range(0, data.size(1) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            # Get mini-batch inputs and targets
            inputs = data[:, i:i + SEQUENCE_LENGTH].to(device)
            targets = data[:, (i + 1):(i + 1) + SEQUENCE_LENGTH].to(device)

            # Forward pass
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // SEQUENCE_LENGTH
            if step % 500 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, EPOCHS, step, num_batches, loss.item(), np.exp(loss.item())))


def test(model, test_ids, num_layers, batch_size, hidden_size):
    criterion = nn.CrossEntropyLoss()
    test_num_batches = test_ids.size(1) // SEQUENCE_LENGTH

    # Test the model
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    test_loss = 0.
    with torch.no_grad():
        for i in range(0, test_ids.size(1) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            # Get mini-batch inputs and targets
            inputs = test_ids[:, i:i + SEQUENCE_LENGTH].to(device)
            targets = test_ids[:, (i + 1):(i + 1) + SEQUENCE_LENGTH].to(device)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            test_loss += criterion(outputs, targets.reshape(-1)).item()

    test_loss = test_loss / test_num_batches
    print('-' * 89)
    print('test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('-' * 89)


def plot_model(model, train_list, test_list, optimizer):
    plt.suptitle(model.name + ' accuracy')
    plt.title(str(type(optimizer).__name__))
    x = [i + 1 for i in range(EPOCHS)]
    plt.plot(x, train_list, 'blue', label='Train data accuracy')
    plt.plot(x, test_list, 'red', label='Test data accuracy')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([75, 100])
    plt.savefig(os.path.join(PLOTS_DIR, model.name + '.png'))
    plt.show()


def main():
    train_data, valid_data, test_data, vocabulary_size = load_data(DATA_DIR)
    model = RNN(vocabulary_size).to(device)

    train_data = np.array(train_data[:(len(train_data) // BATCH_SIZE) * BATCH_SIZE]).reshape((BATCH_SIZE, -1))
    train_data_tensor = torch.from_numpy(train_data).long()
    train(model, train_data_tensor)

    valid_data = np.array(valid_data[:(len(valid_data) // BATCH_SIZE) * BATCH_SIZE]).reshape((BATCH_SIZE, -1))
    valid_data_tensor = torch.from_numpy(valid_data).long()
    test(model, valid_data_tensor, NUM_LAYERS, BATCH_SIZE, HIDDEN_UNITS)


if __name__ == '__main__':
    main()
