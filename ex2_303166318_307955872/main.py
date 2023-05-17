import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from trainer import RNN, train, test, BATCH_SIZE

# ---------------- constants ----------------
DATA_DIR = 'data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

# -------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(root):
    def read_file(path):
        with open(path, encoding='utf-8') as f:
            words = f.read().replace('\n', '<eos>').split(' ')
        words = list(filter(None, words))
        return words

    def batch_data(data):
        data = np.array(data[:(len(data) // BATCH_SIZE) * BATCH_SIZE]).reshape((BATCH_SIZE, -1))
        return torch.from_numpy(data).long()

    train_data = read_file(os.path.join(root, 'ptb.train.txt'))
    valid_data = read_file(os.path.join(root, 'ptb.valid.txt'))
    test_data = read_file(os.path.join(root, 'ptb.test.txt'))

    train_vocabulary = sorted(set(train_data))
    wordToIndex = {word: index for index, word in enumerate(train_vocabulary)}

    train_data = [wordToIndex[word] for word in train_data]
    valid_data = [wordToIndex[word] for word in valid_data]
    test_data = [wordToIndex[word] for word in test_data]

    return batch_data(train_data), batch_data(valid_data), batch_data(test_data), len(train_vocabulary)


def plot_model(model, train_list, test_list, lr):
    plt.title(model.name + '(LR = ' + str(lr) + ')')
    x = [i + 1 for i in range(len(train_list))]
    plt.plot(x, train_list, 'blue', label='Train')
    plt.plot(x, test_list, 'red', label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.ylim([0, 500])
    plt.savefig(os.path.join(PLOTS_DIR, model.name + '.png'))
    plt.show()


def nll_loss(scores, y):
    probabilities = scores.exp() / scores.exp().sum(1, keepdim=True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    return torch.mean(-torch.log(answerprobs) * BATCH_SIZE)


def evaluate_model(train_data, valid_data, test_data, model, optimizer, epochs, sequence_length, valid_stop=None, scheduler=None, use_saved_weights=True):
    print(' -- ' + model.name + ' -- ')
    loss_fn = nll_loss

    if use_saved_weights and os.path.exists(os.path.join(MODELS_DIR, model.name + '.pkl')):
        print('Loading old weights...')
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, model.name + '.pkl')))
        print('Loaded Model - Train Perplexity: {:5.2f}, validation Perplexity: {:5.2f}, test Perplexity: {:5.2f}'.format(
            test(device, model, train_data, loss_fn, sequence_length),
            test(device, model, valid_data, loss_fn, sequence_length),
            test(device, model, test_data, loss_fn, sequence_length)
        ))

    else:
        print('Training Model...')
        lr_start = optimizer.param_groups[0]['lr']

        train_list = []
        test_list = []

        for epoch in range(epochs):
            train(device, model, train_data, loss_fn, optimizer, sequence_length)

            train_list.append(test(device, model, train_data, loss_fn, sequence_length))
            valid_perplexity = test(device, model, valid_data, loss_fn, sequence_length)
            test_list.append(test(device, model, test_data, loss_fn, sequence_length))

            print('Epoch [{}/{}], LR: {:8.6f}, Train Perplexity: {:5.2f}, Validation Perplexity: {:5.2f}'.format(
                epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_list[-1], valid_perplexity))

            if valid_stop and valid_perplexity < valid_stop:
                break

            if scheduler:
                if isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step(valid_perplexity)
                else:
                    scheduler.step()

        # Save the Model
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, model.name + '.pkl'))  # TODO

        # Plot the Model
        plot_model(model, train_list, test_list, lr_start)

    print()


def train_lstm(train_data, valid_data, test_data, vocabulary_size, use_saved_weights):
    model = RNN(nn.LSTM, vocabulary_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 1 / 2 if epoch >= 4 else 1)
    evaluate_model(train_data, valid_data, test_data, model, optimizer, epochs=20, sequence_length=20, scheduler=scheduler, valid_stop=125, use_saved_weights=use_saved_weights)


def train_lstm_with_dropout(train_data, valid_data, test_data, vocabulary_size, use_saved_weights):
    model = RNN(nn.LSTM, vocabulary_size, w_init=0.1, dropout=0.5).to(device)  # TODO w_init=0.2
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, 0.1)
    evaluate_model(train_data, valid_data, test_data, model, optimizer, epochs=100, sequence_length=35, scheduler=scheduler, valid_stop=100, use_saved_weights=use_saved_weights)


def train_gru(train_data, valid_data, test_data, vocabulary_size, use_saved_weights):
    model = RNN(nn.GRU, vocabulary_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.1)
    evaluate_model(train_data, valid_data, test_data, model, optimizer, epochs=20, sequence_length=20, scheduler=scheduler, valid_stop=125, use_saved_weights=use_saved_weights)


def train_gru_with_dropout(train_data, valid_data, test_data, vocabulary_size, use_saved_weights):
    model = RNN(nn.GRU, vocabulary_size, dropout=0.5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
    evaluate_model(train_data, valid_data, test_data, model, optimizer, epochs=100, sequence_length=35, scheduler=scheduler, valid_stop=100, use_saved_weights=use_saved_weights)


def main():
    train_data, valid_data, test_data, vocabulary_size = load_data(DATA_DIR)

    # train_lstm(train_data, valid_data, test_data, vocabulary_size, use_saved_weights=False)
    # train_lstm_with_dropout(train_data, valid_data, test_data, vocabulary_size, use_saved_weights=False)
    # train_gru(train_data, valid_data, test_data, vocabulary_size, use_saved_weights=False)
    train_gru_with_dropout(train_data, valid_data, test_data, vocabulary_size, use_saved_weights=False)


if __name__ == '__main__':
    main()
