import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from trainer import RNN, train, test, BATCH_SIZE

# ---------------- constants ----------------
EPOCHS = 13
LR = 1
LR_DECREASE_START_EPOCH = 7
LR_DECREASE_FACTOR = 2
DROPOUT = 0.2

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


def plot_model(model, train_list, test_list):
    plt.title(model.name)
    x = [i + 1 for i in range(len(train_list))]
    plt.plot(x, train_list, 'blue', label='Train')
    plt.plot(x, test_list, 'red', label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.ylim([0, 1000])
    plt.savefig(os.path.join(PLOTS_DIR, model.name + '.png'))
    plt.show()


def evaluate_model(train_data, valid_data, test_data, model, use_saved_weights):
    print(' -- ' + model.name + ' -- ')

    if use_saved_weights and os.path.exists(os.path.join(MODELS_DIR, model.name + '.pkl')):
        print('Loading old weights...')
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, model.name + '.pkl')))
        print('Loaded Model - Train Perplexity: {:5.2f}, Validation Perplexity: {:5.2f}'.format(
            test(device, model, train_data),
            test(device, model, test_data)
        ))

    else:
        print('Training Model...')

        lr = LR
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        train_list = []
        test_list = []

        for epoch in range(EPOCHS):
            # changed LR according to epoch (from the paper)
            if epoch >= LR_DECREASE_START_EPOCH:
                for param_group in optimizer.param_groups:
                    lr /= LR_DECREASE_FACTOR
                    param_group['lr'] = lr

            train(device, model, train_data, loss_fn, optimizer)
            train_list.append(test(device, model, train_data))
            test_list.append(test(device, model, test_data))

            print('Epoch [{}/{}], LR: {:8.6f}, Train Perplexity: {:5.2f}, Validation Perplexity: {:5.2f}'.format(
                epoch + 1, EPOCHS, lr,
                train_list[-1],
                test(device, model, valid_data)
            ))

        # Save the Model
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, model.name + '.pkl'))  # TODO

        # Plot the Model
        plot_model(model, train_list, test_list)

    print()


def main():
    train_data, valid_data, test_data, vocabulary_size = load_data(DATA_DIR)

    evaluate_model(train_data, valid_data, test_data, RNN(nn.LSTM, vocabulary_size).to(device), use_saved_weights=True)
    evaluate_model(train_data, valid_data, test_data, RNN(nn.LSTM, vocabulary_size, DROPOUT).to(device), use_saved_weights=True)

    evaluate_model(train_data, valid_data, test_data, RNN(nn.GRU, vocabulary_size).to(device), use_saved_weights=True)
    evaluate_model(train_data, valid_data, test_data, RNN(nn.GRU, vocabulary_size, DROPOUT).to(device), use_saved_weights=True)


if __name__ == '__main__':
    main()
