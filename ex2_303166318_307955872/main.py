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
    plt.title(model.name + '(LR = ' + lr + ')')
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


def evaluate_model(train_data, valid_data, test_data, model, epochs, lr_start: float, lr_decrease_start_epoch, lr_decrease_factor, sequence_length, use_saved_weights=True):
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

        optimizer = torch.optim.SGD(model.parameters(), lr=lr_start)
        lr = lr_start
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

        train_list = []
        test_list = []

        for epoch in range(epochs):
            # changed LR according to epoch
            if epoch >= lr_decrease_start_epoch:
                for param_group in optimizer.param_groups:
                    lr /= lr_decrease_factor
                    param_group['lr'] = lr

            train(device, model, train_data, loss_fn, optimizer, sequence_length)
            train_list.append(test(device, model, train_data, loss_fn, sequence_length))
            test_list.append(test(device, model, test_data, loss_fn, sequence_length))

            print('Epoch [{}/{}], LR: {:8.6f}, Train Perplexity: {:5.2f}, Validation Perplexity: {:5.2f}'.format(
                epoch + 1, epochs, lr,
                train_list[-1],
                test(device, model, valid_data, loss_fn, sequence_length)
            ))

        # Save the Model
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, model.name + '.pkl'))  # TODO

        # Plot the Model
        # plot_model(model, train_list, test_list, lr_start)  # TODO

    print()


def main():
    train_data, valid_data, test_data, vocabulary_size = load_data(DATA_DIR)

    # evaluate_model(train_data, valid_data, test_data, RNN(nn.LSTM, vocabulary_size).to(device), 13, 1.0, 4, 2, 20)
    evaluate_model(train_data, valid_data, test_data, RNN(nn.LSTM, vocabulary_size, 0.5).to(device), 50, 1.0, 9, 1.4, 50, use_saved_weights=False)
    # evaluate_model(train_data, valid_data, test_data, RNN(nn.LSTM, vocabulary_size, 0.3).to(device), 13, 1.0, 8, 1.5, 50, use_saved_weights=False)

    # evaluate_model(train_data, valid_data, test_data, RNN(nn.GRU, vocabulary_size).to(device), 20, 0.782, 4, 2, 50)
    # for drop in np.arange(0.1,1,0.1):
    #     print(drop)
    #     evaluate_model(train_data, valid_data, test_data, RNN(nn.GRU, vocabulary_size, 0.2).to(device), 3, lr, 20, 2, use_saved_weights=False)
    # evaluate_model(train_data, valid_data, test_data, RNN(nn.GRU, vocabulary_size, 0.3).to(device), 50, 0.3, 5, 1.2, use_saved_weights=False)  # 114


if __name__ == '__main__':
    main()
