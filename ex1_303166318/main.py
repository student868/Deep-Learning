# We used the pytorch Quickstart guide: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# TODO add ID
import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import Lenet5, Lenet5BatchNorm, Lenet5Dropout
from trainer import train, test
import matplotlib.pyplot as plt

BATCH_SIZE = 64
PRINT_EVERY = 100
EPOCHS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    # Download training data
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_dataloader, test_dataloader


def plot_model(model, train_list, test_list):
    plt.title(model.name + ' accuracy')
    plt.plot(train_list, 'blue', label='Train data accuracy')
    plt.plot(test_list, 'red', label='Test data accuracy')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def evaluate_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, use_saved_weights):
    print(' -- ' + model.name + ' -- ')

    train_correct_list = []
    test_correct_list = []

    if use_saved_weights and os.path.exists(model.name + '.pkl'):
        print('Loading old weights...')
        model.load_state_dict(torch.load(model.name + '.pkl'))
        train_correct, _ = test(device, train_dataloader, model, loss_fn)
        test_correct, _ = test(device, test_dataloader, model, loss_fn)
        print(f"Loaded Model - Train Accuracy: {train_correct * 100 :>0.1f}%, Test Accuracy: {test_correct * 100:>0.1f}%")

    else:
        print('Training Model...')
        for t in range(EPOCHS):
            train(device, train_dataloader, model, loss_fn, optimizer)
            train_correct, _ = test(device, train_dataloader, model, loss_fn)
            train_correct *= 100
            train_correct_list.append(train_correct)
            test_correct, _ = test(device, test_dataloader, model, loss_fn)
            test_correct *= 100
            test_correct_list.append(test_correct)
            print(f"Epoch {t + 1} - Train Accuracy: {train_correct :>0.1f}%, Test Accuracy: {test_correct :>0.1f}%")

        # Save the Model
        torch.save(model.state_dict(), model.name + '.pkl')

        # Plot the Model
        plot_model(model, train_correct_list, test_correct_list)

    print()


def train_original(train_dataloader, test_dataloader, use_saved_weights):
    loss_fn = nn.CrossEntropyLoss()
    model = Lenet5().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    evaluate_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, use_saved_weights)


def train_dropout(train_dataloader, test_dataloader, use_saved_weights):
    loss_fn = nn.CrossEntropyLoss()
    model = Lenet5Dropout().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    evaluate_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, use_saved_weights)


def train_weight_decay(train_dataloader, test_dataloader, use_saved_weights):
    loss_fn = nn.CrossEntropyLoss()
    model = Lenet5()
    model.name = 'Lenet5 with weight decay'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
    evaluate_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, use_saved_weights)


def train_batch_normalization(train_dataloader, test_dataloader, use_saved_weights):
    loss_fn = nn.CrossEntropyLoss()
    model = Lenet5BatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    evaluate_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, use_saved_weights)


def main():
    train_dataloader, test_dataloader = load_data()

    train_original(train_dataloader, test_dataloader, use_saved_weights=True)
    train_dropout(train_dataloader, test_dataloader, use_saved_weights=True)
    train_weight_decay(train_dataloader, test_dataloader, use_saved_weights=True)
    train_batch_normalization(train_dataloader, test_dataloader, use_saved_weights=True)


if __name__ == '__main__':
    main()
