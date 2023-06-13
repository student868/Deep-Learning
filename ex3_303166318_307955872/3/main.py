import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import Vae
from trainer import train_epoch, test_epoch
from sklearn import svm
import pickle

TRAIN_LR_STOP = 1e-6
HIDDEN_SIZE = 600
Z_SIZE = 50
BATCH_SIZE = 64
PRINT_EVERY = 5
MODELS_DIR = 'models'

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    test_data = dataset(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # best initialization even though activation is soft-plus


def svm_train_data(train_dataset, labeled_samples, model_nn):
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

    # Get balanced classes
    x_train, y_train = next(iter(train_dataloader))
    x_train = x_train.flatten(1)
    num_classes = len(y_train.unique())
    indices_of_balanced_classes = torch.cat([(y_train == i).nonzero()[:(labeled_samples // num_classes)].flatten() for i in range(num_classes)])
    x_train = x_train[indices_of_balanced_classes].to(device)
    y_train = y_train[indices_of_balanced_classes]

    # Get latent representation
    mu, sigma, _ = model_nn(x_train)
    x_train = torch.cat([mu, sigma], dim=1)
    x_train = x_train.cpu()

    return x_train.detach().numpy(), y_train.detach().numpy()


def svm_test_data(test_dataset, model_nn):
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Get latent representation
    x_test, y_test = next(iter(test_dataloader))
    x_test = x_test.to(device)
    mu, sigma, _ = model_nn(x_test.flatten(1))
    x_test = torch.cat([mu, sigma], dim=1)
    x_test = x_test.cpu()

    return x_test.detach().numpy(), y_test.detach().numpy()


def train_nn(train_dataloader, test_dataloader, model, construction_loss_fn):
    model.apply(init_weights)

    print('Training NN model...')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10)

    epoch = 0
    while optimizer.param_groups[0]['lr'] > TRAIN_LR_STOP:
        train_epoch(device, train_dataloader, model, construction_loss_fn, optimizer)
        train_loss = test_epoch(device, train_dataloader, model, construction_loss_fn)
        test_loss = test_epoch(device, test_dataloader, model, construction_loss_fn)
        if epoch % PRINT_EVERY == 0:
            print("Epoch {}, LR: {:8.6f}, Train Loss: {:>0.3f}, Test Loss: {:>0.3f}".format(
                epoch + 1, optimizer.param_groups[0]['lr'], train_loss, test_loss))
        scheduler.step(test_loss)
        epoch += 1


def update_nn(train_dataset, test_dataset, use_saved_weights, save=False):
    model_nn = Vae(
        train_dataset[0][0].flatten().size(0),
        HIDDEN_SIZE,
        Z_SIZE,
        train_dataset.__class__.__name__ + '_NN'
    ).to(device)

    save_path = os.path.join(MODELS_DIR, model_nn.name + '.pkl')
    construction_loss_fn = nn.BCELoss(reduction='sum')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if use_saved_weights and os.path.exists(save_path):
        print('Loading NN weights from "' + save_path + '"')
        model_nn.load_state_dict(torch.load(save_path, map_location=torch.device(device)))
    else:
        train_nn(train_dataloader, test_dataloader, model_nn, construction_loss_fn)
        if save:
            print('Saving NN model to "' + save_path + '"')
            torch.save(model_nn.state_dict(), save_path)

    test_loss = test_epoch(device, test_dataloader, model_nn, construction_loss_fn)
    print("NN Test Loss: {:>0.3f}".format(test_loss))
    print()

    return model_nn


def update_svm(train_dataset, labeled_samples, model_nn, use_saved_weights, save=False):
    save_path = os.path.join(MODELS_DIR, train_dataset.__class__.__name__ + '_SVM' + '_' + str(labeled_samples) + '.pkl')

    if use_saved_weights and os.path.exists(save_path):
        print('Loading SVM weights from "' + save_path + '"')
        svm_classifier = pickle.load(open(save_path, 'rb'))
    else:
        print('Training SVM model...')
        svm_classifier = svm.SVC(kernel='poly', degree=5)
        x_train, y_train = svm_train_data(train_dataset, labeled_samples, model_nn)
        svm_classifier.fit(x_train, y_train)
        if save:
            print('Saving SVM model to "' + save_path + '"')
            pickle.dump(svm_classifier, open(save_path, 'wb'))

    return svm_classifier


def evaluate(test_dataset, model_nn, model_svm):
    x_test, y_test = svm_test_data(test_dataset, model_nn)
    pred: np.ndarray = model_svm.predict(x_test)
    accuracy = 100 * (1 - (pred == y_test).mean())
    print('Accuracy Error: {:>0.2f}%'.format(accuracy))
    print()


def main(use_saved_weights=True):
    train_dataset, test_dataset = load_data(datasets.MNIST)
    model_nn = update_nn(train_dataset, test_dataset, use_saved_weights=use_saved_weights)
    for labeled_samples in [100, 600, 1000, 3000]:
        model_svm = update_svm(train_dataset, labeled_samples, model_nn, use_saved_weights=use_saved_weights)
        evaluate(test_dataset, model_nn, model_svm)

    print('#' * 50)
    print()

    train_dataset, test_dataset = load_data(datasets.FashionMNIST)
    model_nn = update_nn(train_dataset, test_dataset, use_saved_weights=use_saved_weights)
    for labeled_samples in [100, 600, 1000, 3000]:
        model_svm = update_svm(train_dataset, labeled_samples, model_nn, use_saved_weights=use_saved_weights)
        evaluate(test_dataset, model_nn, model_svm)


if __name__ == '__main__':
    main(False)
