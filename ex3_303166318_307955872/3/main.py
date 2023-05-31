import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import *
from trainer import train_epoch, test_epoch

EPOCHS = 100
HIDDEN_SIZE = 600
Z_SIZE = 50
BATCH_SIZE = 64
PRINT_EVERY = 5
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'  # TODO


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    training_data = torch.utils.data.Subset(training_data, torch.randperm(len(training_data))[:3000])  # TODO
    test_data = dataset(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data


def train(training_data, test_data, labeled_samples, nn_epochs, use_saved_weights):
    # Train VAE
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = Vae(
        training_data[0][0].flatten().size(0),
        HIDDEN_SIZE,
        Z_SIZE,
        training_data.__class__.__name__ + '_' + str(labeled_samples)
    ).to(device)
    loss_fns = {'reconstruction': nn.BCELoss(), 'z': nn.KLDivLoss}
    optimizer = torch.optim.Adam(model.parameters())

    if use_saved_weights and os.path.exists(os.path.join(MODELS_DIR, model.name + '_NN.pkl')):
        print('Loading old weights...')
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, model.name + '_NN.pkl')))
        test_loss = test_epoch(device, test_dataloader, model, loss_fns)
        print(f"Loaded Model - Test Loss: {test_loss :>0.3f}%")

    else:
        print('Training Model...')
        for t in range(nn_epochs):
            train_epoch(device, train_dataloader, model, loss_fns, optimizer)
            train_loss = test_epoch(device, train_dataloader, model, loss_fns)
            test_loss = test_epoch(device, test_dataloader, model, loss_fns)
            if t % PRINT_EVERY == 0 or t == nn_epochs - 1:
                print(f"Epoch {t + 1} - Train Loss: {train_loss :>0.3f}, Test Loss: {test_loss :>0.3f}")

        # Save the Model
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, model.name + '_NN.pkl'))  # TODO

    # Train SVM
    training_data_subset = labeled_samples  # TODO
    train_dataloader_without_labels = DataLoader(training_data, batch_size=BATCH_SIZE)


def main():
    training_data, test_data = load_data(datasets.MNIST)
    for labeled_samples in [100]:
    # for labeled_samples in [100, 600, 1000, 3000]:  TODO
        train(training_data, test_data, labeled_samples, EPOCHS, use_saved_weights=False)

    exit()
    training_data, test_data = load_data(datasets.FashionMNIST)
    for labeled_samples in [100, 600, 1000, 3000]:
        train(training_data, test_data, labeled_samples, EPOCHS, use_saved_weights=False)


if __name__ == '__main__':
    main()
