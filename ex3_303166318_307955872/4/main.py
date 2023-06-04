import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import GeneratorWGAN, GeneratorDCGAN, DiscriminatorWGAN, DiscriminatorDCGAN
from trainer import train_epoch, test_iteration

BATCH_SIZE = 64
PRINT_EVERY = 1
EPOCHS = 200

DIM = 64
Z_SIZE = 128

MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    training_data = torch.utils.data.Subset(training_data, torch.arange(5000))  # TODO
    test_data = dataset(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data


def init_weights(layer):  # TODO remove?
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # best initialization even though activation is softplus


def train(train_dataset, test_dataset, g, d, use_saved_weights):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    g_save_path = os.path.join(MODELS_DIR, g.name + '.pkl')
    d_save_path = os.path.join(MODELS_DIR, d.name + '.pkl')

    if use_saved_weights and (os.path.exists(g_save_path) and os.path.exists(d_save_path)):
        # Load the models
        print('Loading old weights from "' + g_save_path + '" and "' + d_save_path + '"')
        g.load_state_dict(torch.load(g_save_path, map_location=torch.device(device)))
        d.load_state_dict(torch.load(d_save_path, map_location=torch.device(device)))
        g_test_loss, d_test_loss = test_iteration(device, test_dataloader, g, d)
        print("Loaded models - Test Loss: ({:>0.3f}, {:>0.3f})".format(
            g_test_loss, d_test_loss))

    else:
        print('Training...')
        g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
        d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0001)

        train_list = []
        test_list = []
        for epoch in range(EPOCHS):
            train_epoch(device, train_dataloader, g, d, g_optimizer, d_optimizer)
            train_list.append(test_iteration(device, train_dataloader, g, d))
            test_list.append(test_iteration(device, test_dataloader, g, d))
            if epoch % PRINT_EVERY == 0:
                print("Epoch {:>4}, Train Loss: (G: {:>+10.5f}, D: {:>+10.5f}), Test Loss: (G: {:>+10.5f}, D: {:>+10.5f})".format(
                    epoch + 1, train_list[-1][0], train_list[-1][1], test_list[-1][0], test_list[-1][1]))

            plt.title('Epoch #' + str(epoch + 1))
            noise = torch.randn((100, g.z_size)).to(device)
            img = g(noise).reshape((-1, 1, 28, 28))

            grid = torchvision.utils.make_grid(img, nrow=10).cpu().detach().numpy()
            grid = np.transpose(grid, (1, 2, 0))
            plt.imshow(grid, cmap='gray')
            plt.show()

        # Save the models
        print('Saving models to "' + g_save_path + '" and "' + d_save_path + '"')
        torch.save(g.state_dict(), g_save_path)  # TODO
        torch.save(d.state_dict(), d_save_path)  # TODO


def main():
    training_data, test_data = load_data(datasets.FashionMNIST)
    g = GeneratorWGAN(DIM, Z_SIZE).to(device)
    d = DiscriminatorWGAN(DIM).to(device)
    train(training_data, test_data, g, d, use_saved_weights=False)

    print('#' * 50)
    print()

    # training_data, test_data = load_data(datasets.FashionMNIST)
    # for labeled_samples in [100, 600, 1000, 3000]:
    #     train(training_data, test_data, labeled_samples, use_saved_weights=True)


if __name__ == '__main__':
    main()
