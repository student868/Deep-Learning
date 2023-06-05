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
from trainer import g_train_batch, d_train_batch

BATCH_SIZE = 64
PRINT_EVERY = 200
EPOCHS = 200

DIM = 64
Z_SIZE = 128
DISCRIMINATOR_ITERATIONS = 5  # For WGAN, number of discriminator iterations per generator iterations

MODELS_DIR = 'models'

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)


def plot_loss(lost_list1, lost_list2=None, labels=('W-GAN', 'DC-GAN')):
    if lost_list1 is None:
        raise ValueError('First loss list is None, you must train at least one model to plot the loss')
    if lost_list2 is None:
        print('Second model was not trained, plotting only one loss list')

    plt.title('W-GAN vs DC-GAN Discriminator Loss')
    x = [i + 1 for i in range(len(lost_list1))]
    plt.plot(x, lost_list1, 'blue', label=labels[0])
    if lost_list2 is not None:
        plt.plot(x, lost_list2, 'red', label=labels[1])
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xlim(left=1)
    plt.show()


def plot_samples(g, title, n_samples=100, n_rows=10):
    plt.title(title)
    noise = torch.randn((n_samples, g.z_size)).to(device)
    img = g(noise).reshape((-1, 1, 28, 28))

    grid = torchvision.utils.make_grid(img, nrow=n_rows).cpu().detach().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_one_sample(g, title):
    plot_samples(g, title, n_samples=1, n_rows=1)


def load_models(g, d, g_save_path, d_save_path):
    print('Loading old weights from "' + g_save_path + '" and "' + d_save_path + '"')
    g.load_state_dict(torch.load(g_save_path, map_location=torch.device(device)))
    d.load_state_dict(torch.load(d_save_path, map_location=torch.device(device)))
    print("Loaded models!")


def train_models(dataloader, g, d, epochs, g_save_path, d_save_path, save=False):
    print('Training...')

    g_loss_list = []
    d_loss_list = []

    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        g_optimizer = torch.optim.RMSprop(g.parameters(), lr=5e-5)
        d_optimizer = torch.optim.RMSprop(d.parameters(), lr=5e-5)
    elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
        betas = (0.5, 0.9)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=1e-4, betas=betas)
        d_optimizer = torch.optim.Adam(d.parameters(), lr=1e-4, betas=betas)
    else:
        raise ValueError("Generator / Discriminator Error!")

    for epoch in range(epochs):
        for iteration, (X, _) in enumerate(dataloader):
            d_loss_list.append(d_train_batch(device, dataloader, g, d, d_optimizer, X))

            if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):  # use DISCRIMINATOR_ITERATIONS for WGAN
                if iteration > 0 and iteration % DISCRIMINATOR_ITERATIONS == 0:
                    g_loss_list.append(g_train_batch(device, dataloader, g, d, g_optimizer))
            else:  # train generator after every iteration of the discriminator
                g_loss_list.append(g_train_batch(device, dataloader, g, d, g_optimizer))

            if (iteration + 1) % PRINT_EVERY == 0:
                print("Epoch [{:>4}/{:>4}] Iteration [{:>4}/{:>4}] - Training Loss: (G: {:>+10.5f}, D: {:>+10.5f})".format(
                    epoch + 1, epochs, iteration + 1, len(dataloader), g_loss_list[-1], d_loss_list[-1]))

        plot_samples(g, g.mode + ' Samples - Epoch #' + str(epoch + 1))

    # Save the models
    if save:
        print('Saving models to "' + g_save_path + '" and "' + d_save_path + '"')
        torch.save(g.state_dict(), g_save_path)
        torch.save(d.state_dict(), d_save_path)

    return g_loss_list, d_loss_list


def update_models(dataloader, g, d, epochs, use_saved_weights):
    g_save_path = os.path.join(MODELS_DIR, g.name + '.pkl')
    d_save_path = os.path.join(MODELS_DIR, d.name + '.pkl')

    g.apply(init_weights)
    d.apply(init_weights)

    d_loss_list = None

    if use_saved_weights and (os.path.exists(g_save_path) and os.path.exists(d_save_path)):
        load_models(g, d, g_save_path, d_save_path)
    else:
        g_loss_list, d_loss_list = train_models(dataloader, g, d, epochs, g_save_path, d_save_path)

    plot_samples(g, 'Samples')
    return d_loss_list


def main():
    dataloader = load_data(datasets.FashionMNIST)

    g = GeneratorWGAN(DIM, Z_SIZE).to(device)
    d = DiscriminatorWGAN(DIM).to(device)
    wgan_d_loss = update_models(dataloader, g, d, epochs=EPOCHS, use_saved_weights=False)

    print('#' * 50)
    print()

    g = GeneratorDCGAN(DIM, Z_SIZE).to(device)
    d = DiscriminatorDCGAN(DIM).to(device)
    dcgan_d_loss = update_models(dataloader, g, d, epochs=EPOCHS, use_saved_weights=False)

    plot_loss(wgan_d_loss, dcgan_d_loss)


if __name__ == '__main__':
    main()
