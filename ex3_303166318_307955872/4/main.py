import os.path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from models import GeneratorWGAN, GeneratorDCGAN, DiscriminatorWGAN, DiscriminatorDCGAN
from trainer import train_models, plot_g_samples, plot_one_g_sample, plot_samples

BATCH_SIZE = 64
EPOCHS = 100

DIM = 64
Z_SIZE = 128

MODELS_DIR = 'models'

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(dataset):
    training_data = dataset(root="data", train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


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


def load_models(g, d, g_save_path, d_save_path):
    print('Loading old weights from "' + g_save_path + '" and "' + d_save_path + '"')
    g.load_state_dict(torch.load(g_save_path, map_location=torch.device(device)))
    d.load_state_dict(torch.load(d_save_path, map_location=torch.device(device)))
    print("Loaded models!")


def update_models(dataloader, g, d, epochs, use_saved_weights, save_trained_model=False):
    g_save_path = os.path.join(MODELS_DIR, g.name + '.pkl')
    d_save_path = os.path.join(MODELS_DIR, d.name + '.pkl')

    d_loss_list = None

    if use_saved_weights and (os.path.exists(g_save_path) and os.path.exists(d_save_path)):
        load_models(g, d, g_save_path, d_save_path)
    else:
        g_loss_list, d_loss_list = train_models(device, dataloader, g, d, epochs, g_save_path, d_save_path,
                                                save=save_trained_model)

    return d_loss_list


def main():
    dataloader = load_data(datasets.FashionMNIST)
    plot_samples(next(iter(dataloader))[0], 'Real samples')
    plot_samples(next(iter(dataloader))[0][0], 'Real sample')
    plot_samples(next(iter(dataloader))[0][0], 'Real sample')

    wgan_g = GeneratorWGAN(DIM, Z_SIZE).to(device)
    wgan_d = DiscriminatorWGAN(DIM).to(device)
    wgan_d_loss = update_models(dataloader, wgan_g, wgan_d, EPOCHS, use_saved_weights=False, save_trained_model=True)
    plot_g_samples(device, wgan_g, 'W-GAN')
    plot_one_g_sample(device, wgan_g, 'W-GAN')

    print('#' * 50)
    print()

    dcgan_g = GeneratorDCGAN(DIM, Z_SIZE).to(device)
    dcgan_d = DiscriminatorDCGAN(DIM).to(device)
    dcgan_d_loss = update_models(dataloader, dcgan_g, dcgan_d, EPOCHS, use_saved_weights=False, save_trained_model=True)
    plot_g_samples(device, dcgan_g, 'DC-GAN')
    plot_one_g_sample(device, dcgan_g, 'DC-GAN')

    plot_loss(wgan_d_loss, dcgan_d_loss)


if __name__ == '__main__':
    main()
