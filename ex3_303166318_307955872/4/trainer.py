import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from torch import nn

from models import GeneratorWGAN, DiscriminatorWGAN, GeneratorDCGAN, DiscriminatorDCGAN

DISCRIMINATOR_WEIGHT_CLIP = 0.01
DISCRIMINATOR_ITERATIONS = 5  # For WGAN, number of discriminator iterations per generator iterations

PRINT_EVERY = 500


def wgan_g_loss_fn(d_score_on_fake):
    return 0 - torch.mean(d_score_on_fake)


def wgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    return -1 * (torch.mean(d_score_on_real) - torch.mean(d_score_on_fake))  # -1 to make maximization problem to minimization problem


def dcgan_g_loss_fn(d_score_on_fake):
    return nn.BCEWithLogitsLoss()(d_score_on_fake, torch.full_like(d_score_on_fake, 1))


def dcgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    fake_loss = nn.BCEWithLogitsLoss()(d_score_on_fake, torch.full_like(d_score_on_fake, 0))  # 0 for fake
    real_loss = nn.BCEWithLogitsLoss()(d_score_on_real, torch.full_like(d_score_on_real, 1))  # 1 for real
    return (fake_loss + real_loss) / 2


def g_train_batch(device, dataloader, g, d, optimizer):
    g.train()
    d.eval()

    noise = torch.randn((dataloader.batch_size, g.z_size)).to(device)
    score_on_fake = d(g(noise))

    if isinstance(g, GeneratorWGAN):
        loss = wgan_g_loss_fn(score_on_fake)
    elif isinstance(g, GeneratorDCGAN):
        loss = dcgan_g_loss_fn(score_on_fake)
    else:
        raise ValueError("Generator loss function error!")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def plot_samples(device, g, title, n_samples=25, n_rows=5):
    g.eval()
    plt.title(title)
    noise = torch.randn((n_samples, g.z_size)).to(device)
    img = g(noise).reshape((-1, 1, 28, 28))

    grid = torchvision.utils.make_grid(img, nrow=n_rows).cpu().detach().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_one_sample(device, g, title):
    plot_samples(device, g, title, n_samples=1, n_rows=1)


def d_train_batch(device, dataloader, g, d, optimizer, X):
    d.train()
    g.eval()

    noise = torch.randn((dataloader.batch_size, g.z_size)).to(device)
    score_on_fake = d(g(noise).detach())

    X = X.to(device)
    score_on_real = d(X)

    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        loss = wgan_d_loss_fn(score_on_fake, score_on_real)
    elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
        loss = dcgan_d_loss_fn(score_on_fake, score_on_real)
    else:
        raise ValueError("Discriminator loss function error!")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Weight clipping
    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        with torch.no_grad():
            for param in d.parameters():
                param.clamp_(-DISCRIMINATOR_WEIGHT_CLIP, DISCRIMINATOR_WEIGHT_CLIP)

    return loss.item()


def train_models(device, dataloader, g, d, epochs, g_save_path, d_save_path, save=False):
    print('Training...')

    g_loss_list = []
    d_loss_list = []

    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        g_optimizer = torch.optim.RMSprop(g.parameters(), lr=5e-5)
        d_optimizer = torch.optim.RMSprop(d.parameters(), lr=5e-5)
    elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
        betas = (0.5, 0.999)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=2e-4, betas=betas)
        d_optimizer = torch.optim.Adam(d.parameters(), lr=2e-4, betas=betas)
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

            if ((iteration + 1) % PRINT_EVERY == 0 or iteration + 1 == len(dataloader)) and (len(g_loss_list) > 0) and (len(d_loss_list) > 0):
                print("Epoch [{:>4}/{:>4}] Iteration [{:>4}/{:>4}] - Training Loss: (G: {:>+10.5f}, D: {:>+10.5f})".format(
                    epoch + 1, epochs, iteration + 1, len(dataloader), g_loss_list[-1], d_loss_list[-1]))  # fix g_loss_list is empty for printevery = 1

        plot_samples(device, g, g.mode + ' Samples - Epoch #' + str(epoch + 1))

    # Save the models
    if save:
        print('Saving models to "' + g_save_path + '" and "' + d_save_path + '"')
        torch.save(g.state_dict(), g_save_path)
        torch.save(d.state_dict(), d_save_path)

    return g_loss_list, d_loss_list
