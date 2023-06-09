import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from torch import autograd
from torch import nn

from models import GeneratorWGAN, DiscriminatorWGAN, GeneratorDCGAN, DiscriminatorDCGAN

LAMBDA = 10  # Gradient penalty lambda hyperparameter
DISCRIMINATOR_ITERATIONS = 5  # For WGAN, number of discriminator iterations per generator iterations

PRINT_EVERY = 500


def wgan_g_loss_fn(d_score_on_fake):
    return 0 - torch.mean(d_score_on_fake)


def gradient_penalty(device, d, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1).to(device)

    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    interpolates.requires_grad = True

    d_score_on_interpolates = d(interpolates)

    gradients = autograd.grad(outputs=d_score_on_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(d_score_on_interpolates.shape).to(device),
                              create_graph=True, retain_graph=True)[0]
    slopes = gradients.norm(2, dim=1)
    return ((slopes - 1) ** 2).mean()


def wgan_d_loss_fn(device, d, real_data, fake_data, d_score_on_fake, d_score_on_real):
    return -1 * (torch.mean(d_score_on_real) - torch.mean(d_score_on_fake)) + gradient_penalty(device, d, real_data, fake_data) * LAMBDA
    # -1 to make maximization problem to minimization problem


def dcgan_g_loss_fn(d_score_on_fake):
    return nn.BCEWithLogitsLoss()(d_score_on_fake, torch.full_like(d_score_on_fake, 1))  # 1 to train log(x) rather than log(1-x)


def dcgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    fake_loss = nn.BCEWithLogitsLoss()(d_score_on_fake, torch.full_like(d_score_on_fake, 0))  # 0 for fake
    real_loss = nn.BCEWithLogitsLoss()(d_score_on_real, torch.full_like(d_score_on_real, 1))  # 1 for real
    return (fake_loss + real_loss) / 2  # from official implementation


def g_train_batch(device, g, d, batch_size, optimizer):
    g.train()
    d.eval()

    noise = torch.randn((batch_size, g.z_size)).to(device)
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


def plot_samples(samples, title):
    plt.title(title)
    n_rows = int(np.sqrt(samples.shape[0]))
    grid = torchvision.utils.make_grid(samples, nrow=n_rows).cpu().detach().numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_g_samples(device, g, title, n_samples=25):
    g.eval()
    noise = torch.randn((n_samples, g.z_size)).to(device)
    samples = g(noise).reshape((-1, 1, 28, 28))
    plot_samples(samples, title)


def plot_one_g_sample(device, g, title):
    plot_g_samples(device, g, title, n_samples=1)


def d_train_batch(device, g, d, real_data, optimizer):
    d.train()
    g.eval()

    noise = torch.randn((real_data.shape[0], g.z_size)).to(device)
    fake_data = g(noise).detach()
    score_on_fake = d(fake_data)

    real_data = real_data.to(device).detach()
    score_on_real = d(real_data)

    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        loss = wgan_d_loss_fn(device, d, real_data.flatten(1), fake_data, score_on_fake, score_on_real)
    elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
        loss = dcgan_d_loss_fn(score_on_fake, score_on_real)
    else:
        raise ValueError("Discriminator loss function error!")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_models(device, dataloader, g, d, epochs, g_save_path, d_save_path, save=False):
    print('Training...')

    g_loss_list = []
    d_loss_list = []

    if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
        betas = (0.5, 0.9)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=1e-4, betas=betas)
        d_optimizer = torch.optim.Adam(d.parameters(), lr=1e-4, betas=betas)
    elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
        betas = (0.5, 0.999)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=2e-4, betas=betas)
        d_optimizer = torch.optim.Adam(d.parameters(), lr=2e-4, betas=betas)
    else:
        raise ValueError("Generator / Discriminator Error!")

    for epoch in range(epochs):
        for iteration, (X, _) in enumerate(dataloader):
            d_loss_list.append(d_train_batch(device, g, d, X, d_optimizer))

            if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):  # use DISCRIMINATOR_ITERATIONS
                if iteration > 0 and iteration % DISCRIMINATOR_ITERATIONS == 0:
                    g_loss_list.append(g_train_batch(device, g, d, X.shape[0], g_optimizer))
            else:  # train generator after every iteration of the discriminator
                g_loss_list.append(g_train_batch(device, g, d, X.shape[0], g_optimizer))

            if ((iteration + 1) % PRINT_EVERY == 0 or iteration + 1 == len(dataloader)) and (len(g_loss_list) > 0) and (len(d_loss_list) > 0):
                print("Epoch [{:>4}/{:>4}] Iteration [{:>4}/{:>4}] - Training Loss: (G: {:>+10.5f}, D: {:>+10.5f})".format(
                    epoch + 1, epochs, iteration + 1, len(dataloader), g_loss_list[-1], d_loss_list[-1]))

        plot_g_samples(device, g, g.mode + ' Samples - Epoch #' + str(epoch + 1))

    # Save the models
    if save:
        print('Saving models to "' + g_save_path + '" and "' + d_save_path + '"')
        torch.save(g.state_dict(), g_save_path)
        torch.save(d.state_dict(), d_save_path)

    return g_loss_list, d_loss_list
