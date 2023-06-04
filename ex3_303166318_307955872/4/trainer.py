import torch
from models import GeneratorWGAN, DiscriminatorWGAN, GeneratorDCGAN, DiscriminatorDCGAN

DISCRIMINATOR_WEIGHT_CLIP = 0.1
CRITIC_ITERS = 5  # For WGAN number of critic iters per gen iter


def wgan_g_loss_fn(d_score_on_fake):
    return 0 - torch.mean(d_score_on_fake)


def wgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    return -1 * (torch.mean(d_score_on_real) - torch.mean(d_score_on_fake))  # -1 to make maximization problem to minimization problem


def dcgan_g_loss_fn(d_score_on_fake):
    return torch.mean(torch.log(1 - d_score_on_fake))


def dcgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    return -1 * (torch.mean(torch.log(d_score_on_real)) + torch.mean(torch.log(1 - d_score_on_fake)))  # -1 to make maximization problem to minimization problem


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


def d_train_batch(device, dataloader, g, d, optimizer, X):
    d.train()
    g.eval()

    noise = torch.randn((dataloader.batch_size, g.z_size)).to(device)
    score_on_fake = d(g(noise))

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


def train_epoch(device, dataloader, g, d, g_optimizer, d_optimizer):
    for i, (X, _) in enumerate(dataloader):
        d_train_batch(device, dataloader, g, d, d_optimizer, X)
        if i > 0 and i % CRITIC_ITERS == 0:
            g_train_batch(device, dataloader, g, d, g_optimizer)


def test_iteration(device, dataloader, g, d):
    g.eval()
    d.eval()

    g_loss = 0
    d_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            noise = torch.randn((dataloader.batch_size, g.z_size)).to(device)
            score_on_fake = d(g(noise))

            X = X.to(device)
            score_on_real = d(X)

            if isinstance(g, GeneratorWGAN) and isinstance(d, DiscriminatorWGAN):
                g_loss += wgan_g_loss_fn(score_on_fake)
                d_loss += wgan_d_loss_fn(score_on_fake, score_on_real)
            elif isinstance(g, GeneratorDCGAN) and isinstance(d, DiscriminatorDCGAN):
                g_loss += dcgan_g_loss_fn(score_on_fake)
                d_loss += dcgan_d_loss_fn(score_on_fake, score_on_real)
            else:
                raise ValueError("Generator / Discriminator loss function error!")

    g_loss /= len(dataloader)
    d_loss /= len(dataloader)
    return g_loss, d_loss
