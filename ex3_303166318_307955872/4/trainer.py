import torch
from models import GeneratorWGAN, DiscriminatorWGAN, GeneratorDCGAN, DiscriminatorDCGAN

DISCRIMINATOR_WEIGHT_CLIP = 0.01


def wgan_g_loss_fn(d_score_on_fake):
    return 0 - torch.mean(d_score_on_fake)


def wgan_d_loss_fn(d_score_on_fake, d_score_on_real):
    return -1 * (torch.mean(d_score_on_real) - torch.mean(d_score_on_fake))  # -1 to make maximization problem to minimization problem


# def dcgan_g_loss_fn(d_score_on_fake):
#     return torch.mean(torch.log(1 - d_score_on_fake))
#
#
# def dcgan_d_loss_fn(d_score_on_fake, d_score_on_real):
#     return -1 * (torch.mean(torch.log(d_score_on_real)) + torch.mean(torch.log(1 - d_score_on_fake)))  # -1 to make maximization problem to minimization problem


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

    return loss.item()
