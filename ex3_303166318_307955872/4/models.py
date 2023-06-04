import torch
from torch import nn

OUTPUT_DIM = 28 * 28

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
leaky_relu = nn.LeakyReLU(negative_slope=0.2)


class Generator(nn.Module):
    def __init__(self, mode, model_name, dim, z_size):  # TODO find better names for dim, z_size
        super().__init__()
        self.mode = mode
        self.name = model_name
        self.dim = dim
        self.z_size = z_size

        self.fc = nn.Linear(z_size, 4 * 4 * 4 * dim)
        self.bn1 = nn.BatchNorm1d(4 * 4 * 4 * dim)
        self.deconv1 = nn.ConvTranspose2d(4 * dim, 2 * dim, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.deconv2 = nn.ConvTranspose2d(2 * dim, dim, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.deconv3 = nn.ConvTranspose2d(dim, 1, 4, stride=2, padding=1)

    def forward(self, noise):
        x = self.fc(noise)
        if self.mode == 'WGAN':
            x = self.bn1(x)
        x = relu(x)
        x = torch.reshape(x, (-1, 4 * self.dim, 4, 4))

        x = self.deconv1(x)
        if self.mode == 'WGAN':
            x = self.bn2(x)
        x = relu(x)

        x = x[:, :, :7, :7]

        x = self.deconv2(x)
        if self.mode == 'WGAN':
            x = self.bn3(x)
        x = relu(x)

        x = self.deconv3(x)
        x = sigmoid(x)

        return torch.reshape(x, (-1, OUTPUT_DIM))


class GeneratorWGAN(Generator):
    def __init__(self, dim, z_size):
        super(GeneratorWGAN, self).__init__('WGAN', self.__class__.__name__, dim, z_size)


class GeneratorDCGAN(Generator):
    def __init__(self, dim, z_size):
        super(GeneratorDCGAN, self).__init__('DCGAN', self.__class__.__name__, dim, z_size)


class Discriminator(nn.Module):
    def __init__(self, mode, model_name, dim):  # TODO find better names for dim
        super().__init__()
        self.mode = mode
        self.name = model_name
        self.dim = dim

        self.conv1 = nn.Conv2d(1, dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(2 * dim)
        self.conv3 = nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(4 * dim)
        self.fc = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 28, 28))

        x = self.conv1(x)
        x = leaky_relu(x)

        x = self.conv2(x)
        if self.mode == 'WGAN':
            x = self.bn1(x)
        x = leaky_relu(x)

        x = self.conv3(x)
        if self.mode == 'WGAN':
            x = self.bn2(x)
        x = leaky_relu(x)

        x = torch.reshape(x, (-1, 4 * 4 * 4 * self.dim))
        x = self.fc(x)

        return torch.reshape(x, (-1,))


class DiscriminatorWGAN(Discriminator):
    def __init__(self, dim):
        super(DiscriminatorWGAN, self).__init__('WGAN', self.__class__.__name__, dim)


class DiscriminatorDCGAN(Discriminator):
    def __init__(self, dim):
        super(DiscriminatorDCGAN, self).__init__('DCGAN', self.__class__.__name__, dim)
